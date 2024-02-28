// Author: Wen-jie Lu on 2021/9/14.
#include "cheetah/cheetah-api.h"

#include <seal/seal.h>

#include "gemini/cheetah/shape_inference.h"
#include "gemini/cheetah/tensor_encoder.h"
#include "utils/constants.h"  // ALICE & BOB
#include "utils/net_io_channel.h"
#include "globals.h"
#define LOG_LAYERWISE

#define Arr1DIdxRowM(arr, s0, i) (*((arr) + (i)))
#define Arr2DIdxRowM(arr, s0, s1, i, j) (*((arr) + (i) * (s1) + (j)))
#define Arr3DIdxRowM(arr, s0, s1, s2, i, j, k)                                 \
  (*((arr) + (i) * (s1) * (s2) + (j) * (s2) + (k)))
#define Arr4DIdxRowM(arr, s0, s1, s2, s3, i, j, k, l)                          \
  (*((arr) + (i) * (s1) * (s2) * (s3) + (j) * (s2) * (s3) + (k) * (s3) + (l)))
#define Arr5DIdxRowM(arr, s0, s1, s2, s3, s4, i, j, k, l, m)                   \
  (*((arr) + (i) * (s1) * (s2) * (s3) * (s4) + (j) * (s2) * (s3) * (s4) +      \
     (k) * (s3) * (s4) + (l) * (s4) + (m)))

#define Arr2DIdxColM(arr, s0, s1, i, j) (*((arr) + (j) * (s0) + (i)))

extern uint64_t Conv_sepOfflineTimeInMilliSec;
extern uint64_t Conv_sepOnlineTimeInMilliSec;
extern uint64_t Conv_sepOfflineCommSent;
extern uint64_t Conv_sepOnlineCommSent;
extern uint64_t Matmul_sepOfflineTimeInMilliSec;
extern uint64_t Matmul_sepOnlineTimeInMilliSec;
extern uint64_t Matmul_sepOfflineCommSent;
extern uint64_t Matmul_sepOnlineCommSent;
extern uint64_t BN_sepOfflineTimeInMilliSec;
extern uint64_t BN_sepOnlineTimeInMilliSec;
extern uint64_t BN_sepOfflineCommSent;
extern uint64_t BN_sepOnlineCommSent;
extern uint64_t FusedBN_ReLUOfflineTimeInMilliSec;
extern uint64_t FusedBN_ReLUOnlineTimeInMilliSec;
extern uint64_t FusedBN_ReLUOfflineCommSent;
extern uint64_t FusedBN_ReLUOnlineCommSent;
uint64_t mask_r = (uint64_t)((1ULL << 16) - 1);

template <class CtType>
void send_ciphertext(sci::NetIO *io, const CtType &ct) {
  std::stringstream os;
  uint64_t ct_size;
  ct.save(os);
  ct_size = os.tellp();
  string ct_ser = os.str();
  io->send_data(&ct_size, sizeof(uint64_t));
  io->send_data(ct_ser.c_str(), ct_ser.size());
}

template <class EncVecCtType>
static void send_encrypted_vector(sci::NetIO *io, const EncVecCtType &ct_vec) {
  uint32_t ncts = ct_vec.size();
  io->send_data(&ncts, sizeof(uint32_t));
  for (size_t i = 0; i < ncts; ++i) {
    send_ciphertext(io, ct_vec.at(i));
  }
}

static void recv_encrypted_vector(sci::NetIO *io,
                                  const seal::SEALContext &context,
                                  std::vector<seal::Ciphertext> &ct_vec,
                                  bool is_truncated = false);
static void recv_ciphertext(sci::NetIO *io, const seal::SEALContext &context,
                            seal::Ciphertext &ct, bool is_truncated = false);
void decode_tensor(const gemini::Tensor<uint64_t> &in_tensor, uint64_t *out, int C, int H, int W, int bitlength);
void print_shape(const gemini::Tensor<uint64_t> &tensor);
void print_tensor(const gemini::Tensor<uint64_t> &tensor);
static Code LaunchWorks(gemini::ThreadPool &tpool, size_t num_works,
                        std::function<Code(long wid, size_t start, size_t end)> program);

namespace gemini {

TensorShape GetConv2DOutShape(const HomConv2DSS::Meta &meta) {
  auto o = shape_inference::Conv2D(meta.ishape, meta.fshape, meta.padding,
                                   meta.stride);
  if (!o) {
    printf("GetConv2DOutShape failed\n");
    return TensorShape({0, 0, 0});
  }
  o->Update(0, meta.n_filters);
  return *o;
}

uint64_t CheetahLinear::io_counter() const { return io_ ? io_->counter : 0; }

int64_t CheetahLinear::get_signed(uint64_t x) const {
  if (x >= base_mod_) {
    LOG(FATAL) << "CheetahLinear::get_signed input out-of-bound";
  }

  // [-2^{k-1}, 2^{k-1})
  if (x > positive_upper_)
    return static_cast<int64_t>(x - base_mod_);
  else
    return static_cast<int64_t>(x);
}

uint64_t CheetahLinear::reduce(uint64_t x) const {
  if (barrett_reducer_) {
    return seal::util::barrett_reduce_64(x, *barrett_reducer_);
  } else {
    return x & mod_mask_;
  }
}

CheetahLinear::CheetahLinear(int party, sci::NetIO *io, uint64_t base_mod,
                             size_t nthreads)
    : party_(party), io_(io), nthreads_(nthreads), base_mod_(base_mod) {
  if (base_mod < 2ULL || (int)std::log2(base_mod) >= 45) {
    throw std::logic_error("CheetahLinear: base_mod out-of-bound [2, 2^45)");
  }

  const bool is_mod_2k = IsTwoPower(base_mod_);

  if (is_mod_2k) {
    mod_mask_ = base_mod_ - 1;
    positive_upper_ = base_mod_ / 2;
  } else {
    barrett_reducer_ = seal::Modulus(base_mod_);
    // [0, 7) -> (-4, 3]
    // [0, 8) -> [-4, 4]
    // [0, odd) -> [-floor(odd/2), floor(odd/2)]
    positive_upper_ = (base_mod_ + 1) >> 1;
  }

  const uint64_t plain_mod = base_mod;  // [0, 2^k)

  using namespace seal;
  EncryptionParameters seal_parms(scheme_type::bfv);
  seal_parms.set_n_special_primes(0);
  // We are not exporting the pk/ct with more than 109-bit.
  std::vector<int> moduli_bits{60, 49};

  seal_parms.set_poly_modulus_degree(4096);
  seal_parms.set_coeff_modulus(CoeffModulus::Create(4096, moduli_bits));
  seal_parms.set_plain_modulus(plain_mod);
  context_ =
      std::make_shared<SEALContext>(seal_parms, true, sec_level_type::tc128);

  if (party == sci::BOB) {
    // Bob generate keys
    KeyGenerator keygen(*context_);
    // Keep secret key
    sk_ = std::make_shared<SecretKey>(keygen.secret_key());
    // Send public key
    Serializable<PublicKey> s_pk = keygen.create_public_key();

    std::stringstream os;
    s_pk.save(os);
    uint64_t pk_sze = static_cast<uint64_t>(os.tellp());
    const std::string &keys_str = os.str();

    io_->send_data(&pk_sze, sizeof(uint64_t));
    io_->send_data(keys_str.c_str(), pk_sze);

    conv2d_impl_.setUp(*context_, *sk_);
    fc_impl_.setUp(*context_, *sk_);
    bn_impl_.setUp(base_mod, *context_, *sk_);
  } else {
    pk_ = std::make_shared<PublicKey>();

    uint64_t pk_sze{0};
    io_->recv_data(&pk_sze, sizeof(uint64_t));
    char *key_buf = new char[pk_sze];
    io_->recv_data(key_buf, pk_sze);
    std::stringstream is;
    is.write(key_buf, pk_sze);
    pk_->load(*context_, is);
    delete[] key_buf;

    conv2d_impl_.setUp(*context_, std::nullopt, pk_);
    fc_impl_.setUp(*context_, std::nullopt, pk_);
    bn_impl_.setUp(base_mod, *context_, std::nullopt, pk_);
  }

  if (is_mod_2k) {
    setUpForBN();
  } else {
    std::vector<seal::SEALContext> bn_context{*context_};
    std::vector<std::optional<SecretKey>> bn_opt_sk;
    Code ok;
    if (party == sci::BOB) {
      bn_opt_sk.push_back(*sk_);
      ok = bn_impl_.setUp(plain_mod, bn_context, bn_opt_sk, {});
    } else {
      ok = bn_impl_.setUp(plain_mod, bn_context, bn_opt_sk, {});
    }

    if (ok != Code::OK) {
      throw std::runtime_error("BN setUP failed " + CodeMessage(ok));
    }
  }
}

void CheetahLinear::setUpForBN() {
  using namespace seal;
  size_t ntarget_bits = std::ceil(std::log2(base_mod_));
  size_t crt_bits = 2 * ntarget_bits + 1 + HomBNSS::kStatBits;

  const size_t nbits_per_crt_plain = [](size_t crt_bits) {
    constexpr size_t kMaxCRTPrime = 50;
    for (size_t nCRT = 1;; ++nCRT) {
      size_t np = CeilDiv(crt_bits, nCRT);
      if (np <= kMaxCRTPrime) return np;
    }
  }(crt_bits + 1);

  const size_t nCRT = CeilDiv<size_t>(crt_bits, nbits_per_crt_plain);
  std::vector<int> crt_primes_bits(nCRT, nbits_per_crt_plain);

  const size_t N = 4096;
  auto plain_crts = CoeffModulus::Create(N, crt_primes_bits);
  EncryptionParameters seal_parms(scheme_type::bfv);
  seal_parms.set_n_special_primes(0);
  // We are not exporting the pk/ct with more than 109-bit.
  std::vector<int> cipher_moduli_bits{60, 49};
  seal_parms.set_poly_modulus_degree(N);
  seal_parms.set_coeff_modulus(CoeffModulus::Create(N, cipher_moduli_bits));

  bn_contexts_.resize(nCRT);
  for (size_t i = 0; i < nCRT; ++i) {
    seal_parms.set_plain_modulus(plain_crts[i]);
    bn_contexts_[i] =
        std::make_shared<SEALContext>(seal_parms, true, sec_level_type::tc128);
  }

  std::vector<seal::SEALContext> contexts;
  std::vector<std::optional<SecretKey>> opt_sks;
  if (party_ == sci::BOB) {
    bn_sks_.resize(nCRT);
    for (size_t i = 0; i < nCRT; ++i) {
      KeyGenerator keygen(*bn_contexts_[i]);
      // Keep secret key
      bn_sks_[i] = std::make_shared<SecretKey>(keygen.secret_key());
      // Send public key
      Serializable<PublicKey> s_pk = keygen.create_public_key();

      std::stringstream os;
      s_pk.save(os);
      uint64_t pk_sze = static_cast<uint64_t>(os.tellp());
      const std::string &keys_str = os.str();

      io_->send_data(&pk_sze, sizeof(uint64_t));
      io_->send_data(keys_str.c_str(), pk_sze);
      contexts.emplace_back(*bn_contexts_[i]);
      opt_sks.emplace_back(*bn_sks_[i]);
    }
    auto code = bn_impl_.setUp(base_mod_, contexts, opt_sks, {});
    if (code != Code::OK) {
      throw std::runtime_error("BN setUp failed [" + CodeMessage(code) + "]");
    }
  } else {
    bn_pks_.resize(nCRT);
    for (size_t i = 0; i < nCRT; ++i) {
      bn_pks_[i] = std::make_shared<PublicKey>();
      uint64_t pk_sze{0};
      io_->recv_data(&pk_sze, sizeof(uint64_t));
      char *key_buf = new char[pk_sze];
      io_->recv_data(key_buf, pk_sze);
      std::stringstream is;
      is.write(key_buf, pk_sze);
      bn_pks_[i]->load(*bn_contexts_[i], is);
      delete[] key_buf;
      contexts.emplace_back(*bn_contexts_[i]);
    }

    auto code = bn_impl_.setUp(base_mod_, contexts, opt_sks, bn_pks_);
    if (code != Code::OK) {
      throw std::runtime_error("BN setUp failed [" + CodeMessage(code) + "]");
    }
  }
}

void SummaryTensor(Tensor<double> const &t, std::string tag) {
  double mn = *std::min_element(t.data(), t.data() + t.NumElements());
  double mx = *std::max_element(t.data(), t.data() + t.NumElements());
  std::cout << tag << " shape " << t.shape() << " values in [" << mn << ","
            << mx << "]\n";
}

bool CheetahLinear::verify(const Tensor<uint64_t> &in_tensor_share,
                           const std::vector<Tensor<uint64_t>> &filters,
                           const ConvMeta &meta,
                           const Tensor<uint64_t> &out_tensor_share,
                           int nbit_precision) const {
  size_t n_fitlers = filters.size();
  if (n_fitlers < 1) {
    LOG(WARNING) << "CheetahLinear::verify number of filters = 0";
    return false;
  }

  TensorShape ishape = in_tensor_share.shape();
  TensorShape fshape = filters[0].shape();

  auto oshape = GetConv2DOutShape(meta);

  if (!oshape.IsSameSize(out_tensor_share.shape())) {
    LOG(WARNING) << "CheetahLinear::verify oshape mismatch";
    return false;
  }
  if (!ishape.IsSameSize(meta.ishape)) {
    LOG(WARNING) << "CheetahLinear::verify ishape mismatch";
    return false;
  }
  if (!fshape.IsSameSize(meta.fshape)) {
    LOG(WARNING) << "CheetahLinear::verify fshape mismatch";
    return false;
  }

  if (party_ == sci::BOB) {
    io_->send_data(in_tensor_share.data(),
                   sizeof(uint64_t) * in_tensor_share.NumElements());
    io_->send_data(out_tensor_share.data(),
                   sizeof(uint64_t) * out_tensor_share.NumElements());
  } else {
    std::vector<uint64_t> in_tensor_raw(ishape.num_elements());
    std::vector<uint64_t> out_tensor_raw(oshape.num_elements());
    io_->recv_data(in_tensor_raw.data(),
                   sizeof(uint64_t) * in_tensor_raw.size());
    io_->recv_data(out_tensor_raw.data(),
                   sizeof(uint64_t) * out_tensor_raw.size());

    auto in_tensor = Tensor<uint64_t>::Wrap(in_tensor_raw.data(), ishape);
    auto out_tensor = Tensor<uint64_t>::Wrap(out_tensor_raw.data(), oshape);

    // Reconstruct
    in_tensor.tensor() += in_tensor_share.tensor();
    out_tensor.tensor() += out_tensor_share.tensor();

    in_tensor.tensor() =
        in_tensor.tensor().unaryExpr([this](uint64_t v) { return reduce(v); });
    out_tensor.tensor() =
        out_tensor.tensor().unaryExpr([this](uint64_t v) { return reduce(v); });

    auto cast_to_double = [this](uint64_t v, int nbits) -> double {
      // reduce to [0, p) from [0, 2p)
      int64_t sv = get_signed(reduce(v));
      return sv / std::pow(2., nbits);
    };

    Tensor<double> f64_in(in_tensor.shape());
    f64_in.tensor() = in_tensor.tensor().unaryExpr(
        [&](uint64_t v) { return cast_to_double(v, nbit_precision); });

    SummaryTensor(f64_in, "in_tensor");

    Tensor<uint64_t> ground;
    conv2d_impl_.idealFunctionality(in_tensor, filters, meta, ground, nthreads_);

    int cnt_err{0};
    for (auto c = 0; c < out_tensor.channels(); ++c) {
      for (auto h = 0; h < out_tensor.height(); ++h) {
        for (auto w = 0; w < out_tensor.width(); ++w) {
          int64_t g = get_signed(ground(c, h, w));
          int64_t g_ = get_signed(out_tensor(c, h, w));
          if (g != g_) {
            ++cnt_err;
          }
        }
      }
    }

    if (cnt_err == 0) {
      std::cout << "HomConv: matches" << std::endl;
    } else {
      std::cout << "HomConv: failed" << std::endl;
    }
  }

  return true;
}

void CheetahLinear::fc_offline(const Tensor<uint64_t> &input_vector,
                       const Tensor<uint64_t> &weight_matrix,
                       const FCMeta &meta,
                       Tensor<uint64_t> &out_vec_share) const {
#ifdef LOG_LAYERWISE
    INIT_ALL_IO_DATA_SENT;
    INIT_TIMER;
#endif
  if (!input_vector.shape().IsSameSize(meta.input_shape)) {
    throw std::invalid_argument("CheetahLinear::fc input shape mismatch");
  }

  if (party_ == sci::ALICE &&
      !weight_matrix.shape().IsSameSize(meta.weight_shape)) {
    throw std::invalid_argument("CheetahLinear::fc weight shape mismatch");
  }

  TensorShape out_shape({meta.weight_shape.dim_size(0)});
  if (!out_vec_share.shape().IsSameSize(out_shape)) {
    // NOTE(Wen-jie) If the out_matrix may already wrap some memory
    // Then this Reshape will raise error.
    out_vec_share.Reshape(out_shape);
  }
  int dim1 = 1, dim2 = meta.weight_shape.cols(), dim3 = meta.weight_shape.rows();
  
  Code code;
  int nthreads = nthreads_;
  if (party_ == sci::BOB) {
    // 生成随机矩阵M并加密，发送[M]给ALICE，此处M由外部传入
    std::vector<seal::Serializable<seal::Ciphertext>> M_c;
    code = fc_impl_.encryptInputVector(input_vector, meta, M_c, nthreads);
    if (code != Code::OK) {
      throw std::runtime_error("CheetahLinear::fc encryptInputVector [" +
                                CodeMessage(code) + "]");
    }
    send_encrypted_vector(io_, M_c);
    // 接收[W*M-R]，解密获得W*M-R作为输出<W*X>0
    std::vector<seal::Ciphertext> W_mul_M_sub_R_c;
    recv_encrypted_vector(io_, *context_, W_mul_M_sub_R_c);
    code = fc_impl_.decryptToVector(W_mul_M_sub_R_c, meta, out_vec_share, nthreads);

    if (code != Code::OK) {
      throw std::runtime_error("CheetahLinear::fc decryptToVector [" +
                               CodeMessage(code) + "]");
    }
  } else { //Alice
    std::vector<std::vector<seal::Plaintext>> W_p;
    code = fc_impl_.encodeWeightMatrix(weight_matrix, meta, W_p, nthreads);
    if (code != Code::OK) {
      throw std::runtime_error("CheetahLinear::fc encodeWeightMatrix error [" +
                               CodeMessage(code) + "]");
    }
    // 接收[M]
    std::vector<seal::Ciphertext> M_c;
    recv_encrypted_vector(io_, *context_, M_c);
    // 计算[W*M-R]，将R作为输出out_vec_share
    std::vector<seal::Plaintext> vec_share1;
    std::vector<seal::Ciphertext> W_mul_M_sub_R_c;
    auto code = fc_impl_.matVecMul(W_p, M_c, vec_share1, meta,
                               W_mul_M_sub_R_c, out_vec_share, nthreads);
    if (code != Code::OK) {
      throw std::runtime_error("CheetahLinear::fc matmul2D error [" +
                               CodeMessage(code) + "]");
    }
    send_encrypted_vector(io_, W_mul_M_sub_R_c);
  }
#ifdef LOG_LAYERWISE
    auto temp = TIMER_TILL_NOW;
    uint64_t curComm;
    FIND_ALL_IO_TILL_NOW(curComm);
    Matmul_sepOfflineTimeInMilliSec += temp;
    Matmul_sepOfflineCommSent += curComm;
#endif
}

void CheetahLinear::fc_online(const Tensor<uint64_t> &input_vector,
                        const Tensor<uint64_t> &M,
                        const Tensor<uint64_t> &weight_matrix,
                        const FCMeta &meta, Tensor<uint64_t> &offline_share,
                        Tensor<uint64_t> &out_vec_share) const {
#ifdef LOG_LAYERWISE
    INIT_ALL_IO_DATA_SENT;
    INIT_TIMER;
#endif
  TensorShape out_shape({meta.weight_shape.dim_size(0)});
  if (!out_vec_share.shape().IsSameSize(out_shape)) {
    // NOTE(Wen-jie) If the out_matrix may already wrap some memory
    // Then this Reshape will raise error.
    out_vec_share.Reshape(out_shape);
  }
  int dim1 = 1, dim2 = meta.weight_shape.cols(), dim3 = meta.weight_shape.rows();
  if (party_ == sci::BOB) {
    // 计算并发送X0-M
    uint64_t *X0_sub_MArr = new uint64_t[dim2];
    for (int i = 0; i < dim2; i++) {
      X0_sub_MArr[i] = input_vector.data()[i] - M.data()[i];
    }
    io_->send_data(X0_sub_MArr, dim2 * sizeof(uint64_t));
    for (int i = 0; i < dim3; i++) {
      out_vec_share.data()[i] = offline_share.data()[i];
    }
  } else { // ALICE
    // 接收X0-M
    uint64_t *X0_sub_MArr = new uint64_t[dim2];
    io_->recv_data(X0_sub_MArr, dim2 * sizeof(uint64_t));
    // 计算X-M
    uint64_t *X_sub_MArr = new uint64_t[dim2];
    // for (int i = 0; i < dim2; i++) { 
    //   // X_sub_MArr[i] = (X0_sub_MArr[i] + X1Arr[i]);
    //   X_sub_MArr[i] = (X0_sub_MArr[i] + input_vector.data()[i]);
    // }
    // 计算W*(X-M)+R作为结果
    gemini::Tensor<uint64_t> W_mul_XsbuM;
    // auto eva = [&](long wid, size_t start, size_t end) {
    //   for (int i = start; i < end; i++) {
    //     X_sub_MArr[i] = (X0_sub_MArr[i] + input_vector.data()[i]);
    //     out_vec_share.data()[i] = 0;
    //     for (int j = 0; j < dim3; j++) {
    //       out_vec_share.data()[i] += X_sub_MArr[i] * weight_matrix.data()[j*dim2+i];
    //     }
    //     out_vec_share.data()[i] += offline_share.data()[i];
    //   }
    //   return Code::OK;
    // };
    // gemini::ThreadPool tpool(nthreads_);
    // LaunchWorks(tpool, dim2, eva);

    // 计算X-M
    for (int i = 0; i < dim2; i++) {
      X_sub_MArr[i] = (X0_sub_MArr[i] + input_vector.data()[i]);
    }
    for (int i = 0; i < dim3; i++) {
      out_vec_share.data()[i] = 0;
      for (int j = 0; j < dim2; j++) {
        out_vec_share.data()[i] += X_sub_MArr[j] * weight_matrix.data()[j+i*dim2];
        //printf("i: %d, j: %d, w: %lu, x: %lu \n", i, j, weight_matrix.data()[j+i*dim2], X_sub_MArr[j]);
      }
      out_vec_share.data()[i] += offline_share.data()[i];
    }
    //printf("\nX-M: %lu, %lu, %lu\n", X_sub_MArr[0], X_sub_MArr[1], X_sub_MArr[2]);
  }
#ifdef LOG_LAYERWISE
    auto temp = TIMER_TILL_NOW;
    uint64_t curComm;
    FIND_ALL_IO_TILL_NOW(curComm);
    Matmul_sepOnlineTimeInMilliSec += temp;
    Matmul_sepOnlineCommSent += curComm;
#endif
}

void CheetahLinear::fc(const Tensor<uint64_t> &input_vector,
                       const Tensor<uint64_t> &weight_matrix,
                       const FCMeta &meta,
                       Tensor<uint64_t> &out_vec_share) const {
  // out_matrix = input_matrix * weight_matrix
  if (!input_vector.shape().IsSameSize(meta.input_shape)) {
    throw std::invalid_argument("CheetahLinear::fc input shape mismatch");
  }

  if (party_ == sci::ALICE &&
      !weight_matrix.shape().IsSameSize(meta.weight_shape)) {
    throw std::invalid_argument("CheetahLinear::fc weight shape mismatch");
  }

  TensorShape out_shape({meta.weight_shape.dim_size(0)});
  if (!out_vec_share.shape().IsSameSize(out_shape)) {
    // NOTE(Wen-jie) If the out_matrix may already wrap some memory
    // Then this Reshape will raise error.
    out_vec_share.Reshape(out_shape);
  }

  const auto &impl = fc_impl_;

  Code code;
  int nthreads = nthreads_;
  if (party_ == sci::BOB) {
    {
      std::vector<seal::Serializable<seal::Ciphertext>> ct_buff;
      code = impl.encryptInputVector(input_vector, meta, ct_buff, nthreads);
      if (code != Code::OK) {
        throw std::runtime_error("CheetahLinear::fc encryptInputVector [" +
                                 CodeMessage(code) + "]");
      }
      send_encrypted_vector(io_, ct_buff);
    }

    std::vector<seal::Ciphertext> ct_buff;
    recv_encrypted_vector(io_, *context_, ct_buff);
    code = impl.decryptToVector(ct_buff, meta, out_vec_share, nthreads);

    if (code != Code::OK) {
      throw std::runtime_error("CheetahLinear::fc decryptToVector [" +
                               CodeMessage(code) + "]");
    }
  } else {
    std::vector<std::vector<seal::Plaintext>> encoded_matrix;
    code =
        impl.encodeWeightMatrix(weight_matrix, meta, encoded_matrix, nthreads);
    if (code != Code::OK) {
      throw std::runtime_error("CheetahLinear::fc encodeWeightMatrix error [" +
                               CodeMessage(code) + "]");
    }
    std::vector<seal::Plaintext> vec_share1;
    if (meta.is_shared_input) {
      code = impl.encodeInputVector(input_vector, meta, vec_share1, nthreads);
      if (code != Code::OK) {
        throw std::runtime_error("CheetahLinear::fc encodeInputVector error [" +
                                 CodeMessage(code) + "]");
      }
    }

    uint32_t ncts{0};
    io_->recv_data(&ncts, sizeof(uint32_t));
    std::vector<seal::Ciphertext> vec_share0(ncts);
    for (size_t i = 0; i < ncts; ++i) {
      recv_ciphertext(io_, *context_, vec_share0[i]);
    }

    std::vector<seal::Ciphertext> out_vec_share0;
    auto code = impl.matVecMul(encoded_matrix, vec_share0, vec_share1, meta,
                               out_vec_share0, out_vec_share, nthreads);
    if (code != Code::OK) {
      throw std::runtime_error("CheetahLinear::fc matmul2D error [" +
                               CodeMessage(code) + "]");
    }
    send_encrypted_vector(io_, out_vec_share0);
  }
}

void CheetahLinear::conv_bias_relu_offline(const Tensor<uint64_t> &in_tensor,
              const std::vector<Tensor<uint64_t>> &filters, int T,
              const ConvMeta &meta, Tensor<uint64_t> &out_share0, Tensor<uint64_t> &out_share1,
              int bitlength, int scale) const{
#ifdef LOG_LAYERWISE
    INIT_ALL_IO_DATA_SENT;
    INIT_TIMER;
#endif
  setbuf(stdout,NULL);
  //print_shape(in_tensor);
  //print_shape(filters[0]);
  if (!meta.ishape.IsSameSize(in_tensor.shape())) {
    throw std::invalid_argument("CheetahLinear::conv_bias_relu meta.ishape mismatch");
  }
  if (meta.n_filters != filters.size()) {
    throw std::invalid_argument("CheetahLinear::conv_bias_relu meta.n_filters mismatch");
  }
  for (const auto &f : filters) {
    if (!meta.fshape.IsSameSize(f.shape())) {
      throw std::invalid_argument("CheetahLinear::conv_bias_relu meta.fshape mismatch");
    }
  }
  Code code;
  if (party_ == sci::BOB) {
    std::vector<seal::Serializable<seal::Ciphertext>> M_c;
    code = conv2d_impl_.encryptImage(in_tensor, meta, M_c, nthreads_);// 将X0加密
    if (code != Code:: OK) {
      throw std::runtime_error("CheetahLinear::conv_bias_relu encryptImage " +
                                 CodeMessage(code));
    }
    send_encrypted_vector(io_, M_c); // 发送[M]
    std::vector<seal::Ciphertext> W_mul_M_sub_R0_c, W_mul_M_mul_T_sub_R1_c;
    recv_encrypted_vector(io_, *context_, W_mul_M_sub_R0_c, true);
    recv_encrypted_vector(io_, *context_, W_mul_M_mul_T_sub_R1_c, true);
    // 解密获得W*X-R0
    code = conv2d_impl_.decryptToTensor(W_mul_M_sub_R0_c, meta, out_share0, nthreads_);
    if (code != Code::OK) {
      throw std::runtime_error("CheetahLinear::conv_bias_relu decryptToTensor " +
                               CodeMessage(code));
    }
    // 解密获得W*X*T-R1
    code = conv2d_impl_.decryptToTensor(W_mul_M_mul_T_sub_R1_c, meta, out_share1, nthreads_);
    if (code != Code::OK) {
      throw std::runtime_error("CheetahLinear::conv_bias_relu decryptToTensor " +
                               CodeMessage(code));
    }
  } else { // ALICE
    std::vector<std::vector<seal::Plaintext>> encoded_filters;
    code = conv2d_impl_.encodeFilters(filters, meta, encoded_filters, nthreads_);
    if (code != Code::OK) {
      throw std::runtime_error("CheetahLinear::conv_bias_relu ecnodeFilters " +
                               CodeMessage(code));
    }
    std::vector<seal::Plaintext> zero_p;
    code = conv2d_impl_.encodeImage(in_tensor, meta, zero_p, nthreads_);
    std::vector<seal::Ciphertext> M_c;
    recv_encrypted_vector(io_, *context_, M_c, false); // 接收[M]
    std::vector<seal::Ciphertext> W_mul_M_sub_R0_c;
    code = conv2d_impl_.conv2DSS(M_c, zero_p, encoded_filters, meta, 
                                  W_mul_M_sub_R0_c, out_share0, nthreads_); // out_ct存放[W*M]
    if (code != Code::OK) {
      throw std::runtime_error("CheetahLinear::conv_bias_relu conv2DSS: " +
                               CodeMessage(code));
    }

    // 将filters乘上T倍，这样就能得到W*T，T应为一个随机数，此处先取为2
    TensorShape shape = filters[0].shape();
    int C = shape.channels();
    int H = shape.height();
    int W = shape.width();
    std::vector<Tensor<uint64_t>> scale_filters(filters.size());
    for (auto &f : scale_filters) {
      f.Reshape(shape);
    }
    for (int i = 0; i < filters.size(); i++) {
      for (int c = 0; c < C; c++) {
        for (int h = 0; h < H; h++) { 
          for (int w = 0; w < W; w++) {
            scale_filters.at(i)(c, h, w) = T * filters.at(i)(c, h, w);
          }
        }
      }
    }
    std::vector<std::vector<seal::Plaintext>> scale_encoded_filters;
    code = conv2d_impl_.encodeFilters(scale_filters, meta, scale_encoded_filters, nthreads_);
    if (code != Code::OK) {
      throw std::runtime_error("CheetahLinear::conv_bias_relu ecnodeFilters " +
                               CodeMessage(code));
    }        
    std::vector<seal::Ciphertext> W_mul_M_mul_T_sub_R1_c;
    code = conv2d_impl_.conv2DSS(M_c, zero_p, scale_encoded_filters, meta, 
                                  W_mul_M_mul_T_sub_R1_c, out_share1, nthreads_); // out_ct存放[W*M]
    if (code != Code::OK) {
      throw std::runtime_error("CheetahLinear::conv_bias_relu conv2DSS: " +
                               CodeMessage(code));
    }
    // 发送W_mul_M_c和W_mul_M_T_c
    send_encrypted_vector(io_, W_mul_M_sub_R0_c);
    send_encrypted_vector(io_, W_mul_M_mul_T_sub_R1_c);
  }
#ifdef LOG_LAYERWISE
    auto temp = TIMER_TILL_NOW;
    FusedBN_ReLUOfflineTimeInMilliSec += temp;
    uint64_t curComm;
    FIND_ALL_IO_TILL_NOW(curComm);
    FusedBN_ReLUOfflineCommSent += curComm;
#endif
}

void CheetahLinear::conv_bias_relu_online(const Tensor<uint64_t> &in_tensor,
                const Tensor<uint64_t> &M, const std::vector<Tensor<uint64_t>> &filters, int T,
                const ConvMeta &meta, Tensor<uint64_t> &offline_share0, 
                Tensor<uint64_t> &offline_share1, uint64_t *bias, bool *B, uint64_t *result,
                int bitlength, int scale) const {  
#ifdef LOG_LAYERWISE
    INIT_ALL_IO_DATA_SENT;
    INIT_TIMER;
#endif
  //print_shape(in_tensor);
  if (party_ == sci::BOB) {
    // 将in_tensor解码获得数组X0，将M解码获得数组M
    gemini::TensorShape shape = in_tensor.shape();
    int C = shape.channels(), H = shape.height(), W = shape.width();
    
    uint64_t *X0_sub_MArr = new uint64_t[C * H * W];
    // 发送X0-M到ALICE
    int shape_len = C * H * W;
    for (int i = 0; i < shape_len; i++) {
      X0_sub_MArr[i] = (in_tensor.data()[i] - M.data()[i]);
    }
    io_->send_data(X0_sub_MArr, C * H * W * sizeof(uint64_t));
    // 将offline_share1解码获得V，接收U=(W*(X-M)+b) * T + R1
    gemini::TensorShape shape_out = offline_share1.shape();
    int C_out = shape_out.channels(), H_out = shape_out.height(), W_out = shape_out.width();
    uint64_t *UArr = new uint64_t[C_out * H_out * W_out];
    uint64_t *U_add_V = new uint64_t[C_out * H_out * W_out];
    io_->recv_data(UArr, C_out * H_out * W_out * sizeof(uint64_t));
    // 计算U + V
    int shape_out_len = C_out * H_out * W_out;
    for (int i = 0; i < shape_out_len; i++) {
    }
    uint64_t mask_l = (1ULL << bitlength) - 1;
    for (int i = 0; i < shape_out_len; i++) {
      U_add_V[i] = (UArr[i] + offline_share1.data()[i]) & mask_l;
    }
    // 判断并发送B，返回结果
    for (int i = 0; i < shape_out_len; i++) {
      B[i] = ((U_add_V[i] >> (bitlength - 1)) & 1) ^ 1;
    }
    io_->send_bool(B, C_out * H_out * W_out);
  } else { //ALICE
    // 将in_tensor解码获得数组X1，将offline_share0解码获得R0，将offline_share1解码获得R1
    gemini::TensorShape shape = in_tensor.shape();
    int C = shape.channels(), H = shape.height(), W = shape.width();
    int shape_len = C * H * W;
    uint64_t *X0_sub_MArr = new uint64_t[C * H * W];
    uint64_t *X_sub_MArr = new uint64_t[C * H * W];
    // 将b扩展开来，b的长度为CO
    int CO = meta.n_filters; // 每个filter对应一个
    // 接收X0-M
    io_->recv_data(X0_sub_MArr, C * H * W * sizeof(uint64_t));
    // 计算X-M = X0-M + X1
    gemini::Tensor<uint64_t> X_sub_MTen(meta.ishape);
    
    // auto eva_X_sub_M = [&](long wid, size_t start, size_t end) {
      // for (int i = start; i < end; i++) {
      for (int i = 0; i < shape_len; i++) { 
        // X_sub_MArr[i] = (X0_sub_MArr[i] + X1Arr[i]);
        X_sub_MTen.data()[i] = (X0_sub_MArr[i] + in_tensor.data()[i]);
      }
    //   return Code::OK;
    // };
    // gemini::ThreadPool tpool(nthreads_);
    // LaunchWorks(tpool, shape_len, eva_X_sub_M);
    // 计算<W*X>1 = W*(X-M) + b + R0（卷积计算）和 U=(W*(X-M)+b) * T + R1
    gemini::Tensor<uint64_t> W_mul_XsbuM;
    conv2d_impl_.idealFunctionality(X_sub_MTen, filters, meta, W_mul_XsbuM, nthreads_);
    gemini::TensorShape shape_out = W_mul_XsbuM.shape();
    int C_out = shape_out.channels(), H_out = shape_out.height(), W_out = shape_out.width();
    int shape_out_len = C_out * H_out * W_out;
    uint64_t *UArr = new uint64_t[shape_out_len];
    // auto eva_result_U = [&](long wid, size_t start, size_t end) {
      // for (int i = 0; i < shape_out_len; i++) {
      for (int i = 0; i < shape_out_len; i++) { 
        // 计算<W*X>1 = W*(X-M) + b + R0（卷积计算）
        result[i] = W_mul_XsbuM.data()[i] + bias[i/(H_out*W_out)] + offline_share0.data()[i];
        // 计算U=(W*(X-M)+b) * T + R1
        UArr[i] = T * (W_mul_XsbuM.data()[i] + bias[i/(H_out*W_out)]) + offline_share1.data()[i];
      }
    //   return Code::OK;
    // };
    // LaunchWorks(tpool, shape_len, eva_result_U);
    // 发送U
    io_->send_data(UArr, C_out*H_out*W_out*sizeof(uint64_t));
    // 接收B
    io_->recv_bool(B, C_out * H_out * W_out);
  }
#ifdef LOG_LAYERWISE
    auto temp = TIMER_TILL_NOW;
    FusedBN_ReLUOnlineTimeInMilliSec += temp;
    uint64_t curComm;
    FIND_ALL_IO_TILL_NOW(curComm);
    FusedBN_ReLUOnlineCommSent += curComm;
#endif
}

void CheetahLinear::fc_get_random_zero_tensor(Tensor<uint64_t> &in_tensor, const FCMeta &meta, bool zero) const {
  const int r = meta.input_shape.rows();
  const int c = meta.input_shape.rows();
  uint64_t *tmp = new uint64_t[r*c];
  sci::PRG128 prg;
  if (zero == false) {
    prg.random_data(tmp, r * c * sizeof(uint64_t));
  } else {
    memset(tmp, 0, r * c * sizeof(uint64_t));
  }
  for (int i = 0; i < r; i++) {
    for (int j = 0; j < c; j++) {
      in_tensor(i, j) = tmp[i * c + j] & mask_r;
      if (in_tensor(i, j) == 0) {
        do {
          prg.random_data(&tmp[i * c + j], sizeof(uint64_t));
          in_tensor(i, j) = tmp[i * c + j] & mask_r;
        } while(in_tensor(i, j) == 0);
      }
    }
  }
}

void CheetahLinear::get_random_zero_tensor(Tensor<uint64_t> &in_tensor, const ConvMeta &meta, bool zero) const {
  const int H = meta.ishape.height();
  const int W = meta.ishape.width();
  const int CI = meta.ishape.channels();
  uint64_t *tmp = new uint64_t[H*W*CI];
  sci::PRG128 prg;
  if (zero == false) {
    prg.random_data(tmp, H * W * CI * sizeof(uint64_t));
  } else {
    memset(tmp, 0, H*W*CI * sizeof(uint64_t));
  }
  for (int j = 0; j < H; j++) {
      for (int k = 0; k < W; k++) {
        for (int p = 0; p < CI; p++) {
          in_tensor(p, j, k) = tmp[j*W + k*CI + p] & mask_r;
          if (in_tensor(p, j, k) == 0) {// 避免产生随机数为0
            do {
                prg.random_data(&tmp[j*W + k*CI + p], sizeof(uint64_t));
                in_tensor(p, j, k) = tmp[j*W + k*CI + p] & mask_r;
            } while(in_tensor(p, j, k) == 0);
          }
        }
      }
    }
}

void CheetahLinear::conv2d_offline(const Tensor<uint64_t> &in_tensor,
                           const std::vector<Tensor<uint64_t>> &filters,
                           const ConvMeta &meta,Tensor<uint64_t> &out_tensor, bool BN) const {
#ifdef LOG_LAYERWISE
    INIT_ALL_IO_DATA_SENT;
    INIT_TIMER;
#endif
  setbuf(stdout,NULL);
  if (meta.n_filters != filters.size()) {
    throw std::invalid_argument("CheetahLinear::conv_bias_relu meta.n_filters mismatch");
  }
  for (const auto &f : filters) {
    if (!meta.fshape.IsSameSize(f.shape())) {
      throw std::invalid_argument("CheetahLinear::conv_bias_relu meta.fshape mismatch");
    }
  }
  Code code;
  if (party_ == sci::BOB) {
    // 将随机矩阵M加密，发送[M]给ALICE
    std::vector<seal::Serializable<seal::Ciphertext>> M_c;
    code = conv2d_impl_.encryptImage(in_tensor, meta, M_c, nthreads_);// 将X0加密
    if (code != Code:: OK) {
      throw std::runtime_error("CheetahLinear::conv_bias_relu encryptImage " +
                                 CodeMessage(code));
    }
    send_encrypted_vector(io_, M_c); // 发送[M]
    // 接收[W*M-R]，解密获得W*M-R作为输出<W*X>0
    std::vector<seal::Ciphertext> W_mul_M_sub_R_c;
    recv_encrypted_vector(io_, *context_, W_mul_M_sub_R_c);
    code = conv2d_impl_.decryptToTensor(W_mul_M_sub_R_c, meta, out_tensor, nthreads_);
    if (code != Code::OK) {
      throw std::runtime_error("CheetahLinear::conv_bias_relu decryptToTensor " +
                               CodeMessage(code));
    }
  } else {
    // 接收[M]
    std::vector<seal::Ciphertext> M_c;
    recv_encrypted_vector(io_, *context_, M_c, false); // 接收[M]
    // 计算[W*M-R]，将R作为输出
    std::vector<std::vector<seal::Plaintext>> encoded_filters;
    code = conv2d_impl_.encodeFilters(filters, meta, encoded_filters, nthreads_);
    if (code != Code::OK) {
      throw std::runtime_error("CheetahLinear::conv_bias_relu ecnodeFilters " +
                               CodeMessage(code));
    }
    std::vector<seal::Plaintext> zero_p;
    code = conv2d_impl_.encodeImage(in_tensor, meta, zero_p, nthreads_);
    std::vector<seal::Ciphertext> W_mul_M_sub_R_c;
    code = conv2d_impl_.conv2DSS(M_c, zero_p, encoded_filters, meta, 
                                  W_mul_M_sub_R_c, out_tensor, nthreads_); // out_ct存放[W*M]
    if (code != Code::OK) {
      throw std::runtime_error("CheetahLinear::conv_bias_relu conv2DSS: " +
                               CodeMessage(code));
    }
    // 发送[W*M-R]给BOB
    send_encrypted_vector(io_, W_mul_M_sub_R_c);
  }
#ifdef LOG_LAYERWISE
    auto temp = TIMER_TILL_NOW;
    uint64_t curComm;
    FIND_ALL_IO_TILL_NOW(curComm);
if (BN == false) {
    Conv_sepOfflineTimeInMilliSec += temp;
    Conv_sepOfflineCommSent += curComm;
} else {
    BN_sepOfflineTimeInMilliSec += temp;
    BN_sepOfflineCommSent += curComm;
}
#endif
}

void CheetahLinear::conv2d_online(const Tensor<uint64_t> &in_tensor,
                const Tensor<uint64_t> &M, const std::vector<Tensor<uint64_t>> &filters,
                const ConvMeta &meta, Tensor<uint64_t> &offline_share,
                uint64_t *result, bool BN) const {
#ifdef LOG_LAYERWISE
    INIT_ALL_IO_DATA_SENT;
    INIT_TIMER;
#endif
  if (!meta.ishape.IsSameSize(in_tensor.shape())) {
    throw std::invalid_argument("CheetahLinear::conv_bias_relu meta.ishape mismatch");
  }
  if (party_ == sci::BOB) {
    // 将in_tensor解码获得数组X0，将M解码获得数组M
    gemini::TensorShape shape = in_tensor.shape();
    int C = shape.channels(), H = shape.height(), W = shape.width();
    uint64_t *X0_sub_MArr = new uint64_t[C * H * W];
    int shape_len = C * H * W;
    for (int i = 0; i < shape_len; i++) {
      X0_sub_MArr[i] = in_tensor.data()[i] - M.data()[i];
    }
    // 发送X0-M
    io_->send_data(X0_sub_MArr, C * H * W * sizeof(uint64_t));
  } else {
    // 将in_tensor解码获得数组X1，将offline_share0解码获得R
    gemini::TensorShape shape = in_tensor.shape();
    int C = shape.channels(), H = shape.height(), W = shape.width();
    uint64_t *X0_sub_MArr = new uint64_t[C * H * W];
    // 接收X0-M
    io_->recv_data(X0_sub_MArr, C * H * W * sizeof(uint64_t));
    
    // 计算X-M

    int shape_len = C * H * W;
    gemini::Tensor<uint64_t> X_sub_MTen(meta.ishape);
    for (int i = 0; i < shape_len; i++) { 
      // X_sub_MArr[i] = (X0_sub_MArr[i] + X1Arr[i]);
      X_sub_MTen.data()[i] = (X0_sub_MArr[i] + in_tensor.data()[i]);
    }
    // 计算W*(X-M)+R作为结果
    gemini::Tensor<uint64_t> W_mul_XsbuM;
    gemini::Tensor<uint64_t> W_mul_X1;
    
    conv2d_impl_.idealFunctionality(X_sub_MTen, filters, meta, W_mul_XsbuM, nthreads_);
    gemini::TensorShape shape_out = W_mul_XsbuM.shape();
    W_mul_X1.Reshape(shape_out);
    int C_out = shape_out.channels(), H_out = shape_out.height(), W_out = shape_out.width();
    int shape_out_len = C_out * H_out * W_out;

    // auto eva_W_mul_X1 = [&](long wid, size_t start, size_t end) {
      // for (int i = start; i < end; i++) {
      for (int i = 0; i < shape_out_len; i++) {  
        result[i] = W_mul_XsbuM.data()[i] + offline_share.data()[i];
      }
    //   return Code::OK;
    // };
    // LaunchWorks(tpool, shape_out_len, eva_W_mul_X1);
  }
  
#ifdef LOG_LAYERWISE
    auto temp = TIMER_TILL_NOW;
    uint64_t curComm;
    FIND_ALL_IO_TILL_NOW(curComm);
if (BN == false) {
    Conv_sepOnlineTimeInMilliSec += temp;
    Conv_sepOnlineCommSent += curComm;
} else {
    BN_sepOnlineTimeInMilliSec += temp;
    BN_sepOnlineCommSent += curComm;
}
#endif
}

void CheetahLinear::conv2d(const Tensor<uint64_t> &in_tensor,
                           const std::vector<Tensor<uint64_t>> &filters,
                           const ConvMeta &meta,
                           Tensor<uint64_t> &out_tensor) const {
  if (!meta.ishape.IsSameSize(in_tensor.shape())) {
    throw std::invalid_argument("CheetahLinear::conv2d meta.ishape mismatch");
  }
  if (meta.n_filters != filters.size()) {
    throw std::invalid_argument(
        "CheetahLinear::conv2d meta.n_filters mismatch");
  }
  for (const auto &f : filters) {
    if (!meta.fshape.IsSameSize(f.shape())) {
      throw std::invalid_argument("CheetahLinear::conv2d meta.fshape mismatch");
    }
  }

  const auto &impl = conv2d_impl_;

  Code code;
  if (party_ == sci::BOB) {
    {
      std::vector<seal::Serializable<seal::Ciphertext>> ct_buff;
      code = impl.encryptImage(in_tensor, meta, ct_buff, nthreads_); // 将X0加密
      if (code != Code::OK) {
        throw std::runtime_error("CheetahLinear::conv2d encryptImage " +
                                 CodeMessage(code));
      }
      send_encrypted_vector(io_, ct_buff); // 发送[X0]
    }

    // Wait for result
    std::vector<seal::Ciphertext> ct_buff;
    recv_encrypted_vector(io_, *context_, ct_buff, true);

    code = impl.decryptToTensor(ct_buff, meta, out_tensor, nthreads_);
    if (code != Code::OK) {
      throw std::runtime_error("CheetahLinear::conv2d decryptToTensor " +
                               CodeMessage(code));
    }
  } else {
    std::vector<std::vector<seal::Plaintext>> encoded_filters;
    code = impl.encodeFilters(filters, meta, encoded_filters, nthreads_); // 将Filters进行编码，这个在BN中是？可以看看尺寸
    if (code != Code::OK) {
      throw std::runtime_error("CheetahLinear::conv2d ecnodeFilters " +
                               CodeMessage(code));
    }

    std::vector<seal::Plaintext> encoded_share;
    if (meta.is_shared_input) {
      code = impl.encodeImage(in_tensor, meta, encoded_share, nthreads_); // 编码X1
      if (code != Code::OK) {
        throw std::runtime_error("CheetahLinear::conv2d encodeImage " +
                                 CodeMessage(code));
      }
    }

    std::vector<seal::Ciphertext> ct_buff;
    recv_encrypted_vector(io_, *context_, ct_buff, false); // 接收[X0]

    std::vector<seal::Ciphertext> out_ct;
    auto code = impl.conv2DSS(ct_buff, encoded_share, encoded_filters, meta,
                              out_ct, out_tensor, nthreads_); // 计算[W * (X0 + X1) - R]放到out_ct中，输出R到out_tensor中
    if (code != Code::OK) {
      throw std::runtime_error("CheetahLinear::conv2d conv2DSS: " +
                               CodeMessage(code));
    }
    send_encrypted_vector(io_, out_ct);
  }
}

void CheetahLinear::bn(const Tensor<uint64_t> &input_vector,
                       const Tensor<uint64_t> &scale_vector, const BNMeta &meta,
                       Tensor<uint64_t> &out_vector) const {
  if (meta.is_shared_input &&
      !input_vector.shape().IsSameSize(meta.vec_shape)) {
    throw std::runtime_error("bn input_vector shape mismatch");
  }
  Code code;
  if (party_ == sci::BOB) {
    {
      std::vector<seal::Serializable<seal::Ciphertext>> ct_buff;
      code = bn_impl_.encryptVector(input_vector, meta, ct_buff, nthreads_);
      if (code != Code::OK) {
        throw std::runtime_error("bn encryptVector [" + CodeMessage(code) +
                                 "]");
      }
      code = bn_impl_.sendEncryptVector(io_, ct_buff, meta);
      if (code != Code::OK) {
        throw std::runtime_error("bn sendEncryptVector [" + CodeMessage(code) +
                                 "]");
      }
    }

    std::vector<seal::Ciphertext> ct_buff;
    code = bn_impl_.recvEncryptVector(io_, ct_buff, meta);
    if (code != Code::OK) {
      throw std::runtime_error("bn recvEncryptVector [" + CodeMessage(code) +
                               "]");
    }

    code = bn_impl_.decryptToVector(ct_buff, meta, out_vector, nthreads_);
    if (code != Code::OK) {
      throw std::runtime_error("bn decryptToVector [" + CodeMessage(code) +
                               "]");
    }
  } else { // ALICE
    if (!scale_vector.shape().IsSameSize(meta.vec_shape)) {
      throw std::runtime_error("bn scale_vector shape mismatch");
    }

    std::vector<seal::Plaintext> encoded_vector;
    std::vector<seal::Plaintext> encoded_scales;
    if (meta.is_shared_input) {
      code =
          bn_impl_.encodeVector(input_vector, meta, encoded_vector, nthreads_);
      if (code != Code::OK) {
        throw std::runtime_error("bn encodeVector [" + CodeMessage(code) + "]");
      }
    }

    code = bn_impl_.encodeScales(scale_vector, meta, encoded_scales, nthreads_);
    if (code != Code::OK) {
      throw std::runtime_error("bn encodeScales [" + CodeMessage(code) + "]");
    }

    std::vector<seal::Ciphertext> encrypted_vector;
    code = bn_impl_.recvEncryptVector(io_, encrypted_vector, meta);
    if (code != Code::OK) {
      throw std::runtime_error("bn recvEncryptVector [" + CodeMessage(code) +
                               "]");
    }
    if (encrypted_vector.size() != encoded_scales.size()) {
      LOG(FATAL) << "vector / scales size mismatch";
    }
    std::vector<seal::Ciphertext> out_ct;
    code = bn_impl_.bn(encrypted_vector, encoded_vector, encoded_scales, meta,
                       out_ct, out_vector, nthreads_);
    if (code != Code::OK) {
      throw std::runtime_error("bn failed [" + CodeMessage(code) + "]");
    }

    code = bn_impl_.sendEncryptVector(io_, out_ct, meta);
    if (code != Code::OK) {
      throw std::runtime_error("bn sendEncryptVector [" + CodeMessage(code) +
                               "]");
    }
  }
}

void CheetahLinear::bn_direct(const Tensor<uint64_t> &input_tensor,
                              const Tensor<uint64_t> &scale_vector,
                              const BNMeta &meta,
                              Tensor<uint64_t> &out_tensor) const {
  if (meta.is_shared_input && !input_tensor.shape().IsSameSize(meta.ishape)) {
    throw std::runtime_error("bn_direct input_vector shape mismatch");
  }
  Code code;
  if (party_ == sci::BOB) {
    {
      std::vector<seal::Serializable<seal::Ciphertext>> ct_buff;
      code = bn_impl_.encryptTensor(input_tensor, meta, ct_buff, nthreads_); // 加密X0
      if (code != Code::OK) {
        throw std::runtime_error("bn_direct encryptVector [" +
                                 CodeMessage(code) + "]");
      }
      send_encrypted_vector(io_, ct_buff); // 发送[X0]
    }

    std::vector<seal::Ciphertext> ct_buff;
    recv_encrypted_vector(io_, *context_, ct_buff); // 接收[scale*X-R]

    code = bn_impl_.decryptToTensor(ct_buff, meta, out_tensor, nthreads_); // 解密放到out_tensor中
    if (code != Code::OK) {
      throw std::runtime_error("bn_direct decryptToTensor [" +
                               CodeMessage(code) + "]");
    }
  } else {
    if (scale_vector.dims() != 1 ||
        scale_vector.length() != input_tensor.channels()) {
      throw std::runtime_error("bn_direct scale_vector shape mismatch");
    }

    std::vector<seal::Plaintext> encoded_tensor;
    if (meta.is_shared_input) {
      code =
          bn_impl_.encodeTensor(input_tensor, meta, encoded_tensor, nthreads_); // 编码X1
      if (code != Code::OK) {
        throw std::runtime_error("bn_direct encodeVector [" +
                                 CodeMessage(code) + "]");
      }
    }

    std::vector<seal::Ciphertext> encrypted_tensor;
    recv_encrypted_vector(io_, *context_, encrypted_tensor); // 接收[X0]

    std::vector<seal::Ciphertext> out_ct;
    code = bn_impl_.bn_direct(encrypted_tensor, encoded_tensor, scale_vector,
                              meta, out_ct, out_tensor, nthreads_);
    if (code != Code::OK) {
      throw std::runtime_error("bn_direct failed [" + CodeMessage(code) + "]");
    }
    send_encrypted_vector(io_, out_ct); // 发送[scale*X-R]
  }
}

}  // namespace gemini

void recv_encrypted_vector(sci::NetIO *io, const seal::SEALContext &context,
                           std::vector<seal::Ciphertext> &ct_vec,
                           bool is_truncated) {
  uint32_t ncts{0};
  io->recv_data(&ncts, sizeof(uint32_t));
  if (ncts > 0) {
    ct_vec.resize(ncts);
    for (size_t i = 0; i < ncts; ++i) {
      recv_ciphertext(io, context, ct_vec[i], is_truncated);
    }
  }
}

void recv_ciphertext(sci::NetIO *io, const seal::SEALContext &context,
                     seal::Ciphertext &ct, bool is_truncated) {
  std::stringstream is;
  uint64_t ct_size;
  io->recv_data(&ct_size, sizeof(uint64_t));
  char *c_enc_result = new char[ct_size];
  io->recv_data(c_enc_result, ct_size);
  is.write(c_enc_result, ct_size);
  if (is_truncated) {
    ct.unsafe_load(context, is);
  } else {
    ct.load(context, is);
  }
  delete[] c_enc_result;
}

void decode_tensor(const gemini::Tensor<uint64_t> &in_tensor, uint64_t *out, int C, int H, int W, int bitlength) {
  for (int i = 0; i < C; i++) {
    for (int j = 0; j < H; j++) {
      for (int k = 0; k < W; k++) {
        out[i*H + j*W + k] = in_tensor(i, j, k);
      }
    }
  }
}

void print_shape(const gemini::Tensor<uint64_t> &tensor) {
  gemini::TensorShape shape = tensor.shape();
  int C = shape.channels(), H = shape.height(), W = shape.width();
  printf("tensor size :C=%d H=%d W=%d\n", C, H, W);
}

void print_tensor(const gemini::Tensor<uint64_t> &tensor) {
  gemini::TensorShape shape = tensor.shape();
  int C = shape.channels(), H = shape.height(), W = shape.width();
  for (int i = 0; i < C; i++) {
    for (int j = 0; j < H; j++) {
      for (int k = 0; k < W; k++) {
        printf("%d ", tensor(i, j, k));
      }
      printf("\n");
    }
  }
}

static Code LaunchWorks(
    gemini::ThreadPool &tpool, size_t num_works,
    std::function<Code(long wid, size_t start, size_t end)> program) {
  if (num_works == 0) return Code::OK;
  const long pool_sze = tpool.pool_size();
  if (pool_sze <= 1L) {
    return program(0, 0, num_works);
  } else {
    Code code;
    std::vector<std::future<Code>> futures;
    size_t work_load = (num_works + pool_sze - 1) / pool_sze;
    for (long wid = 0; wid < pool_sze; ++wid) {
      size_t start = wid * work_load;
      size_t end = std::min(start + work_load, num_works);
      futures.push_back(tpool.enqueue(program, wid, start, end));
    }

    code = Code::OK;
    for (auto &&work : futures) {
      Code c = work.get();
      if (code == Code::OK && c != Code::OK) {
        code = c;
      }
    }
    return code;
  }
}
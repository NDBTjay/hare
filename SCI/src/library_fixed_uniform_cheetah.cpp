// Author: Wen-jie Lu
// Adapter for the SCI's implementation using Cheetah's linear protocols.
#if USE_CHEETAH

#include <gemini/cheetah/tensor.h>

#include "cheetah/cheetah-api.h"
#include "defines_uniform.h"
#include "globals.h"

#define VERIFY_LAYERWISE
#define LOG _LAYERWISE
#undef VERIFY_LAYERWISE // undefine this to turn OFF the verifcation
//#undef LOG_LAYERWISE // undefine this to turn OFF the log

#ifndef SCI_OT
extern int64_t getSignedVal(uint64_t x);
extern uint64_t getRingElt(int64_t x);
#else
extern uint64_t prime_mod;
extern uint64_t moduloMask;
extern uint64_t moduloMidPt;

static inline int64_t getSignedVal(uint64_t x) {
  assert(x < prime_mod);
  int64_t sx = x;
  if (x >= moduloMidPt) sx = x - prime_mod;
  return sx;
}

static inline uint64_t getRingElt(int64_t x) {
  return ((uint64_t)x) & moduloMask;
}
#endif

extern uint64_t SecretAdd(uint64_t x, uint64_t y);

void get_zero_tensor_1d(gemini::Tensor<uint64_t> &in_tensor) {
  gemini::TensorShape shape = in_tensor.shape();
  int len = shape.length();
  for (int i = 0; i < len; i++) {
    in_tensor.data()[i] = 0;
  }
  return;
}

void get_random_tensor_1d(gemini::Tensor<uint64_t> &in_tensor, int bitlength) {
  uint64_t mask_l = (uint64_t)((1ULL << bitlength) - 1);
  uint64_t mask_r = (uint64_t)((1ULL << 5) - 1);
  gemini::TensorShape shape = in_tensor.shape();
  int len = shape.length();
  uint64_t *tmp = new uint64_t[len];
  sci::PRG128 prg;
  prg.random_data(tmp, len * sizeof(uint64_t));

  for (int i = 0; i < len; i++) {
    in_tensor.data()[i] = tmp[i] & mask_r;
    if (in_tensor.data()[i] == 0) {
      do {
        prg.random_data(&tmp[i], sizeof(uint64_t));
        in_tensor.data()[i] = tmp[i] & mask_r;
      } while(in_tensor.data()[i] == 0);
    }
  }
}

void get_zero_tensor_2d(gemini::Tensor<uint64_t> &in_tensor) {
  gemini::TensorShape shape = in_tensor.shape();
  const int r = shape.rows();
  const int c = shape.cols();
  int len = r * c;
  for (int i = 0; i < len; i++) {
    in_tensor.data()[i] = 0;
  }
}

void get_random_tensor_2d(gemini::Tensor<uint64_t> &in_tensor, int bitlength) {
  uint64_t mask_l = (uint64_t)((1ULL << bitlength) - 1);
  uint64_t mask_r = (uint64_t)((1ULL << 5) - 1);
  gemini::TensorShape shape = in_tensor.shape();
  const int r = shape.rows();
  const int c = shape.cols();
  uint64_t *tmp = new uint64_t[r*c];
  sci::PRG128 prg;
  prg.random_data(tmp, r * c * sizeof(uint64_t));

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

void get_zero_tensor(gemini::Tensor<uint64_t> &in_tensor) {
  gemini::TensorShape shape = in_tensor.shape();
  int C = shape.channels(), H = shape.height(), W = shape.width();
  int len = C * H * W;
  for (int i = 0; i < len; i++) {
    in_tensor.data()[i] = 0;
  }
}

void get_random_tensor(gemini::Tensor<uint64_t> &in_tensor, int bitlength) {
  uint64_t mask_l = (uint64_t)((1ULL << bitlength) - 1);
  uint64_t mask_r = (uint64_t)((1ULL << 5) - 1);
  gemini::TensorShape shape = in_tensor.shape();
  int C = shape.channels(), H = shape.height(), W = shape.width();
  uint64_t *tmp = new uint64_t[H*W*C];
  sci::PRG128 prg;
  printf("H:%d, W:%d, C:%d\n", H, W, C);
  prg.random_data(tmp, C * H * W * sizeof(uint64_t));
  // for (int i = 0; i < H*W*C; i++) {
  //   tmp[i] = i+1;
  // }
  for (int i = 0; i < H; i++) {
    for (int j = 0; j < W; j++) {
      for (int k = 0; k < C; k++) {
        in_tensor(k, i, j) = tmp[i*W*C + j*C + k] & mask_r;
        if (in_tensor(k, i, j) == 0) {
          do {
            prg.random_data(&tmp[i*W*C + j*C + k], sizeof(uint64_t));
            in_tensor(k, i, j) = tmp[i*W*C + j*C + k] & mask_r;
          } while(in_tensor(k, i, j) == 0);
        }
      }
    }
  }
}

#ifdef LOG_LAYERWISE
#include <vector>

typedef std::vector<uint64_t> uint64_1D;
typedef std::vector<std::vector<uint64_t>> uint64_2D;
typedef std::vector<std::vector<std::vector<uint64_t>>> uint64_3D;
typedef std::vector<std::vector<std::vector<std::vector<uint64_t>>>> uint64_4D;

extern void funcReconstruct2PCCons(signedIntType *y, const intType *x, int len);

// Helper functions for computing the ground truth
// See `cleartext_library_fixed_uniform.h`
extern void Conv2DWrapper_pt(uint64_t N, uint64_t H, uint64_t W, uint64_t CI,
                             uint64_t FH, uint64_t FW, uint64_t CO,
                             uint64_t zPadHLeft, uint64_t zPadHRight,
                             uint64_t zPadWLeft, uint64_t zPadWRight,
                             uint64_t strideH, uint64_t strideW,
                             uint64_4D &inputArr, uint64_4D &filterArr,
                             uint64_4D &outArr);

extern void MatMul2DEigen_pt(int64_t i, int64_t j, int64_t k, uint64_2D &A,
                             uint64_2D &B, uint64_2D &C, int64_t consSF);

extern void ElemWiseActModelVectorMult_pt(uint64_t s1, uint64_1D &arr1,
                                          uint64_1D &arr2, uint64_1D &outArr);
#endif

void MatMul2D_seperate(int32_t d0, int32_t d1, int32_t d2, const intType *mat_A,
              const intType *mat_B, intType *mat_C, bool is_A_weight_matrix) {
#ifdef LOG_LAYERWISE
  INIT_ALL_IO_DATA_SENT;
  INIT_TIMER;
#endif
  using namespace gemini;
  CheetahLinear::FCMeta meta;

  TensorShape mat_A_shape({d0, d1});
  TensorShape mat_B_shape({d1, d2});

  TensorShape input_shape = is_A_weight_matrix ? mat_B_shape : mat_A_shape;
  TensorShape weight_shape = is_A_weight_matrix ? mat_A_shape : mat_B_shape;
  meta.input_shape = TensorShape({input_shape.dim_size(1)});
  // Transpose
  meta.weight_shape =
      TensorShape({weight_shape.dim_size(1), weight_shape.dim_size(0)});
  meta.is_shared_input = kIsSharedInput;

  auto weight_mat = is_A_weight_matrix ? mat_A : mat_B;
  auto input_mat = is_A_weight_matrix ? mat_B : mat_A;

  Tensor<intType> weight_matrix;
  if (cheetah_linear->party() == SERVER) {
    // Transpose the weight matrix and convert the uint64_t to ring element
    weight_matrix.Reshape(meta.weight_shape);
    const size_t nrows = weight_shape.dim_size(0);
    const size_t ncols = weight_shape.dim_size(1);
    printf("nrows: %d, ncols: %d\n", nrows, ncols);
    for (long r = 0; r < nrows; ++r) {
      for (long c = 0; c < ncols; ++c) {
        Arr2DIdxRowM(weight_matrix.data(), ncols, nrows, c, r) =
            getRingElt(Arr2DIdxRowM(weight_mat, nrows, ncols, r, c)); // 倒序？
      }
    }
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        printf("%lu ", weight_matrix.data()[j*2+i]);
      }
    }
    printf("\n");
  }
  for (long r = 0; r < input_shape.rows(); ++r) { // 逐行做矩阵向量积
    // row-major
    const intType *input_row = input_mat + r * input_shape.cols();

    Tensor<intType> input_vector;
    if (meta.is_shared_input) {
      input_vector = Tensor<intType>::Wrap(const_cast<intType *>(input_row),
                                           meta.input_shape);
    } else {
      input_vector.Reshape(meta.input_shape);
      std::transform(input_row, input_row + meta.input_shape.num_elements(),
                     input_vector.data(),
                     [](uint64_t v) { return getRingElt(v); });
    }

    gemini::Tensor<uint64_t> out_vec(gemini::TensorShape({meta.weight_shape.dim_size(0)}));
    //cheetah_linear->fc(input_vector, weight_matrix, meta, out_vec);
    gemini::Tensor<uint64_t> M(meta.input_shape);
    gemini::Tensor<uint64_t> offline_share(gemini::TensorShape({meta.weight_shape.dim_size(0)}));
    if (party == CLIENT) { // CLIENT端选择M
      get_random_tensor_1d(M, bitlength);
    } else {
      get_zero_tensor_1d(M);
    }
    
    cheetah_linear->fc_offline(M, weight_matrix, meta, offline_share);
    cheetah_linear->fc_online(input_vector, M, weight_matrix, meta, offline_share, out_vec);
    
    std::copy_n(out_vec.data(), out_vec.shape().num_elements(),
              mat_C + r * input_shape.cols());
  }
  if (cheetah_linear->party() == SERVER) {
    cheetah_linear->safe_erase(weight_matrix.data(),
                               meta.weight_shape.num_elements());
  }
#ifdef LOG_LAYERWISE
  auto temp = TIMER_TILL_NOW;
  Matmul_sepTimeInMilliSec += temp;
#ifndef NO_PRINT
  std::cout << "Time in sec for current matmul = " << (temp / 1000.0)
            << std::endl;
#endif
  uint64_t curComm;
  FIND_ALL_IO_TILL_NOW(curComm);
  Matmul_sepCommSent += curComm;
#endif
}

void MatMul2D(int32_t d0, int32_t d1, int32_t d2, const intType *mat_A,
              const intType *mat_B, intType *mat_C, bool is_A_weight_matrix) {
#ifdef LOG_LAYERWISE
  INIT_ALL_IO_DATA_SENT;
  INIT_TIMER;
#endif

  using namespace gemini;
  CheetahLinear::FCMeta meta;

  TensorShape mat_A_shape({d0, d1});
  TensorShape mat_B_shape({d1, d2});

  TensorShape input_shape = is_A_weight_matrix ? mat_B_shape : mat_A_shape;
  TensorShape weight_shape = is_A_weight_matrix ? mat_A_shape : mat_B_shape;
  meta.input_shape = TensorShape({input_shape.dim_size(1)});
  // Transpose
  meta.weight_shape =
      TensorShape({weight_shape.dim_size(1), weight_shape.dim_size(0)});
  meta.is_shared_input = kIsSharedInput;

  auto weight_mat = is_A_weight_matrix ? mat_A : mat_B;
  auto input_mat = is_A_weight_matrix ? mat_B : mat_A;

  Tensor<intType> weight_matrix;
  if (cheetah_linear->party() == SERVER) {
    // Transpose the weight matrix and convert the uint64_t to ring element
    weight_matrix.Reshape(meta.weight_shape);
    const size_t nrows = weight_shape.dim_size(0);
    const size_t ncols = weight_shape.dim_size(1);
    for (long r = 0; r < nrows; ++r) {
      for (long c = 0; c < ncols; ++c) {
        Arr2DIdxRowM(weight_matrix.data(), ncols, nrows, c, r) =
            getRingElt(Arr2DIdxRowM(weight_mat, nrows, ncols, r, c)); // 倒序？
      }
    }
  }
  for (long r = 0; r < input_shape.rows(); ++r) { // 逐行做矩阵向量积
    // row-major
    const intType *input_row = input_mat + r * input_shape.cols();

    Tensor<intType> input_vector;
    if (meta.is_shared_input) {
      input_vector = Tensor<intType>::Wrap(const_cast<intType *>(input_row),
                                           meta.input_shape);
    } else {
      input_vector.Reshape(meta.input_shape);
      std::transform(input_row, input_row + meta.input_shape.num_elements(),
                     input_vector.data(),
                     [](uint64_t v) { return getRingElt(v); });
    }

    Tensor<uint64_t> out_vec;
    cheetah_linear->fc(input_vector, weight_matrix, meta, out_vec);
    std::copy_n(out_vec.data(), out_vec.shape().num_elements(),
                mat_C + r * input_shape.cols());
  }

  if (cheetah_linear->party() == SERVER) {
    cheetah_linear->safe_erase(weight_matrix.data(),
                               meta.weight_shape.num_elements());
  }
#ifdef LOG_LAYERWISE
  auto temp = TIMER_TILL_NOW;
  MatMulTimeInMilliSec += temp;
#ifndef NO_PRINT
  std::cout << "Time in sec for current matmul = " << (temp / 1000.0)
            << std::endl;
#endif
  uint64_t curComm;
  FIND_ALL_IO_TILL_NOW(curComm);
  MatMulCommSent += curComm;
#endif

#ifdef VERIFY_LAYERWISE
  int s1 = d0;
  int s2 = d1;
  int s3 = d2;
  auto A = mat_A;
  auto B = mat_B;
  auto C = mat_C;
#ifdef SCI_HE
  for (int i = 0; i < s1; i++) {
    for (int j = 0; j < s3; j++) {
      assert(Arr2DIdxRowM(C, s1, s3, i, j) < prime_mod);
    }
  }
#endif
  if (party == SERVER) {
    funcReconstruct2PCCons(nullptr, A, s1 * s2);
    funcReconstruct2PCCons(nullptr, B, s2 * s3);
    funcReconstruct2PCCons(nullptr, C, s1 * s3);
  } else {
    signedIntType *VA = new signedIntType[s1 * s2];
    funcReconstruct2PCCons(VA, A, s1 * s2);
    signedIntType *VB = new signedIntType[s2 * s3];
    funcReconstruct2PCCons(VB, B, s2 * s3);
    signedIntType *VC = new signedIntType[s1 * s3];
    funcReconstruct2PCCons(VC, C, s1 * s3);

    std::vector<std::vector<uint64_t>> VAvec;
    std::vector<std::vector<uint64_t>> VBvec;
    std::vector<std::vector<uint64_t>> VCvec;
    VAvec.resize(s1, std::vector<uint64_t>(s2, 0));
    VBvec.resize(s2, std::vector<uint64_t>(s3, 0));
    VCvec.resize(s1, std::vector<uint64_t>(s3, 0));

    for (int i = 0; i < s1; i++) {
      for (int j = 0; j < s2; j++) {
        VAvec[i][j] = getRingElt(Arr2DIdxRowM(VA, s1, s2, i, j));
      }
    }
    for (int i = 0; i < s2; i++) {
      for (int j = 0; j < s3; j++) {
        VBvec[i][j] = getRingElt(Arr2DIdxRowM(VB, s2, s3, i, j));
      }
    }

    MatMul2DEigen_pt(s1, s2, s3, VAvec, VBvec, VCvec, 0);

    bool pass = true;
    for (int i = 0; i < s1; i++) {
      for (int j = 0; j < s3; j++) {
        int64_t gnd = getSignedVal(VCvec[i][j]);
        int64_t cmp = Arr2DIdxRowM(VC, s1, s3, i, j);
        if (gnd != cmp) {
          if (pass) {
            std::cout << gnd << " => " << cmp << "\n";
          }
		  if (pass && std::abs(gnd - cmp) > 1) {
            pass = false;
		  }
        }
      }
    }
    if (pass == true)
      std::cout << GREEN << "MatMul Output Matches" << RESET << std::endl;
    else
      std::cout << RED << "MatMul Output Mismatch" << RESET << std::endl;

    delete[] VA;
    delete[] VB;
    delete[] VC;
  }
#endif
}

void Conv_bias_ReLU(signedIntType N, signedIntType H, signedIntType W,
                   signedIntType CI, signedIntType FH, signedIntType FW,
                   signedIntType CO, signedIntType zPadHLeft,
                   signedIntType zPadHRight, signedIntType zPadWLeft,
                   signedIntType zPadWRight, signedIntType strideH,
                   signedIntType strideW, intType *inputArr, intType *filterArr,
                   intType *biasArr, intType *outArr, int bitlength, int scale) {
#ifdef LOG_LAYERWISE
  INIT_ALL_IO_DATA_SENT;
  INIT_TIMER;
#endif

// 让偏左、偏上的填充更大
  if (zPadWLeft < zPadWRight) {
    std::swap(zPadWLeft, zPadWRight);
  }
  if (zPadHLeft < zPadHRight) {
    std::swap(zPadHLeft, zPadHRight);
  }
  static int ctr = 1;
  signedIntType newH = (((H + (zPadHLeft + zPadHRight) - FH) / strideH) + 1);
  signedIntType newW = (((W + (zPadWLeft + zPadWRight) - FW) / strideW) + 1);

  gemini::CheetahLinear::ConvMeta meta;
  meta.ishape = gemini::TensorShape({CI, H, W});
  meta.fshape = gemini::TensorShape({CI, FH, FW});
  meta.n_filters = CO;

  std::vector<gemini::Tensor<intType>> filters(CO);
  std::vector<gemini::Tensor<intType>> bias(CO);
  for (auto &f : filters) {
    f.Reshape(meta.fshape);
  }

  for (int i = 0; i < FH; i++) {
    for (int j = 0; j < FW; j++) {
      for (int k = 0; k < CI; k++) {
        for (int p = 0; p < CO; p++) {
          filters.at(p)(k, i, j) =
              getRingElt(Arr4DIdxRowM(filterArr, FH, FW, CI, CO, i, j, k, p));
        }
      }
    }
  }

  const int npads = zPadHLeft + zPadHRight + zPadWLeft + zPadWRight;
  meta.padding = npads == 0 ? gemini::Padding::VALID : gemini::Padding::SAME;
  meta.stride = strideH;
  meta.is_shared_input = kIsSharedInput;
#ifndef NO_PRINT
  printf(
      "HomConv #%d called N=%ld, H=%ld, W=%ld, CI=%ld, FH=%ld, FW=%ld, "
      "CO=%ld, S=%ld, Padding %s (%d %d %d %d)\n",
      ctr++, N, meta.ishape.height(), meta.ishape.width(),
      meta.ishape.channels(), meta.fshape.height(), meta.fshape.width(),
      meta.n_filters, meta.stride,
      (meta.padding == gemini::Padding::VALID ? "VALID" : "SAME"), zPadHLeft,
      zPadHRight, zPadWLeft, zPadWRight);
#endif
#ifdef LOG_LAYERWISE
  const int64_t io_counter = cheetah_linear->io_counter();
#endif

  for (int i = 0; i < N; ++i) {
    gemini::Tensor<intType> image(meta.ishape);
    for (int j = 0; j < H; j++) {
      for (int k = 0; k < W; k++) {
        for (int p = 0; p < CI; p++) {
          image(p, j, k) =
              getRingElt(Arr4DIdxRowM(inputArr, N, H, W, CI, i, j, k, p));
        }
      }
    }
    // 主要操作，调用conv_bias_relu
    // 生成一个与image大小一致的随机矩阵
    gemini::Tensor<uint64_t> M(meta.ishape);
    if (party == CLIENT) { // CLIENT端选择M
      get_random_tensor(M, bitlength);
    } else {
      get_zero_tensor(M);
    }
    gemini::Tensor<intType> offline_share0;
    gemini::Tensor<intType> offline_share1;
    //cheetah_linear->conv2d(image, filters, meta, out_tensor);
    int T = 2;
    cheetah_linear->conv_bias_relu_offline(M, filters, T, meta, offline_share0, offline_share1, bitlength, scale);
    // 执行完上面的函数后，BOB方，有M、offline_share0([W*X]0)、offline_share1(V)
    // ALICE方，有filters(W)、T、 offline_share0(R0)、offline_share1(R1)  
    gemini::TensorShape shape_out = offline_share1.shape();
    int C_out = shape_out.channels(), H_out = shape_out.height(), W_out = shape_out.width();
    bool* B = new bool[C_out * H_out * W_out];
    uint64_t* result = new uint64_t[C_out * H_out * W_out];
    cheetah_linear->conv_bias_relu_online(image, M, filters, T, meta, offline_share0, offline_share1, biasArr, B, result, bitlength, scale);   
    for (int j = 0; j < H_out; j++) {
      for (int k = 0; k < W_out; k++) {
        for (int p = 0; p < CO; p++) {
          if (Arr4DIdxRowM(B, N, newH, newW, CO, i, j, k, p) == false) {
            Arr4DIdxRowM(outArr, N, newH, newW, CO, i, j, k, p) = 0;
          } else {
            if (party == SERVER) {
              Arr4DIdxRowM(outArr, N, newH, newW, CO, i, j, k, p) = result[p*newH*newW + j*newH + k];
            } else {
              Arr4DIdxRowM(outArr, N, newH, newW, CO, i, j, k, p) = offline_share0.data()[p*newH*newW + j*newH + k];
            }
          }
        }
      }
    }
  }

#ifdef LOG_LAYERWISE
  auto temp = TIMER_TILL_NOW;
  FusedBN_ReLUTimeInMilliSec += temp; // 记录卷积时间
  const int64_t nbytes_sent = cheetah_linear->io_counter() - io_counter;
#ifndef NO_PRINT
  std::cout << "Time in sec for current conv_bias_relu = [" << (temp / 1000.0)
            << "] sent [" << (nbytes_sent / 1024. / 1024.) << "] MB"
            << std::endl;
#endif
  uint64_t curComm;
  FIND_ALL_IO_TILL_NOW(curComm);
  FusedBN_ReLUCommSent += curComm;
#endif
}
// 如果BN为True，说明是由BN调用的，否则是Conv调用的
void Conv2DWrapper_seperate(signedIntType N, signedIntType H, signedIntType W,
                   signedIntType CI, signedIntType FH, signedIntType FW,
                   signedIntType CO, signedIntType zPadHLeft,
                   signedIntType zPadHRight, signedIntType zPadWLeft,
                   signedIntType zPadWRight, signedIntType strideH,
                   signedIntType strideW, intType *inputArr, intType *filterArr,
                   intType *outArr, bool BN) {
#ifdef LOG_LAYERWISE
  INIT_ALL_IO_DATA_SENT;
  INIT_TIMER;
#endif
  // 让偏左、偏上的填充更大
  if (zPadWLeft < zPadWRight) {
    std::swap(zPadWLeft, zPadWRight);
  }
  if (zPadHLeft < zPadHRight) {
    std::swap(zPadHLeft, zPadHRight);
  }
  static int ctr = 1;
  signedIntType newH = (((H + (zPadHLeft + zPadHRight) - FH) / strideH) + 1);
  signedIntType newW = (((W + (zPadWLeft + zPadWRight) - FW) / strideW) + 1);

  gemini::CheetahLinear::ConvMeta meta;
  meta.ishape = gemini::TensorShape({CI, H, W});
  meta.fshape = gemini::TensorShape({CI, FH, FW});
  meta.n_filters = CO;

  std::vector<gemini::Tensor<intType>> filters(CO);
  for (auto &f : filters) {
    f.Reshape(meta.fshape);
  }

  for (int i = 0; i < FH; i++) {
    for (int j = 0; j < FW; j++) {
      for (int k = 0; k < CI; k++) {
        for (int p = 0; p < CO; p++) {
          filters.at(p)(k, i, j) =
              getRingElt(Arr4DIdxRowM(filterArr, FH, FW, CI, CO, i, j, k, p));
        }
      }
    }
  }

  const int npads = zPadHLeft + zPadHRight + zPadWLeft + zPadWRight;
  meta.padding = npads == 0 ? gemini::Padding::VALID : gemini::Padding::SAME;
  meta.stride = strideH;
  meta.is_shared_input = kIsSharedInput;
#ifndef NO_PRINT
  printf(
      "HomConv_seperate #%d called N=%ld, H=%ld, W=%ld, CI=%ld, FH=%ld, FW=%ld, "
      "CO=%ld, S=%ld, Padding %s (%d %d %d %d)\n",
      ctr++, N, meta.ishape.height(), meta.ishape.width(),
      meta.ishape.channels(), meta.fshape.height(), meta.fshape.width(),
      meta.n_filters, meta.stride,
      (meta.padding == gemini::Padding::VALID ? "VALID" : "SAME"), zPadHLeft,
      zPadHRight, zPadWLeft, zPadWRight);
#endif
#ifdef LOG_LAYERWISE
  const int64_t io_counter = cheetah_linear->io_counter();
#endif

  for (int i = 0; i < N; ++i) {
    gemini::Tensor<intType> image(meta.ishape);
    for (int j = 0; j < H; j++) {
      for (int k = 0; k < W; k++) {
        for (int p = 0; p < CI; p++) {
          image(p, j, k) =
              getRingElt(Arr4DIdxRowM(inputArr, N, H, W, CI, i, j, k, p));
        }
      }
    }
    // 主要操作，调用conv2d
    // gemini::Tensor<intType> out_tensor;
    // cheetah_linear->conv2d(image, filters, meta, out_tensor);
    // 主要操作，调用conv2d_offline和conv2d_online
    gemini::Tensor<uint64_t> M(meta.ishape);
    if (party == CLIENT) { // CLIENT端选择M
      get_random_tensor(M, bitlength);
    } else {
      get_zero_tensor(M);
    }
    gemini::Tensor<intType> offline_share;
    cheetah_linear->conv2d_offline(M, filters, meta, offline_share, BN);
    gemini::TensorShape shape_out = offline_share.shape();
    int C_out = shape_out.channels(), H_out = shape_out.height(), W_out = shape_out.width();
    uint64_t* result = new uint64_t[C_out * H_out * W_out];
    cheetah_linear->conv2d_online(image, M, filters, meta, offline_share, result, BN);

    // for (int j = 0; j < newH; j++) {
    //   for (int k = 0; k < newW; k++) {
    //     for (int p = 0; p < CO; p++) {
    //       if (party == SERVER) {
    //         Arr4DIdxRowM(outArr, N, newH, newW, CO, i, j, k, p) = Arr3DIdxRowM(result, newH, newW, CO, j, k, p);
    //       } else {
    //         Arr4DIdxRowM(outArr, N, newH, newW, CO, i, j, k, p) = offline_share.data()[j*newW*CO + k*CO + p];
    //       }
    //     }
    //   }
    // }
    
    for (int j = 0; j < newH; j++) {
      for (int k = 0; k < newW; k++) {
        for (int p = 0; p < CO; p++) {
          if (party == SERVER) {
            //Arr4DIdxRowM(outArr, N, newH, newW, CO, i, j, k, p) = Arr3DIdxRowM(result, newH, newW, CO, j, k, p);
            Arr4DIdxRowM(outArr, N, newH, newW, CO, i, j, k, p) = result[p*newH*newW + j*newH + k];
          } else {
            Arr4DIdxRowM(outArr, N, newH, newW, CO, i, j, k, p) = offline_share.data()[p*newH*newW + j*newH + k];
          }
        }
      }
    }
  }

#ifdef LOG_LAYERWISE
  auto temp = TIMER_TILL_NOW;
if (BN == false) {
  Conv_sepTimeInMilliSec += temp; // 记录卷积时间
} else {
  BN_sepTimeInMilliSec += temp;
}
  const int64_t nbytes_sent = cheetah_linear->io_counter() - io_counter;
#ifndef NO_PRINT
  std::cout << "Time in sec for current conv = [" << (temp / 1000.0)
            << "] sent [" << (nbytes_sent / 1024. / 1024.) << "] MB"
            << std::endl;
#endif
  uint64_t curComm;
  FIND_ALL_IO_TILL_NOW(curComm);
if (BN == false) {
  Conv_sepCommSent += curComm;
} else {
  BN_sepCommSent += curComm;
}
#endif
}

void Conv2DWrapper(signedIntType N, signedIntType H, signedIntType W,
                   signedIntType CI, signedIntType FH, signedIntType FW,
                   signedIntType CO, signedIntType zPadHLeft,
                   signedIntType zPadHRight, signedIntType zPadWLeft,
                   signedIntType zPadWRight, signedIntType strideH,
                   signedIntType strideW, intType *inputArr, intType *filterArr,
                   intType *outArr, bool BN) {
#ifdef LOG_LAYERWISE
  INIT_ALL_IO_DATA_SENT;
  INIT_TIMER;
#endif
  // 让偏左、偏上的填充更大
  if (zPadWLeft < zPadWRight) {
    std::swap(zPadWLeft, zPadWRight);
  }
  if (zPadHLeft < zPadHRight) {
    std::swap(zPadHLeft, zPadHRight);
  }
  static int ctr = 1;
  signedIntType newH = (((H + (zPadHLeft + zPadHRight) - FH) / strideH) + 1);
  signedIntType newW = (((W + (zPadWLeft + zPadWRight) - FW) / strideW) + 1);

  gemini::CheetahLinear::ConvMeta meta;
  meta.ishape = gemini::TensorShape({CI, H, W});
  meta.fshape = gemini::TensorShape({CI, FH, FW});
  meta.n_filters = CO;

  std::vector<gemini::Tensor<intType>> filters(CO);
  for (auto &f : filters) {
    f.Reshape(meta.fshape);
  }

  for (int i = 0; i < FH; i++) {
    for (int j = 0; j < FW; j++) {
      for (int k = 0; k < CI; k++) {
        for (int p = 0; p < CO; p++) {
          filters.at(p)(k, i, j) =
              getRingElt(Arr4DIdxRowM(filterArr, FH, FW, CI, CO, i, j, k, p));
        }
      }
    }
  }

  const int npads = zPadHLeft + zPadHRight + zPadWLeft + zPadWRight;
  meta.padding = npads == 0 ? gemini::Padding::VALID : gemini::Padding::SAME;
  meta.stride = strideH;
  meta.is_shared_input = kIsSharedInput;
#ifndef NO_PRINT
  printf(
      "HomConv #%d called N=%ld, H=%ld, W=%ld, CI=%ld, FH=%ld, FW=%ld, "
      "CO=%ld, S=%ld, Padding %s (%d %d %d %d)\n",
      ctr++, N, meta.ishape.height(), meta.ishape.width(),
      meta.ishape.channels(), meta.fshape.height(), meta.fshape.width(),
      meta.n_filters, meta.stride,
      (meta.padding == gemini::Padding::VALID ? "VALID" : "SAME"), zPadHLeft,
      zPadHRight, zPadWLeft, zPadWRight);
#endif
#ifdef LOG_LAYERWISE
  const int64_t io_counter = cheetah_linear->io_counter();
#endif

  for (int i = 0; i < N; ++i) {
    gemini::Tensor<intType> image(meta.ishape);
    for (int j = 0; j < H; j++) {
      for (int k = 0; k < W; k++) {
        for (int p = 0; p < CI; p++) {
          image(p, j, k) =
              getRingElt(Arr4DIdxRowM(inputArr, N, H, W, CI, i, j, k, p));
        }
      }
    }
    // 主要操作，调用conv2d
    gemini::Tensor<intType> out_tensor;
    cheetah_linear->conv2d(image, filters, meta, out_tensor);

    for (int j = 0; j < newH; j++) {
      for (int k = 0; k < newW; k++) {
        for (int p = 0; p < CO; p++) {
          Arr4DIdxRowM(outArr, N, newH, newW, CO, i, j, k, p) =
              out_tensor(p, j, k);
        }
      }
    }
  }

#ifdef LOG_LAYERWISE
  auto temp = TIMER_TILL_NOW;
  ConvTimeInMilliSec += temp; // 记录卷积时间
  const int64_t nbytes_sent = cheetah_linear->io_counter() - io_counter;
#ifndef NO_PRINT
  std::cout << "Time in sec for current conv = [" << (temp / 1000.0)
            << "] sent [" << (nbytes_sent / 1024. / 1024.) << "] MB"
            << std::endl;
#endif
  uint64_t curComm;
  FIND_ALL_IO_TILL_NOW(curComm);
  ConvCommSent += curComm;
#endif

#ifdef VERIFY_LAYERWISE
#ifdef SCI_HE
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < newH; j++) {
      for (int k = 0; k < newW; k++) {
        for (int p = 0; p < CO; p++) {
          assert(Arr4DIdxRowM(outArr, N, newH, newW, CO, i, j, k, p) <
                 prime_mod);
        }
      }
    }
  }
#endif  // SCI_HE

  if (party == SERVER) {
    funcReconstruct2PCCons(nullptr, inputArr, N * H * W * CI);
    funcReconstruct2PCCons(nullptr, filterArr, FH * FW * CI * CO);
    funcReconstruct2PCCons(nullptr, outArr, N * newH * newW * CO);
  } else {
    signedIntType *VinputArr = new signedIntType[N * H * W * CI];
    funcReconstruct2PCCons(VinputArr, inputArr, N * H * W * CI);
    signedIntType *VfilterArr = new signedIntType[FH * FW * CI * CO];
    funcReconstruct2PCCons(VfilterArr, filterArr, FH * FW * CI * CO);
    signedIntType *VoutputArr = new signedIntType[N * newH * newW * CO];
    funcReconstruct2PCCons(VoutputArr, outArr, N * newH * newW * CO);

    std::vector<std::vector<std::vector<std::vector<uint64_t>>>> VinputVec;
    VinputVec.resize(N, std::vector<std::vector<std::vector<uint64_t>>>(
                            H, std::vector<std::vector<uint64_t>>(
                                   W, std::vector<uint64_t>(CI, 0))));

    std::vector<std::vector<std::vector<std::vector<uint64_t>>>> VfilterVec;
    VfilterVec.resize(FH, std::vector<std::vector<std::vector<uint64_t>>>(
                              FW, std::vector<std::vector<uint64_t>>(
                                      CI, std::vector<uint64_t>(CO, 0))));

    std::vector<std::vector<std::vector<std::vector<uint64_t>>>> VoutputVec;
    VoutputVec.resize(N, std::vector<std::vector<std::vector<uint64_t>>>(
                             newH, std::vector<std::vector<uint64_t>>(
                                       newW, std::vector<uint64_t>(CO, 0))));

    for (int i = 0; i < N; i++) {
      for (int j = 0; j < H; j++) {
        for (int k = 0; k < W; k++) {
          for (int p = 0; p < CI; p++) {
            VinputVec[i][j][k][p] =
                getRingElt(Arr4DIdxRowM(VinputArr, N, H, W, CI, i, j, k, p));
          }
        }
      }
    }
    for (int i = 0; i < FH; i++) {
      for (int j = 0; j < FW; j++) {
        for (int k = 0; k < CI; k++) {
          for (int p = 0; p < CO; p++) {
            VfilterVec[i][j][k][p] = getRingElt(
                Arr4DIdxRowM(VfilterArr, FH, FW, CI, CO, i, j, k, p));
          }
        }
      }
    }

    Conv2DWrapper_pt(N, H, W, CI, FH, FW, CO, zPadHLeft, zPadHRight, zPadWLeft,
                     zPadWRight, strideH, strideW, VinputVec, VfilterVec,
                     VoutputVec);

    bool pass = true;
    int err_cnt = 0;
    int pos_one = 0, neg_one = 0;
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < newH; j++) {
        for (int k = 0; k < newW; k++) {
          for (int p = 0; p < CO; p++) {
            int64_t gnd = Arr4DIdxRowM(VoutputArr, N, newH, newW, CO, i, j, k, p);
            int64_t cmp = getSignedVal(VoutputVec[i][j][k][p]);
            int64_t diff = gnd - cmp;

            if (diff != 0) {

              if (diff > 0 && pos_one < 2) {
                std::cout << "expect " << gnd << " but got " << cmp << "\n";
              } 

			  if (diff < 0 && neg_one < 2) {
                std::cout << "expect " << gnd << " but got " << cmp << "\n";
              }

              pos_one += (diff > 0);
              neg_one += (diff < 0);
			  if (pass && std::abs(diff) > 1) {
				pass = false;
			  }
              ++err_cnt;
            }
          }
        }
      }
    }

    if (pass == true) {
      std::cout << GREEN << "Convolution Output Matches" << RESET << std::endl;
    } else {
      std::cout << RED << "Convolution Output Mismatch" << RESET << std::endl;
      printf("Error count %d (%d +1, %d -1). %f\%\n", err_cnt, pos_one,
             neg_one, static_cast<double>(err_cnt) * 100. / (N * newH * newW * CO));
    }

    delete[] VinputArr;
    delete[] VfilterArr;
    delete[] VoutputArr;
  }
#endif  // VERIFY_LAYERWISE
}

void BatchNorm(int32_t B, int32_t H, int32_t W, int32_t C,
               const intType *inputArr, const intType *scales,
               const intType *bias, intType *outArr) {
#ifdef LOG_LAYERWISE
  INIT_ALL_IO_DATA_SENT;
  INIT_TIMER;
#endif
  static int batchNormCtr = 1;

  gemini::CheetahLinear::BNMeta meta;
  meta.target_base_mod = prime_mod;
  meta.is_shared_input = kIsSharedInput;
  meta.ishape = gemini::TensorShape({C, H, W});
#ifndef NO_PRINT
  std::cout << "HomBN #" << batchNormCtr << " on shape " << meta.ishape
            << std::endl;
#endif
  batchNormCtr++;

  gemini::Tensor<intType> scale_vec;
  scale_vec.Reshape(gemini::TensorShape({C}));
  if (cheetah_linear->party() == SERVER) {
    std::transform(scales, scales + C, scale_vec.data(), getRingElt);
  }

  gemini::Tensor<intType> in_tensor(meta.ishape);
  gemini::Tensor<intType> out_tensor;
  for (int b = 0; b < B; ++b) {
    for (int32_t h = 0; h < H; ++h) {
      for (int32_t w = 0; w < W; ++w) {
        for (int32_t c = 0; c < C; ++c) {
          in_tensor(c, h, w) =
              getRingElt(Arr4DIdxRowM(inputArr, B, H, W, C, b, h, w, c));
        }
      }
    }
    // 调用CheetahLinear::bn_direct进行BN计算
    cheetah_linear->bn_direct(in_tensor, scale_vec, meta, out_tensor);

    for (int32_t h = 0; h < H; ++h) {
      for (int32_t w = 0; w < W; ++w) {
        for (int32_t c = 0; c < C; ++c) {
          Arr4DIdxRowM(outArr, B, H, W, C, b, h, w, c) =
              SecretAdd(out_tensor(c, h, w), bias[c]);//加上bias
        }
      }
    }
  }

  if (cheetah_linear->party() == SERVER) {
    cheetah_linear->safe_erase(scale_vec.data(), scale_vec.NumElements());
  }

#ifdef LOG_LAYERWISE
  auto temp = TIMER_TILL_NOW;
  BatchNormInMilliSec += temp;
  uint64_t curComm;
  FIND_ALL_IO_TILL_NOW(curComm);
  BatchNormCommSent += curComm;
#ifndef NO_PRINT
  std::cout << "Time in sec for current BN = [" << (temp / 1000.0) << "] sent ["
            << (curComm / 1024. / 1024.) << "] MB" << std::endl;
#endif
#endif
}

void ElemWiseActModelVectorMult(int32_t size, intType *inArr,
                                intType *multArrVec, intType *outputArr) {
#ifdef LOG_LAYERWISE
  INIT_ALL_IO_DATA_SENT;
  INIT_TIMER;
#endif

  static int batchNormCtr = 1;
#ifndef NO_PRINT
  printf("HomBN #%d via element-wise mult on %d points\n", batchNormCtr++,
         size);
#endif
  gemini::CheetahLinear::BNMeta meta;
  meta.target_base_mod = prime_mod;
  meta.is_shared_input = kIsSharedInput;
  meta.vec_shape = gemini::TensorShape({size});

  gemini::Tensor<intType> in_vec;
  gemini::Tensor<intType> scale_vec;
  scale_vec.Reshape(meta.vec_shape);
  if (cheetah_linear->party() == SERVER) {
    std::transform(multArrVec, multArrVec + size, scale_vec.data(), getRingElt);
  }

  if (meta.is_shared_input) {
    in_vec = gemini::Tensor<intType>::Wrap(inArr, meta.vec_shape);
  } else {
    in_vec.Reshape(meta.vec_shape);
    std::transform(inArr, inArr + size, in_vec.data(), getRingElt);
  }
  gemini::Tensor<intType> out_vec;
  // 调用CheetahLinear::bn进行计算
  cheetah_linear->bn(in_vec, scale_vec, meta, out_vec);
  std::copy_n(out_vec.data(), out_vec.shape().num_elements(), outputArr);

  if (cheetah_linear->party() == SERVER) {
    cheetah_linear->safe_erase(scale_vec.data(), scale_vec.NumElements());
  }

#ifdef LOG_LAYERWISE
  auto temp = TIMER_TILL_NOW;
  BatchNormInMilliSec += temp;
  uint64_t curComm;
  FIND_ALL_IO_TILL_NOW(curComm);
  BatchNormCommSent += curComm;
#endif

#ifdef VERIFY_LAYERWISE
  for (int i = 0; i < size; i++) {
    assert(outputArr[i] < prime_mod);
  }

  if (party == SERVER) {
    funcReconstruct2PCCons(nullptr, inArr, size);
    funcReconstruct2PCCons(nullptr, multArrVec, size);
    funcReconstruct2PCCons(nullptr, outputArr, size);
  } else {
    signedIntType *VinArr = new signedIntType[size];
    funcReconstruct2PCCons(VinArr, inArr, size);
    signedIntType *VmultArr = new signedIntType[size];
    funcReconstruct2PCCons(VmultArr, multArrVec, size);
    signedIntType *VoutputArr = new signedIntType[size];
    funcReconstruct2PCCons(VoutputArr, outputArr, size);

    std::vector<uint64_t> VinVec(size);
    std::vector<uint64_t> VmultVec(size);
    std::vector<uint64_t> VoutputVec(size);

    for (int i = 0; i < size; i++) {
      VinVec[i] = getRingElt(VinArr[i]);
      VmultVec[i] = getRingElt(VmultArr[i]);
    }

    ElemWiseActModelVectorMult_pt(size, VinVec, VmultVec, VoutputVec);

    bool pass = true;
    for (int i = 0; i < size; i++) {
      int64_t gnd = getSignedVal(VoutputVec[i]);
      int64_t cmp = VoutputArr[i];
      if (gnd != cmp) {
        if (pass) {
          std::cout << RED << gnd << " ==> " << cmp << RESET << std::endl;
        }
        pass = false;
      }
    }
    if (pass == true)
      std::cout << GREEN << "ElemWiseSecretVectorMult Output Matches" << RESET
                << std::endl;
    else
      std::cout << RED << "ElemWiseSecretVectorMult Output Mismatch" << RESET
                << std::endl;

    delete[] VinArr;
    delete[] VmultArr;
    delete[] VoutputArr;
  }
#endif
}
#endif

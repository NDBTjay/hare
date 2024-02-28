#ifndef RELU_HE_H__
#define RELU_HE_H__

#include "globals.h"
#include "library_fixed_uniform.h"
#include <seal/seal.h>
#include "NonLinear/relu-interface.h"
#include "BuildingBlocks/aux-protocols.h"
#include "cheetah/cheetah-api.h"
#include "utils/emp-tool.h"
#include "defines_uniform.h"

using namespace seal;
using namespace std;
using namespace sci;
extern sci::NetIO *ioArr[4];
extern uint64_t ReluSetupTimeInMilliSec;
extern uint64_t ReluOfflineTimeInMilliSec;
extern uint64_t ReluOnlineTimeInMilliSec;
extern uint64_t ReluSetupCommSent;
extern uint64_t ReluOfflineCommSent;
extern uint64_t ReluOnlineCommSent;
template <typename IO, typename type>
class ReLUSsProtocol : public ReLUProtocol<IO, type> {
private:
    Encryptor *encryptor_;
    Evaluator *evaluator_;
    Decryptor *decryptor_;
    BatchEncoder *batch_encoder_;
    SEALContext *context_;
    SecretKey sk_;
    PublicKey pk_;

public:
    IO *io_ = nullptr;
    int party;
    int num_relu;
    int l;
    sci::OTPack<IO> *otpack;
    TripleGenerator<IO> *triple_gen = nullptr;
    MillionaireProtocol<IO> *millionaire = nullptr;
    uint64_t mask_l;
    uint64_t mask_u = (uint64_t)((1ULL << 3) - 1);
    uint64_t mask_r = (uint64_t)((1ULL << 10) - 1);
    uint64_t poly_modulus_degree;
    PRG128 prg;
    uint64_t max;

    ReLUSsProtocol(int party, IO *io, int l, sci::OTPack<IO> *otpack) {
        // printf("party : %d, io: %lu, otpack: %lu\n", party, io, otpack);
        this->party = party;
        this->io_ = io;
        this->l = l;
        this->max = 0;
        this->otpack = otpack;
        this->millionaire = new MillionaireProtocol<IO>(party, io, otpack);
        this->triple_gen = this->millionaire->triple_gen;
        configure();
        setup();
    }

    virtual ~ReLUSsProtocol() { return; }

    void configure() {
        if (this->l != 32 && this->l != 64) {
            mask_l = (type)((1ULL << l) - 1);
        } else if (this->l == 32) {
            mask_l = -1;
        } else {  // l = 64
            mask_l = -1ULL;
        }
    }

    void setup() {
#ifdef LOG_LAYERWISE
    INIT_ALL_IO_DATA_SENT;
    INIT_TIMER;
#endif
        EncryptionParameters seal_parms(scheme_type::bfv);
        seal_parms.set_n_special_primes(0);
        std::vector<int> moduli_bits{60, 49};
        poly_modulus_degree = 4096;
        seal_parms.set_poly_modulus_degree(poly_modulus_degree);
        seal_parms.set_coeff_modulus(CoeffModulus::Create(4096, moduli_bits));
        auto coeff_modulu = seal_parms.coeff_modulus();
        seal_parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, l+3));
        auto plain_modulu = seal_parms.plain_modulus();
        printf("coeff_modulu: %lu, plain_modulu: %lu\n", coeff_modulu, plain_modulu);
        context_ = new SEALContext(seal_parms, true, seal::sec_level_type::none);
        if (party == sci::BOB) { // BOB generate sk、pk and send pk to ALICE
            KeyGenerator keygen(*context_);
            sk_ = keygen.secret_key();
            keygen.create_public_key(pk_);
            Serializable<PublicKey> s_pk = keygen.create_public_key();
            std::stringstream os;
            s_pk.save(os);//将s_pk序列化后存储到os这个stream中
            uint64_t pk_sze = static_cast<uint64_t>(os.tellp());
            //uint64_t pk_sze = os.tellp();
            const std::string &keys_str = os.str();
            io_->send_data(&pk_sze, sizeof(uint64_t));//发送pk长度
            printf("send pk_sze: %lu\n", pk_sze);
            io_->send_data(keys_str.c_str(), pk_sze);//发送pk
            printf("send second");
            encryptor_ = new Encryptor(*context_, pk_);
            evaluator_ = new Evaluator(*context_);
            decryptor_ = new Decryptor(*context_, sk_);
            batch_encoder_ = new BatchEncoder(*context_);
        } else { // ALICE receive pk
            uint64_t pk_sze{0};
            //uint64_t pk_sze;
            io_->recv_data(&pk_sze, sizeof(uint64_t));
            printf("recv pk_sze: %lu\n", pk_sze);
            char *key_buf = new char[pk_sze];
            io_->recv_data(key_buf, pk_sze);//在fread卡住了
            printf("recv second\n");
            std::stringstream is;
            is.write(key_buf, pk_sze);
            pk_.load(*context_, is);
            delete[] key_buf;
            encryptor_ = new Encryptor(*context_, pk_);
            evaluator_ = new Evaluator(*context_);
            batch_encoder_ = new BatchEncoder(*context_);
        }
        io_->flush();
#ifdef LOG_LAYERWISE
    auto temp = TIMER_TILL_NOW;
    ReluSetupTimeInMilliSec += temp;
    uint64_t curComm;
    FIND_ALL_IO_TILL_NOW(curComm);
    ReluSetupCommSent += curComm;
#endif
    }

    void offline(uint64_t *u, uint64_t *v, uint64_t *s, uint64_t *t, int size) {
#ifdef LOG_LAYERWISE
    INIT_ALL_IO_DATA_SENT;
    INIT_TIMER;
#endif
        if (party == sci::BOB) {
            // 阶段1，产生并发送[u1], [v1]
            vector<uint64_t> u1(u, u+size), v1(v, v+size), s1(s, s+size), t1(t, t+size);
            Plaintext u1_p, v1_p;
            Ciphertext u1_s, v1_s;
            batch_encoder_->encode(u1, u1_p);
            batch_encoder_->encode(v1, v1_p);
            vector<uint64_t>().swap(u1); // 释放vector空间
            vector<uint64_t>().swap(v1);
            encryptor_->encrypt(u1_p, u1_s);
            encryptor_->encrypt(v1_p, v1_s);
            send_ciphertext(io_, u1_s);
            send_ciphertext(io_, v1_s);
            // 阶段2，接收[s1], [t1]，并进行解密
            Ciphertext u1_mul_v2_sub_s2_s, u2_mul_v1_sub_t2_s;
            recv_ciphertext(io_, *context_, u1_mul_v2_sub_s2_s, false);
            recv_ciphertext(io_, *context_, u2_mul_v1_sub_t2_s, false);
            Plaintext s1_p, t1_p;
            decryptor_->decrypt(u1_mul_v2_sub_s2_s, s1_p);
            decryptor_->decrypt(u2_mul_v1_sub_t2_s, t1_p);
            batch_encoder_->decode(s1_p, s1);
            batch_encoder_->decode(t1_p, t1);
            for (int i = 0; i < size; i++) {
                s1[i] &= mask_l;
                t1[i] &= mask_l;
            }
            memcpy(s, &s1[0], size * sizeof(uint64_t));
            memcpy(t, &t1[0], size * sizeof(uint64_t));
            vector<uint64_t>().swap(s1);
            vector<uint64_t>().swap(t1);
        } else { //ALICE
            //阶段1，接收[u1], [v1]
            Ciphertext u1_s, v1_s;
            recv_ciphertext(io_, *context_, u1_s, false);
            recv_ciphertext(io_, *context_, v1_s, false);
            vector<uint64_t> u2(u, u+size), v2(v, v+size), s2(s, s+size), t2(t, t+size);
            Plaintext u2_p, v2_p, s2_p, t2_p;
            batch_encoder_->encode(u2, u2_p);
            batch_encoder_->encode(v2, v2_p);
            batch_encoder_->encode(s2, s2_p);
            batch_encoder_->encode(t2, t2_p);
            vector<uint64_t>().swap(u2);
            vector<uint64_t>().swap(v2);
            vector<uint64_t>().swap(s2);
            vector<uint64_t>().swap(t2);
            //阶段2，计算并发送[s1]=[u1*v2-s2], [t1]=[u2*v1-t2]
            Ciphertext u1_mul_v2_s, u1_mul_v2_sub_s2_s, u2_mul_v1_s, u2_mul_v1_sub_t2_s;
            evaluator_->multiply_plain(u1_s, v2_p, u1_mul_v2_s);
            evaluator_->sub_plain(u1_mul_v2_s, s2_p, u1_mul_v2_sub_s2_s);
            evaluator_->multiply_plain(v1_s, u2_p, u2_mul_v1_s);
            evaluator_->sub_plain(u2_mul_v1_s, t2_p, u2_mul_v1_sub_t2_s);
            send_ciphertext(io_, u1_mul_v2_sub_s2_s);
            send_ciphertext(io_, u2_mul_v1_sub_t2_s);
        }
        io_->flush();
#ifdef LOG_LAYERWISE
    auto temp = TIMER_TILL_NOW;
    ReluOfflineTimeInMilliSec += temp;
    uint64_t curComm;
    FIND_ALL_IO_TILL_NOW(curComm);
    ReluOfflineCommSent += curComm;
#endif
    }

    void online(uint64_t *u, uint64_t *v, uint64_t *s, uint64_t *t, int size,
        uint64_t *result, uint64_t *share, uint8_t *drelu_res = nullptr) {
#ifdef LOG_LAYERWISE
    INIT_ALL_IO_DATA_SENT;
    INIT_TIMER;
#endif
        if (party == sci::BOB) {
            //printf("x1: %lu u1: %lu, v1:%lu, s1:%lu, t1:%lu\n", share[0], u[0], r[0], s[0], t[0]);
            // 计算x1-v1， 接收x2-v2
            uint64_t *x1_sub_v1 = new uint64_t[size];
            uint64_t *x2_sub_v2 = new uint64_t[size];
            for (int i = 0; i < size; i++) {
                x1_sub_v1[i] = (share[i] - v[i]) & mask_l;
            }
            io_->send_data(x1_sub_v1, size * sizeof(uint64_t));
            io_->recv_data(x2_sub_v2, size * sizeof(uint64_t));
            // 计算k1=(u1*x2_sub_v2 + s1 +t1)，进而计算k1+x1*u1，接收k2+x2*u2
            uint64_t *k1 = new uint64_t[size];
            uint64_t *x1_mul_u1_add_k1 = new uint64_t[size];
            uint64_t *x2_mul_u2_add_k2 = new uint64_t[size];
            for (int i = 0; i < size; i++) {
                k1[i] = (u[i] * x2_sub_v2[i] + s[i] +t[i]) & mask_l;
                x1_mul_u1_add_k1[i] = (share[i] * u[i] + k1[i]) & mask_l;
            }
            io_->send_data(x1_mul_u1_add_k1, size * sizeof(uint64_t));
            io_->recv_data(x2_mul_u2_add_k2, size * sizeof(uint64_t));
            // 计算V=(x2_mul_u2_add_k2 + k1 + x1 * u1)
            uint64_t *K = new uint64_t[size];
            for (int i = 0; i < size; i++) {
                K[i] = (x2_mul_u2_add_k2[i] + k1[i] + share[i] * u[i]) & mask_l;
                //printf("%d relu: u1: %lu V: %lu\n", i + 1, u[i], K[i]);
            }
            // 输出drelu和result
            for (int i = 0; i < size; i++) {
                if (K[i] >> (l-1) & 1) {//最高位为1，则为负数
                    if (drelu_res != nullptr) {
                        drelu_res[i] = 0;
                    }
                    if (result != nullptr) {
                        result[i] = 0;
                    }
                } else { //为正数，保持share不变
                    if (drelu_res != nullptr) {
                        drelu_res[i] = 1;
                    }
                    if (result != nullptr) {
                        result[i] = share[i];
                    }
                }
            }
            delete(x1_sub_v1), x1_sub_v1 = NULL;
            delete(x2_sub_v2), x2_sub_v2 = NULL;
            delete(k1), k1 = NULL;
            delete(x1_mul_u1_add_k1), x1_mul_u1_add_k1 = NULL;
            delete(x2_mul_u2_add_k2), x2_mul_u2_add_k2 = NULL;
            delete(K), K = NULL;
        } else {
            //printf("x2: %lu, u2: %lu, v2:%lu, s2:%lu, t2:%lu\n", share[0], u[0], r[0], s[0], t[0]);
            // 计算x2-v2，接收x1-v1
            uint64_t *x1_sub_v1 = new uint64_t[size];
            uint64_t *x2_sub_v2 = new uint64_t[size];
            for (int i = 0; i < size; i++) {
                x2_sub_v2[i] = (share[i] - v[i]) & mask_l;
            }
            io_->recv_data(x1_sub_v1, size * sizeof(uint64_t));
            io_->send_data(x2_sub_v2, size * sizeof(uint64_t));
            // 计算k2=(u2*x1_sub_v1 + s2 +t2)，进而// 计算k2+x2*u2，接收k1+x1*u1
            uint64_t *k2 = new uint64_t[size];
            uint64_t *x1_mul_u1_add_k1 = new uint64_t[size];
            uint64_t *x2_mul_u2_add_k2 = new uint64_t[size];
            for (int i = 0; i < size; i++) {
                k2[i] = (u[i] * x1_sub_v1[i] + s[i] +t[i]) & mask_l;
                x2_mul_u2_add_k2[i] = (share[i] * u[i] + k2[i]) & mask_l;
            }
            io_->recv_data(x1_mul_u1_add_k1, size * sizeof(uint64_t));
            io_->send_data(x2_mul_u2_add_k2, size * sizeof(uint64_t));
            // 计算V=(x1_mul_u1_add_k1 + k2 + x2 * u2)
            uint64_t *K = new uint64_t[size];
            for (int i = 0; i < size; i++) {
                K[i] = (x1_mul_u1_add_k1[i] + k2[i] + share[i] * u[i]) & mask_l;
                //printf("%d relu: u2: %lu V: %lu\n", i + 1, u[i], K[i]);
            }
            // 输出drelu和result
            for (int i = 0; i < size; i++) {
                if (K[i] >> (l-1) & 1) {//最高位为1，则为负数
                    if (drelu_res != nullptr) {
                        drelu_res[i] = 0;
                    }
                    if (result != nullptr) {
                        result[i] = 0;
                    }
                } else { //为正数，保持share不变
                    if (drelu_res != nullptr) {
                        drelu_res[i] = 1;
                    }
                    if (result != nullptr) {
                        result[i] = share[i];
                    }
                }
            }
            delete(x1_sub_v1), x1_sub_v1 = NULL;
            delete(x2_sub_v2), x2_sub_v2 = NULL;
            delete(k2), k2 = NULL;
            delete(x1_mul_u1_add_k1), x1_mul_u1_add_k1 = NULL;
            delete(x2_mul_u2_add_k2), x2_mul_u2_add_k2 = NULL;
            delete(K), K = NULL;
        }
#ifdef LOG_LAYERWISE
    auto temp = TIMER_TILL_NOW;
    ReluOnlineTimeInMilliSec += temp;
    uint64_t curComm;
    FIND_ALL_IO_TILL_NOW(curComm);
    ReluOnlineCommSent += curComm;
#endif
    }

    void relu(uint64_t *result, uint64_t *share, int num_relu,
            uint8_t *drelu_res = nullptr, bool skip_ot = false) {
        this->num_relu = num_relu;
        uint64_t *u = new uint64_t[num_relu];
        uint64_t *v = new uint64_t[num_relu];
        uint64_t *s = new uint64_t[num_relu];
        uint64_t *t = new uint64_t[num_relu];
        // uint64_t *max_ = new uint64_t[num_relu];
        // for (int i = 0; i < num_relu; i++) {
        //     if ((share[i] >> 63) == 1) {
        //         max_[i] = 0;
        //     } else {
        //         max_[i] = share[i];
        //     }
        //     share[i] &= mask_l;
        // }
        // for (int i = 0; i < num_relu; i++) {
        //     if (max_[i] > max) {
        //         max = max_[i];
        //     }
        // }
        // //printf("max of input: %lu\n", max);
        if (party == sci::BOB) {
            generate_rand(u, mask_u, num_relu);
            generate_rand(v, mask_r, num_relu);
        } else {
            generate_rand(u, mask_u, num_relu);
            generate_rand(v, mask_r, num_relu);
            generate_rand(s, mask_r, num_relu);
            generate_rand(t, mask_r, num_relu);
        }
        uint64_t chunk_size = poly_modulus_degree;
        uint64_t num_chunk = (num_relu + chunk_size - 1) / chunk_size;
        uint64_t rest = num_relu % chunk_size;
        uint64_t *result_pos;
        uint8_t *drelu_res_pos;

        for (int i = 0; i < num_chunk; i++) {
            if (result == nullptr) { //后面部分代码调用存在result = nullptr，此处为方便处理多种情况，在这里设置值
                result_pos = nullptr;
            } else {
                result_pos = result + i * chunk_size;
            }
            if (drelu_res == nullptr) { //后面代码调用存在drelu = nullptr
                drelu_res_pos = nullptr;
            } else {
                drelu_res_pos = drelu_res + i * chunk_size;
            }
            if (i == num_chunk - 1 && rest != 0) {
                offline((u + i * chunk_size), (v + i * chunk_size), 
                    (s + i * chunk_size), (t + i * chunk_size), rest);
                online((u + i *chunk_size), (v + i * chunk_size), 
                    (s + i * chunk_size), (t + i * chunk_size), rest,
                    result_pos, (share + i * chunk_size), drelu_res_pos);
            } else {
                offline((u + i * chunk_size), (v + i * chunk_size), 
                    (s + i * chunk_size), (t + i * chunk_size), chunk_size);
                online((u + i *chunk_size), (v + i * chunk_size), 
                    (s + i * chunk_size), (t + i * chunk_size), chunk_size,
                    result_pos, (share + i * chunk_size), drelu_res_pos);
            }
        }

        delete(u), u = NULL;
        delete(v), v = NULL;
        delete(s), s = NULL;
        delete(t), t = NULL;
    }

    void generate_rand(uint64_t *num, uint64_t mask, int num_relu) {
        prg.random_data(num, num_relu * sizeof(uint64_t));
        for (int i = 0; i < num_relu; i++) {
            num[i] &= mask;
            if (num[i] == 0) {// 避免产生随机数为0
                do {
                    prg.random_data(&num[i], sizeof(uint64_t));
                    num[i] &= mask;
                } while(num[i] == 0);
            }
        }
    }

    void send_ciphertext(IO *io_, seal::Ciphertext &ct) {
        std::stringstream os;
        uint64_t ct_size;
        ct.save(os);
        ct_size = os.tellp();
        string ct_ser = os.str();
        //printf("send ct_size: %lu\n", ct_size);
        io_->send_data(&ct_size, sizeof(uint64_t));
        io_->send_data(ct_ser.c_str(), ct_ser.size());
    }

    void recv_ciphertext(IO *io_, const seal::SEALContext &context,
                        seal::Ciphertext &ct, bool is_truncated) {
        std::stringstream is;
        uint64_t ct_size;
        io_->recv_data(&ct_size, sizeof(uint64_t));
        //printf("recv ct_size: %lu\n", ct_size);
        char *c_enc_result = new char[ct_size];
        io_->recv_data(c_enc_result, ct_size);
        is.write(c_enc_result, ct_size);
        if (is_truncated) {
            ct.unsafe_load(context, is);
        } else {
            ct.load(context, is);
        }
        delete[] c_enc_result;
    }
};

#endif //RELU_HE_H__
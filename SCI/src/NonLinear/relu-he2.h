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
extern uint64_t ReluOfflineTimeInMilliSec1;
extern uint64_t ReluOfflineTimeInMilliSec;
extern uint64_t ReluOnlineTimeInMilliSec;
extern uint64_t ReluSetupCommSent;
extern uint64_t ReluOfflineCommSent1;
extern uint64_t ReluOfflineCommSent;
extern uint64_t ReluOnlineCommSent;
extern uint64_t PoolOfflineTimeInMilliSec;
extern uint64_t PoolOfflineCommSent;
extern uint64_t PoolOnlineTimeInMilliSec;
extern uint64_t PoolOnlineCommSent;
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
    bool rand_set;
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
        this->rand_set = false;
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
        // printf("coeff_modulu: %lu, plain_modulu: %lu\n", coeff_modulu, plain_modulu);
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
            io_->send_data(keys_str.c_str(), pk_sze);//发送pk
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
    uint64_t curComm;
    FIND_ALL_IO_TILL_NOW(curComm);
if (io_ == ioArr[0]) {
    ReluSetupTimeInMilliSec += temp;
    ReluSetupCommSent += curComm;
#endif
}
    }

    void offline(uint64_t *Vect1, uint64_t *Vect2, int size, bool skip_ot) {
        if (party == sci::BOB) {
#ifdef LOG_LAYERWISE
    INIT_ALL_IO_DATA_SENT;
    INIT_TIMER;
#endif
            if (rand_set == false) {
                generate_rand(Vect1 ,mask_r, num_relu); // 随机生成M
                rand_set = true;
            }
            // 阶段1，发送[M]
            vector<uint64_t> M_v(Vect1, Vect1+size), V_v(Vect2, Vect2+size);
            Plaintext M_p;
            Ciphertext M_s;
            batch_encoder_->encode(M_v, M_p);
            vector<uint64_t>().swap(M_v); // 释放vector空间
            encryptor_->encrypt(M_p, M_s);
            send_ciphertext(io_, M_s);
#ifdef LOG_LAYERWISE
    auto temp1 = TIMER_TILL_NOW;
    uint64_t curComm1;
    FIND_ALL_IO_TILL_NOW(curComm1);
if (io_ == ioArr[0]) {
    if (skip_ot == true) { // relu type
        ReluOfflineTimeInMilliSec1 += temp1;
        ReluOfflineCommSent1 += curComm1;
    }
}
#endif
            // 阶段2，接收[M ⊗ T - R]，并解密到V
            Ciphertext M_mul_T_sub_R_s;
            Plaintext V_p;
            recv_ciphertext(io_, *context_, M_mul_T_sub_R_s, false);
            decryptor_->decrypt(M_mul_T_sub_R_s, V_p);
            batch_encoder_->decode(V_p, V_v);
            for (int i = 0; i < size; i++) {
                V_v[i] &= mask_l;
            }
            memcpy(Vect2, &V_v[0], size * sizeof(uint64_t)); //将结果保存到V
            vector<uint64_t>().swap(V_v);
            //io_->flush();
        } else {
#ifdef LOG_LAYERWISE
    INIT_ALL_IO_DATA_SENT;
    INIT_TIMER;
#endif
            if (rand_set == false) {
                generate_rand(Vect1 ,mask_u, num_relu); // 随机生成T
                generate_rand(Vect2 ,mask_r, num_relu); // 随机生成R
                rand_set = true;
            }
            // 阶段1，接收[M]
            vector<uint64_t> T_v(Vect1, Vect1+size), R_v(Vect2,Vect2+size);
            Ciphertext M_s, T_s, R_s;
            Plaintext M_p, T_p, R_p;
            batch_encoder_->encode(T_v, T_p);
            batch_encoder_->encode(R_v, R_p);
            recv_ciphertext(io_, *context_, M_s, false);
#ifdef LOG_LAYERWISE
    auto temp1 = TIMER_TILL_NOW;
    uint64_t curComm1;
    FIND_ALL_IO_TILL_NOW(curComm1);
if (io_ == ioArr[0]) {
    if (skip_ot == true) { // relu type
        ReluOfflineTimeInMilliSec1 += temp1;
        ReluOfflineCommSent1 += curComm1;
    }
#endif
}
        vector<uint64_t>().swap(T_v);
        vector<uint64_t>().swap(R_v);
        // 阶段2，计算并发送[M ⊗ T - R]
        Ciphertext M_mul_T_s, M_mul_T_sub_R_s;
        evaluator_->multiply_plain(M_s, T_p, M_mul_T_s);
        evaluator_->sub_plain(M_mul_T_s, R_p, M_mul_T_sub_R_s);
        send_ciphertext(io_, M_mul_T_sub_R_s);
        //io_->flush();
        }
    }

    void online(uint64_t *Vect1, uint64_t *Vect2, int size, uint64_t *result, 
                    uint64_t *share, uint8_t *drelu_res = nullptr) {
        if (party == sci::BOB) {
            uint64_t *X0_sub_M = new uint64_t[size];
            for (int i = 0; i < size; i++) {
                X0_sub_M[i] = (share[i] - Vect1[i]) & mask_l;
            }
            io_->send_data(X0_sub_M, size * sizeof(uint64_t));
            // 阶段2，接收U=(X-M)⊗T+R，计算V+U以及B
            uint64_t *U = new uint64_t[size];
            uint64_t *U_add_V = new uint64_t[size];// 这个之后可以省略
            uint64_t *B = new uint64_t[size]; //这个之后可以改成bool看看
            io_->recv_data(U, size * sizeof(uint64_t));
            for (int i = 0; i < size; i++) {
                U_add_V[i] = (U[i] + Vect2[i]) & mask_l;
                B[i] = (U_add_V[i] >> (l-1)) & 1 ^ 1;//取符号位，并与1异或，得到drelu结果
            }
            io_->send_data(B, size * sizeof(uint64_t));
            // 输出结果
            if (drelu_res != nullptr) {
                for (int i = 0; i < size; i++) {
                    drelu_res[i] = B[i]; //可以用memcpy
                }
            }
            if (result != nullptr) {
                for (int i = 0; i < size; i++) {
                    if (B[i] == 1) {
                        result[i] = share[i];
                    } else {
                        result[i] = 0;
                    }
                } 
            }
            io_->flush();
            delete(X0_sub_M), X0_sub_M = NULL;
            delete(U), U = NULL;
            delete(U_add_V), U_add_V = NULL;
            delete(B), B = NULL;
        } else { // ALICE
            // 阶段1，接收X0-M
            uint64_t *X0_sub_M = new uint64_t[size];
            uint64_t *U = new uint64_t[size];
            io_->recv_data(X0_sub_M, size * sizeof(uint64_t));
            // 阶段2，计算并发送U
            for (int i = 0; i < size; i++) {
                U[i] = ((X0_sub_M[i] + share[i]) * Vect1[i] - Vect2[i]) & mask_l;
            }
            io_->send_data(U, size * sizeof(uint64_t));
            // 阶段3，接收B
            uint64_t *B = new uint64_t[size]; //这个之后可以改成bool看看
            io_->recv_data(B, size * sizeof(uint64_t));
            // 输出结果
            if (drelu_res != nullptr) {
                for (int i = 0; i < size; i++) {
                    drelu_res[i] = B[i]; //可以用memcpy
                }
            }
            if (result != nullptr) {
                for (int i = 0; i < size; i++) {
                    if (B[i] == 1) {
                        result[i] = share[i];
                    } else {
                        result[i] = 0;
                    }
                } 
            }
            io_->flush();
            delete(X0_sub_M), X0_sub_M = NULL;
            delete(U), U = NULL;
            delete(B), B = NULL;
        }
    }

    void relu(uint64_t *result, uint64_t *share, int num_relu, 
            uint8_t *drelu_res = nullptr, bool skip_ot = false) {
        this->num_relu = num_relu;
        uint64_t chunk_size = poly_modulus_degree;
        //uint64_t chunk_size = 16*16;
        uint64_t num_chunk = (num_relu + chunk_size - 1) / chunk_size;
        uint64_t rest = num_relu % chunk_size;
        uint64_t *result_pos;
        uint8_t *drelu_res_pos;
        uint64_t *Vec1 = new uint64_t[num_relu];
        uint64_t *Vec2 = new uint64_t[num_relu];
        int temp_size;
        this->rand_set = false;
        {
#ifdef LOG_LAYERWISE
    INIT_ALL_IO_DATA_SENT;
    INIT_TIMER;
#endif
            for (int i = 0; i < num_chunk; i++) {
                if (i == num_chunk - 1 && rest != 0) { // 如果最后一块不为空
                    temp_size = rest;
                } else {
                    temp_size = chunk_size;
                }
                offline(Vec1+i*chunk_size, Vec2+i*chunk_size, temp_size, skip_ot);
            }

#ifdef LOG_LAYERWISE
    auto temp = TIMER_TILL_NOW;
    uint64_t curComm;
    FIND_ALL_IO_TILL_NOW(curComm);
if (io_ == ioArr[0]) {
    if (skip_ot == true) { // relu type
        ReluOfflineTimeInMilliSec += temp;
        ReluOfflineCommSent += curComm;
    } else { // pool type
        PoolOfflineTimeInMilliSec += temp;
        PoolOfflineCommSent += curComm;
    }
#endif
}
        }
        {
#ifdef LOG_LAYERWISE
    INIT_ALL_IO_DATA_SENT;
    INIT_TIMER;
#endif
            for (int i = 0; i < num_chunk; i++) {
                // 部分代码调用存在result = nullptr，此处为方便处理多种情况，在这里设置值
                result_pos = (result == nullptr) ? nullptr : result + i * chunk_size;
                // 代码调用存在drelu = nullptr
                drelu_res_pos = (drelu_res_pos == nullptr) ? nullptr : drelu_res + i * chunk_size;
                
                if (i == num_chunk - 1 && rest != 0) { // 如果最后一块不为空
                    temp_size = rest;
                } else {
                    temp_size = chunk_size;
                }
                online(Vec1+i*chunk_size, Vec2+i*chunk_size, temp_size, result_pos, share+i*chunk_size, drelu_res_pos);
            }
            delete(Vec1), Vec1 = NULL;
            delete(Vec2), Vec2 = NULL;

#ifdef LOG_LAYERWISE
    auto temp2 = TIMER_TILL_NOW;
    uint64_t curComm2;
    FIND_ALL_IO_TILL_NOW(curComm2);
if (io_ == ioArr[0]) {
    if (skip_ot == true) { // relu
        ReluOnlineTimeInMilliSec += temp2;
        ReluOnlineCommSent += curComm2;
    } else { // pool
        PoolOnlineTimeInMilliSec += temp2;
        PoolOnlineCommSent += curComm2;
    }
#endif
}
        }
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
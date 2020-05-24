#include "tisaNET.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <string>
#include <algorithm>

#define format_key {'t','i','s','a','N','E','T'}
#define f_k_size 7

#define data_head {'D','A','T','A'}
#define d_size 4

#define format_key_size sizeof(char) * 7
#define data_head_size sizeof(char) * 4

#define mnist_pict_offset 16
#define mnist_lab_offset 8

#define mnist_image_size 784

#define mnist_train_d "train-images.idx3-ubyte"
#define mnist_train_l "train-labels.idx1-ubyte"
#define mnist_test_d "t10k-images.idx3-ubyte"
#define mnist_test_l "t10k-labels.idx1-ubyte"

#define judge 0.05

#define progress_bar_length 30

namespace tisaNET{
    //MNISTからデータを作る
    bool load_MNIST(const char* path, Data_set& train_data, Data_set& test_data, int sample_size,int test_size, bool single_output) {
        std::random_device seed_gen;
        std::default_random_engine rand_gen(seed_gen());
        std::string folder = path;

        unsigned int train_data_start = rand_gen() % 60000;
        unsigned int test_data_start = rand_gen() % 10000;
        //ここから訓練のデータ
        {
            std::string filename = folder + "\\" + mnist_train_d;
            std::ifstream file_d(filename,std::ios::binary);
            if (!file_d) {
                printf("Can not open file : %s\n", filename);
                exit(EXIT_FAILURE);
            }

            filename = folder + "\\" + mnist_train_l;
            std::ifstream file_l(filename, std::ios::binary);
            if (!file_l) {
                printf("Can not open file : %s\n", filename);
                exit(EXIT_FAILURE);
            }

            file_d.seekg(mnist_pict_offset + (long)(train_data_start * mnist_image_size));
            file_l.seekg(mnist_lab_offset + (long)train_data_start);

            std::vector<uint8_t> tmp_d(mnist_image_size);
            uint8_t tmp_for_l;
            for (int i = 0,index=train_data_start;i < sample_size;i++,index++) {
                std::vector<uint8_t> tmp_l;
                if (index >= 60000) {
                    index = 0;
                    file_d.seekg(mnist_pict_offset);
                    file_l.seekg(mnist_lab_offset);
                }
                file_d.read(reinterpret_cast<char*>(&tmp_d[0]), sizeof(uint8_t) * mnist_image_size);
                train_data.data.push_back(tmp_d);
                file_l.read(reinterpret_cast<char*>(&tmp_for_l), sizeof(uint8_t));
                /*
                //debug
                bool error_check = file_d.good();
                int tell_d = file_d.tellg();
                tell_d = (tell_d - mnist_pict_offset) / mnist_image_size;
                int tell_l = file_l.tellg();
                tell_l = tell_l - mnist_lab_offset;
                //end_debug
                */
                if (single_output) {
                    tmp_l.push_back(tmp_for_l);
                }
                else {
                    for (int bit = 0; bit < 10;bit++) {
                        if (bit == tmp_for_l) {
                            tmp_l.push_back(1);
                        }
                        else {
                            tmp_l.push_back(0);
                        }
                    }
                }
                train_data.answer.push_back(tmp_l);
                /*
                //debug
                std::vector<std::vector<uint8_t>> debug_d;
                for (int row = 0,dd_count=0;row < 28;row++) {
                    std::vector<uint8_t> tmp;
                    for (int column = 0; column < 28;column++,dd_count++) {
                        tmp.push_back(train_data.data.back()[dd_count]);
                    }
                    debug_d.push_back(tmp);
                }
                tisaMat::matrix debug_dm(debug_d);
                debug_dm.show();
                printf("%d\n",train_data.answer.back()[0]);
                //end_debug
                */
            }
        }

        //ここから評価のデータ
        {
            std::string filename = folder + '\\' + mnist_test_d;
            std::ifstream file_d(filename, std::ios::binary);
            if (!file_d) {
                printf("Can not open file : %s\n", filename);
                exit(EXIT_FAILURE);
            }
            filename = folder + '\\' + mnist_test_l;
            std::ifstream file_l(filename, std::ios::binary);
            if (!file_l) {
                printf("Can not open file : %s\n", filename);
                exit(EXIT_FAILURE);
            }

            file_d.seekg(mnist_pict_offset + (long)(test_data_start * mnist_image_size));
            file_l.seekg((long)mnist_lab_offset + (long)test_data_start);

            std::vector<uint8_t> tmp_d(mnist_image_size);
            
            uint8_t tmp_for_l;
            for (int i = 0, index = test_data_start; i < test_size; i++, index++) {
                std::vector<uint8_t> tmp_l;
                if (index >= 60000) {
                    file_d.seekg(mnist_pict_offset);

                }
                file_d.read(reinterpret_cast<char*>(&tmp_d[0]), sizeof(uint8_t) * mnist_image_size);
                test_data.data.push_back(tmp_d);
                file_l.read(reinterpret_cast<char*>(&tmp_for_l), sizeof(uint8_t));
                if (single_output) {
                    tmp_l.push_back(tmp_for_l);
                }
                else {
                    for (int bit = 0; bit < 10; bit++) {
                        if (bit == tmp_for_l) {
                            tmp_l.push_back(1);
                        }
                        else {
                            tmp_l.push_back(0);
                        }
                    }
                }
                test_data.answer.push_back(tmp_l);
            }
        }

        printf("  :>  loaded MNIST successfully\n");
    }
   
    double step(double X) {
        double Y;
        if (X == 0.0) {
            Y = 0;
        }
        else {
            Y = (X / fabs(X) + 1) / 2;
        }
        return Y;
    }

    double sigmoid(double X) {
        double Y;
        Y = 1 / (1 + exp(-1 * X));
        return Y;
    }

    double ReLU(double X) {
        if (X > 0) {
            return X;
        }
        else {
            return 0;
        }
    }

    //SOFTMAXが呼ばれたとき専用のあくまでも部品
    double softmax(double X) {
        return exp(X);
    }

    //おまけ
    bool print01(int bit, long a) {
        char* bi;
        bi = (char*)malloc(sizeof(char) * bit);
        if (bi == NULL) {
            return NULL;
        }
        else {}
        for (int i = 0; i < bit; i++) {
            char tmp;
            tmp = 0b1 & (a >> i);
            if (tmp == 0) {
                bi[bit - 1 - i] = '0';
            }
            else {
                bi[bit - 1 - i] = '1';
            }
        }
        for (int i = 0; i < bit; i++) {
            printf("%c", bi[i]);
        }
        free(bi);
        return true;
    }

    double mean_squared_error(std::vector<std::vector<uint8_t>>& teacher, std::vector<std::vector<double>>& output) {
        int sample_size = output.size();
        int output_num = output[0].size();
        double tmp = 0.0;
        for (int i = 0;i < sample_size;i++) {
            for (int j = 0;j < output_num;j++) {
                tmp += (teacher[i][j] - output[i][j]) * (teacher[i][j] - output[i][j]);
            }
        }
        
        tmp /= sample_size;
        return tmp;
    }
    
    double cross_entropy_error(std::vector<std::vector<uint8_t>>& teacher, std::vector<std::vector<double>>& output) {
        int sample_size = output.size();
        int output_num = output[0].size();
        double tmp = 0.0;

        //log()が-∞にならないように、せめて0より大きくなるようちいちゃい数を足す
        double delta = 1e-7;
        for (int i = 0; i < sample_size; i++) {
            for (int j = 0; j < output_num; j++) {
                tmp -= teacher[i][j] * log(output[i][j] + delta) + (1.0 - teacher[i][j]) * log(1 - output[i][j] + delta);
            }
        }
        
        tmp /= sample_size;
        return tmp;
    }

    void Model::Create_Layer(int nodes, uint8_t Activation) {
        layer tmp;
        if (Activation != INPUT) {
            int input = net_layer.back().node;
            tmp.node = nodes;
            tmp.Activation_f = Activation;
            tmp.W = new tisaMat::matrix(input, nodes);
            tmp.B = std::vector<double>(nodes);
            tmp.Output = std::vector<double>(nodes);
        }
        else {
            tmp.node = nodes;
            tmp.Activation_f = Activation;
            tmp.Output = std::vector<double>(nodes);
        }
        net_layer.push_back(tmp);
    }
    //おまけ(重みとバイアスを任意の値で初期化)
    void Model::Create_Layer(int nodes, uint8_t Activation,double init) {
        layer tmp;
        if (Activation != INPUT) {
            int input = net_layer.back().node;
            tmp.node = nodes;
            tmp.Activation_f = Activation;
            tmp.W = new tisaMat::matrix(input, nodes, init);
            tmp.B = std::vector<double>(nodes,init);
            tmp.Output = std::vector<double>(nodes);
        }
        else {
            tmp.node = nodes;
            tmp.Activation_f = Activation;
            tmp.Activation_f = NULL;
            tmp.Output = std::vector<double>(nodes);
        }
        net_layer.push_back(tmp);
    }
/*
    void Model::input_data(std::vector<double>& data) {
        int input_num = data.size();
        if (net_layer.front().Output.size() != input_num) {
            printf("input error|!|\n");
        }
        else {
            net_layer.front().Output = data;
        }
    }
*/
    tisaMat::matrix Model::F_propagate(tisaMat::matrix& Input_data) {
        int sample_size = Input_data.mat_RC[0];
        tisaMat::matrix output_matrix(sample_size, net_layer.back().Output.size());
        for (int data_index = 0;data_index < sample_size;data_index++) {
            input_data(Input_data.elements[data_index]);
            for (int i = 1; i < number_of_layer(); i++) {
                std::vector<double> X = tisaMat::vector_multiply(net_layer[i - 1].Output, *net_layer[i].W);
                X = tisaMat::vector_add(X, net_layer[i].B);
                //ソフトマックス関数を使うときはまず最大値を全部から引く
                if (net_layer[i].Activation_f == SOFTMAX) {
                    double max = *std::max_element(X.begin(), X.end());
                    for (int X_count = 0; X_count < X.size(); X_count++) {
                        X[X_count] -= max;
                    }
                }

                for (int j = 0; j < X.size(); j++) {
                    net_layer[i].Output[j] = (*Af[net_layer[i].Activation_f])(X[j]);
                    if (isnan(net_layer[i].Output[j])) {
                        bool nan_flug = 1;
                    }
                }

                if (net_layer[i].Activation_f == SOFTMAX) {
                    double sigma = 0.0;
                    for (int node = 0; node < net_layer[i].Output.size(); node++) {
                        sigma += net_layer[i].Output[node];
                    }
                    tisaMat::vector_multiscalar(net_layer[i].Output, 1.0 / sigma);
                }
            }
            output_matrix.elements[data_index] = net_layer.back().Output;
        }
        return output_matrix;
    }
    //訓練用
    tisaMat::matrix Model::F_propagate(tisaMat::matrix& Input_data,std::vector<Trainer>& trainer) {
        int sample_size = Input_data.mat_RC[0];
        tisaMat::matrix output_matrix(sample_size, net_layer.back().Output.size());
        int layer_num = number_of_layer();
        for (int data_index = 0; data_index < sample_size; data_index++) {
            input_data(Input_data.elements[data_index]);
            for (int i = 1; i < layer_num; i++) {
                std::vector<double> X = tisaMat::vector_multiply(net_layer[i - 1].Output, *net_layer[i].W);//入力が変わってない説
                X = tisaMat::vector_add(X, net_layer[i].B);
                //ここまでで、活性化関数を使う前の計算が終了　なんかずっと同じ値になってる？？？

                //ソフトマックス関数を使うときはまず最大値を全部から引く
                if (net_layer[i].Activation_f == SOFTMAX) {
                    double max = *std::max_element(X.begin(),X.end());
                    for (int X_count = 0; X_count < X.size(); X_count++) {
                        X[X_count] -= max;
                    }
                }

                for (int j = 0; j < X.size(); j++) {
                    net_layer[i].Output[j] = (*Af[net_layer[i].Activation_f])(X[j]);
                    if (isnan(net_layer[i].Output[j])) {
                        bool nan_flug = 1;
                    }
                }

                if (net_layer[i].Activation_f == SOFTMAX) {
                    double sigma = 0.0;
                    for (int node = 0; node < net_layer[i].Output.size(); node++) {
                        sigma += net_layer[i].Output[node];
                    }
                    tisaMat::vector_multiscalar(net_layer[i].Output, 1.0 / sigma);
                }

                trainer[i - 1].Y[data_index] = net_layer[i].Output;
            }
            output_matrix.elements[data_index] = net_layer.back().Output;
        }
        return output_matrix;
    }

    void Model::B_propagate(std::vector<std::vector<uint8_t>>& teacher, tisaMat::matrix& output,uint8_t error_func, std::vector<Trainer>& trainer,double lr,tisaMat::matrix& input_batch) {
        int output_num = output.mat_RC[1];
        int batch_size = output.mat_RC[0];
        bool cross_sig_flag = (error_func == CROSS_ENTROPY_ERROR) && ((net_layer.back().Activation_f == SIGMOID) || (net_layer.back().Activation_f == SOFTMAX));
        //初回限定で誤差をセットして学習率もかける

        tisaMat::matrix error_matrix(teacher);
        //多クラス分類でなければy-tが使える
        error_matrix = tisaMat::matrix_subtract(output, error_matrix);
        //printf("error_matrix for propagate\n");
        //error_matrix.show();

        if (error_func == CROSS_ENTROPY_ERROR && cross_sig_flag != 1) {
            tisaMat::matrix tmp_for_crossE(batch_size, output_num, 1.0);
            tmp_for_crossE = tisaMat::matrix_subtract(tmp_for_crossE, output);
            tmp_for_crossE = tisaMat::Hadamard_product(tmp_for_crossE, output);//たまに0になる要素がある

            //tmp_for_crossEが小さいと、最後の割り算で発散するので防止のためちいちゃい数を足す
            tisaMat::matrix tmp_delta(batch_size, output_num, 1e-10);
            tmp_for_crossE = tisaMat::matrix_add(tmp_for_crossE, tmp_delta);

            error_matrix = tisaMat::Hadamard_division(error_matrix, tmp_for_crossE);//誤差がすごいことになってる(errorが少ないと発散してるっぽい)
        }

        error_matrix.multi_scalar(lr);

        //ほんとは出力層の微分は特別扱いで計算したい


        //重みとかの更新量を求める前にリフレッシュ
        for (int current_layer = 0; current_layer < net_layer.size() - 1;current_layer++) {
            trainer[current_layer].dW->multi_scalar(0);
            tisaMat::vector_multiscalar(trainer[current_layer].dB, 0);
        }
        
        //重みとかの更新量の平均を出す 具体的にはバッチのパターンごとに更新量を出して、あとでバッチサイズで割る
        for (int batch_segment = 0; batch_segment < batch_size; batch_segment++) {

            //伝播していく行列(秘伝のたれ) あとで行列積をつかいたいのでベクトルではなく行列として用意します(アダマール積もつかいますが)
            std::vector<std::vector<double>> tmp(1, error_matrix.elements[batch_segment]);
            tisaMat::matrix propagate_matrix(tmp);
            bool reduction_flag = cross_sig_flag;
            for (int current_layer = net_layer.size() - 1; current_layer > 0; current_layer--) {
                //秘伝のたれを仕込む(伝播する行列)
                    //ノードごとの活性化関数の微分
                tisaMat::matrix dAf(1, net_layer[current_layer].node);
                if (reduction_flag) {
                    reduction_flag = 0;
                    for (int i = 0; i < net_layer[current_layer].node; i++) {
                        dAf.elements[0][i] = 1;
                    }
                }
                else {
                    switch (net_layer[current_layer].Activation_f) {
                    case SIGMOID:
                        for (int i = 0; i < net_layer[current_layer].node; i++) {
                            double Y = trainer[current_layer - 1].Y[batch_segment][i];
                            dAf.elements[0][i] = Y * (1 - Y);
                        }
                        break;
                    case SOFTMAX:
                    {
                        double tmp_softmax = 0.0;
                        for (int count = 0; count < net_layer[current_layer].node; count++) {
                            tmp_softmax += error_matrix.elements[batch_segment][count] * trainer[current_layer - 1].Y[batch_segment][count];
                        }
                        for (int i = 0; i < net_layer[current_layer].node; i++) {
                            double Y = trainer[current_layer - 1].Y[batch_segment][i];
                            dAf.elements[0][i] = Y * (1 - Y) - Y * (tmp_softmax - Y * error_matrix.elements[batch_segment][i]);
                        }
                    }
                    break;
                    case RELU:
                        for (int i = 0; i < net_layer[current_layer].node; i++) {
                            dAf.elements[0][i] = 1;
                        }
                        break;
                    case STEP:
                        for (int i = 0; i < net_layer[current_layer].node; i++) {
                            dAf.elements[0][i] = 1;
                        }
                        break;
                    }
                }

                //活性化関数の微分行列と秘伝のタレのアダマール積
                propagate_matrix = tisaMat::Hadamard_product(dAf, propagate_matrix);

                //今の層の重み、バイアスの更新量を計算する
                    //重みは順伝播のときの入力も使う
                tisaMat::matrix W_tmp(0,0);
                if((current_layer - 1) > 0){
                    W_tmp = tisaMat::vector_to_matrix(trainer[current_layer - 2].Y[batch_segment]);//current_layer-2のトレーナーは、前の層のトレーナー
                    W_tmp = tisaMat::matrix_transpose(W_tmp);
                    W_tmp = tisaMat::matrix_multiply(W_tmp, propagate_matrix);
                }
                else {
                    W_tmp = tisaMat::vector_to_matrix(input_batch.elements[batch_segment]);
                    W_tmp = tisaMat::matrix_transpose(W_tmp);
                    W_tmp = tisaMat::matrix_multiply(W_tmp, propagate_matrix);
                }
                
                *(trainer[current_layer - 1].dW) = tisaMat::matrix_add(*trainer[current_layer - 1].dW,W_tmp);
                    //バイアス
                trainer[current_layer - 1].dB = tisaMat::vector_add(trainer[current_layer - 1].dB,propagate_matrix.elements[0]);

                //今の層の重みの転置行列を秘伝のたれのうしろから行列積で次の層へ
                W_tmp = tisaMat::matrix_transpose(*(net_layer[current_layer].W));
                propagate_matrix = tisaMat::matrix_multiply(propagate_matrix, W_tmp);
            }
        }

        //ミニバッチ学習の場合、重みとかの更新量を平均する
        if (batch_size > 1) {
            for (int i = 0;i < trainer.size();i++) {
                trainer[i].dW->multi_scalar(1.0 / batch_size);
                tisaMat::vector_multiscalar(trainer[i].dB,1.0 / batch_size);
            }
        }

    }

    //ごちゃごちゃ変えて実験する用
    void Model::B_propagate2(std::vector<std::vector<uint8_t>>& teacher, tisaMat::matrix& output, uint8_t error_func, std::vector<Trainer>& trainer, double lr, tisaMat::matrix& input_batch) {
        int output_num = output.mat_RC[1];
        int batch_size = output.mat_RC[0];
        bool isMCE = (error_func == CROSS_ENTROPY_ERROR) && (output_num > 1);
        //初回限定で誤差をセットして学習率もかける

        tisaMat::matrix error_matrix(teacher);
        error_matrix = tisaMat::matrix_subtract(output, error_matrix);
        //printf("error_matrix for propagate\n");
        //error_matrix.show();

        error_matrix.multi_scalar(lr);

        //ほんとは出力層の微分は特別扱いで計算したい


        //重みとかの更新量を求める前にリフレッシュ
        for (int current_layer = 0; current_layer < net_layer.size() - 1; current_layer++) {
            trainer[current_layer].dW->multi_scalar(0);
            tisaMat::vector_multiscalar(trainer[current_layer].dB, 0);
        }

        //重みとかの更新量の平均を出す 具体的にはバッチのパターンごとに更新量を出して、あとでバッチサイズで割る
        for (int batch_segment = 0; batch_segment < batch_size; batch_segment++) {

            //伝播していく行列(秘伝のたれ) あとで行列積をつかいたいのでベクトルではなく行列として用意します(アダマール積もつかいますが)
            std::vector<std::vector<double>> tmp(1, error_matrix.elements[batch_segment]);
            tisaMat::matrix propagate_matrix(tmp);

            for (int current_layer = net_layer.size() - 1; current_layer > 0; current_layer--) {
                //秘伝のたれを仕込む(伝播する行列)
                    //ノードごとの活性化関数の微分
                tisaMat::matrix dAf(1, net_layer[current_layer].node);
                switch (net_layer[current_layer].Activation_f) {
                case SIGMOID:
                    for (int i = 0; i < net_layer[current_layer].node; i++) {
                        double Y = trainer[current_layer - 1].Y[batch_segment][i];
                        dAf.elements[0][i] = Y * (1 - Y);
                    }
                    break;
                case SOFTMAX:
                    for (int i = 0; i < net_layer[current_layer].node; i++) {
                        dAf.elements[0][i] = 1;
                    }
                    break;
                case RELU:
                    for (int i = 0; i < net_layer[current_layer].node; i++) {
                        dAf.elements[0][i] = 1;
                    }
                    break;
                case STEP:
                    for (int i = 0; i < net_layer[current_layer].node; i++) {
                        dAf.elements[0][i] = 1;
                    }
                    break;
                }
                //活性化関数の微分行列と秘伝のタレのアダマール積
                propagate_matrix = tisaMat::Hadamard_product(dAf, propagate_matrix);

                //今の層の重み、バイアスの更新量を計算する
                    //重みは順伝播のときの入力も使う
                tisaMat::matrix W_tmp(0, 0);
                if ((current_layer - 1) > 0) {
                    W_tmp = tisaMat::vector_to_matrix(trainer[current_layer - 2].Y[batch_segment]);//current_layer-2のトレーナーは、前の層のトレーナー
                    W_tmp = tisaMat::matrix_transpose(W_tmp);
                    W_tmp = tisaMat::matrix_multiply(W_tmp, propagate_matrix);
                }
                else {
                    W_tmp = tisaMat::vector_to_matrix(input_batch.elements[batch_segment]);
                    W_tmp = tisaMat::matrix_transpose(W_tmp);
                    W_tmp = tisaMat::matrix_multiply(W_tmp, propagate_matrix);
                }

                *(trainer[current_layer - 1].dW) = tisaMat::matrix_add(*trainer[current_layer - 1].dW, W_tmp);
                //バイアス
                trainer[current_layer - 1].dB = tisaMat::vector_add(trainer[current_layer - 1].dB, propagate_matrix.elements[0]);

                //今の層の重みの転置行列を秘伝のたれのうしろから行列積で次の層へ
                W_tmp = tisaMat::matrix_transpose(*(net_layer[current_layer].W));
                propagate_matrix = tisaMat::matrix_multiply(propagate_matrix, W_tmp);
            }
        }

        //ミニバッチ学習の場合、重みとかの更新量を平均する
        if (batch_size > 1) {
            for (int i = 0; i < trainer.size(); i++) {
                trainer[i].dW->multi_scalar(1.0 / batch_size);
                tisaMat::vector_multiscalar(trainer[i].dB, 1.0 / batch_size);
            }
        }

    }

    int Model::number_of_layer() {
        return net_layer.size();
    }

    void Model::initialize() {
        std::random_device seed_gen;
        std::default_random_engine rand_gen(seed_gen());
        std::normal_distribution<> dist;

        //0番の層は入力の分配にしかつかわないので１番の層から
        for (int current_layer = 1; current_layer < number_of_layer();current_layer++) {
            int prev_nodes = net_layer[current_layer - 1].node;
            switch (net_layer[current_layer].Activation_f) {
            case SIGMOID://Xaivierの初期値
                {
                    std::normal_distribution<>::param_type param(0.0, sqrt(1.0 / prev_nodes));
                    dist.param(param);
                }
                break;
            default://Heの初期値
                {
                    std::normal_distribution<>::param_type param(0.0, sqrt(2.0 / prev_nodes));
                    dist.param(param);
                }
                break;
            }

            int W_row = net_layer[current_layer].W->mat_RC[0];
            int W_column = net_layer[current_layer].W->mat_RC[1];
            for (int R = 0; R < W_row;R++) {
                for (int C = 0; C < W_column;C++) {
                    net_layer[current_layer].W->elements[R][C] = dist(rand_gen);
                }
            }
        }
    }

    void Model::load_model(const char* tp_file) {
        std::ifstream file(tp_file,std::ios::binary);
        if (!file) {
            printf("Can not open file : %s\n",tp_file);
            exit(EXIT_FAILURE);
        }

        char file_check[7];
        const char format_checker[f_k_size] = format_key;
        file.read(file_check, format_key_size);
        for (int i = 0; i < 7; i++) {
            if (file_check[i] != format_checker[i]) {
                printf("The file format is incorrect : %s\n", tp_file);
                exit(EXIT_FAILURE);
            }
        }

        int layer = 0;
        file.read(reinterpret_cast<char*>(&layer),sizeof(int));
        int *node = new int[layer];
        uint8_t *Activation_f = new uint8_t[layer];
        file.read(reinterpret_cast<char*>(node),layer * sizeof(int));
        file.read(reinterpret_cast<char*>(Activation_f),layer * sizeof(uint8_t));

        const char Data_head[d_size] = data_head;
        file.read(file_check, data_head_size);
        for (int i = 0; i < 4; i++) {
            if (file_check[i] != Data_head[i]) {
                printf("failed to read parameter\n");
                printf("The file maybe corrapted : %s\n", tp_file);
                delete[] node;
                delete[] Activation_f;
                exit(EXIT_FAILURE);
            }
        }

        for (int current_layer = 0; current_layer < layer;current_layer++) {
            Create_Layer(node[current_layer],Activation_f[current_layer]);
        }

        for (int current_layer = 1; current_layer < layer; current_layer++) {
            int input = net_layer[current_layer - 1].node;
            int current_node = net_layer[current_layer].node;
            std::vector<double> tmp_row(current_node);

            tisaMat::matrix tmp_W(input,current_node);
            //重み行列を読みだす
            for (int W_row=0; W_row < input; W_row++){
                file.read(reinterpret_cast<char*>(&tmp_row[0]),current_node * sizeof(double));
                tmp_W.elements[W_row] = tmp_row;
            }
            net_layer[current_layer].W->elements = tmp_W.elements;

            //tmp_rowを使いまわしてバイアスを読みだす
            file.read(reinterpret_cast<char*>(&tmp_row[0]),current_node * sizeof(double));
            net_layer[current_layer].B = tmp_row;
        }

        delete[] node;
        delete[] Activation_f;
    }

    void Model::save_model(const char* filename) {
        std::ofstream file(filename,std::ios::binary);
        if (!file) {
            printf("failed to open file : %s\n",filename);
            exit(EXIT_FAILURE);
        }
        //INPUTのモードの層が入力層でないと読み込めないので、チェック
        if (net_layer[0].Activation_f != INPUT) {
            printf("first layer is not ""INPUT""\n");
            exit(EXIT_FAILURE);
        }

        const char Format_key[f_k_size] = format_key;
        file.write(Format_key, format_key_size);
        int layer = number_of_layer();
        file.write(reinterpret_cast<char*>(&layer),sizeof(int));
        for (int current_layer = 0;current_layer < layer;current_layer++) {
            int node = net_layer[current_layer].node;
            file.write(reinterpret_cast<char*>(&node),sizeof(int));
        }
        for (int current_layer = 0; current_layer < layer; current_layer++) {
            uint8_t Af = net_layer[current_layer].Activation_f;
            file.write(reinterpret_cast<char*>(&Af), sizeof(uint8_t));
        }
        const char Data_head[d_size] = data_head;
        file.write(Data_head, data_head_size);

        //ここからモデルのパラメーターをファイルに書き込んでいく
        for (int current_layer = 1; current_layer < layer; current_layer++) {
            int W_row = net_layer[current_layer].W->mat_RC[0];
            int node = net_layer[current_layer].W->mat_RC[1];
            //重み行列を書き込む
            for (int r = 0; r < W_row; r++) {
                file.write(reinterpret_cast<char*>(&net_layer[current_layer].W->elements[r][0]),node * sizeof(double));
            }
            //バイアスを書き込む
            file.write(reinterpret_cast<char*>(&net_layer[current_layer].B[0]), node * sizeof(double));
        }

        printf("  :)  The file was output successfully!!! : %s\n",filename);
    }

    void Model::monitor_accuracy(bool monitor_accuracy) {
        monitoring_accuracy = monitor_accuracy;
    }

    void Model::logging_error(const char* log_file) {
        log_error = true;
        log_filename = log_file;
    }

    void Model::train(double learning_rate,Data_set& train_data, Data_set& test_data, int epoc, int iteration, uint8_t Error_func) {
        if (net_layer[0].node != train_data.data[0].size()) {
            printf("Input size and number of input layer's nodes do not match");
            exit(EXIT_FAILURE);
        }
        if (net_layer.back().node != train_data.answer[0].size()) {
            printf("Output size and number of output layer's nodes do not match");
            exit(EXIT_FAILURE);
        }
        
        int output_num = net_layer.back().node;
        int batch_size = train_data.data.size() / iteration;
        tisaMat::matrix output_iterate(batch_size,output_num);
        tisaMat::matrix input_iterate(batch_size, train_data.data[0].size());
        tisaMat::matrix answer_iterate(batch_size, output_num);
        double error;
        tisaMat::matrix test_mat(test_data.data);
        std::vector<std::vector<uint8_t>> teach_iterate(batch_size);

        //バックプロパゲーションの時に重みの更新量を記憶するトレーナーをつくる
        std::vector<Trainer> trainer;
        for (int i=0; i < net_layer.size()-1; i++){
            Trainer tmp;
            //1で初期化しないと、更新量計算するときに掛け算できなくなる
            tmp.dW = new tisaMat::matrix(net_layer[i+1].W->mat_RC[0], net_layer[i + 1].W->mat_RC[1],1);
            tmp.dB = std::vector<double>(net_layer[i + 1].node,1);
            for (int j = 0; j < batch_size;j++) {
                tmp.Y.push_back(std::vector<double>(net_layer[i + 1].node));
            }
            trainer.push_back(tmp);
        }

        //CSV形式で誤差を記録する準備
        if (log_error) {
            std::ofstream o_file(log_filename);
            if (!o_file) {
                printf("failed to open file : %s\n", log_filename);
                exit(EXIT_FAILURE);
            }
            o_file << "epoc,Error" << '\n';

            for (int ep = 0; ep < epoc; ep++) {
                printf("| epoc : %6d |", ep);
                for (int i = 0; i < iteration; i++) {

                    if (train_data.data.size() < batch_size * i) break;

                    //次のイテレーションでつかう入力の行列をつくる
                    for (int j = 0; j < batch_size; j++) {
                        input_iterate.elements[j] = tisaMat::vector_cast<double>(train_data.data[(batch_size * i) + j]);
                        teach_iterate[j] = train_data.answer[(batch_size * i) + j];
                    }

                    //printf("| epoc : %6d |",ep);


                    //printf("input\n");
                    //input_iterate.show();
                    output_iterate = F_propagate(input_iterate, trainer);

                    //printf("output\n");
                    //output_iterate.show();
                    //printf("answer\n");
                    //tisaMat::matrix answer_matrix(teach_iterate);
                    //answer_matrix.show();


                    error = (*Ef[Error_func])(teach_iterate, output_iterate.elements);//ここではじめて？nan
                    //printf("|%6d iterate|Error   :  %lf\n", i,error);

                    o_file << float(ep) + float(i+1.0)/float(iteration) << ',' << error << '\n';

                    bool have_error = 0;
                    if (error != 0.0) {
                        have_error = 1;
                    }

                    if (have_error) {
                        B_propagate(teach_iterate, output_iterate, Error_func, trainer, learning_rate, input_iterate);
                        //B_propagate2(teach_iterate, output_iterate, Error_func, trainer, learning_rate, input_iterate);
                        //トレーナーの値を使って重みを調整する
                        for (int layer = 1; layer < net_layer.size(); layer++) {
                            //重み
                            *(net_layer[layer].W) = tisaMat::matrix_subtract(*net_layer[layer].W, *trainer[layer - 1].dW);
                            //printf("%d layer dW\n", layer);
                            //trainer[layer - 1].dW->show();
                            //バイアス
                            net_layer[layer].B = tisaMat::vector_subtract(net_layer[layer].B, trainer[layer - 1].dB);
                            //printf("%d layer dB\n", layer);
                            //tisaMat::vector_show(trainer[layer - 1].dB);
                        }
                    }

                    show_train_progress(iteration, i);
                }
                printf("\n");
                /*
                //今の重みとか表示(デバッグ用)
                for (int layer = 1; layer < net_layer.size(); layer++) {
                    //重み
                    printf("W\n");
                    net_layer[layer].W->show();
                    //バイアス
                    printf("B\n");
                    tisaMat::vector_show(net_layer[layer].B);
                }
                */

                //テスト(output_iterateは使いまわし)
                output_iterate = F_propagate(test_mat);
                printf("| TEST |");
                //printf("test_input\n");
                //test_mat.show();
                //printf("test_output\n");
                //output_iterate.show();
                error = (*Ef[Error_func])(test_data.answer, output_iterate.elements);
                printf("Error : %lf\n",error);

                if (monitoring_accuracy == true) {
                    m_a(output_iterate.elements, test_data.answer, Error_func);
                }

                if (error == 0.0) {
                    break;
                }
            }

        }
        else {
            for (int ep = 0; ep < epoc; ep++) {
                printf("| epoc : %6d |", ep);
                for (int i = 0; i < iteration; i++) {

                    if (train_data.data.size() < batch_size * i) break;

                    //次のイテレーションでつかう入力の行列をつくる
                    for (int j = 0; j < batch_size; j++) {
                        input_iterate.elements[j] = tisaMat::vector_cast<double>(train_data.data[(batch_size * i) + j]);
                        teach_iterate[j] = train_data.answer[(batch_size * i) + j];
                    }

                    //printf("| epoc : %6d |",ep);


                    //printf("input\n");
                    //input_iterate.show();
                    output_iterate = F_propagate(input_iterate, trainer);

                    //printf("output\n");
                    //output_iterate.show();
                    //printf("answer\n");
                    //tisaMat::matrix answer_matrix(teach_iterate);
                    //answer_matrix.show();


                    error = (*Ef[Error_func])(teach_iterate, output_iterate.elements);//ここではじめて？nan
                    //printf("|%6d iterate|Error   :  %lf\n", i, error);

                    bool have_error = 0;
                    if (error != 0.0) {
                        have_error = 1;
                    }

                    if (have_error) {
                        B_propagate(teach_iterate, output_iterate, Error_func, trainer, learning_rate, input_iterate);
                        //B_propagate2(teach_iterate, output_iterate, Error_func, trainer, learning_rate, input_iterate);
                        //トレーナーの値を使って重みを調整する
                        for (int layer = 1; layer < net_layer.size(); layer++) {
                            //重み
                            *(net_layer[layer].W) = tisaMat::matrix_subtract(*net_layer[layer].W, *trainer[layer - 1].dW);
                            //printf("%d layer dW\n", layer);
                            //trainer[layer - 1].dW->show();
                            //バイアス
                            net_layer[layer].B = tisaMat::vector_subtract(net_layer[layer].B, trainer[layer - 1].dB);
                            //printf("%d layer dB\n", layer);
                            //tisaMat::vector_show(trainer[layer - 1].dB);
                        }
                    }

                    show_train_progress(iteration,i);
                }
                printf("\n");
                /*
                //今の重みとか表示(デバッグ用)
                for (int layer = 1; layer < net_layer.size(); layer++) {
                    //重み
                    printf("W\n");
                    net_layer[layer].W->show();
                    //バイアス
                    printf("B\n");
                    tisaMat::vector_show(net_layer[layer].B);
                }
                */

                //テスト(output_iterateは使いまわし)
                output_iterate = F_propagate(test_mat);
                printf("| TEST |");
                //printf("test_input\n");
                //test_mat.show();
                //printf("test_output\n");
                //output_iterate.show();
                error = (*Ef[Error_func])(test_data.answer, output_iterate.elements);
                printf("Error : %lf\n", error);

                if (monitoring_accuracy == true) {
                    m_a(output_iterate.elements, test_data.answer, Error_func);
                }

                if (error == 0.0) {
                    break;
                }
            }
        }
    }

    void Model::m_a(std::vector<std::vector<double>>& output, std::vector<std::vector<uint8_t>>& answer,uint8_t error_func) {
        int total_size = answer.size();
        int correct_count = 0;
        for (int i = 0;i < total_size;i++) {
            std::vector<std::vector<double>> tmp_out(1,output[i]);
            std::vector<std::vector<uint8_t>> tmp_ans(1,answer[i]);
            double error = (*Ef[error_func])(tmp_ans,tmp_out);
            if (error < judge) {
                correct_count++;
            }
        }
        printf("accuracy : %d / %d\n",correct_count,total_size);
    }

    void show_train_progress(int total_iteration, int now_iteration) {
        printf("\033[17G|");
        double progress = float(now_iteration+1) / float(total_iteration);
        int bar_num = (progress+0.01) * progress_bar_length;
        for (int i = 0;i < bar_num - 1;i++) {
            printf("=");
        }
        printf(">");
        for (int i = 0; i < progress_bar_length - bar_num;i++) {
            printf("-");
        }
        printf("| %5.2lf%%", progress*100.0);
    }
}
#include "tisaNET.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <fstream>

#define format_key "tisaNET"
#define data_head "DATA"

#define format_key_size sizeof(char) * 7
#define data_head_size sizeof(char) * 4


namespace tisaNET{

    double step(double a) {
        double Y;
        if (a == 0.0) {
            Y = 0;
        }
        else {
            Y = (a / fabs(a) + 1) / 2;
        }
        return Y;
    }

    double sigmoid(double a) {
        double Y;
        Y = 1 / (1 + exp(-1 * a));
        return Y;
    }

    double ReLU(double a) {
        if (a > 0) {
            return a;
        }
        else {
            return 0;
        }
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

    std::vector<double> mean_squared_error(std::vector<std::vector<double>>& teacher, std::vector<std::vector<double>>& output) {
        int sample_size = output.size();
        std::vector<double> tmp(output[0].size(),0);
        for (int i = 0;i < sample_size;i++) {
            for (int j = 0;j < tmp.size();j++) {
                tmp[j] += (teacher[i][j] - output[i][j]) * (teacher[i][j] - output[i][j]);
            }
        }
        for (int j = 0; j < tmp.size(); j++) {
            tmp[j] /= sample_size;
        }
        return tmp;
    }
    //stub
    std::vector<double> cross_entropy(std::vector<std::vector<double>>& teacher, std::vector<std::vector<double>>& output) {
        int sample_size = output.size();
        std::vector<double> tmp(output[0].size(), 0);
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

    void Model::input_data(std::vector<double>& data) {
        int input_num = data.size();
        if (net_layer.front().Output.size() != input_num) {
            printf("input error|!|\n");
        }
        else {
            net_layer.front().Output = data;
        }
    }

    tisaMat::matrix Model::F_propagate(tisaMat::matrix& Input_data) {
        int sample_size = Input_data.mat_RC[0];
        tisaMat::matrix output_matrix(sample_size, net_layer.back().Output.size());
        for (int data_index = 0;data_index < sample_size;data_index++) {
            input_data(Input_data.elements[data_index]);
            for (int i = 1; i < number_of_layer(); i++) {
                std::vector<double> X = *(tisaMat::vector_multiply(net_layer[i - 1].Output, *net_layer[i].W));
                X = *(tisaMat::vector_add(X, net_layer[i].B));
                for (int j = 0; j < X.size(); j++) {
                    net_layer[i].Output[j] = (*Af[net_layer[i].Activation_f])(X[j]);
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
        for (int data_index = 0; data_index < sample_size; data_index++) {
            input_data(Input_data.elements[data_index]);
            for (int i = 1; i < number_of_layer(); i++) {
                std::vector<double> X = *(tisaMat::vector_multiply(net_layer[i - 1].Output, *net_layer[i].W));
                X = *(tisaMat::vector_add(X, net_layer[i].B));
                for (int j = 0; j < X.size(); j++) {
                    net_layer[i].Output[j] = (*Af[net_layer[i].Activation_f])(X[j]);

                    trainer[i - 1].Y[data_index] = net_layer[i].Output;
                }
            }
            output_matrix.elements[data_index] = net_layer.back().Output;
        }
        return output_matrix;
    }

    void Model::B_propagate(std::vector<std::vector<double>>& teacher, tisaMat::matrix& output,uint8_t error_func, std::vector<Trainer>& trainer,double lr,tisaMat::matrix& input_batch) {
        int output_num = output.mat_RC[1];
        int batch_size = output.mat_RC[0];

        //初回限定で誤差をセットして学習率もかける
        tisaMat::matrix error_matrix(teacher);
        error_matrix = *(tisaMat::matrix_subtract(output,error_matrix));
        printf("error_matrix for propagate\n");
        error_matrix.show();
        error_matrix.multi_scalar(lr);

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

            for (int current_layer = net_layer.size() - 1; current_layer > 0; current_layer--) {
                //秘伝のたれを仕込む(伝播する行列)
                    //ノードごとの活性化関数の微分
                tisaMat::matrix dAf(1, net_layer[current_layer].node);
                switch (net_layer[current_layer].Activation_f) {
                case SIGMOID:
                    for (int i = 0; i < net_layer[current_layer].node; i++) {
                        double Y = trainer[current_layer-1].Y[batch_segment][i];
                        dAf.elements[0][i] = Y * (1 - Y);
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
                propagate_matrix = *(tisaMat::matrix_Hadamard(dAf, propagate_matrix));

                //今の層の重み、バイアスの更新量を計算する
                    //重みは順伝播のときの入力も使う
                tisaMat::matrix* W_tmp;
                if((current_layer - 1) > 0){
                    W_tmp = tisaMat::vector_to_matrix(trainer[current_layer - 2].Y[batch_segment]);//current_layer-2のトレーナーは、前の層のトレーナー
                    W_tmp = tisaMat::matrix_transpose(*W_tmp);
                    W_tmp = tisaMat::matrix_multiply(*W_tmp, propagate_matrix);
                }
                else {
                    W_tmp = tisaMat::vector_to_matrix(input_batch.elements[batch_segment]);
                    W_tmp = tisaMat::matrix_transpose(*W_tmp);
                    W_tmp = tisaMat::matrix_multiply(*W_tmp, propagate_matrix);
                }
                
                trainer[current_layer - 1].dW = tisaMat::matrix_add(*trainer[current_layer - 1].dW,*W_tmp);
                    //バイアス
                trainer[current_layer - 1].dB = *(tisaMat::vector_add(trainer[current_layer - 1].dB,propagate_matrix.elements[0]));

                //今の層の重みの転置行列を秘伝のたれのうしろから行列積で次の層へ
                W_tmp = tisaMat::matrix_transpose(*(net_layer[current_layer].W));
                propagate_matrix = *(tisaMat::matrix_multiply(propagate_matrix, *W_tmp));
            }
        }

        //ミニバッチ学習の場合、重みとかの更新量を平均する
        if (batch_size > 1) {
            for (int i = 0;i < trainer.size();i++) {
                trainer[i].dW->multi_scalar(1.0 / batch_size);
                tisaMat::vector_multiscalar(trainer[i].dB,1.0 / batch_size);
            }
        }else{
        }

    }

    int Model::number_of_layer() {
        return net_layer.size();
    }

    void Model::initialize() {
        std::random_device seed_gen;
        std::default_random_engine rand_gen;
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
        const char* format_checker = format_key;
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

        const char* Data_head = data_head;
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

        const char* Format_key = format_key;
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
        const char* Data_head = data_head;
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

        printf("The file was output successfully\n");
    }

    void Model::train(double learning_rate,Data_set& train_data, Data_set& test_data, int epoc, int iteration, uint8_t Error_func) {
        if (net_layer[0].node != train_data.sample_data[0].size()) {
            printf("Input size and number of input layer's nodes do not match");
            exit(EXIT_FAILURE);
        }
        if (net_layer.back().node != train_data.answer[0].size()) {
            printf("Output size and number of output layer's nodes do not match");
            exit(EXIT_FAILURE);
        }
        
        int output_num = net_layer.back().node;
        int batch_size = train_data.sample_data.size() / iteration;
        tisaMat::matrix output_iterate(batch_size,output_num);
        tisaMat::matrix input_iterate(batch_size, train_data.sample_data[0].size());
        tisaMat::matrix answer_iterate(batch_size, output_num);
        std::vector<double> error(output_num);
        tisaMat::matrix test_mat(test_data.sample_data);
        std::vector<std::vector<double>> teach_iterate(batch_size);

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

        for (int ep = 0; ep < epoc; ep++) {
            for (int i = 0; i < iteration; i++) {

                if (train_data.sample_data.size() < batch_size * i) break;

                //次のイテレーションでつかう入力の行列をつくる
                for (int j = 0; j < batch_size; j++) {
                    input_iterate.elements[j] = train_data.sample_data[(batch_size * i) + j];
                    teach_iterate[j] = train_data.answer[(batch_size * i) + j];
                }
                printf("| %d time |\n",ep);
                printf("input\n");
                input_iterate.show();
                output_iterate = F_propagate(input_iterate, trainer);

                printf("output\n");
                output_iterate.show();
                printf("answer\n");
                tisaMat::matrix answer_matrix(teach_iterate);
                answer_matrix.show();
                
                error = (*Ef[Error_func])(teach_iterate, output_iterate.elements);
                printf("Error (error_func:mode %d)  ", Error_func);
                for (int i = 0; i < error.size(); i++) {
                    printf("%lf ", error[i]);
                }
                printf("\n");

                bool have_error = 0;
                for (int i = 0; i < error.size();i++) {
                    if (error[i] != 0.0) {
                        have_error = 1;
                        break;
                    }
                }


                if (have_error) {
                    B_propagate(teach_iterate, output_iterate, Error_func, trainer, learning_rate, input_iterate);
                    //トレーナーの値を使って重みを調整する
                    for (int layer = 1; layer < net_layer.size(); layer++) {
                        //重み
                        net_layer[layer].W = tisaMat::matrix_subtract(*net_layer[layer].W, *trainer[layer - 1].dW);
                        printf("%d layer dW\n", layer);
                        trainer[layer - 1].dW->show();
                        //バイアス
                        net_layer[layer].B = *(tisaMat::vector_subtract(net_layer[layer].B, trainer[layer - 1].dB));
                        printf("%d layer dB\n", layer);
                        tisaMat::vector_show(trainer[layer - 1].dB);
                    }
                }
            }

            //今の重みとか表示(デバッグ用)
            for (int layer = 1; layer < net_layer.size(); layer++) {
                //重み
                printf("W\n");
                net_layer[layer].W->show();
                //バイアス
                printf("B\n");
                tisaMat::vector_show(net_layer[layer].B);
            }

            //テスト(output_iterateは使いまわし)
            output_iterate = F_propagate(test_mat);
            printf("| TEST |\n");
            printf("test_input\n");
            test_mat.show();
            printf("test_output\n");
            output_iterate.show();
            error = (*Ef[Error_func])(test_data.answer, output_iterate.elements);
            printf("Error : ");
            tisaMat::vector_show(error);

            bool have_error = 0;
            for (int i = 0; i < error.size(); i++) {
                if (error[i] != 0.0) {
                    have_error = 1;
                    break;
                }
            }

            if (have_error == 0) {
                break;
            }
        }
    }
}
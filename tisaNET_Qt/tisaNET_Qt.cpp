#include "tisaNET_Qt.h"
#include <fstream>

namespace tisaNET_Qt
{
    void Model::monitor_network(bool monitor_network) {
        net_view = monitor_network;
        //net_viewer->show();
    }

    void Model::save_model(const char* tp_file) {
        save_model_file = tp_file;
        save_model_flag = true;
    }

    Q_INVOKABLE void Model::train() {
        if (net_layer[0].node != train_data->data[0].size()) {
            printf("Input size and number of input layer's nodes do not match");
            exit(EXIT_FAILURE);
        }
        if (net_layer.back().node != train_data->answer[0].size()) {
            printf("Output size and number of output layer's nodes do not match");
            exit(EXIT_FAILURE);
        }

        int output_num = net_layer.back().node;
        int iteration = train_data->data.size() / batch_size;

        if (iteration < 1) {
            printf("batch size is over sample size|!|\n");
            exit(EXIT_FAILURE);
        }

        tisaMat::matrix output_iterate(batch_size, output_num);
        tisaMat::matrix input_iterate(batch_size, train_data->data[0].size());
        tisaMat::matrix answer_iterate(batch_size, output_num);
        double error;
        tisaMat::matrix test_mat(test_data->data);
        std::vector<std::vector<uint8_t>> teach_iterate(batch_size);

        //バックプロパゲーションの時に重みの更新量を記憶するトレーナーをつくる
        std::vector<tisaNET::Trainer> trainer;
        for (int i = 0; i < net_layer.size() - back_prop_offset; i++) {
            tisaNET::Trainer tmp;
            if (!(net_layer[i].is_pool_layer())) {
                tmp.dW = new tisaMat::matrix(net_layer[i + back_prop_offset].W->mat_RC[0], net_layer[i + back_prop_offset].W->mat_RC[1]);
            }

            if (net_layer[i].is_conv_layer()) {
                tmp.dB = std::vector<double>(net_layer[i].filter_num);
                int tmpRC[2] = { net_layer[i].Output_mat.front().mat_RC[0], net_layer[i].Output_mat.front().mat_RC[1] };
                for (int j = 0; j < batch_size; j++) {
                    tmp.Y_mat.push_back(std::vector<tisaMat::matrix>(net_layer[i].filter_num, tisaMat::matrix(tmpRC[0], tmpRC[1])));
                }
            }
            else if (net_layer[i].is_pool_layer()) {
                int tmpRC[2] = { net_layer[i].Output_mat.front().mat_RC[0], net_layer[i].Output_mat.front().mat_RC[1] };
                int outputRC[2] = { net_layer[i].output_dim3[0], net_layer[i].output_dim3[1] };
                for (int j = 0; j < batch_size; j++) {
                    tmp.Y_mat.push_back(std::vector<tisaMat::matrix>(net_layer[i].output_dim3[2], tisaMat::matrix(tmpRC[0], tmpRC[1])));
                    tmp.pool_index.push_back(std::vector < std::vector < std::vector < std::array<int, 3>>>>(net_layer[i].output_dim3[2], std::vector < std::vector < std::array<int, 3>>>(outputRC[0], std::vector<std::array<int, 3>>(outputRC[1], std::array<int, 3>{0, 0, 0}))));
                }
            }
            else if (!(net_layer[i].is_pool_layer())) {
                tmp.dB = std::vector<double>(net_layer[i + back_prop_offset].node);
                for (int j = 0; j < batch_size; j++) {
                    tmp.Y.push_back(std::vector<double>(net_layer[i + back_prop_offset].node));
                }
            }
            trainer.push_back(tmp);
        }
        if (conv_count > 0) {
            for (int i = 0; i < batch_size; i++) {
                trainer[conv_count - 1].Y.push_back(std::vector<double>(net_layer[conv_count - 1 + back_prop_offset].output_dim3[0]
                    * net_layer[conv_count - 1 + back_prop_offset].output_dim3[1]
                    * net_layer[conv_count - 1 + back_prop_offset].output_dim3[2]));
            }
        }

        char ts[20] = { "\0" };
        time_t t = time(nullptr);
        std::tm timestr;
#ifdef _MSC_VER
        localtime_r(&t, &timestr);
#else
        localtime_s(&timestr, &t);
#endif
        strftime(ts, 20, "%Y/%m/%d %H:%M:%S", &timestr);
        printf("<trainning started at %s>\n", ts);

        //CSV形式で誤差を記録する準備
        if (log_error) {
            std::ofstream o_file(log_filename);
            if (!o_file) {
                printf("failed to open file : %s\n", log_filename);
                exit(EXIT_FAILURE);
            }
            o_file << "epoc,Error" << '\n';

            for (int ep = 0; ep < epoc; ep++) {
                if (stop_flag && stop_mode == Current_Epoc) {
                    goto finish;
                }

                printf("| epoc : %6d / %6d|\n", ep + 1, epoc);

                tisaNET::data_shuffle(*train_data);

                for (int i = 0; i < iteration; i++) {
                    if (pause_flag) {
                        while (pause_flag) { QThread::msleep(loop_msec); }
                    }
                    if (stop_flag && stop_mode == Just_Now) {
                        goto finish;
                    }

                    if (train_data->data.size() < batch_size * i) break;

                    //次のイテレーションでつかう入力の行列をつくる
                    for (int j = 0; j < batch_size; j++) {
                        input_iterate.elements[j] = tisaMat::vector_cast<double>(train_data->data[(batch_size * i) + j]);
                        teach_iterate[j] = train_data->answer[(batch_size * i) + j];
                    }

                    //printf("| epoc : %6d |",ep);


                    //printf("input\n");
                    //input_iterate.show();
                    output_iterate = feed_forward(input_iterate, trainer);

                    //printf("output\n");
                    //output_iterate.show();
                    //printf("answer\n");
                    //tisaMat::matrix answer_matrix(teach_iterate);
                    //answer_matrix.show();


                    error = (*Ef[Error_func])(teach_iterate, output_iterate.elements);//ここではじめて？nan
                    //printf("|%6d iterate|Error   :  %lf\n", i,error);

                    o_file << float(ep) + float(i + 1.0) / float(iteration) << ',' << error << '\n';

                    bool have_error = 0;
                    if (error != 0.0) {
                        have_error = 1;
                    }

                    if (have_error) {
                        B_propagate(teach_iterate, output_iterate, Error_func, trainer, learning_rate, input_iterate);
                        //B_propagate2(teach_iterate, output_iterate, Error_func, trainer, learning_rate, input_iterate);
                        //トレーナーの値を使って重みを調整する
                        for (int layer = back_prop_offset; layer < net_layer.size(); layer++) {
                            if (!net_layer[layer].is_pool_layer()) {
                                //重み
                                *(net_layer[layer].W) = tisaMat::matrix_subtract(*net_layer[layer].W, *trainer[layer - back_prop_offset].dW);

                                if (net_layer[layer].is_conv_layer()) {
                                    for (int filter = 0; filter < net_layer[layer].filter_num;filter++) {
                                        std::vector<tisaMat::matrix> tmp_data = tisaNET::conv_vect_to_mat3D(net_layer[layer].W->elements[filter], net_layer[layer].filter_dim3);
                                        emit filter_changed(layer + 1,filter,tmp_data);
                                    }
                                }

                                //printf("%d layer dW\n", layer);
                                //trainer[layer - 1].dW->show();
                                //バイアス
                                net_layer[layer].B = tisaMat::vector_subtract(net_layer[layer].B, trainer[layer - back_prop_offset].dB);
                                //printf("%d layer dB\n", layer);
                                //tisaMat::vector_show(trainer[layer - 1].dB);
                            }
                        }
                    }

                    tisaNET::show_train_progress(iteration, i);
                }
                printf("\n");
                /*
                //今の重みとか表示(デバッグ用)
                for (int layer = back_prop_offset; layer < net_layer.size(); layer++) {
                    //重み
                    printf("W\n");
                    net_layer[layer].W->show();
                    //バイアス
                    printf("B\n");
                    tisaMat::vector_show(net_layer[layer].B);
                }
                */

                //テスト(output_iterateは使いまわし)
                output_iterate = feed_forward(test_mat);
                printf("| TEST |");
                //printf("test_input\n");
                //test_mat.show();
                //printf("test_output\n");
                //output_iterate.show();
                t = time(nullptr);
#ifdef _MSC_VER
                localtime_r(&t, &timestr);
#else
                localtime_s(&timestr, &t);
#endif
                strftime(ts, 20, "%Y/%m/%d %H:%M:%S", &timestr);
                error = (*Ef[Error_func])(test_data->answer, output_iterate.elements);
                printf("Error : %lf <timestamp : %s>\n", error, ts);
                tisaNET::clear_under_cl(net_layer.size());
                if (monitoring_accuracy == true) {
                    m_a(output_iterate.elements, test_data->answer, Error_func);
                }

                if (error == 0.0) {
                    break;
                }
            }

        }
        else {
            for (int ep = 0; ep < epoc; ep++) {
                if (stop_flag && stop_mode == Current_Epoc) {
                    goto finish;
                }
                printf("\033[2K");
                printf("| epoc : %6d / %6d|\n", ep + 1, epoc);

                data_shuffle(*train_data);

                for (int i = 0; i < iteration; i++) {
                    if (pause_flag) {
                        while (pause_flag) { QThread::msleep(loop_msec); }
                    }
                    if (stop_flag && stop_mode == Just_Now) {
                        goto finish;
                    }

                    if (train_data->data.size() < batch_size * i) break;

                    //次のイテレーションでつかう入力の行列をつくる
                    for (int j = 0; j < batch_size; j++) {
                        input_iterate.elements[j] = tisaMat::vector_cast<double>(train_data->data[(batch_size * i) + j]);
                        teach_iterate[j] = train_data->answer[(batch_size * i) + j];
                    }

                    //printf("| epoc : %6d |",ep);


                    //printf("input\n");
                    //input_iterate.show();
                    output_iterate = feed_forward(input_iterate, trainer);

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
                        for (int layer = back_prop_offset; layer < net_layer.size(); layer++) {
                            if (!net_layer[layer].is_pool_layer()) {
                                //重み
                                *(net_layer[layer].W) = tisaMat::matrix_subtract(*net_layer[layer].W, *trainer[layer - back_prop_offset].dW);

                                if (net_layer[layer].is_conv_layer()) {
                                    for (int filter = 0; filter < net_layer[layer].filter_num; filter++) {
                                        std::vector<tisaMat::matrix> tmp_data = tisaNET::conv_vect_to_mat3D(net_layer[layer].W->elements[filter], net_layer[layer].filter_dim3);
                                        emit filter_changed(layer + 1, filter, tmp_data);
                                    }
                                }
                                //printf("%d layer dW\n", layer);
                                //trainer[layer - 1].dW->show();
                                //バイアス
                                net_layer[layer].B = tisaMat::vector_subtract(net_layer[layer].B, trainer[layer - back_prop_offset].dB);
                                //printf("%d layer dB\n", layer);
                                //tisaMat::vector_show(trainer[layer - 1].dB);
                            }
                            if (net_layer[layer].is_conv_layer()) {
                                net_layer[layer].W_normalization();
                            }
                        }
                    }

                    tisaNET::show_train_progress(iteration, i);
                }
                printf("\n");
                /*
                //今の重みとか表示(デバッグ用)
                for (int layer = back_prop_offset; layer < net_layer.size(); layer++) {
                    //重み
                    printf("W\n");
                    net_layer[layer].W->show();
                    //バイアス
                    printf("B\n");
                    tisaMat::vector_show(net_layer[layer].B);
                }
                */

                //テスト(output_iterateは使いまわし)
                output_iterate = feed_forward(test_mat);
                printf("| TEST |");
                //printf("test_input\n");
                //test_mat.show();
                //printf("test_output\n");
                //output_iterate.show();
                t = time(nullptr);
#ifdef _MSC_VER
                localtime_r(&t, &timestr);
#else
                localtime_s(&timestr, &t);
#endif
                strftime(ts, 20, "%Y/%m/%d %H:%M:%S", &timestr);
                error = (*Ef[Error_func])(test_data->answer, output_iterate.elements);
                printf("Error : %lf <timestamp : %s>\n", error, ts);
                tisaNET::clear_under_cl(net_layer.size());
                if (monitoring_accuracy == true) {
                    m_a(output_iterate.elements, test_data->answer, Error_func);
                }

                if (error == 0.0) {
                    break;
                }
            }
        }

        finish:
        printf("\ntraining complete!\n");
        if (save_model_flag) {
            save_model();
        }
    }

    void Model::save_model() {
        std::ofstream file(save_model_file, std::ios::binary);
        if (!file) {
            printf("failed to open file : %s\n", save_model_file);
            exit(EXIT_FAILURE);
        }
        //comvolute層のため廃止
        /*
        //INPUTのモードの層が入力層でないと読み込めないので、チェック
        if (net_layer[0].Activation_f != INPUT) {
            printf("first layer is not ""INPUT""\n");
            exit(EXIT_FAILURE);
        }
        */

        const char Format_key[f_k_size] = format_key;
        file.write(Format_key, format_key_size);
        uint8_t layer = number_of_layer();
        file.write(reinterpret_cast<char*>(&layer), sizeof(uint8_t));
        for (int current_layer = 0; current_layer < layer; current_layer++) {
            uint16_t node = net_layer[current_layer].node;
            file.write(reinterpret_cast<char*>(&node), sizeof(uint16_t));
        }
        for (int current_layer = 0; current_layer < layer; current_layer++) {
            //uint8_t Af = net_layer[current_layer].Activation_f | (net_layer[current_layer].is_conv << 4);
            uint8_t Af = net_layer[current_layer].Activation_f;
            file.write(reinterpret_cast<char*>(&Af), sizeof(uint8_t));
        }
        const char Data_head[d_size] = data_head;
        file.write(Data_head, data_head_size);

        //畳み込み層の概形を作るためのデータを書き込む
        if (conv_count > 0) {
            file.write(reinterpret_cast<char*>(&conv_count), sizeof(uint8_t));
            for (int i = 0; i < conv_count; i++) {
                file.write(reinterpret_cast<char*>(&net_layer[i].stride), sizeof(uint8_t));
                file.write(reinterpret_cast<char*>(&net_layer[i].input_dim3), 3 * sizeof(uint16_t));
                file.write(reinterpret_cast<char*>(&net_layer[i].filter_dim3), 3 * sizeof(uint8_t));
                file.write(reinterpret_cast<char*>(&net_layer[i].filter_num), sizeof(uint8_t));
            }
        }
        else {
            file.write(reinterpret_cast<char*>(&conv_count), sizeof(uint8_t));
        }

        //ここからモデルのパラメーターをファイルに書き込んでいく
        for (int current_layer = back_prop_offset + conv_count; current_layer < layer; current_layer++) {
            int W_row = net_layer[current_layer].W->mat_RC[0];
            int node = net_layer[current_layer].W->mat_RC[1];
            //重み行列を書き込む
            for (int r = 0; r < W_row; r++) {
                file.write(reinterpret_cast<char*>(&net_layer[current_layer].W->elements[r][0]), node * sizeof(double));
            }
            //バイアスを書き込む
            file.write(reinterpret_cast<char*>(&net_layer[current_layer].B[0]), node * sizeof(double));
        }

        const char exp_key[e_x_size] = expand_key;
        file.write(exp_key, expand_key_size);

        //畳み込み層のフィルターとバイアスのデータを書き込む
        for (int i = 0; i < conv_count; i++) {
            if (!net_layer[i].is_pool_layer()) {
                for (int row = 0; row < net_layer[i].filter_num; row++) {
                    file.write(reinterpret_cast<char*>(&net_layer[i].W->elements[row][0]),
                        net_layer[i].filter_dim3[0] * net_layer[i].filter_dim3[1] * net_layer[i].filter_dim3[2] * sizeof(double));
                }

                file.write(reinterpret_cast<char*>(&net_layer[i].B[0]), net_layer[i].filter_num * sizeof(double));
            }
        }

        printf("  :)  The file was output successfully!!! : %s\n", save_model_file);
    }

}
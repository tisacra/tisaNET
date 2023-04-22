#include "tisaNET_Qt.h"
#include <fstream>

namespace tisaNET_Qt
{
    void Model::monitor_network(bool monitor_network) {
        net_view = monitor_network;
        //net_viewer->show();
    }

    void Model::save_model(std::string tp_file) {
        save_model_file = tp_file;
        save_model_flag = true;
    }

    Q_INVOKABLE void Model::train() {
        tisaNET::Model::train(learning_rate,*train_data,*test_data,epoc,batch_size,Error_func);
        tisaNET::clear_under_cl(net_layer.size());
        if (save_model_flag) {
            tisaNET::Model::save_model(save_model_file);
        }
    }

    /*
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

        //�o�b�N�v���p�Q�[�V�����̎��ɏd�݂̍X�V�ʂ��L������g���[�i�[������
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

        std::ofstream o_file;
        if (log_error) {
            o_file = std::ofstream(log_filename);
            if (!o_file) {
                printf("failed to open file : %s\n", log_filename);
                exit(EXIT_FAILURE);
            }
            o_file << "epoc,Error" << '\n';
        }

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

                //���̃C�e���[�V�����ł������͂̍s�������
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


                error = (*Ef[Error_func])(teach_iterate, output_iterate.elements);//�����ł͂��߂āHnan
                //printf("|%6d iterate|Error   :  %lf\n", i, error);
                if (log_error) {
                    o_file << float(ep) + float(i + 1.0) / float(iteration) << ',' << error << '\n';
                }


                bool have_error = 0;
                if (error != 0.0) {
                    have_error = 1;
                }

                if (have_error) {
                    B_propagate(teach_iterate, output_iterate, Error_func, trainer, learning_rate, input_iterate);
                    //B_propagate2(teach_iterate, output_iterate, Error_func, trainer, learning_rate, input_iterate);
                    //�g���[�i�[�̒l���g���ďd�݂𒲐�����
                    for (int layer = back_prop_offset; layer < net_layer.size(); layer++) {
                        if (!net_layer[layer].is_pool_layer()) {
                            //�d��
                            *(net_layer[layer].W) = tisaMat::matrix_subtract(*net_layer[layer].W, *trainer[layer - back_prop_offset].dW);

                            if (net_layer[layer].is_conv_layer()) {
                                for (int filter = 0; filter < net_layer[layer].filter_num; filter++) {
                                    std::vector<tisaMat::matrix> tmp_data = tisaNET::conv_vect_to_mat3D(net_layer[layer].W->elements[filter], net_layer[layer].filter_dim3);
                                    emit filter_changed(layer * 2 + 1, filter, tmp_data);
                                }
                            }
                            //printf("%d layer dW\n", layer);
                            //trainer[layer - 1].dW->show();
                            //�o�C�A�X
                            net_layer[layer].B = tisaMat::vector_subtract(net_layer[layer].B, trainer[layer - back_prop_offset].dB);
                        }
                        if (net_layer[layer].is_conv_layer()) {
                            net_layer[layer].W_normalization();
                        }
                    }
                }

                tisaNET::show_train_progress(iteration, i);
            }
            printf("\n");
            output_iterate = feed_forward(test_mat);
            printf("| TEST |");
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

        finish:
        printf("\n \033[2K training complete!\n");
        tisaNET::clear_under_cl(net_layer.size());
        if (save_model_flag) {
            tisaNET::Model::save_model(save_model_file);
        }
    }
    */
}
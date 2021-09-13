#include "tisaNET.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <string>
#include <ctime>

#define format_key {'t','i','s','a','N','E','T'}
#define f_k_size 7

#define data_head {'D','A','T','A'}
#define d_size 4

#define expand_key {'E','X','P','A','N','D'}
#define e_x_size 6

#define format_key_size sizeof(char) * 7
#define data_head_size sizeof(char) * 4
#define expand_key_size sizeof(char) * 6

#define mnist_pict_offset 16
#define mnist_lab_offset 8

#define mnist_image_size 784

#define mnist_train_d "train-images.idx3-ubyte"
#define mnist_train_l "train-labels.idx1-ubyte"
#define mnist_test_d "t10k-images.idx3-ubyte"
#define mnist_test_l "t10k-labels.idx1-ubyte"

#define judge 0.05

#define progress_bar_length 40

#ifdef _MSC_VER
struct tm* localtime_r(const time_t* time, struct tm* resultp){
    if (localtime_s(resultp, time))
        return nullptr;
    return resultp;
}
#endif

namespace tisaNET{

    bool is_conv_layer(uint8_t i) {
        if ((0B00010000 & i) != 0) {
            return true;
        }
        else {
            return false;
        }
    }

    uint8_t get_Af(uint8_t i) {
        return 0B00001111 & i;
    }

    //MNISTからデータを作る
    void load_MNIST(const char* path, Data_set& train_data, Data_set& test_data,int sample_size,int test_size, bool single_output) {
        std::random_device seed_gen;
        std::default_random_engine rand_gen(seed_gen());
        std::string folder = path;

        unsigned int train_data_start = rand_gen() % 60000;
        unsigned int test_data_start = rand_gen() % 10000;
        //ここから訓練のデータ
        if(sample_size > 0){
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
            std::vector<double> tmp_d_use(mnist_image_size);
            uint8_t tmp_for_l;
            for (int i = 0,index=train_data_start;i < sample_size;i++,index++) {
                std::vector<uint8_t> tmp_l;
                if (index >= 60000) {
                    index = 0;
                    file_d.seekg(mnist_pict_offset);
                    file_l.seekg(mnist_lab_offset);
                }
                file_d.read(reinterpret_cast<char*>(&tmp_d[0]), sizeof(uint8_t) * mnist_image_size);
                for (int j = 0; j < mnist_image_size; j++) {
                    tmp_d_use[j] = tmp_d[j] / 256.;
                }
                train_data.data.push_back(tmp_d_use);
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
                /*//debug
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
            std::vector<double> tmp_d_use(mnist_image_size);

            uint8_t tmp_for_l;
            for (int i = 0, index = test_data_start; i < test_size; i++, index++) {
                std::vector<uint8_t> tmp_l;
                if (index >= 60000) {
                    file_d.seekg(mnist_pict_offset);

                }
                file_d.read(reinterpret_cast<char*>(&tmp_d[0]), sizeof(uint8_t) * mnist_image_size);
                for (int j = 0; j < mnist_image_size; j++) {
                    tmp_d_use[j] = tmp_d[j] / 256.;
                }
                test_data.data.push_back(tmp_d_use);
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

    void load_MNIST(const char* path, Data_set& test_data,const int test_size, bool single_output) {
        std::random_device seed_gen;
        std::default_random_engine rand_gen(seed_gen());
        std::string folder = path;

        unsigned int test_data_start = rand_gen() % 10000;
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
            std::vector<double> tmp_d_use(mnist_image_size);

            uint8_t tmp_for_l;
            for (int i = 0, index = test_data_start; i < test_size; i++, index++) {
                std::vector<uint8_t> tmp_l;
                if (index >= 60000) {
                    file_d.seekg(mnist_pict_offset);

                }
                file_d.read(reinterpret_cast<char*>(&tmp_d[0]), sizeof(uint8_t) * mnist_image_size);
                for (int j = 0; j < mnist_image_size; j++) {
                    tmp_d_use[j] = tmp_d[j] / 256.;
                }
                test_data.data.push_back(tmp_d_use);
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

    //256色BMPファイルから一次配列をつくる
    std::vector<uint8_t> vec_from_256bmp(const char* bmp_file) {
        std::ifstream file(bmp_file, std::ios::binary);
        if (!file) {
            printf("failed to open file : %s\n", bmp_file);
            exit(EXIT_FAILURE);
        }
        char file_check[2];
        file.read(file_check,sizeof(char)*2);
        if (file_check[0] != 'B' || file_check[1] != 'M') {
            printf("%s is not BMP file\n",bmp_file);
            exit(EXIT_FAILURE);
        }

        int filesize;
        char intbuf[4];
        file.read(intbuf, sizeof(int) * 1);
        filesize = intbuf[3] << 24 | intbuf[2] << 16 | intbuf[1] << 8 | intbuf[0];

        file.seekg(sizeof(char)*10,std::ios_base::beg);
        int todata;
        file.read(intbuf,sizeof(int)*1);
        todata = intbuf[3] << 24 | intbuf[2] << 16 | intbuf[1] << 8 | intbuf[0];

        int widge, height;
        file.seekg(sizeof(char) * 18, std::ios_base::beg);
        file.read(intbuf, sizeof(int) * 1);
        widge = intbuf[3] << 24 | intbuf[2] << 16 | intbuf[1] << 8 | intbuf[0];
        file.read(intbuf, sizeof(int) * 1);
        height = intbuf[3] << 24 | intbuf[2] << 16 | intbuf[1] << 8 | intbuf[0];

        file.seekg(sizeof(char)*(todata),std::ios_base::beg);
        std::vector<uint8_t> tmp(filesize - todata);
        std::vector<uint8_t> FF(tmp.size(),0xff);
        file.read(reinterpret_cast<char*>(&tmp[0]),sizeof(uint8_t)*(filesize - todata));
        tmp = tisaMat::vector_subtract(FF, tmp);
        
        //上下反転
        for (int i = 0; i < height/2;i++) {
            for (int j = 0; j < widge;j++) {
                std::swap(tmp[widge*i + j],tmp[widge*(height-1-i) + j]);
            }
        }
        return tmp;
    }

    void data_shuffle(Data_set& sdata) {
        int center_index = sdata.data.size() / 2;
        int sample_size = sdata.data.size();
        std::random_device seed_gen;
        std::default_random_engine rand_gen(seed_gen());
        std::vector<int> page;
        
        //シャッフルの基準になる配列(これをシャッフルして、要素の並びをまねさせる)
        for (int i = 0; i < sample_size;i++) {
            page.push_back(i);
        }
        std::shuffle(page.begin(),page.end(),rand_gen);

        for (int i = sample_size-1; i > 0;i--) {
            int swap_index = std::distance(page.begin(),std::find(page.begin(),page.end(),i));
            if (i != swap_index) {
                std::swap(page[swap_index],page[i]);
                std::swap(sdata.data[swap_index], sdata.data[i]);
                std::swap(sdata.answer[swap_index], sdata.answer[i]);
            }
        }
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

    tisaMat::matrix dilate(tisaMat::matrix& mat, uint8_t d) {
        int row = mat.mat_RC[0];
        int col = mat.mat_RC[1];

        tisaMat::matrix tmp((row * (d + 1) - d),(col * (d + 1) - d));
        for (int i = 0; i < row;i++) {
            for (int j = 0; j < col;j++) {
                tmp.elements[i * (d + 1)][j * (d + 1)] = mat.elements[i][j];
            }
        }
        return tmp;
    }

    tisaMat::matrix zero_padding(tisaMat::matrix& mat, uint8_t p,uint8_t q) {
        int row = mat.mat_RC[0];
        int col = mat.mat_RC[1];

        tisaMat::matrix tmp(row + (2 * p),col + (2 * p));
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                tmp.elements[i + p][j + q] = mat.elements[i][j];
            }
        }
        return tmp;
    }

    tisaMat::matrix zero_padding_half(tisaMat::matrix& mat, uint8_t p, uint8_t q) {
        int row = mat.mat_RC[0];
        int col = mat.mat_RC[1];

        tisaMat::matrix tmp(row + p, col + p);
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                tmp.elements[i][j] = mat.elements[i][j];
            }
        }
        return tmp;
    }

    void Model::Create_Layer(int nodes, uint8_t Activation) {
        layer tmp;
        if (Activation < INPUT) {
            int input = net_layer.back().Output.size();
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
            back_prop_offset++;
        }
        net_layer.push_back(tmp);
    }
    //おまけ(重みとバイアスを任意の値で初期化)
    void Model::Create_Layer(int nodes, uint8_t Activation,double init) {
        layer tmp;
        if (Activation < INPUT) {
            int input = net_layer.back().Output.size();
            tmp.node = nodes;
            tmp.Activation_f = Activation;
            tmp.W = new tisaMat::matrix(input, nodes, init);
            tmp.B = std::vector<double>(nodes,init);
            tmp.Output = std::vector<double>(nodes);
        }
        else {
            tmp.node = nodes;
            tmp.Activation_f = Activation;
            tmp.Output = std::vector<double>(nodes);
            back_prop_offset++;
        }
        net_layer.push_back(tmp);
    }

    void Model::Create_Comvolute_Layer(uint8_t Activation,int input_shape[3], int filter_shape[3], int f_num, int st) {
        layer tmp;
        tmp.is_conv = true;
        tmp.Activation_f = Activation;
        tmp.input_dim3[0] = input_shape[0];
        tmp.input_dim3[1] = input_shape[1];
        tmp.input_dim3[2] = input_shape[2];
        tmp.node = input_shape[0] * input_shape[1];
        tmp.filter_num = f_num;
        tmp.W = new tisaMat::matrix(f_num, filter_shape[0] * filter_shape[1] * filter_shape[2]);
        tmp.B = std::vector<double>(f_num);
        tmp.filter_dim3[0] = filter_shape[0];
        tmp.filter_dim3[1] = filter_shape[1];
        tmp.filter_dim3[2] = filter_shape[2];
        tmp.stride = st;

        //入力よりフィルタが小さいかチェック
        bool size_check = false;
        for (int i = 0; i < 3; i++) {
            if (tmp.input_dim3[i] < tmp.filter_dim3[i]) {
                size_check = true;
            }
        }
        if (size_check) {
            printf("|!|ERROR|!| filter size is bigger than input!\n");
            printf("filter : ( %4d, %4d , %4d) , input : ( %4d, %4d , %4d)\n", tmp.filter_dim3[0], tmp.filter_dim3[1], tmp.filter_dim3[2],
                tmp.input_dim3[0], tmp.input_dim3[1], tmp.input_dim3[2]);
            exit(EXIT_FAILURE);
        }

        //paddingサイズきめる(ハーフパディング)
        tmp.pad[0] = (((input_shape[0] - filter_shape[0]) / st + 1) * st - (input_shape[0] - filter_shape[0])) % st;
        tmp.pad[1] = (((input_shape[1] - filter_shape[1]) / st + 1) * st - (input_shape[1] - filter_shape[1])) % st;

        if (tmp.pad[0] != 0 || tmp.pad[1] != 0) {
            tmp.padding_flag = true;
        }

        //元サイズ、フィルター、ストライド、フィルター数が決まれば出力サイズ確定
        tmp.output_dim3[0] = (input_shape[0] - filter_shape[0] + tmp.pad[0]) / st + 1;
        tmp.output_dim3[1] = (input_shape[1] - filter_shape[1] + tmp.pad[1]) / st + 1;
        tmp.output_dim3[2] = (input_shape[2] / filter_shape[2]) * f_num;

        tmp.Output = std::vector<double>(tmp.output_dim3[0] * tmp.output_dim3[1] * tmp.output_dim3[2]);
        tmp.Output_mat = new tisaMat::matrix(tmp.output_dim3[2], (tmp.output_dim3[0] * tmp.output_dim3[1]));
        //back_prop_offset++;
        comv_count++;
        net_layer.push_back(tmp);
    }

    void Model::Create_Comvolute_Layer(uint8_t Activation,int filter_shape[3], int f_num, int st) {
        if (net_layer.size() == 0) {
            printf("Can't Create network without specify input shape\n");
            exit(EXIT_FAILURE);
        }
        layer tmp;
        tmp.is_conv = true;
        tmp.Activation_f = Activation;
        tmp.input_dim3[0] = net_layer.back().output_dim3[0];
        tmp.input_dim3[1] = net_layer.back().output_dim3[1];
        tmp.input_dim3[2] = net_layer.back().output_dim3[2];
        tmp.node = net_layer.back().output_dim3[0] * net_layer.back().output_dim3[1];
        tmp.filter_num = f_num;
        tmp.W = new tisaMat::matrix(f_num, filter_shape[0] * filter_shape[1] * filter_shape[2]);
        tmp.B = std::vector<double>(f_num);
        tmp.filter_dim3[0] = filter_shape[0];
        tmp.filter_dim3[1] = filter_shape[1];
        tmp.filter_dim3[2] = filter_shape[2];
        tmp.stride = st;

        //入力よりフィルタが小さいかチェック
        bool size_check = false;
        for (int i = 0; i < 3; i++) {
            if (tmp.input_dim3[i] < tmp.filter_dim3[i]) {
                size_check = true;
            }
        }
        if (size_check) {
            printf("|!|ERROR|!| filter size is bigger than input!\n");
            printf("filter : ( %4d, %4d , %4d) , input : ( %4d, %4d , %4d)\n",tmp.filter_dim3[0], tmp.filter_dim3[1], tmp.filter_dim3[2],
                                                                              tmp.input_dim3[0], tmp.input_dim3[1], tmp.input_dim3[2]);
            exit(EXIT_FAILURE);
        }

        //paddingサイズきめる(ハーフパディング)
        tmp.pad[0] = (((net_layer.back().output_dim3[0] - filter_shape[0]) / st + 1) * st - (net_layer.back().output_dim3[0] - filter_shape[0])) % st;
        tmp.pad[1] = (((net_layer.back().output_dim3[1] - filter_shape[1]) / st + 1) * st - (net_layer.back().output_dim3[1] - filter_shape[1])) % st;

        if (tmp.pad[0] != 0 || tmp.pad[1] != 0) {
            tmp.padding_flag = true;
        }

        //元サイズ、フィルター、ストライド、フィルター数が決まれば出力サイズ確定
        tmp.output_dim3[0] = (net_layer.back().output_dim3[0] - filter_shape[0] + tmp.pad[0]) / st + 1;
        tmp.output_dim3[1] = (net_layer.back().output_dim3[1] - filter_shape[1] + tmp.pad[1]) / st + 1;
        tmp.output_dim3[2] = (net_layer.back().output_dim3[2] / filter_shape[2]) * f_num;

        tmp.Output = std::vector<double>(tmp.output_dim3[0] * tmp.output_dim3[1] * tmp.output_dim3[2]);
        tmp.Output_mat = new tisaMat::matrix(tmp.output_dim3[2], (tmp.output_dim3[0] * tmp.output_dim3[1]));
        //back_prop_offset++;
        comv_count++;
        net_layer.push_back(tmp);
    }
    
    //1次元になった2次元データを畳み込む
    void layer::comvolute(std::vector<double>& input) {
        double tmp_sum;
        uint16_t row = input_dim3[0];
        uint16_t column = input_dim3[1];
        uint8_t filter_row = filter_dim3[0];
        uint8_t filter_column = filter_dim3[1];
        uint8_t dpt = filter_dim3[2];
        uint8_t f_num = filter_num;
        uint16_t route_row = output_dim3[0];
        uint16_t route_col = output_dim3[1];
        uint16_t filter_size2D = filter_row * filter_column;
        uint16_t feature_size = (row + pad[0]) * (column + pad[1]);
        double tmp_max = 0.0;

        std::vector<double> input_use;
        //必要ならハーフパディング
        if (padding_flag) {
            uint16_t origin_shape[3] = {row,column,dpt};
            std::vector<tisaMat::matrix> tmp_mat = comv_vect_to_mat(input,origin_shape);
            for (int i = 0; i < dpt;i++) {
                tmp_mat[i] = zero_padding_half(tmp_mat[i], pad[0], pad[1]);
            }
            input_use = comv_mat_to_vect(tmp_mat);
        }
        else {
            input_use = input;
        }

        //畳み込みできるかチェック
        if (feature_size * dpt != input_use.size()) {
            printf("input shape incorrect|!| input : %4d setting : ( %4d, %4d) * %3d\n",input_use.size(),row,column,dpt);
            exit(EXIT_FAILURE);
        }

        for (int c_f = 0; c_f < f_num;c_f++) {
            for (int base_row = 0; base_row < route_row; base_row++) {
                for (int base_col = 0; base_col < route_col; base_col++) {
                    tmp_sum = 0.;
                    //畳み込む(フィルターかける)
                    for (int c_f_dpt = 0; c_f_dpt < dpt;c_f_dpt++) {
                        for (int i = 0; i < filter_row; i++) {
                            for (int j = 0; j < filter_column; j++) {
                                tmp_sum += W->elements[c_f][(c_f_dpt * filter_size2D) + i * filter_column + j] * input_use[(c_f_dpt * feature_size) + ((i + base_row * stride) * input_dim3[1]) + (j + base_col * stride)];
                                
                            }
                        }
                    }
                    
                    tmp_sum += B[c_f];
                    Output[(c_f * route_row * route_col) + (base_row * route_col) + base_col] = tmp_sum;
                    if (tmp_sum > tmp_max) {
                        tmp_max = tmp_sum;
                    }
                }
            }
        }
        
        //最大値で平均
        if (tmp_max != 0.0) {
            tisaMat::vector_multiscalar(Output,1. / tmp_max);
        }
        //sigmoid関数つかう
        /*
        for (int i = 0; i < Output.size();i++) {
            Output[i] = sigmoid(Output[i]);
        }
        */
    }

    void layer::comvolute(tisaMat::matrix& input) {
        double tmp_sum;
        uint16_t row = input_dim3[0];
        uint16_t column = input_dim3[1];
        uint8_t filter_row = filter_dim3[0];
        uint8_t filter_column = filter_dim3[1];
        uint8_t dpt = filter_dim3[2];
        uint8_t f_num = filter_num;
        uint16_t route_row = output_dim3[0];
        uint16_t route_col = output_dim3[1];
        uint16_t route_dpt = input_dim3[2] / filter_dim3[2];
        uint16_t filter_size2D = filter_row * filter_column;
        uint16_t feature_size = (row + pad[0]) * (column + pad[1]);
        double tmp_max = 0.0;

        tisaMat::matrix input_use(route_dpt,feature_size);
        //必要ならハーフパディング
        if (padding_flag) {
            uint16_t origin_shape[2] = {row,column};

            for (int i = 0; i < dpt; i++) {
                tisaMat::matrix tmp_mat = comv_vect_to_mat2D(input.elements[i], origin_shape);
                tmp_mat = zero_padding_half(tmp_mat, pad[0], pad[1]);
                input_use.elements[i] = comv_mat_to_vect2D(tmp_mat);
            }
        }
        else {
            input_use = input;
        }

        //畳み込みできるかチェック
        if ((feature_size != input_use.mat_RC[1]) || (input_use.mat_RC[0] != route_dpt)) {
            printf("input shape incorrect|!| input( row * column , feature map number) : ( %4d , %4d) setting : ( %4d, %4d ,%4d) * %3d\n",input_use.mat_RC[1],input_use.mat_RC[0],row,column,dpt,f_num);
            exit(EXIT_FAILURE);
        }

        for (int c_f = 0; c_f < f_num;c_f++) {
            for (int current_map = 0;current_map < route_dpt;current_map++) {
                for (int base_row = 0; base_row < route_row; base_row++) {
                    for (int base_col = 0; base_col < route_col; base_col++) {
                        tmp_sum = 0.;
                        //畳み込む(フィルターかける)
                        for (int c_f_dpt = 0; c_f_dpt < dpt; c_f_dpt++) {
                            for (int i = 0; i < filter_row; i++) {
                                for (int j = 0; j < filter_column; j++) {
                                    tmp_sum += W->elements[c_f][(c_f_dpt * filter_size2D) + i * filter_column + j]
                                               * input_use.elements[current_map][(c_f_dpt * feature_size) + ((i + base_row * stride) * input_dim3[1]) + (j + base_col * stride)];
                                }
                            }
                        }
                        tmp_sum += B[c_f];
                        Output_mat->elements[(c_f * route_dpt) + current_map][(base_row * route_col) + base_col] = tmp_sum;
                        if (tmp_sum > tmp_max) {
                            tmp_max = tmp_sum;
                        }
                    }
                }
            }
        }
        
        //最大値で平均
        if (tmp_max != 0.0) {
            Output_mat->multi_scalar(1. / tmp_max);
        }
        //sigmoid関数つかう
        /*
        for (int i = 0; i < Output_mat->mat_RC[0];i++) {
            for (int j = 0; j < Output_mat->mat_RC[1];j++) {
                Output_mat->elements[i][j] = sigmoid(Output_mat->elements[i][j]);
            }
        }
        */
    }

    void layer::comvolute_test(tisaMat::matrix& input) {
        double tmp_sum;
        uint16_t row = 5;
        uint16_t column = 5;
        uint8_t filter_row = 2;
        uint8_t filter_column = 2;
        uint8_t dpt = 1;
        uint8_t f_num = 1;
        uint16_t route_row = (row + 1 - filter_row) / 2;
        uint16_t route_col = (column + 1 - filter_column) / 2;
        uint16_t route_dpt = 1 / 1;
        uint16_t filter_size2D = filter_row * filter_column;
        uint16_t feature_size = row * column;
        double sum_max = 0.;

        //畳み込みできるかチェック
        if ((feature_size != input.mat_RC[1]) || (input.mat_RC[0] != route_dpt)) {
            printf("input shape incorrect|!| input( row * column , feature map number) : ( %4d , %4d) setting : ( %4d, %4d ,%4d) * %3d\n", input.mat_RC[1], input.mat_RC[0], row, column, dpt, f_num);
            exit(EXIT_FAILURE);
        }

        std::vector<std::vector<double>> tmpWV = { {1,1,2,2} };
        tisaMat::matrix testW(tmpWV);
        tisaMat::matrix O(f_num * route_dpt,route_col * route_row);

        for (int c_f = 0; c_f < f_num; c_f++) {
            for (int current_map = 0; current_map < route_dpt; current_map++) {
                for (int base_row = 0; base_row < route_row; base_row++) {
                    for (int base_col = 0; base_col < route_col; base_col++) {
                        tmp_sum = 0.;
                        //畳み込む(フィルターかける)
                        for (int c_f_dpt = 0; c_f_dpt < dpt; c_f_dpt++) {
                            for (int i = 0; i < filter_row; i++) {
                                for (int j = 0; j < filter_column; j++) {
                                    tmp_sum += testW.elements[c_f][(c_f_dpt * filter_size2D) + i * filter_column + j]
                                        * input.elements[current_map][(c_f_dpt * feature_size) + ((i + base_row * 2) * 5) + (j + base_col * 2)];
                                }
                            }
                        }
                        tmp_sum += B[c_f];
                        O.elements[(c_f * route_dpt) + current_map][(base_row * route_col) + base_col] = tmp_sum;
                        if (tmp_sum > sum_max) {
                            sum_max = tmp_sum;
                        }
                    }
                }
            }
        }
        //正規化する
        Output_mat->multi_scalar(1. / sum_max);


    }

    void layer::output_vec_to_mat() {
        uint16_t out_shape[3] = { output_dim3[0] ,
                                  output_dim3[1] ,
                                  output_dim3[2]};
        uint16_t output2D = out_shape[0] * out_shape[1];

        for (int d = 0; d < out_shape[2];d++) {
            for (int r = 0; r < out_shape[0];r++) {
                for (int c = 0; c < out_shape[1];c++) {
                    Output_mat->elements[d][(r * out_shape[1] + c)] = Output[(d * output2D) + (r * out_shape[1] + c)];
                }
            }
        }
    }

    void layer::output_mat_to_vec() {
        for (int r = 0; r < Output_mat->mat_RC[0]; r++) {
            for (int c = 0; c < Output_mat->mat_RC[1]; c++) {
                Output[r * Output_mat->mat_RC[1] + c] = Output_mat->elements[r][c];
            }
        }
    }

    tisaMat::matrix Model::feed_forward(tisaMat::matrix& Input_data) {
        int sample_size = Input_data.mat_RC[0];
        tisaMat::matrix output_matrix(sample_size, net_layer.back().Output.size());
        for (int data_index = 0;data_index < sample_size;data_index++) {
            input_data(Input_data.elements[data_index]);
            for (int i = back_prop_offset + comv_count; i < number_of_layer(); i++) {
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
                    /*if (isnan(net_layer[i].Output[j])) {
                        bool nan_flug = 1;
                    }*/
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
    tisaMat::matrix Model::feed_forward(tisaMat::matrix& Input_data,std::vector<Trainer>& trainer) {
        int sample_size = Input_data.mat_RC[0];
        tisaMat::matrix output_matrix(sample_size, net_layer.back().Output.size());
        int layer_num = number_of_layer();
        for (int data_index = 0; data_index < sample_size; data_index++) {
            input_data(Input_data.elements[data_index],trainer,data_index);

            for (int i = back_prop_offset + comv_count; i < layer_num; i++) {
                std::vector<double> X = tisaMat::vector_multiply(net_layer[i - 1].Output, *net_layer[i].W);
                X = tisaMat::vector_add(X, net_layer[i].B);
                //ここまでで、活性化関数を使う前の計算が終了

                //ソフトマックス関数を使うときはまず最大値を全部から引く
                if (net_layer[i].Activation_f == SOFTMAX) {
                    double max = *std::max_element(X.begin(),X.end());
                    for (int X_count = 0; X_count < X.size(); X_count++) {
                        X[X_count] -= max;
                    }
                }

                for (int j = 0; j < X.size(); j++) {
                    net_layer[i].Output[j] = (*Af[net_layer[i].Activation_f])(X[j]);
                    /*if (isnan(net_layer[i].Output[j])) {
                        bool nan_flug = 1;
                    }*/
                }

                if (net_layer[i].Activation_f == SOFTMAX) {
                    double sigma = 0.0;
                    for (int node = 0; node < net_layer[i].Output.size(); node++) {
                        sigma += net_layer[i].Output[node];
                    }
                    tisaMat::vector_multiscalar(net_layer[i].Output, 1.0 / sigma);
                }

                trainer[i-back_prop_offset].Y[data_index] = net_layer[i].Output;
            }
            output_matrix.elements[data_index] = net_layer.back().Output;
            //デバッグ用の分散確認
            double dist = output_matrix.distributed();
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
        for (int current_layer = 0; current_layer < net_layer.size() - back_prop_offset;current_layer++) {
            trainer[current_layer].dW->multi_scalar(0);
            tisaMat::vector_multiscalar(trainer[current_layer].dB, 0);
        }
        
        //重みとかの更新量の平均を出す 具体的にはバッチのパターンごとに更新量を出して、あとでバッチサイズで割る
        for (int batch_segment = 0; batch_segment < batch_size; batch_segment++) {

            //伝播していく行列(秘伝のたれ) あとで行列積をつかいたいのでベクトルではなく行列として用意します(アダマール積もつかいますが)
            std::vector<std::vector<double>> tmp(1, error_matrix.elements[batch_segment]);
            tisaMat::matrix propagate_matrix(tmp);
            bool reduction_flag = cross_sig_flag;
            for (int current_layer = net_layer.size() - 1; current_layer >= back_prop_offset + comv_count; current_layer--) {
                int trainer_layer = current_layer - back_prop_offset;
                
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
                            double Y = trainer[trainer_layer].Y[batch_segment][i];
                            dAf.elements[0][i] = Y * (1 - Y);
                        }
                        break;
                    case SOFTMAX:
                    {
                        double tmp_softmax = 0.0;
                        for (int count = 0; count < net_layer[current_layer].node; count++) {
                            tmp_softmax += error_matrix.elements[batch_segment][count] * trainer[trainer_layer].Y[batch_segment][count];
                        }
                        for (int i = 0; i < net_layer[current_layer].node; i++) {
                            double Y = trainer[trainer_layer].Y[batch_segment][i];
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
                if((current_layer) > back_prop_offset + comv_count){
                    W_tmp = tisaMat::vector_to_matrix(trainer[trainer_layer - 1].Y[batch_segment]);//trainer_layer-1のトレーナーは、前の層のトレーナー
                    W_tmp = tisaMat::matrix_transpose(W_tmp);
                    W_tmp = tisaMat::matrix_multiply(W_tmp, propagate_matrix);
                }
                else {
                    if (net_layer[current_layer - 1].Activation_f == INPUT) {
                        W_tmp = tisaMat::vector_to_matrix(input_batch.elements[batch_segment]);//trainer_layer-1のトレーナーは、前の層のトレーナー
                        W_tmp = tisaMat::matrix_transpose(W_tmp);
                        W_tmp = tisaMat::matrix_multiply(W_tmp, propagate_matrix);
                    }
                    else {
                        W_tmp = tisaMat::vector_to_matrix(trainer[trainer_layer - 1].Y[batch_segment]);//trainer_layer-1のトレーナーは、前の層のトレーナー
                        W_tmp = tisaMat::matrix_transpose(W_tmp);
                        W_tmp = tisaMat::matrix_multiply(W_tmp, propagate_matrix);
                    }

                }
                
                *(trainer[trainer_layer].dW) = tisaMat::matrix_add(*trainer[trainer_layer].dW,W_tmp);
                //バイアス
                trainer[trainer_layer].dB = tisaMat::vector_add(trainer[trainer_layer].dB,propagate_matrix.elements[0]);

                //今の層の重みの転置行列を秘伝のたれのうしろから行列積で次の層へ
                W_tmp = tisaMat::matrix_transpose(*(net_layer[current_layer].W));
                propagate_matrix = tisaMat::matrix_multiply(propagate_matrix, W_tmp);
            }

            //ここから畳み込み層の更新量計算

            //もし全結合層手前で特徴マップ全部を一本にラスタライズされてたら、秘伝のタレ(層の出力の誤差)をマップ(ごとにラスタライズ)の形に直す
            //もしじゃなくても一回はするかも
            if (comv_count != 0) {
                uint16_t size2D = net_layer[comv_count - 1].Output_mat->mat_RC[1];
                uint16_t map_num = net_layer[comv_count - 1].Output_mat->mat_RC[0];
                tisaMat::matrix tmp(map_num,size2D);

                for (int i = 0; i < map_num;i++) {
                    for (int j = 0; j < size2D;j++) {
                        tmp.elements[i][j] = propagate_matrix.elements[0][(i * size2D) + j];
                    }
                }
                propagate_matrix = tmp;
            }

            for (int current_layer = comv_count-1; current_layer > back_prop_offset - 1;current_layer--) {
                uint8_t st = net_layer[current_layer].stride;

                uint16_t in_3d[3] = {net_layer[current_layer].input_dim3[0],
                                     net_layer[current_layer].input_dim3[1],
                                     net_layer[current_layer].input_dim3[2]};

                //単発の入力サイズ
                uint16_t in_segment_3d[3] = { net_layer[current_layer].input_dim3[0],
                                              net_layer[current_layer].input_dim3[1],
                                              net_layer[current_layer].filter_dim3[2] };

                //dW計算のcomvolute用
                uint16_t in_seg_3d[3] = { net_layer[current_layer].input_dim3[0] + net_layer[current_layer].pad[0],
                                          net_layer[current_layer].input_dim3[1] + net_layer[current_layer].pad[1],
                                          net_layer[current_layer].filter_dim3[2] };

                uint8_t fil_3d[3] = { net_layer[current_layer].filter_dim3[0],
                                      net_layer[current_layer].filter_dim3[1],
                                      net_layer[current_layer].filter_dim3[2] };

                //出力は単発では深さ1
                uint16_t out_3d[3] = { net_layer[current_layer].output_dim3[0],
                                       net_layer[current_layer].output_dim3[1],
                                       1};

                uint8_t fnum = net_layer[current_layer].filter_num;
                uint16_t feature_num = (in_3d[2] / fil_3d[2]);

                //次のpropagate_matrixの準備
                tisaMat::matrix tmp_prop(in_3d[2],in_3d[0] * in_3d[1]);

                for (int current_filter = 0; current_filter < fnum;current_filter++) {
                    for (int current_X = 0; current_X < feature_num;current_X++) {
                    //今の出力に対する更新量を計算
                        //重み(フィルター)の更新量は入力を誤差で畳み込み
                        std::vector<tisaMat::matrix> input;
                        if (current_layer == back_prop_offset) {
                            input = comv_vect_to_mat<uint16_t>(input_batch.elements[batch_segment], in_segment_3d);
                        }
                        else {
                            input = comv_vect_to_mat<uint16_t>(trainer[current_layer - 1].Y_mat[batch_segment].elements[current_X], in_segment_3d);
                        }
                        
                        //この層がパディングしていた層なら、同じように入力をパディング
                        if (net_layer[current_layer].padding_flag) {
                            for (int i = 0; i < in_segment_3d[2];i++) {
                                input[i] = zero_padding_half(input[i],net_layer[current_layer].pad[0], net_layer[current_layer].pad[1]);
                            }
                        }

                        std::vector<tisaMat::matrix> E = comv_vect_to_mat<uint16_t>(propagate_matrix.elements[(current_filter * feature_num) + current_X],out_3d);

                        //comvolute_layerの畳み込みにシグモイド関数を実装したので、その微分項を計算
                        /*
                        tisaMat::matrix tmp_dsig(out_3d[0],out_3d[1]);
                        for (int i = 0; i < out_3d[0];i++) {
                            for (int j = 0; j < out_3d[1];j++) {
                                double tmp_Y = trainer[current_layer].Y_mat[batch_segment].elements[current_filter * feature_num + current_X][i * out_3d[1] + j];
                                tmp_dsig.elements[i][j] = tmp_Y * (1 - tmp_Y);
                            }
                        }
                        //秘伝のタレとシグモイド微分のアダマール積
                        E[0] = tisaMat::Hadamard_product(E[0],tmp_dsig);
                        */

                        std::vector<tisaMat::matrix> E_for_dW;
                        for (int i = 0; i < out_3d[2];i++) {
                            E_for_dW.push_back(dilate(E[0], st - 1));
                        }

                        uint16_t EdW_3d[3] = {E_for_dW[0].mat_RC[0],E_for_dW[0].mat_RC[1],out_3d[2]};
                        trainer[current_layer].dW->elements[current_filter] = tisaMat::vector_add(trainer[current_layer].dW->elements[current_filter],
                                                                                                  comvolute(input,E_for_dW,in_seg_3d,EdW_3d,1));

                        for (int row = 0; row < out_3d[0];row++) {
                            for (int col = 0; col < out_3d[1];col++) {
                                trainer[current_layer].dB[current_filter] += E[0].elements[row][col];
                            }
                        }
                        trainer[current_layer].dB[current_filter] /= out_3d[0] * out_3d[1];

                        if (current_layer == back_prop_offset) {
                            break;
                        }

                        //ここから秘伝のタレつくる(今のレイヤーにとっての入力サイズのタレ)
                        tisaMat::matrix tmp_E(0, 0);
                        //誤差をstride-1でdilate
                        if (st > 1) {
                            tmp_E = dilate(E[0],st - 1);
                        }
                        else {
                            tmp_E = E[0];
                        }
                        //tmp_Eをフィルターのサイズ-1でpadding
                        if ((fil_3d[0] * fil_3d[1]) != 1) {
                            tmp_E = zero_padding(tmp_E,fil_3d[0] - 1,fil_3d[1] - 1);
                            //ハーフパディングが必要か判定
                            uint8_t pad_p = in_3d[0] - ((tmp_E.mat_RC[0] - fil_3d[0]) + 1);
                            uint8_t pad_q = in_3d[1] - ((tmp_E.mat_RC[1] - fil_3d[1]) + 1);

                            if ((pad_p + pad_q) > 0) {
                                tmp_E = zero_padding_half(tmp_E,pad_p,pad_q);
                            }
                        }
                        //フィルターを縦横に反転
                        std::vector<tisaMat::matrix> tmp_W(fil_3d[2],tisaMat::matrix(fil_3d[0],fil_3d[1]));
                        uint16_t fil_size2D = fil_3d[0] * fil_3d[1];
                        for (int i = 0; i < fil_3d[2];i++) {
                            for (int j = 0; j < fil_3d[0];j++) {
                                for (int k = 0; k < fil_3d[1];k++) {
                                    tmp_W[i].elements[j][k] = net_layer[current_layer].W->elements[current_filter][((fil_3d[2] - i - 1) * fil_size2D) + ((fil_3d[1] - j - 1) * fil_3d[1]) + (fil_3d[0] - k - 1)];
                                }
                            }
                        }
                        //強制stride=1でtmp_Eをtmp_Wで畳み込み
                        std::vector<std::vector<double>> tmp_propagate = b_p_decomv(tmp_E, tmp_W);
                        //tmp_propにtmp_propagateを記録
                        for (int d = 0; d < in_segment_3d[2];d++) {
                            tmp_prop.elements[(current_X * in_segment_3d[2]) + d] = tisaMat::vector_add(tmp_prop.elements[(current_X * in_segment_3d[2]) + d],tmp_propagate[d]);
                        }
                    }
                }
                trainer[current_layer].dW->multi_scalar(1.0 / feature_num);
                tisaMat::vector_multiscalar(trainer[current_layer].dB, 1.0 / feature_num);
                propagate_matrix = tmp_prop;
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

    int Model::number_of_layer() {
        return net_layer.size();
    }

    void Model::initialize() {
        std::random_device seed_gen;
        std::default_random_engine rand_gen(seed_gen());
        std::normal_distribution<> dist;

        for (int current_layer = back_prop_offset; current_layer < comv_count + back_prop_offset; current_layer++) {
            int W_row = net_layer[current_layer].W->mat_RC[0];
            int W_column = net_layer[current_layer].W->mat_RC[1];
            
            int prev_nodes = net_layer[current_layer].node;

            std::normal_distribution<>::param_type param(0.0, sqrt(2.0 / prev_nodes));
            dist.param(param);

            for (int R = 0; R < W_row; R++) {
                for (int C = 0; C < W_column; C++) {
                    net_layer[current_layer].W->elements[R][C] = dist(rand_gen);
                }
            }
        }

        //最初の層は入力の分配とかにしかつかわないのでかくれ層から
        for (int current_layer = back_prop_offset + comv_count; current_layer < number_of_layer();current_layer++) {
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

    std::vector<double> comv_mat_to_vect(std::vector<tisaMat::matrix>& origin){
        uint16_t shape[3] = {origin[0].mat_RC[0],
                             origin[0].mat_RC[1],
                             origin.size()};

        std::vector<double> tmp;
        for (int i = 0; i < shape[2];i++) {
            for (int j = 0; j < shape[0];j++) {
                tmp.insert(tmp.end(),origin[i].elements[j].begin(), origin[i].elements[j].end());
            }
        }

        return tmp;
    }

    std::vector<double> comv_mat_to_vect2D(tisaMat::matrix& origin) {
        uint16_t shape[2] = { origin.mat_RC[0],
                              origin.mat_RC[1]};

        std::vector<double> tmp;
        for (int j = 0; j < shape[0]; j++) {
            tmp.insert(tmp.end(), origin.elements[j].begin(), origin.elements[j].end());
        }
        
        return tmp;
    }

    std::vector<double> comvolute(std::vector<tisaMat::matrix> input, std::vector<tisaMat::matrix> filter,uint16_t *input_dim3,uint16_t *filter_dim3,uint8_t stride) {
        double tmp_sum;
        uint16_t row = input_dim3[0];
        uint16_t column = input_dim3[1];
        uint8_t filter_row = filter_dim3[0];
        uint8_t filter_column = filter_dim3[1];
        uint8_t dpt = filter_dim3[2];

        //入力よりフィルタが小さいかチェック
        bool size_check = false;
        for (int i = 0; i < 3; i++) {
            if (input_dim3[i] < filter_dim3[i]) {
                size_check = true;
            }
        }
        if (size_check) {
            printf("|!|ERROR|!| filter size is bigger than input!\n");
            printf("filter : ( %4d, %4d , %4d) , input : ( %4d, %4d , %4d)\n", filter_dim3[0], filter_dim3[1], filter_dim3[2],
                input_dim3[0], input_dim3[1], input_dim3[2]);
            exit(EXIT_FAILURE);
        }

        std::vector<tisaMat::matrix> input_use(input_dim3[2],tisaMat::matrix(row,column));
        uint8_t pad_p = (((row - filter_row) / stride + 1) * stride - (row - filter_row)) % stride;
        uint8_t pad_q = (((column - filter_column) / stride + 1) * stride - (column - filter_column)) % stride;
        if (pad_p != 0 || pad_q != 0) {
            for (int i = 0; i < input_dim3[2];i++) {
                input_use[i] = zero_padding_half(input[i],pad_p,pad_q);
            }
        }
        else {
            for (int i = 0; i < input_dim3[2]; i++) {
                input_use[i] = input[i];
            }
        }

        uint16_t route_row = (row - filter_row + pad_p) / stride + 1;
        uint16_t route_col = (column - filter_column + pad_q) / stride + 1;

        uint16_t route_dpt = input_dim3[2] / filter_dim3[2];
        uint16_t size2D = route_row * route_col;
        //double sum_max = 0.;

        std::vector<double> tmp(route_row * route_col * route_dpt);

        for (int current_map = 0; current_map < route_dpt; current_map++) {
            for (int base_row = 0; base_row < route_row; base_row++) {
                for (int base_col = 0; base_col < route_col; base_col++) {
                    tmp_sum = 0.;
                    //畳み込む(フィルターかける)
                    for (int c_f_dpt = 0; c_f_dpt < dpt; c_f_dpt++) {
                        for (int i = 0; i < filter_row; i++) {
                            for (int j = 0; j < filter_column; j++) {
                                tmp_sum += filter[c_f_dpt].elements[i][j]
                                        * input_use[current_map].elements[i + base_row * stride][j + base_col * stride];
                            }
                        }
                    }
                    tmp[(size2D * current_map) + (base_row * route_col) + base_col] = tmp_sum;
                    //if (tmp_sum > sum_max) {
                    //    sum_max = tmp_sum;
                    //}
                }
            }
        }
        //平均する
        //tisaMat::vector_multiscalar(tmp, 1. / sum_max);
        return tmp;
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

        uint8_t layer = 0;
        file.read(reinterpret_cast<char*>(&layer),sizeof(uint8_t));
        uint16_t *node = new uint16_t[layer];
        uint8_t *Activation_f = new uint8_t[layer];
        file.read(reinterpret_cast<char*>(node),layer * sizeof(uint16_t));
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

        //畳み込み層の数確認
        uint8_t comvs = 0;
        file.read(reinterpret_cast<char*>(&comvs),sizeof(uint8_t));
        std::vector<uint8_t> stride_tmp;
        std::vector<std::vector<uint16_t>> dim3_tmp;
        std::vector<std::vector<uint8_t>> filter_tmp;
        std::vector<uint8_t> fnum_tmp;

        if (comvs > 0) {
            for (int i = 0; i < comvs;i++) {
                uint8_t st;
                std::vector<uint16_t> dim3(3);
                std::vector<uint8_t> filt(3);
                uint8_t fnum;

                file.read(reinterpret_cast<char*>(&st), sizeof(uint8_t));
                file.read(reinterpret_cast<char*>(&dim3[0]), 3 * sizeof(uint16_t));
                file.read(reinterpret_cast<char*>(&filt[0]), 3 * sizeof(uint8_t));
                file.read(reinterpret_cast<char*>(&fnum), sizeof(uint8_t));

                stride_tmp.push_back(st);
                dim3_tmp.push_back(dim3);
                filter_tmp.push_back(filt);
                fnum_tmp.push_back(fnum);
            }
        }

        for (int current_layer = 0; current_layer < layer;current_layer++) {
            if (is_conv_layer(Activation_f[current_layer])) {
                int input[3] = { dim3_tmp[current_layer][0],dim3_tmp[current_layer][1],dim3_tmp[current_layer][2]};
                int filter[3] = { filter_tmp[current_layer][0], filter_tmp[current_layer][1], filter_tmp[current_layer][2] };
                Create_Comvolute_Layer(get_Af(Activation_f[current_layer]),input,filter,fnum_tmp[current_layer], stride_tmp[current_layer]);
                /*
                Create_Comvolute_Layer(dim2_tmp[current_layer][0], dim2_tmp[current_layer][1],
                                       filter_tmp[current_layer][0], filter_tmp[current_layer][1],
                                       filter_tmp[current_layer][2],fnum_tmp[current_layer],stride_tmp[current_layer]);
                */
            }else{
                Create_Layer(node[current_layer], Activation_f[current_layer]);
            }
        }

        for (int current_layer = back_prop_offset + comv_count; current_layer < layer; current_layer++) {
            int input = net_layer[current_layer - 1].Output.size();
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

        const char Expand[e_x_size] = expand_key;
        file.read(file_check, expand_key_size);
        for (int i = 0; i < 6; i++) {
            if (file_check[i] != Expand[i]) {
                printf("failed to read parameter\n");
                printf("The file maybe corrapted : %s\n", tp_file);
                delete[] node;
                delete[] Activation_f;
                exit(EXIT_FAILURE);
            }
        }

        //畳み込み層のフィルターとバイアス読み込み
        for (int i = 0; i < comvs;i++) {
            uint8_t f_R = net_layer[i].filter_dim3[0];
            uint8_t f_C = net_layer[i].filter_dim3[1];
            uint8_t f_d = net_layer[i].filter_dim3[2];
            uint8_t fnum = net_layer[i].filter_num;

            std::vector<double> tmp_fil(f_R * f_C * f_d);
            std::vector<double> tmp_B(fnum);
            for (int row = 0; row < fnum;row++) {
                file.read(reinterpret_cast<char*>(&tmp_fil[0]),f_R * f_C * f_d * sizeof(double));
                net_layer[i].W->elements[row] = tmp_fil;
            }

            file.read(reinterpret_cast<char*>(&tmp_B[0]),fnum * sizeof(double));
            net_layer[i].B = tmp_B;
        }

        delete[] node;
        delete[] Activation_f;

        printf("|loaded model|\n");
        //ロードしたモデルの概形を表示する
        for (int i = 0;i < net_layer.size();i++) {
            printf("Layer : %s (node : %5d)",Af_name[net_layer[i].Activation_f],net_layer[i].node);
            if (net_layer[i].is_conv) {
                printf(" || input : ( %4d , %4d , %4d) filter : ( %3d, %3d, %3d) * %3d\n",net_layer[i].input_dim3[0], net_layer[i].input_dim3[1], net_layer[i].input_dim3[2],
                                                                                          net_layer[i].filter_dim3[0], net_layer[i].filter_dim3[1], net_layer[i].filter_dim3[2],
                                                                                          net_layer[i].filter_num);
            }
            else {
                printf("\n");
            }
        }
    }

    void Model::save_model(const char* filename) {
        std::ofstream file(filename,std::ios::binary);
        if (!file) {
            printf("failed to open file : %s\n",filename);
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
        file.write(reinterpret_cast<char*>(&layer),sizeof(uint8_t));
        for (int current_layer = 0;current_layer < layer;current_layer++) {
            uint16_t node = net_layer[current_layer].node;
            file.write(reinterpret_cast<char*>(&node),sizeof(uint16_t));
        }
        for (int current_layer = 0; current_layer < layer; current_layer++) {
            uint8_t Af = net_layer[current_layer].Activation_f | (net_layer[current_layer].is_conv << 4);
            file.write(reinterpret_cast<char*>(&Af), sizeof(uint8_t));
        }
        const char Data_head[d_size] = data_head;
        file.write(Data_head, data_head_size);

        //畳み込み層の概形を作るためのデータを書き込む
        if (comv_count > 0) {
            file.write(reinterpret_cast<char*>(&comv_count), sizeof(uint8_t));
            for (int i = 0; i < comv_count;i++) {
                file.write(reinterpret_cast<char*>(&net_layer[i].stride), sizeof(uint8_t));
                file.write(reinterpret_cast<char*>(&net_layer[i].input_dim3), 3 * sizeof(uint16_t));
                file.write(reinterpret_cast<char*>(&net_layer[i].filter_dim3), 3 * sizeof(uint8_t));
                file.write(reinterpret_cast<char*>(&net_layer[i].filter_num), sizeof(uint8_t));
            }
        }
        else { 
            file.write(reinterpret_cast<char*>(&comv_count),sizeof(uint8_t)); 
        }
        
        //ここからモデルのパラメーターをファイルに書き込んでいく
        for (int current_layer = back_prop_offset + comv_count; current_layer < layer; current_layer++) {
            int W_row = net_layer[current_layer].W->mat_RC[0];
            int node = net_layer[current_layer].W->mat_RC[1];
            //重み行列を書き込む
            for (int r = 0; r < W_row; r++) {
                file.write(reinterpret_cast<char*>(&net_layer[current_layer].W->elements[r][0]),node * sizeof(double));
            }
            //バイアスを書き込む
            file.write(reinterpret_cast<char*>(&net_layer[current_layer].B[0]), node * sizeof(double));
        }

        const char exp_key[e_x_size] = expand_key;
        file.write(exp_key,expand_key_size);

        //畳み込み層のフィルターとバイアスのデータを書き込む
        for (int i = 0; i < comv_count;i++) {
            for (int row = 0; row < net_layer[i].filter_num;row++) {
                file.write(reinterpret_cast<char*>(&net_layer[i].W->elements[row][0]),
                                                   net_layer[i].filter_dim3[0] * net_layer[i].filter_dim3[1] * net_layer[i].filter_dim3[2] * sizeof(double));
            }

            file.write(reinterpret_cast<char*>(&net_layer[i].B[0]),net_layer[i].filter_num * sizeof(double));
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

    void Model::train(double learning_rate,Data_set& train_data, Data_set& test_data, int epoc, int batch_size, uint8_t Error_func) {
        if (net_layer[0].node != train_data.data[0].size()) {
            printf("Input size and number of input layer's nodes do not match");
            exit(EXIT_FAILURE);
        }
        if (net_layer.back().node != train_data.answer[0].size()) {
            printf("Output size and number of output layer's nodes do not match");
            exit(EXIT_FAILURE);
        }
        
        int output_num = net_layer.back().node;
        int iteration = train_data.data.size() / batch_size;

        if (iteration < 1) {
            printf("batch size is over sample size|!|\n");
            exit(EXIT_FAILURE);
        }

        tisaMat::matrix output_iterate(batch_size,output_num);
        tisaMat::matrix input_iterate(batch_size, train_data.data[0].size());
        tisaMat::matrix answer_iterate(batch_size, output_num);
        double error;
        tisaMat::matrix test_mat(test_data.data);
        std::vector<std::vector<uint8_t>> teach_iterate(batch_size);

        //バックプロパゲーションの時に重みの更新量を記憶するトレーナーをつくる
        std::vector<Trainer> trainer;
        for (int i=0; i < net_layer.size()-back_prop_offset; i++){
            Trainer tmp;
            tmp.dW = new tisaMat::matrix(net_layer[i+ back_prop_offset].W->mat_RC[0], net_layer[i + back_prop_offset].W->mat_RC[1]);

            if (net_layer[i].is_conv) {
                tmp.dB = std::vector<double>(net_layer[i + back_prop_offset].filter_num);
                for (int j = 0; j < batch_size; j++) {
                    tmp.Y_mat.push_back(tisaMat::matrix(net_layer[i + back_prop_offset].Output_mat->mat_RC[0], net_layer[i + back_prop_offset].Output_mat->mat_RC[1]));
                }
            }
            else {
                tmp.dB = std::vector<double>(net_layer[i + back_prop_offset].node);
                for (int j = 0; j < batch_size; j++) {
                    tmp.Y.push_back(std::vector<double>(net_layer[i + back_prop_offset].node));
                }
            }
            trainer.push_back(tmp);
        }
        if (comv_count > 0) {
            for (int i = 0; i < batch_size;i++) {
                trainer[comv_count - 1].Y.push_back(std::vector<double>(net_layer[comv_count - 1 + back_prop_offset].Output_mat->mat_RC[0] * net_layer[comv_count - 1 + back_prop_offset].Output_mat->mat_RC[1]));
            }
        }


        char ts[20] = { "\0" };
        time_t t = time(nullptr);
        std::tm timestr;
        localtime_r(&t ,&timestr);
        strftime(ts, 20, "%Y/%m/%d %H:%M:%S", &timestr);
        printf("<trainning started at %s>\n",ts);

        //CSV形式で誤差を記録する準備
        if (log_error) {
            std::ofstream o_file(log_filename);
            if (!o_file) {
                printf("failed to open file : %s\n", log_filename);
                exit(EXIT_FAILURE);
            }
            o_file << "epoc,Error" << '\n';

            for (int ep = 0; ep < epoc; ep++) {
                printf("| epoc : %6d / %6d|\n", ep+1, epoc);

                data_shuffle(train_data);

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
                    output_iterate = feed_forward(input_iterate, trainer);

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
                        for (int layer = back_prop_offset; layer < net_layer.size(); layer++) {
                            //重み
                            *(net_layer[layer].W) = tisaMat::matrix_subtract(*net_layer[layer].W, *trainer[layer - back_prop_offset].dW);
                            //printf("%d layer dW\n", layer);
                            //trainer[layer - 1].dW->show();
                            //バイアス
                            net_layer[layer].B = tisaMat::vector_subtract(net_layer[layer].B, trainer[layer - back_prop_offset].dB);
                            //printf("%d layer dB\n", layer);
                            //tisaMat::vector_show(trainer[layer - 1].dB);
                        }
                    }

                    show_train_progress(iteration, i);
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
                localtime_r(&t, &timestr);
                strftime(ts, 20, "%Y/%m/%d %H:%M:%S", &timestr);
                error = (*Ef[Error_func])(test_data.answer, output_iterate.elements);
                printf("Error : %lf <timestamp : %s>\n", error, ts);

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
                printf("| epoc : %6d / %6d|\n", ep+1,epoc);

                data_shuffle(train_data);

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
                            //重み
                            *(net_layer[layer].W) = tisaMat::matrix_subtract(*net_layer[layer].W, *trainer[layer - back_prop_offset].dW);
                            //printf("%d layer dW\n", layer);
                            //trainer[layer - 1].dW->show();
                            //バイアス
                            net_layer[layer].B = tisaMat::vector_subtract(net_layer[layer].B, trainer[layer - back_prop_offset].dB);
                            //printf("%d layer dB\n", layer);
                            //tisaMat::vector_show(trainer[layer - 1].dB);
                        }
                    }

                    show_train_progress(iteration,i);
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
                localtime_r(&t, &timestr);
                strftime(ts,20,"%Y/%m/%d %H:%M:%S",&timestr);
                error = (*Ef[Error_func])(test_data.answer, output_iterate.elements);
                printf("Error : %lf <timestamp : %s>\n", error,ts);

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

    std::vector<std::vector<double>> tisaNET::Model::b_p_decomv(tisaMat::matrix input, std::vector<tisaMat::matrix> filter) {
        double tmp_sum;
        uint16_t row = input.mat_RC[0];
        uint16_t column = input.mat_RC[1];
        uint8_t filter_row = filter[0].mat_RC[0];
        uint8_t filter_column = filter[0].mat_RC[1];
        uint8_t dpt = filter.size();
        uint16_t route_row = (row - filter_row) + 1;
        uint16_t route_col = (column - filter_column) + 1;
        uint16_t size2D = route_row * route_col;
        //double sum_max = 0.;

        std::vector<std::vector<double>> tmp(dpt,std::vector<double>(route_row * route_col));

        for (int current_map = 0; current_map < dpt; current_map++) {
            for (int base_row = 0; base_row < route_row; base_row++) {
                for (int base_col = 0; base_col < route_col; base_col++) {
                    tmp_sum = 0.;
                    //畳み込む(フィルターかける)    誤差逆伝播ではチャンネル別、合計しない
                    //for (int c_f_dpt = 0; c_f_dpt < dpt; c_f_dpt++) {
                        for (int i = 0; i < filter_row; i++) {
                            for (int j = 0; j < filter_column; j++) {
                                tmp_sum += filter[current_map].elements[i][j]
                                    * input.elements[i + base_row][j + base_col];
                            }
                        }
                    //}
                    tmp[current_map][(base_row * route_col) + base_col] = tmp_sum;
                    //if (tmp_sum > sum_max) {
                    //    sum_max = tmp_sum;
                    //}
                }
            }
        }
        //平均する
        //tisaMat::vector_multiscalar(tmp, 1. / sum_max);
        return tmp;
    }

    void show_train_progress(int total_iteration, int now_iteration) {
        printf("\r|");
        double progress = float(now_iteration+1) / float(total_iteration);
        int bar_num = (progress+0.01) * progress_bar_length;
        for (int i = 0;i < bar_num - 1;i++) {
            printf("=");
        }
        printf(">");
        for (int i = 0; i < progress_bar_length - bar_num;i++) {
            printf("-");
        }
        printf("| %5.2lf%% ", progress*100.0);
    }
}
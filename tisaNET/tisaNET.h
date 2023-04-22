#pragma once
/*～簡単な使い方～
0.Data_setは訓練データ、評価データで分けて作ってください
  sample_dataの第2インデックスの大きさがニューラルネットへの入力の数に対応します

1.Modelクラスをインスタンス化する(ニューラルネットのモデルを表します)
2.Model.Create_Layerすると、最後に作った層の後ろに新しく層をつなげます
	最初は必ず第二引数(Activation)をINPUTにしてください(入力層として使います)
3.一番最後に作った層が出力層になります
*/
#include <tisaMat.h>
#include <cstdint>
#include <vector>
#include <random>
#include <algorithm>
#include <array>

#define SIGMOID 0
#define RELU 1
#define STEP 2
#define SOFTMAX 3
#define INPUT 4
#define SIMPLE_CONVOLUTE 5
#define NORMALIZE 6

#define MEAN_SQUARED_ERROR 0
#define CROSS_ENTROPY_ERROR 1

#define CONVOLUTE 0x10
#define POOLING 0x20
#define MAX_POOL 0
#define AVE_POOL 1
#define SUM_POOL 2


#define format_key {'t','i','s','a','N','E','T'}
#define f_k_size 7

#define data_head {'D','A','T','A'}
#define d_size 4

#define expand_key {'E','X','P','A','N','D'}
#define e_x_size 6

#define format_key_size sizeof(char) * 7
#define data_head_size sizeof(char) * 4
#define expand_key_size sizeof(char) * 6



#ifdef _MSC_VER
struct tm* localtime_r(const time_t* time, struct tm* resultp);
#endif

enum stop_mode_name {
	Just_Now,
	Current_Epoc,
};

namespace tisaNET {

	static const char* Af_name[7] = { "SIGMOID","RELU   ","STEP   ","SOFTMAX","INPUT  ","SIMPLE_CONVOLUTE","NORMALINE" };
	static const char* pool_mode[3] = {"MAX_POOL","AVERAGE_POOL","SUM_POOL"};

	double step(double X);
	double sigmoid(double X);
	double ReLU(double X);
	double softmax(double X);

	//bool is_conv_layer(uint8_t i);

	uint8_t get_Af(uint8_t i);

	double variance(std::vector<tisaMat::matrix> mat);

	struct Data_set {
		std::vector<std::vector<double>> data;
		std::vector<std::vector<uint8_t>> answer;
	};

	void data_shuffle(Data_set& sdata);

	struct Trainer {
		tisaMat::matrix* dW = nullptr;
		std::vector<double> dB;
		std::vector<std::vector<double>> Y;
		std::vector<std::vector<tisaMat::matrix>> Y_mat;
		std::vector < std::vector<std::vector<std::vector<std::array<int,3>>>>> pool_index;
	};

	class layer {
	public:
		uint8_t Activation_f = 0;
		uint16_t node = 0;
		tisaMat::matrix *W = nullptr;
		std::vector<double> B;
		std::vector<double> Output;

		//畳み込み層のためのデータ
		//Wで代用のため廃止
		//std::vector<tisaMat::matrix> filter;
		
		//is_conv_layer関数実装に伴い廃止
		//bool is_conv = false;
		uint8_t stride = 1;
		uint16_t input_dim3[3];
		//filter_dimをpoolingにも流用する
		uint8_t filter_dim3[3];
		uint16_t output_dim3[3];
		uint16_t filter_output_dim3[3];
		uint8_t pad[2] = {0,0};
		bool padding_flag = false;
		uint8_t filter_num = 1;

		std::vector<tisaMat::matrix> Output_mat;
		double (*Af[4])(double) = { sigmoid,ReLU,step,softmax };
		void convolute(std::vector<double>& input);
		void convolute(std::vector<tisaMat::matrix>& input);
		void max_pooling(std::vector<double>& input);
		void max_pooling(std::vector<tisaMat::matrix>& input);
		void max_pooling(std::vector<double>& input,Trainer& trainer,int batch_index);
		void max_pooling(std::vector<tisaMat::matrix>& input, Trainer& trainer, int batch_index);
		void convolute_test(tisaMat::matrix& input);
		void output_vec_to_mat();
		void output_mat_to_vec();
		bool is_conv_layer();
		bool is_pool_layer();
		bool pool_mode();

		void W_normalization();
	};

	template <typename T>
	std::vector<tisaMat::matrix> conv_vect_to_mat(std::vector<double> input, T* shape) {
		T row = shape[0];
		T col = shape[1];
		T dpt = shape[2];
		T size2D = row * col;

		std::vector<tisaMat::matrix> tmp(dpt,tisaMat::matrix(row,col));

		for (int i = 0; i < dpt;i++) {
			for (int j = 0; j < row;j++) {
				for (int k = 0; k < col;k++) {
					tmp[i].elements[j][k] = input[(i * size2D) + (j * col) + k];
				}
			}
		}
		return tmp;
	}
	
	template <typename T>
	tisaMat::matrix conv_vect_to_mat2D(std::vector<double> input, T* shape) {
		T row = shape[0];
		T col = shape[1];
		T size2D = row * col;

		tisaMat::matrix tmp(row, col);

		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				tmp.elements[i][j] = input[(i * col) + j];
			}
		}
		return tmp;
	}

	template <typename T>
	std::vector<tisaMat::matrix> conv_vect_to_mat3D(std::vector<double> input, T* shape) {
		T row = shape[0];
		T col = shape[1];
		T dpt = shape[2];
		T size2D = row * col;

		std::vector<tisaMat::matrix> tmp(dpt, tisaMat::matrix(row,col));

		for (int k = 0; k < dpt;k++) {
			for (int i = 0; i < row; i++) {
				for (int j = 0; j < col; j++) {
					tmp[k].elements[i][j] = input[(k * size2D) + (i * col) + j];
				}
			}
		}
		
		return tmp;
	}

	std::vector<double> conv_mat_to_vect(std::vector<tisaMat::matrix>& origin);

	std::vector<double> conv_mat_to_vect2D(tisaMat::matrix& origin);

	std::vector<double> convolute(std::vector<tisaMat::matrix> input, std::vector<tisaMat::matrix> filter, uint16_t* input_dim3, uint16_t* filter_dim3,uint8_t stride);
	std::vector<double> max_pooling(std::vector<tisaMat::matrix> input, uint16_t* input_dim3, uint16_t* filter_dim3);


	//MNISTからデータを作る(csv形式のMNISTデータセットから <https://github.com/pjreddie/mnist-csv-png>)
	void load_MNIST(const char* path,Data_set& train_data,Data_set& test_data, int sample_size,int test_size, bool single_output);
	void load_MNIST(const char* path,Data_set& test_data,int test_size, bool single_output);
	
	//void load_MNIST_csv(const char* path, Data_set& train_data, Data_set& test_data, int sample_size, int test_size, bool single_output);
	//void load_MNIST_csv(const char* path, Data_set& test_data, int test_size, bool single_output);

	//256色BMPファイルから一次配列を作る
	std::vector<uint8_t> vec_from_256bmp(const char *bmp_file);

	//数値をバイナリで表示
	bool print01(int bit, long Value);
	
	//平均二乗誤差関数
	double mean_squared_error(std::vector<std::vector<uint8_t>>& teacher, std::vector<std::vector<double>>& output);

	//交差エントロピー関数
	double cross_entropy_error(std::vector<std::vector<uint8_t>>& teacher, std::vector<std::vector<double>>& output);

	tisaMat::matrix dilate(tisaMat::matrix &mat,uint8_t d);
	
	tisaMat::matrix zero_padding(tisaMat::matrix &mat,uint8_t p,uint8_t q);

	tisaMat::matrix zero_padding_half(tisaMat::matrix& mat, char p, char q);

	class Model {
	public:

		std::vector<layer> net_layer;

		//ネットワークの一番うしろに層をつけ足す(initは重みを初期化するときの値、省略すると乱数)
		void Create_Layer(int Outputs, uint8_t Activation);
		void Create_Layer(int nodes, uint8_t Activation, double init);

		//畳み込み層をつくる
		//フィルター手動設定(保留)
		/*
		void Create_Comvolute_Layer(int row,int column, std::vector<std::vector<double>>& filter);
		void Create_Comvolute_Layer(int row, int column, std::vector<std::vector<double>>& filter, int stride);
		*/
		//フィルター指定なし
		//void Create_Comvolute_Layer(int input_shape[3], int filter_shape[3], int filter_num);
		void Create_Convolute_Layer(uint8_t Activation,int input_shape[3], int filter_shape[3], int filter_num,int stride);
		//input_shape[3]の代わりに前の層のoutput_dim3を参照
		void Create_Convolute_Layer(uint8_t Activation,int filter_shape[3], int filter_num, int stride);

		//プーリング層をつくる
		void Create_Pooling_Layer(uint8_t Activation, int input_shape[3], int filter_shape[3]);
		void Create_Pooling_Layer(uint8_t Activation, int filter_shape[3]);


		//入力層(最初の層のこと)にネットワークへの入力をいれる
		virtual void input_data(std::vector<double>& data) {
			int input_num = data.size();
			if (net_layer.front().node != input_num) {
				printf("input error|!|\n");
				exit(EXIT_FAILURE);
			}
			else {
				std::vector<double> input = data;
				if (net_layer.front().is_conv_layer()) {
					if (net_layer.front().is_conv_layer()) net_layer.front().convolute(input);
					else {
						switch (net_layer.front().Activation_f ^ POOLING) {
						case MAX_POOL:
							net_layer.front().max_pooling(input);
							break;
						}
					}
					/*デバッグ用
					std::vector<std::vector<double>> testinputV = { {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25} };
					tisaMat::matrix testinput(testinputV);
					net_layer.front().comvolute_test(testinput);
					*/

					//vectorからmatrixへ
					net_layer.front().output_vec_to_mat();
					int i = 1;
					for (; net_layer[i].is_conv_layer() || net_layer[i].is_pool_layer(); i++) {
						if (net_layer[i].is_conv_layer()) net_layer[i].convolute(net_layer[i - 1].Output_mat);
						else {
							switch (net_layer[i].Activation_f ^ POOLING) {
							case MAX_POOL:
								net_layer[i].max_pooling(net_layer[i - 1].Output_mat);
								break;
							}
						}
					}
					//畳み込み最終段でvectorになおす
					net_layer[i - 1 + back_prop_offset].output_mat_to_vec();
				}
				else if (net_layer.front().Activation_f == INPUT) {
					net_layer.front().Output = input;
				}
			}
		}

		//訓練用入力
		void input_data(std::vector<double>& data, std::vector<Trainer> &trainer,int index) {
			int input_num = data.size();
			if (net_layer.front().node != input_num) {
				printf("input error|!|\n");
				exit(EXIT_FAILURE);
			}
			else {
				std::vector<double> input = data;

				if (net_layer.front().is_conv_layer() || net_layer.front().is_pool_layer()) {
					if (net_layer.front().is_conv_layer()) net_layer.front().convolute(input);
					else {
						switch (net_layer.front().Activation_f ^ POOLING) {
							case MAX_POOL:
								net_layer.front().max_pooling(input,trainer.front(),index);
								break;
						}
					}
					/*デバッグ用
					std::vector<std::vector<double>> testinputV = { {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25} };
					tisaMat::matrix testinput(testinputV);
					net_layer.front().comvolute_test(testinput);
					*/

					//vectorからmatrixへ
					net_layer.front().output_vec_to_mat();
					//入力は一つなので、output_matの深さはfilter_num
					trainer.front().Y_mat[index] = net_layer.front().Output_mat;

					//分散の表示
					double var = net_layer.front().Output_mat.front().variance();
					printf("\n| Layer %d | variance : %lf", 0, var);

					int i = 1;
					for (; net_layer[i].is_conv_layer() || net_layer[i].is_pool_layer(); i++) {
						if (net_layer[i].is_conv_layer()) net_layer[i].convolute(net_layer[i - 1].Output_mat);
						else {
							switch (net_layer[i].Activation_f ^ POOLING) {
							case MAX_POOL:
								net_layer[i].max_pooling(net_layer[i - 1].Output_mat, trainer[i],index);
								break;
							}
						}
						trainer[i].Y_mat[index] = net_layer[i].Output_mat;
						
						//分散の表示
						double var = variance(net_layer[i].Output_mat);
						printf("\n| Layer %d | variance : %lf", i, var);
					}

					//畳み込み最終段でvectorになおす
					net_layer[i - 1 + back_prop_offset].output_mat_to_vec();
					trainer[i - 1].Y[index] = conv_mat_to_vect(trainer[i - 1].Y_mat[index]);

				}
				else if (net_layer.front().Activation_f == INPUT) {
					net_layer.front().Output = input;
				}
			}
		}

		//順伝播する
		template <typename T>
		//単発
		std::vector<double> feed_forward(std::vector<T>& Input_data) {
			std::vector<double> output_vecter(net_layer.back().Output.size());
			std::vector<double> input = tisaMat::vector_cast<double>(Input_data);
			input_data(input);
			for (int i = back_prop_offset + conv_count; i < number_of_layer(); i++) {
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
				}

				if (net_layer[i].Activation_f == SOFTMAX) {
					double sigma = 0.0;
					for (int node = 0; node < net_layer[i].Output.size(); node++) {
						sigma += net_layer[i].Output[node];
					}
					tisaMat::vector_multiscalar(net_layer[i].Output, 1.0 / sigma);
				}
			}
			output_vecter = net_layer.back().Output;
			if(net_view) emit_output(output_vecter);
			return output_vecter;
		}

		tisaMat::matrix feed_forward(tisaMat::matrix& Input_data);
		tisaMat::matrix feed_forward(tisaMat::matrix& Input_data, std::vector<Trainer>& trainer);

		//Qt併用時のための関数
		virtual void emit_output(std::vector<double>) {  };

		//逆誤差伝播する
		void B_propagate(std::vector<std::vector<uint8_t>>& teacher, tisaMat::matrix& output, uint8_t error_func, std::vector<Trainer>& trainer,double lr, tisaMat::matrix& input_batch);

		//ネットワークの層の数を取り出す
		int number_of_layer();

		//モデルの重みとかを初期化する
		void initialize();

		//モデルを訓練する
		void train(double learning_rate, Data_set& train_data, Data_set& test_data, int epoc, int batch_size, uint8_t Error_func);

		//モデルのファイル(.tp)を読み込む
		void load_model(const char* tp_file);

		//モデルをファイルに出力する
		virtual void save_model(std::string tp_file);

		//正答率を表示/非表示にする
		void monitor_accuracy(bool monitor_accuracy);

		//訓練時の誤差をイテレーションごとに記録する(csv形式で)
		void logging_error(const char* log_file);


	protected:
		bool monitoring_accuracy = false;
		bool log_error = false;
		uint8_t back_prop_offset = 0;
		uint8_t conv_count = 0;
		std::string log_filename;
		bool rasterized = false;
		double (*Ef[2])(std::vector<std::vector<uint8_t>>&, std::vector<std::vector<double>>&) = { mean_squared_error,cross_entropy_error };
		double (*Af[4])(double) = { sigmoid,ReLU,step,softmax };
		void m_a(std::vector<std::vector<double>>& output, std::vector<std::vector<uint8_t>>& answer, uint8_t error_func);
		std::vector<std::vector<double>> b_p_deconv(tisaMat::matrix input, std::vector<tisaMat::matrix> filter);
		std::vector<std::vector<double>> de_max_pool(tisaMat::matrix E, std::vector<std::vector<std::array<int, 3>>> trainer, uint16_t* input_shape, uint8_t* filt_shape);
		virtual void show_net_process(std::vector<double>& data) {};
		virtual void pause_trainning() {};
		bool pause_flag = false;
		int stop_mode;
		bool stop_flag = false;
		bool net_view = false;
	};

	void show_train_progress(int total_iteration,int now_iteration);
	void clear_under_cl(int lines);
}
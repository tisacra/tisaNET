#pragma once
/*〜簡単な使い方〜
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

#define SIGMOID 0
#define RELU 1
#define STEP 2
#define SOFTMAX 3
#define INPUT 4
#define COMVOLUTE 5
#define NORMALIZE 6

#define MEAN_SQUARED_ERROR 0
#define CROSS_ENTROPY_ERROR 1

namespace tisaNET {

	static const char* Af_name[7] = { "SIGMOID","RELU","STEP","SOFTMAX","INPUT","COMVOLUTE","NORMALINE" };

	struct Data_set {
		std::vector<std::vector<double>> data;
		std::vector<std::vector<uint8_t>> answer;
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
		uint8_t stride = 1;
		uint16_t input_dim3[3];
		uint8_t filter_dim3[3];
		uint8_t filter_num = 1;
		tisaMat::matrix* Output_mat = nullptr;
		void comvolute(std::vector<double>& input);
		void comvolute(tisaMat::matrix& input);
		void comvolute_test(tisaMat::matrix& input);
		void output_vec_to_mat();
		void output_mat_to_vec();
	};

	struct Trainer {
		tisaMat::matrix* dW = nullptr;
		std::vector<double> dB;
		std::vector<std::vector<double>> Y;
	};

	//MNISTからデータを作る
	void load_MNIST(const char* path,Data_set& train_data,Data_set& test_data, int sample_size,int test_size, bool single_output);
	void load_MNIST(const char* path,Data_set& test_data,int test_size, bool single_output);

	//256色BMPファイルから一次配列を作る
	std::vector<uint8_t> vec_from_256bmp(const char *bmp_file);

	double step(double X);
	double sigmoid(double X);
	double ReLU(double X);
	double softmax(double X);

	//数値をバイナリで表示
	bool print01(int bit, long Value);
	
	//平均二乗誤差関数
	double mean_squared_error(std::vector<std::vector<uint8_t>>& teacher, std::vector<std::vector<double>>& output);

	//交差エントロピー関数
	double cross_entropy_error(std::vector<std::vector<uint8_t>>& teacher, std::vector<std::vector<double>>& output);

	class Model {
	public:

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
		void Create_Comvolute_Layer(int input_shape[3], int filter_shape[3], int filter_num,int stride);


		//入力層(最初の層のこと)にネットワークへの入力をいれる
		template <typename T>
		void input_data(std::vector<T>& data) {
			int input_num = data.size();
			if (net_layer.front().node != input_num) {
				printf("input error|!|\n");
				exit(EXIT_FAILURE);
			}
			else {
				std::vector<double> input = tisaMat::vector_cast<double>(data);
				if (net_layer.front().Activation_f == COMVOLUTE) {
					net_layer.front().comvolute(input);

					/*デバッグ用
					std::vector<std::vector<double>> testinputV = { {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25} };
					tisaMat::matrix testinput(testinputV);
					net_layer.front().comvolute_test(testinput);
					*/

					//vectorからmatrixへ
					net_layer.front().output_vec_to_mat();
					int i = 1;
					for (;net_layer[i].Activation_f == COMVOLUTE;i++) {
						net_layer[i].comvolute(*(net_layer[i-1].Output_mat));
					}
					//畳み込み最終段でvectorになおす
					net_layer[i-1].output_mat_to_vec();
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
			input_data(Input_data);
			for (int i = back_prop_offset; i < number_of_layer(); i++) {
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
			return output_vecter;
		}

		tisaMat::matrix feed_forward(tisaMat::matrix& Input_data);
		tisaMat::matrix feed_forward(tisaMat::matrix& Input_data, std::vector<Trainer>& trainer);

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
		void save_model(const char* tp_file);

		//正答率を表示/非表示にする
		void monitor_accuracy(bool monitor_accuracy);

		//訓練時の誤差をイテレーションごとに記録する(csv形式で)
		void logging_error(const char* log_file);

	private:
		bool monitoring_accuracy = false;
		bool log_error = false;
		uint8_t back_prop_offset = 0;
		uint8_t comv_count = 0;
		std::string log_filename;
		std::vector<layer> net_layer;
		double (*Ef[2])(std::vector<std::vector<uint8_t>>&, std::vector<std::vector<double>>&) = { mean_squared_error,cross_entropy_error };
		double (*Af[4])(double) = { sigmoid,ReLU,step,softmax };
		void m_a(std::vector<std::vector<double>>& output, std::vector<std::vector<uint8_t>>& answer, uint8_t error_func);
		void B_propagate2(std::vector<std::vector<uint8_t>>& teacher, tisaMat::matrix& output, uint8_t error_func, std::vector<Trainer>& trainer, double lr, tisaMat::matrix& input_batch);

	};

	void show_train_progress(int total_iteration,int now_iteration);
}
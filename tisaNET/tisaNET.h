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

#define SIGMOID 0
#define RELU 1
#define STEP 2
#define INPUT 3

#define MEAN_SQUARED_ERROR 0
#define CROSS_ENTROPY 1

namespace tisaNET {

	struct Data_set {
		std::vector<std::vector<double>> sample_data;
		std::vector<std::vector<double>> answer;
	};

	struct layer {
		uint8_t Activation_f;
		uint8_t node;
		tisaMat::matrix* W;
		std::vector<double> B;
		std::vector<double> Output;
	};

	struct Trainer {
		tisaMat::matrix* dW;
		std::vector<double> dB;
		std::vector<std::vector<double>> Y;
	};

	double step(double X);
	double sigmoid(double X);
	double ReLU(double X);

	//数値をバイナリで表示
	bool print01(int bit, long Value);
	
	//平均二乗誤差関数
	std::vector<double> mean_squared_error(std::vector<std::vector<double>>&, std::vector<std::vector<double>>& output);

	//交差エントロピー関数
	std::vector<double> cross_entropy(std::vector<std::vector<double>>&, std::vector<std::vector<double>>& output);

	class Model {
	public:

		//ネットワークの一番うしろに層をつけ足す(initは重みを初期化するときの値、省略すると乱数)
		void Create_Layer(int Outputs, uint8_t Activation);
		void Create_Layer(int nodes, uint8_t Activation, double init);

		//入力層(最初の層のこと)にネットワークへの入力をいれる
		void input_data(std::vector<double>& data);

		//順伝播する
		tisaMat::matrix F_propagate(tisaMat::matrix& Input_data);
		tisaMat::matrix F_propagate(tisaMat::matrix& Input_data, std::vector<Trainer>& trainer);

		//逆誤差伝播する
		void B_propagate(std::vector<std::vector<double>>& teacher, tisaMat::matrix& output, uint8_t error_func, std::vector<Trainer>& trainer,double lr, tisaMat::matrix& input_batch);

		//ネットワークの層の数を取り出す
		int number_of_layer();

		//モデルを訓練する
		void train(double learning_rate, Data_set& train_data, Data_set& test_data, int epoc, int iteration, uint8_t Error_func);

	private:
		std::vector<layer> net_layer;
		std::vector<double> (*Ef[2])(std::vector<std::vector<double>>&, std::vector<std::vector<double>>&) = { mean_squared_error,cross_entropy };
		double (*Af[3])(double) = { sigmoid,ReLU,step };
		std::random_device rnd;
	};
}
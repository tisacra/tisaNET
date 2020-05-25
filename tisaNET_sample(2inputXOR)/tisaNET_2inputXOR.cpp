#include <tisaNET.h>

int main() {
	//訓練用データ(シャッフルしても、増やしてもデータと答えの並びが対応していればOK)
	//二次元配列の形になることに注意! 中身の配列の要素数がネットワークへの入力になります(ここでは2入力)
	tisaNET::Data_set train_data;
	train_data.data = { {0,0},
						{0,1},
						{1,0},
						{1,1} };
	//複数出力する場合に対応させるため、答えも二次元配列にします
	train_data.answer = { {0},
						  {1},
						  {1},
						  {0} };

	//評価用データ
	tisaNET::Data_set test_data;
	test_data.data = { {0,0},{0,1},{1,0},{1,1} };
	test_data.answer = { {0},{1},{1},{0} };

	//ここからネットワークを「組み立てて」いきます
	tisaNET::Model model;
	//最初の層は入力を分配させるものとして使います
	//入力するデータのサイズと最初の層のノード数があっていないとエラーになります
	//(ノードの数, 活性化関数の種類)
	model.Create_Layer(2, INPUT);		//始めの層は活性化関数の代わりにINPUTを指定する必要があります
	model.Create_Layer(2, SIGMOID);		//ノード2つ、活性化関数はシグモイド関数である層
	model.Create_Layer(1, STEP);		//最後の層は出力層になります　今回は１出力で活性化関数はステップ関数
										//ReLU関数,Softmax関数も使えます(使うときは全て大文字)

	//重みなどをXaivierの初期値、Heの初期値にしたがって初期化します
	model.initialize();
	//訓練時にそのエポックごとの正確さを表示させます
	model.monitor_accuracy(true);

	//いよいよ学習します
	//(学習率, 訓練用データ, 評価用データ, エポック(全体を何セット学習するか), バッチサイズ(全体を何個ずつに分けて学習するか), 目的(誤差)関数)
	//目的関数には、平均二乗誤差(MEAN_SQUARED_ERROR)とクロスエントロピー誤差(CROSS_ENTROPY_ERROR)を用意しています
	model.train(0.5,train_data,test_data,1000,4,MEAN_SQUARED_ERROR);//訓練データが4つで1エポックに4回伝播なので、この時はインライン学習です

	//モデルのパラメーターをファイルに保存できます(拡張子は自由です)
	model.save_model("2inXOR_model.tp");
	return 0;
}
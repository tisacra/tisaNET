#pragma once

#include "tisanet_qt_global.h"
#include <tisaMat.h>
#include <tisaNET.h>
#include <FilterViewer.h>
#include <QMainWindow>
#include <QObject>
#include <QPushButton>
#include <QFrame>
#include <QMenu>
#include <QThread>

#define loop_msec 100

namespace tisaNET_Qt {
	class Model: public QObject, public tisaNET::Model
	{
		Q_OBJECT

		public:
		
			Model(double learning_rate, tisaNET::Data_set& train_data, tisaNET::Data_set& test_data, int epoc, int batch_size, uint8_t Error_func,QObject* parent=0)
				:QObject(parent)
			{
				this->learning_rate = learning_rate;
				this->train_data = &train_data;
				this->test_data = &test_data;
				this->epoc = epoc;
				this->batch_size = batch_size;
				this->Error_func = Error_func;
			};
			~Model() {};

			//ネットワークの一番うしろに層をつけ足す(initは重みを初期化するときの値、省略すると乱数)
			//void Create_Layer(int Outputs, uint8_t Activation);
			//void Create_Layer(int nodes, uint8_t Activation, double init);

			//畳み込み層をつくる
			//フィルター手動設定(保留)
			/*
			void Create_Comvolute_Layer(int row,int column, std::vector<std::vector<double>>& filter);
			void Create_Comvolute_Layer(int row, int column, std::vector<std::vector<double>>& filter, int stride);
			*/
			//フィルター指定なし
			//void Create_Comvolute_Layer(int input_shape[3], int filter_shape[3], int filter_num);
			//void Create_Convolute_Layer(uint8_t Activation, int input_shape[3], int filter_shape[3], int filter_num, int stride);
			//input_shape[3]の代わりに前の層のoutput_dim3を参照
			//void Create_Convolute_Layer(uint8_t Activation, int filter_shape[3], int filter_num, int stride);

			//プーリング層をつくる
			//void Create_Pooling_Layer(uint8_t Activation, int input_shape[3], int filter_shape[3]);
			//void Create_Pooling_Layer(uint8_t Activation, int filter_shape[3]);

			using tisaNET::Model::train;
			Q_INVOKABLE void train();



			//モデルをファイルに出力する
			void save_model(const char* tp_file) override;
			void save_model();

			//入力層(最初の層のこと)にネットワークへの入力をいれる
			void input_data(std::vector<double>& data) override {
				int input_num = data.size();
				if (net_layer.front().node != input_num) {
					printf("input error|!|\n");
					exit(EXIT_FAILURE);
				}
				else {
					std::vector<double> input = tisaMat::vector_cast<double>(data);
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
			void input_data(std::vector<double>& data, std::vector<tisaNET::Trainer>& trainer, int index) override {
				int input_num = data.size();
				if (net_layer.front().node != input_num) {
					printf("input error|!|\n");
					exit(EXIT_FAILURE);
				}
				else {
					std::vector<double> input = tisaMat::vector_cast<double>(data);

					if (net_view) {
						//net_viewer->filter = tisaNET::conv_vect_to_mat3D(data,net_layer.front().input_dim3);
						emit filter_changed(0,0,tisaNET::conv_vect_to_mat3D(data, net_layer.front().input_dim3));
						//net_viewer->repaint();
					}

					if (net_layer.front().is_conv_layer() || net_layer.front().is_pool_layer()) {
						if (net_layer.front().is_conv_layer()) net_layer.front().convolute(input);
						else {
							switch (net_layer.front().Activation_f ^ POOLING) {
							case MAX_POOL:
								net_layer.front().max_pooling(input, trainer.front(), index);
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
									net_layer[i].max_pooling(net_layer[i - 1].Output_mat, trainer[i], index);
									break;
								}
							}
							trainer[i].Y_mat[index] = net_layer[i].Output_mat;

							//分散の表示
							double var = tisaNET::variance(net_layer[i].Output_mat);
							printf("\n| Layer %d | variance : %lf", i, var);
						}

						//畳み込み最終段でvectorになおす
						net_layer[i - 1 + back_prop_offset].output_mat_to_vec();
						trainer[i - 1].Y[index] = tisaNET::conv_mat_to_vect(trainer[i - 1].Y_mat[index]);

					}
					else if (net_layer.front().Activation_f == INPUT) {
						net_layer.front().Output = input;
					}
				}
			}

			using tisaNET::Model::feed_forward;

			//順伝播する
			template <typename T>
			//単発
			std::vector<double> feed_forward(std::vector<T>& Input_data) {
				std::vector<double> output_vecter(net_layer.back().Output.size());
				input_data(Input_data);
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
				emit output_updated(output_vector);
				return output_vecter;
			}

			void emit_output(std::vector<double> data) override {
				emit output_updated(data);
			}

			//Qtの機能を使ってモデルの中身を表示する
			void monitor_network(bool monitor_network);
	
		signals:
			void filter_changed(const int ,const int ,const std::vector<tisaMat::matrix>);
			void output_updated(const std::vector<double>);

		public slots:
			void pause(bool is_runnning) {
				pause_flag = !is_runnning;
			}

			void stop(int mode) {
				stop_flag = true;
				stop_mode = mode;
			}

		private:
			bool net_view = false;
			//FilterViewer* net_viewer;

			double learning_rate;
			tisaNET::Data_set* train_data;
			tisaNET::Data_set* test_data;
			int epoc;
			int batch_size;
			uint8_t Error_func;
			bool save_model_flag = false;
			std::string save_model_file;
			bool pause_flag = false;
			int stop_mode;
			bool stop_flag = false;
	};

	class NetViewer : public QWidget {
		Q_OBJECT

		public:
			NetViewer(Model* ref_model,std::vector<QString>* tag = nullptr, QWidget* parent = nullptr) :QWidget(parent){
				this->ref_model = ref_model;

				if (tag != nullptr)tags = *tag;

				net_layout = new QHBoxLayout(this);
				layers.push_back(new LayerViewer("Input"));
				layers.front()->addfilter(ref_model->net_layer.front().input_dim3);
				net_layout->addWidget(layers.back());
				
				emitter_array.push_back(new emitter());
				connect(emitter_array.back(),SIGNAL(filter_update_call(const int, const std::vector<tisaMat::matrix>)),
						layers.back(),SLOT(filter_update_call(int , std::vector<tisaMat::matrix>)));

				for (int i = 0; i < ref_model->net_layer.size();i++) {
					layers.push_back(new LayerViewer("Layer " + QString::number(i)));
					for (int j = 0; j < ref_model->net_layer[i].filter_num;j++) {
						layers.back()->addfilter(ref_model->net_layer[i].filter_dim3);
					}
					net_layout->addWidget(layers.back());

					emitter_array.push_back(new emitter());
					connect(emitter_array.back(), SIGNAL(filter_update_call(const int, const std::vector<tisaMat::matrix>)),
							layers.back(), SLOT(filter_update_call(int, std::vector<tisaMat::matrix>)));
				}

				if (tags.size() == ref_model->net_layer.back().Output.size()) {
					layers.push_back(new LayerViewer("Output"));
					layers.back()->addanswer(tags);
					net_layout->addWidget(layers.back());

					emitter_array.push_back(new emitter());
					connect(ref_model, SIGNAL(output_updated(const std::vector<double>)),
						layers.back(), SLOT(output_update(std::vector<double>)));
				}
					
				connect(ref_model, SIGNAL(filter_changed(const int, const int, const std::vector<tisaMat::matrix>)),
						this, SLOT(filter_update(int, int, std::vector<tisaMat::matrix>)));

				this->setLayout(net_layout);
				show();
			}
			~NetViewer() {}

		public slots:
			void filter_update(int layer,int filter,std::vector<tisaMat::matrix> data) {
				emit emitter_array[layer]->filter_update_call(filter,data);
			}

		private:
			QHBoxLayout* net_layout;
			Model* ref_model;
			std::vector<LayerViewer*> layers;
			std::vector<emitter*> emitter_array;
			std::vector<QString> tags;
	};

	enum stop_mode_name {
		Just_Now,
		Current_Epoc,
	};

	const QString stop_mode_names[] = {"Just Now",
									   "Current Epoc"};

	class ControlPanel : public QMainWindow {
		Q_OBJECT

		public:
			ControlPanel(Model* ref_model, bool show_model, std::vector<QString>* tag = nullptr, QMainWindow* parent = nullptr) :QMainWindow(parent) {
				QWidget* cent_widget = new QWidget(this);
				QGridLayout* main_layout = new QGridLayout();

				QFrame* buttons_frame = new QFrame(this);
				buttons_frame->setFrameShape(QFrame::Box);

				QGridLayout* buttons = new QGridLayout(buttons_frame);

				pause_button = new QPushButton("pause",buttons_frame);
				pause_button->setCheckable(true);
				connect(pause_button,SIGNAL(toggled(bool)),this,SLOT(toggle_pause(bool)));
				buttons->addWidget(pause_button,0,0,1,2);


				stop_button = new QPushButton("stop",buttons_frame);
				connect(stop_button, SIGNAL(clicked(bool)), this, SLOT(stop()));
				buttons->addWidget(stop_button,1,0);

				stop_mode_button = new QPushButton(stop_mode_names[Just_Now],buttons_frame);
				QMenu* stop_mode_menu = new QMenu(stop_mode_button);
				QAction* just_now = new QAction(stop_mode_names[Just_Now]);
				connect(just_now, SIGNAL(triggered()), this, SLOT(stop_JustNow()));
				stop_mode_menu->addAction(just_now);
				QAction* current_epoc = new QAction(stop_mode_names[Current_Epoc]);
				connect(current_epoc,SIGNAL(triggered()),this,SLOT(stop_CurrentEpoc()));
				stop_mode_menu->addAction(current_epoc);
				
				stop_mode_button->setMenu(stop_mode_menu);
				buttons->addWidget(stop_mode_button,1,1);

				buttons_frame->setMaximumWidth(200);
				if (show_model) {
					NV = new NetViewer(ref_model,tag, this);
					main_layout->addWidget(NV, 0, 0);
				}
				else {
				}

				main_layout->addWidget(buttons_frame, 0, 1);
				cent_widget->setLayout(main_layout);
				setCentralWidget(cent_widget);

				connect(this, SIGNAL(pause(const bool)), ref_model, SLOT(pause(bool)), Qt::DirectConnection);
				connect(this, SIGNAL(stop(const int)), ref_model, SLOT(stop(int)), Qt::DirectConnection);
				show();
			}
			~ControlPanel() {}


		signals:
			void pause(const bool);
			void stop(const int);

		public slots:
			void toggle_pause(bool checked) {
				if (checked) {
					pause_button->setText("restart");
				}
				else {
					pause_button->setText("pause");
				}
				is_running = !is_running;
				emit pause(is_running);
			}

			void stop_JustNow() {
				stop_mode_button->setText(stop_mode_names[Just_Now]);
				stop_mode = Just_Now;
			}

			void stop_CurrentEpoc() {
				stop_mode_button->setText(stop_mode_names[Current_Epoc]);
				stop_mode = Current_Epoc;
			}

			void stop() {
				emit pause(true);
				emit stop(stop_mode);
			}

		private:
			Model* ref_model;
			QPushButton* pause_button;
			QPushButton* stop_button;
			QPushButton* stop_mode_button;
			NetViewer* NV;
			int stop_mode = 0;
			bool is_running = true;
	};
}

#pragma once

#define TISANET_QT

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

			using tisaNET::Model::train;
			Q_INVOKABLE void train();

			//モデルをファイルに出力する
			void save_model(std::string tp_file) override;

			void emit_output(std::vector<double> data) override {
				emit output_updated(data);
			}

			void pause_trainning() override{
				while (pause_flag) { QThread::msleep(loop_msec); }
			}

			void show_net_process(std::vector<double>& data) override{

				emit filter_changed(0, 0, tisaNET::conv_vect_to_mat3D(data, net_layer.front().input_dim3));

				for (int i = 0; i < net_layer.size();i++) {
					if (net_layer[i].is_conv_layer()) {

						int filt_out_num = net_layer[i].filter_output_dim3[2];
						for (int j = 0; j < net_layer[i].filter_num; j++) {
							std::vector<tisaMat::matrix> tmp_data = tisaNET::conv_vect_to_mat3D(net_layer[i].W->elements[j], net_layer[i].filter_dim3);
							emit filter_changed(i * 2 + 1, j, tmp_data);
							
							std::vector<tisaMat::matrix> tmp_output;
							for (int k = 0; k < filt_out_num; k++) {
								tmp_output.push_back(net_layer[i].Output_mat[j * filt_out_num + k]);
							}

							emit filter_changed(i * 2 + 2, j, tmp_output);
						}
					}
					else {
						switch (net_layer.front().Activation_f ^ POOLING) {
						case MAX_POOL:
							break;
						}
					}
				}
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
			//FilterViewer* net_viewer;

			double learning_rate;
			tisaNET::Data_set* train_data;
			tisaNET::Data_set* test_data;
			int epoc;
			int batch_size;
			uint8_t Error_func;
			bool save_model_flag = false;
			std::string save_model_file;
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


					//中間出力も表示してみる
					layers.push_back(new LayerViewer("Middle " + QString::number(i)));
					for (int j = 0; j < ref_model->net_layer[i].filter_num; j++) {
						layers.back()->addfilter(ref_model->net_layer[i].filter_output_dim3);
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
					NV_area = new QScrollArea(this);
					NV = new NetViewer(ref_model,tag, NV_area);

					NV_area->setWidgetResizable(true);
					NV_area->setWidget(NV);
					main_layout->addWidget(NV_area, 0, 0);
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

			QScrollArea* NV_area;
			NetViewer* NV;
			int stop_mode = 0;
			bool is_running = true;
	};
}

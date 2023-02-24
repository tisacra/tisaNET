#pragma once

#include <QWidget>
#include <QLabel>
#include <QFrame>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QScrollArea>
#include <QSizePolicy>
#include <QPainter>
#include "ui_FilterViewer.h"

#include <tisaMat.h>

#define output_bar_length 100
#define output_bar_width 10

class FilterViewerElement : public QFrame {
	Q_OBJECT

	public:
		FilterViewerElement(float _zoom,QWidget* parent = nullptr)
			: QFrame(parent)
		{
			setFrameShape(QFrame::Box);
			setLineWidth(1);
			setSizePolicy(QSizePolicy::Policy::Minimum, QSizePolicy::Policy::Minimum);
			zoom = _zoom;
		}
		~FilterViewerElement() {}

		float zoom;

	public slots:
		void image_update(QImage);

	protected:
		void paintEvent(QPaintEvent*);
	private:
		QImage img;
};

class emitter : public QObject {
	Q_OBJECT
	public:
		emitter(QObject* parent = nullptr) :QObject(parent) {}
		~emitter() {}
	signals:
		void filter_update_call(const int, const std::vector<tisaMat::matrix>);
		void filter_update_call(const std::vector<tisaMat::matrix>);
		void filter_image_update(const QImage data);
};

class FilterViewer : public QWidget
{
	Q_OBJECT

	signals:

	public:
		template<typename T>
		FilterViewer(T *filter_dim3,QWidget* parent)
			: QWidget(parent)
		{
			ui->setupUi(this);
			setSizePolicy(QSizePolicy::Policy::Minimum, QSizePolicy::Policy::Minimum);
			filter_layout = new QHBoxLayout(this);
			feature2d[0] = filter_dim3[0];
			feature2d[1] = filter_dim3[1];
			filter_depth = filter_dim3[2];
			max = 1.;
			min = 0.;

			margin = {10,10};
			zoom = 3.;
			QSize size = QSize(feature2d[0] * filter_depth, feature2d[1]) * zoom;
			setFixedSize(size + margin);

			for (int i = 0; i < filter_depth;i++) {
				addfilter();
			}
		}

		int feature2d[2];
		int filter_depth;
		double max;
		double min;

		~FilterViewer() {}
	
		//std::vector<tisaMat::matrix> filter;
		//std::vector<QImage> *img;
		void addfilter() {
			filt_elements.push_back(new FilterViewerElement(zoom,this));
			filter_layout->addWidget(filt_elements.back());

			emitter_array.push_back(new emitter());
			connect(emitter_array.back(), SIGNAL(filter_image_update(const QImage)),
					filt_elements.back(), SLOT(image_update(QImage)));
		}

		float zoom;
	public slots:
		void filter_update(std::vector<tisaMat::matrix> data);

	private:
		Ui::FilterViewerClass *ui;
		//std::vector<QImage> m_img;
		QHBoxLayout* filter_layout;
		std::vector<FilterViewerElement*> filt_elements;
		std::vector<emitter*> emitter_array;
		QSize margin;
};

enum GraphBar_Mode {
	LeftToRight,
};

class GraphBar : public QFrame {
	Q_OBJECT
	public:
		GraphBar(int mode, int width, int length,QWidget* parent) :QFrame(parent) {
			graph_mode = mode;
			bar_width = width;
			full_length = length;
			switch (mode) {
				case LeftToRight:
					setFixedSize(length,width);
					break;
			}
		}
		~GraphBar(){}

		int full_length;
		int current_length;
		int bar_width;

		void plot_graph(int val) {
			current_length = val;
			repaint();
		}

	protected:
		void paintEvent(QPaintEvent* event) override{
			QPainter painter(this);

			switch (graph_mode) {
				case LeftToRight:
					painter.setBrush(Qt::blue);
					painter.drawRect(0, 0, current_length, bar_width);
					break;
			}

			QFrame::paintEvent(event);
		}

	private:
		int graph_mode;

};

class LayerViewer : public QFrame {
	Q_OBJECT

	public:
		LayerViewer(QString str, QWidget* parent = nullptr) :QFrame(parent) {
			layer_label = new QLabel(this);
			layer_label->setText(str);

			main_layout = new QVBoxLayout(this);
			main_layout->addWidget(layer_label);

			filter_scroll_area = new QScrollArea(this);
			filter_area_frame = new QFrame(filter_scroll_area);
			
			filter_area_frame->setSizePolicy(QSizePolicy::Policy::Minimum, QSizePolicy::Policy::Minimum);
			filter_scroll_area->setWidgetResizable(true);
			filter_scroll_area->setWidget(filter_area_frame);

			main_layout->addWidget(filter_scroll_area);
		}
		~LayerViewer() {}

		std::vector<FilterViewer*> filters;

		template<typename T>
		void addfilter(T *filter_dim3) {
			if (filters.size() == 0) {
				layer_layout = new QVBoxLayout(filter_area_frame);
				layer_layout->setSizeConstraint(QLayout::SetMinimumSize);
				layer_layout->setSpacing(3);

				filter_area_frame->setLayout(layer_layout);
			}
			//filter_scroll_area->takeWidget();
			filters.push_back(new FilterViewer(filter_dim3, filter_area_frame));
			layer_layout->addWidget(filters.back());
			//filter_scroll_area->setWidget(filter_area_frame);

			emitter_array.push_back(new emitter());
			connect(emitter_array.back(), SIGNAL(filter_update_call(const std::vector<tisaMat::matrix>)),
				filters.back(), SLOT(filter_update(std::vector<tisaMat::matrix>)));
		}

		void addanswer(std::vector<QString> &tag) {
			QGridLayout* answer_layout = new QGridLayout(filter_area_frame);
			answer_layout->setSizeConstraint(QLayout::SetMinimumSize);
			for (int i = 0; i < tag.size();i++) {
				answer_tag.push_back(new QLabel( filter_area_frame));
				answer_tag.back()->setText(tag[i]);

				answer_output_graph.push_back(new GraphBar(LeftToRight,output_bar_width,output_bar_length,filter_area_frame));

				answer_output.push_back(new QLabel(filter_area_frame));
				answer_output.back()->setText("start");
				
				answer_layout->addWidget(answer_tag[i], i, 0);
				answer_layout->addWidget(answer_output_graph[i],i,1);
				answer_layout->addWidget(answer_output[i],i,2);
			}
			filter_area_frame->setLayout(answer_layout);
		}

	public slots:
		void filter_update_call(int filter, std::vector<tisaMat::matrix> data) {
			emit emitter_array[filter]->filter_update_call(data);
		}

		void output_update(std::vector<double> output) {
			for (int i = 0; i < output.size();i++) {
				answer_output[i]->setText(QString::number(output[i]));
				answer_output_graph[i]->plot_graph(output_bar_length * output[i]);
			}
		}

	private:
		QLabel* layer_label;
		QVBoxLayout* layer_layout;
		QVBoxLayout* main_layout;
		QScrollArea* filter_scroll_area;
		QFrame* filter_area_frame;
		std::vector<emitter*> emitter_array;
		std::vector<QLabel*> answer_tag;
		std::vector<GraphBar*> answer_output_graph;
		std::vector<QLabel*> answer_output;
};


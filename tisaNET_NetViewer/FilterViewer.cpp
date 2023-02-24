#include "FilterViewer.h"

void FilterViewer::filter_update(std::vector<tisaMat::matrix> data) {
	//filter = data;
	feature2d[0] = data.front().mat_RC[0];
	feature2d[1] = data.front().mat_RC[1];
	double tmp_min = tisaMat::min(data);
	double tmp_max = tisaMat::max(data);
	QImage tmp_img = QImage(feature2d[1], feature2d[0],QImage::Format_RGBX64);
	//this->sizeHint = QSize(feature2d[0] * filter_depth,feature2d[1]);


	for (int k = 0; k < filter_depth; k++) {
		for (int i = 0; i < feature2d[0]; i++) {
			for (int j = 0; j < feature2d[1]; j++) {
				QRgba64 rgba64;
				uint16_t color_base = (data[k].elements[i][j] - min) / (max - min) * 0xFFFF;
				rgba64.setRed(color_base);
				rgba64.setGreen(color_base);
				rgba64.setBlue(color_base);
				QColor color = QColor(rgba64);
				tmp_img.setPixelColor(j,i, color);
			}
		}
		emit emitter_array[k]->filter_image_update(tmp_img);
	}
	//QSize size = sizeHint();
	//setFixedSize((QSize(feature2d[0] * filter_depth, feature2d[1]) * zoom + margin));
}

void FilterViewerElement::image_update(QImage tmp_img) {
	img = tmp_img;
	repaint();
}

void FilterViewerElement::paintEvent(QPaintEvent* event) {
	QPainter painter(this);
	painter.scale(zoom, zoom);
	painter.drawImage(0,0,img);
	QFrame::paintEvent(event);
	//img->clear();
	//QPainter p(this);
	/*
	for (int k = 0; k < feature[2];k++) {
		painter.drawImage(0,k * feature[1],m_img[k]);
	}
	*/
	resize(img.width() * zoom +1,img.height() * zoom + 1);
}
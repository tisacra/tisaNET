#include <QApplication>
#include <QThread>
#include <QMetaObject>
#include <tisaNET_Qt.h>
#include <iostream>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    //Q_DECLARE_METATYPE(std::vector<tisaMat::matrix>)
    qRegisterMetaType<std::vector<tisaMat::matrix>>("std::vector<tisaMat::matrix>");
    qRegisterMetaType<std::vector<double>>("std::vector<double>");

    tisaNET::Data_set train_data;
    tisaNET::Data_set test_data;

    std::vector<QString> tags = {"0","1","2","3" ,"4" ,"5" ,"6" ,"7" ,"8","9" };

    tisaNET::load_MNIST("..\\..\\MNIST", train_data, test_data, 50000, 10000, false);

    tisaNET_Qt::Model model(0.001, train_data, test_data, 1, 20, CROSS_ENTROPY_ERROR);

    bool use_load = 1;
    bool model_save = 1;

    const char* model_file = "..\\model_files\\mnist_0422_1.tp";
    if (use_load) {
        model.load_model(model_file);
    }
    else {
        int filt_1[3] = { 10,10,1 };
        int input_shape[3] = { 28,28,1 };
        model.Create_Convolute_Layer(SIGMOID, input_shape, filt_1, 15, 3);
        int filt_3[3] = { 7,7,1 };
        model.Create_Convolute_Layer(RELU, filt_3, 12, 1);
        /*
        int filt_2[3] = { 2,2,1 };
        model.Create_Pooling_Layer(MAX_POOL,filt_2);
        */
        model.Create_Layer(100,RELU);
        model.Create_Layer(100, SIGMOID);
        model.Create_Layer(10, SOFTMAX);
        model.initialize();
    }

    model.monitor_accuracy(true);
    //model.logging_error("log_mnist2023_0420_1.csv");

    model.monitor_network(true);

    if(model_save) model.save_model(model_file);

    tisaNET_Qt::ControlPanel CP(&model,true,&tags);

    QThread* thread = new QThread();
    model.moveToThread(thread);
    QMetaObject::invokeMethod(&model, "train", Qt::QueuedConnection);
    thread->start();
    return a.exec();
}
#include <tisaNET.h>
#include <iostream>

int main()
{
    tisaNET::Data_set train_data;
    tisaNET::Data_set test_data;
    /*
    train_data.data = { {0,0,0},{0,0,1},{0,1,0},{0,1,1},{1,0,0},{1,0,1},{1,1,0},{1,1,1} };
    train_data.answer = { {0,1},{1,0},{0,1},{0,1},{1,0},{1,0},{0,1},{1,0} };
    test_data.data = { {0,0,0},{0,0,1},{0,1,0},{0,1,1},{1,0,0},{1,0,1},{1,1,0},{1,1,1} };
    test_data.answer = { {0,1},{1,0},{0,1},{0,1},{1,0},{1,0},{0,1},{1,0} };
    */
    tisaNET::load_MNIST("..\\..\\MNIST", train_data, test_data, 7000, 1000, false);

    tisaNET::Model model;

    /*
    std::vector<std::vector<double>> filter11 = { {1.0, 0.0, 1.0, 0.0, 1.0},
                                                 {0.0, 0.5, 0.7, 0.5, 0.0},
                                                 {1.0, 0.7, 1.0, 0.7, 1.0},
                                                 {0.0, 0.5, 0.7, 0.5, 0.0},
                                                 {1.0, 0.0, 1.0, 0.0, 1.0} };

    std::vector<std::vector<double>> filter12 = { {1.0, 0.3, 1.0},
                                                 {0.3, 1.0, 0.3},
                                                 {1.0, 0.3, 1.0} };
    */
    bool use_load = 0;
    if (use_load) {
        model.load_model("mnist_0125_1.tp");
    }
    else {
        //MNISTは784
        model.Create_Layer(784,INPUT);
        int filter[3] = { 5,5,1 };
        int input[3] = {28,28,1};
        model.Create_Layer(100, RELU);
        model.Create_Layer(30, SIGMOID);
        model.Create_Layer(10, SOFTMAX);
        model.initialize();
    }

    model.monitor_accuracy(true);
    //model.logging_error("log_mnist0125.csv");
    model.train(0.01,train_data,test_data,5,10,CROSS_ENTROPY_ERROR);
    model.save_model("mnist_0809_1.tp");
    return 0;
}
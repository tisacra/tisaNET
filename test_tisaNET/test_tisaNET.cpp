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
    tisaNET::load_MNIST("..\\..\\MNIST", train_data, test_data, 50000, 10000, false);

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
    bool use_load = 1;
    if (use_load) {
        model.load_model("mnist_0213_3.tp");
    }
    else {
        //MNISTは784
        int filt_1[3] = {7,7,1};
        int input_shape[3] = { 28,28,1 };
        model.Create_Convolute_Layer(RELU,input_shape,filt_1,15,3);

        int filt_2[3] = { 2,2,1 };
        model.Create_Pooling_Layer(MAX_POOL,filt_2);

        int filt_3[3] = { 4,4,1 };
        model.Create_Convolute_Layer(RELU, filt_3, 20, 1);
        //model.Create_Layer(28*28, INPUT);
        model.Create_Layer(700, SIGMOID);
        model.Create_Layer(500, RELU);
        model.Create_Layer(300, RELU);
        model.Create_Layer(10, SOFTMAX);
        model.initialize();
    }

    model.monitor_accuracy(true);
    model.logging_error("log_mnist2023_0213_3.csv");
    model.train(0.005,train_data,test_data,10,20,CROSS_ENTROPY_ERROR);
    model.save_model("mnist_0213_3.tp");
    return 0;
}

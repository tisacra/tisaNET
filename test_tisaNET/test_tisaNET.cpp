﻿#include <tisaNET.h>
#include <iostream>

int main()
{
    tisaNET::Data_set train_data;
    //train_data.data = { {0,0,0},{0,0,1},{0,1,0},{0,1,1},{1,0,0},{1,0,1},{1,1,0},{1,1,1} };
    //train_data.answer = { {0,1},{1,0},{0,1},{0,1},{1,0},{1,0},{0,1},{1,0} };
    tisaNET::Data_set test_data;
    //test_data.data = { {0,0,0},{0,0,1},{0,1,0},{0,1,1},{1,0,0},{1,0,1},{1,1,0},{1,1,1} };
    //test_data.answer = { {0,1},{1,0},{0,1},{0,1},{1,0},{1,0},{0,1},{1,0} };
    tisaNET::load_MNIST("..\\..\\..\\..\\MNIST",train_data,test_data,50000,10000,false);

    tisaNET::Model model;

    /*
    std::vector<std::vector<double>> filter1 = { {1.0, 0.0, 1.0, 0.0, 1.0},
                                                 {0.0, 0.5, 0.7, 0.5, 0.0},
                                                 {1.0, 0.7, 1.0, 0.7, 1.0},
                                                 {0.0, 0.5, 0.7, 0.5, 0.0},
                                                 {1.0, 0.0, 1.0, 0.0, 1.0} };

    std::vector<std::vector<double>> filter2 = { {1.0, 0.3, 1.0},
                                                 {0.3, 1.0, 0.3},
                                                 {1.0, 0.3, 1.0} };
    */
    /**/
    model.Create_Comvolute_Layer(28,28,5,5,1,5,2);
    model.Create_Comvolute_Layer(11,11,5,5,5,10);
    model.Create_Layer(32, RELU);
    model.Create_Layer(16, SIGMOID);
    model.Create_Layer(10, SOFTMAX);
    model.initialize();
    /**/
    //model.load_model("mnist_1212_2.tp");

    model.monitor_accuracy(true);
    //model.logging_error("log_mnist1211.csv");
    model.train(0.03,train_data,test_data,10,10,CROSS_ENTROPY_ERROR);
    model.save_model("mnist_1212_4.tp");
    return 0;
}
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
    tisaNET::load_MNIST("..\\..\\..\\..\\MNIST", train_data, test_data, 50000, 10000, false);

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

    //MNISTは784
    //model.Create_Layer(3,INPUT);
    /**/ 
    int input1[3] = {28,28,1};
    int filter1[3] = {5,5,1};
    model.Create_Comvolute_Layer(input1,filter1,5,2);
    int filter2[3] = { 5,5,1 };
    model.Create_Comvolute_Layer(filter2,5,2);
    int filter3[3] = { 5,5,1 };
    model.Create_Comvolute_Layer(filter3, 10, 1);
    /**/
    /**/
    model.Create_Layer(16, RELU);
    model.Create_Layer(16, SIGMOID);
    model.Create_Layer(10, SOFTMAX);
    model.initialize();
    /**/
    //model.load_model("mnist_1230_1.tp");

    model.monitor_accuracy(true);
    model.logging_error("log_mnist0101.csv");
    model.train(0.01,train_data,test_data,10,10,CROSS_ENTROPY_ERROR);
    model.save_model("mnist_0101_1.tp");
    return 0;
}
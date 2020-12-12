#include <tisaNET.h>
#include <iostream>

int main()
{
    tisaNET::Data_set train_data;
    //train_data.data = { {0,0,0},{0,0,1},{0,1,0},{0,1,1},{1,0,0},{1,0,1},{1,1,0},{1,1,1} };
    //train_data.answer = { {0,1},{1,0},{0,1},{0,1},{1,0},{1,0},{0,1},{1,0} };
    tisaNET::Data_set test_data;
    //test_data.data = { {0,0,0},{0,0,1},{0,1,0},{0,1,1},{1,0,0},{1,0,1},{1,1,0},{1,1,1} };
    //test_data.answer = { {0,1},{1,0},{0,1},{0,1},{1,0},{1,0},{0,1},{1,0} };
    //tisaNET::load_MNIST("..\\..\\..\\..\\MNIST", train_data, test_data, 50, 10, false);

    tisaNET::Model model;

    std::vector<uint8_t> tracer = {0,1,2,3,4,5,6,7,8,9,
                                   0,1,2,3,4,5,6,7,8,9, 
                                   0,1,2,3,4,5,6,7,8,9,
                                   0,1,2,3,4,5,6,7,8,9, 
                                   0,1,2,3,4,5,6,7,8,9,
                                   0,1,2,3,4,5,6,7,8,9,
                                   0,1,2,3,4,5,6,7,8,9,
                                   0,1,2,3,4,5,6,7,8,9, 
                                   0,1,2,3,4,5,6,7,8,9,
                                   0,1,2,3,4,5,6,7,8,9};

    /**/
    std::vector<std::vector<double>> filter = { {1.0, 0.3, 1.0},
                                                {0.3, 1.0, 0.3},
                                                {1.0, 0.3, 1.0} };
    /*
    model.Create_Comvolute_Layer(10, 10, 3, 3, 1, 3);
    model.Create_Comvolute_Layer(7, 7, 3, 3, 3, 3);
    model.Create_Layer(32, RELU);
    model.Create_Layer(10, SOFTMAX);
    model.initialize();
    */

    model.load_model("mnist_1212_3.tp");

    //model.monitor_accuracy(true);
    //model.logging_error("log_mnist1211.csv");
    //model.train(0.05, train_data, test_data, 10, 10, CROSS_ENTROPY_ERROR);
    model.feed_forward(tracer);
    model.save_model("mnist_1212_3.tp");
    return 0;
}
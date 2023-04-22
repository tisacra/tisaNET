#include <tisaNET.h>
#include <iostream>

int main(int argc,char* argv[])
{
    tisaNET::Data_set train_data;
    tisaNET::Data_set test_data;
    
    tisaNET::load_MNIST("..\\..\\MNIST", train_data, test_data, 50000, 10000, false);

    tisaNET::Model model;

    bool use_load = 1;
    if (use_load) {
        model.load_model("mnist_0422_1.tp");
    }
    else {
        int filt_1[3] = {10,10,1};
        int input_shape[3] = { 28,28,1 };
        model.Create_Convolute_Layer(RELU,input_shape,filt_1,7,3);
        int filt_2[3] = { 8,8,1 };
        model.Create_Convolute_Layer(RELU, filt_2, 10, 1);
        /*
        int filt_2[3] = { 2,2,1 };
        model.Create_Pooling_Layer(MAX_POOL,filt_2);
        */
        model.Create_Layer(400, SIGMOID);
        model.Create_Layer(10, SOFTMAX);
        model.initialize();
    }

    model.monitor_accuracy(true);
    //model.logging_error("log_mnist2023_0214_1.csv");
    model.train(0.01,train_data,test_data,10,20,CROSS_ENTROPY_ERROR);
    model.save_model("mnist_0422_1.tp");
    return 0;
}

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
    tisaNET::load_MNIST("..\\..\\MNIST",train_data,test_data,100,20,false);

    tisaNET::Model model;
    model.Create_Layer(784,INPUT);
    model.Create_Layer(100, SIGMOID);
    model.Create_Layer(10, SOFTMAX);
    model.initialize();

    model.monitor_accuracy(true);
    model.train(0.001,train_data,test_data,300,10,CROSS_ENTROPY_ERROR);
    model.save_model("test_mnist.tp");
    return 0;
}
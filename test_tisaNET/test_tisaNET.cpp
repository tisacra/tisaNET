#include <tisaNET.h>
#include <iostream>

int main()
{
    tisaNET::Data_set train_data;
    tisaNET::Data_set test_data;
    tisaNET::load_MNIST("C:\\Users\\ssskai\\Downloads\\MNIST",train_data,test_data,500,70,false);

    tisaNET::Model model;
    model.Create_Layer(784,INPUT);
    model.Create_Layer(100,RELU);
    model.Create_Layer(60,RELU);
    model.Create_Layer(10,SOFTMAX);
    model.initialize();

    model.train(0.001,train_data,test_data,200,500,CROSS_ENTROPY_ERROR);
    model.save_model("test_mnist.tp");
    return 0;
}
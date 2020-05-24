#include <tisaNET.h>
#include <iostream>

int main()
{
    tisaNET::Data_set train_data;
    
    tisaNET::Data_set test_data;
    
    tisaNET::load_MNIST("..\\..\\MNIST", train_data, test_data, 10000, 5000, false);

    tisaNET::Model model;
    model.load_model("..\\test_tisaNET\\test_mnist.tp");
    std::vector<double> output = model.F_propagate(test_data.data[10]);
    tisaMat::vector_show(output);
    printf("answer\n");
    tisaMat::vector_show(test_data.answer[10]);
    return 0;
}
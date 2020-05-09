#include <tisaNET.h>
#include <iostream>

int main()
{
    tisaNET::Data_set train_data;
    train_data.sample_data = { {0,0},{0,1},{1,0},{1,1} };
    train_data.answer = { {0},{1},{1},{0} };

    tisaNET::Data_set test_data;
    test_data.sample_data = { {0,0},{0,1},{1,0},{1,1} };
    test_data.answer = { {0},{1},{1},{0} };

    tisaNET::Model model;
    model.Create_Layer(2, INPUT);
    model.Create_Layer(2, SIGMOID);
    model.Create_Layer(1, STEP);
    
    model.initialize();
    model.train(0.5,train_data,test_data,600,4,MEAN_SQUARED_ERROR);
    return 0;
}
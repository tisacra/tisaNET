#include <tisaNET.h>
#include <iostream>

int main()
{
    
    /*for (int k = 0; k < 28; k++) {
        for (int i = 0; i < 28; i++) {
            int tmp = eval[i + k*28];
            printf("%3u ", tmp);
        }
        printf("\n");
    }
    */

    tisaNET::Model model;
    model.load_model("..\\test_tisaNET\\mnist_1212_1.tp");

    /*
    tisaNET::Data_set eval;
    tisaNET::load_MNIST("..\\..\\..\\MNIST",eval,10,false);
    
    for (int i = 0; i < 10;i++) {
        std::vector<double> output = model.feed_forward(eval.data[i]);
        for (int k = 0; k < 28; k++) {
            for (int j = 0; j < 28; j++) {
                int tmp = eval.data[i][k * 28 + j] * 256;
                printf("%3u ", tmp);
            }
            printf("\n");
        }
        printf("|answer|\n");
        tisaMat::vector_show(output);
        double percent = *(std::max_element(output.begin(), output.end())) * 100.0;
        int tag = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
        printf("%lf%% で 「%d」です\n", percent, tag);
    }
    */

    std::vector<uint8_t> input = tisaNET::vec_from_256bmp("..\\..\\..\\MNIST\\6-001.bmp");
    std::vector<double> output = model.feed_forward(input);
    for (int k = 0; k < 28; k++) {
        for (int j = 0; j < 28; j++) {
            int tmp = input[k * 28 + j];
            printf("%3u ", tmp);
        }
        printf("\n");
    }
    printf("|answer|\n");
    tisaMat::vector_show(output);
    double percent = *(std::max_element(output.begin(), output.end())) * 100.0;
    int tag = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
    printf("%lf%% で 「%d」です\n", percent, tag);

    return 0;
}
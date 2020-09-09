#include <tisaNET.h>
#include <iostream>

int main()
{
    tisaNET::Data_set eval;
    
    /*for (int k = 0; k < 28; k++) {
        for (int i = 0; i < 28; i++) {
            int tmp = eval[i + k*28];
            printf("%3u ", tmp);
        }
        printf("\n");
    }
    */
    tisaNET::load_MNIST("..\\..\\MNIST",eval,10,false);
    
    tisaNET::Model model;
    model.load_model("..\\test_tisaNET\\test_mnist.tp");
    for (int i = 0; i < 10;i++) {
        std::vector<double> output = model.feed_forward(eval.data[i]);
        for (int k = 0; k < 28; k++) {
            for (int i = 0; i < 28; i++) {
                int tmp = eval.data[i][k * 28];
                printf("%3u ", tmp);
            }
            printf("\n");
        }
        printf("|answer|\n");
        tisaMat::vector_show(output);
        double percent = *(std::max_element(output.begin(), output.end())) * 100.0;
        int tag = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
        printf("%lf%% ‚Å u%dv‚Å‚·\n", percent, tag);
    }
    
    return 0;
}
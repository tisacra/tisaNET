#include <tisaNET.h>
#include <iostream>

int main()
{
    std::vector<uint8_t> eval = tisaNET::vec_from_256bmp("..\\..\\my9.bmp");
    for (int k = 0; k < 28; k++) {
        for (int i = 0; i < 28; i++) {
            int tmp = eval[i + k*28];
            printf("%3u ", tmp);
        }
        printf("\n");
    }
    
    tisaNET::Model model;
    model.load_model("..\\test_tisaNET\\test_mnist.tp");
    std::vector<double> output = model.feed_forward(eval);
    printf("|answer|\n");
    tisaMat::vector_show(output);
    double percent = *(std::max_element(output.begin(),output.end())) * 100.0;
    int tag = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
    printf("%lf%% ‚Å u%dv‚Å‚·\n",percent,tag);
    return 0;
}
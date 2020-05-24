#pragma once
/*�`�ȒP�Ȏg�����`
0.Data_set�͌P���f�[�^�A�]���f�[�^�ŕ����č���Ă�������
  sample_data�̑�2�C���f�b�N�X�̑傫�����j���[�����l�b�g�ւ̓��͂̐��ɑΉ����܂�

1.Model�N���X���C���X�^���X������(�j���[�����l�b�g�̃��f����\���܂�)
2.Model.Create_Layer����ƁA�Ō�ɍ�����w�̌��ɐV�����w���Ȃ��܂�
	�ŏ��͕K��������(Activation)��INPUT�ɂ��Ă�������(���͑w�Ƃ��Ďg���܂�)
3.��ԍŌ�ɍ�����w���o�͑w�ɂȂ�܂�
*/
#include <tisaMat.h>
#include <cstdint>
#include <vector>
#include <random>

#define SIGMOID 0
#define RELU 1
#define STEP 2
#define SOFTMAX 3
#define INPUT 4

#define MEAN_SQUARED_ERROR 0
#define CROSS_ENTROPY_ERROR 1

namespace tisaNET {
	class Data_set {
	public:
		std::vector<std::vector<uint8_t>> data;
		std::vector<std::vector<uint8_t>> answer;
	};

	struct layer {
		uint8_t Activation_f = 0;
		unsigned short node = 0;
		tisaMat::matrix *W = nullptr;
		std::vector<double> B;
		std::vector<double> Output;
	};

	struct Trainer {
		tisaMat::matrix* dW = nullptr;
		std::vector<double> dB;
		std::vector<std::vector<double>> Y;
	};

	//MNIST����f�[�^�����
	bool load_MNIST(const char* path,Data_set& train_data,Data_set& test_data, int sample_size,int test_size, bool single_output);

	double step(double X);
	double sigmoid(double X);
	double ReLU(double X);
	double softmax(double X);

	//���l���o�C�i���ŕ\��
	bool print01(int bit, long Value);
	
	//���ϓ��덷�֐�
	double mean_squared_error(std::vector<std::vector<uint8_t>>& teacher, std::vector<std::vector<double>>& output);

	//�����G���g���s�[�֐�
	double cross_entropy_error(std::vector<std::vector<uint8_t>>& teacher, std::vector<std::vector<double>>& output);

	class Model {
	public:

		//�l�b�g���[�N�̈�Ԃ�����ɑw��������(init�͏d�݂�����������Ƃ��̒l�A�ȗ�����Ɨ���)
		void Create_Layer(int Outputs, uint8_t Activation);
		void Create_Layer(int nodes, uint8_t Activation, double init);

		//���͑w(�ŏ��̑w�̂���)�Ƀl�b�g���[�N�ւ̓��͂������
		template <typename T>
		void input_data(std::vector<T>& data) {
			int input_num = data.size();
			if (net_layer.front().Output.size() != input_num) {
				printf("input error|!|\n");
			}
			else {
				std::vector<double> input = tisaMat::vector_cast<double>(data);
				net_layer.front().Output = input;
			}
		}

		//���`�d����
		template <typename T>
		//�P��
		std::vector<double> F_propagate(std::vector<T>& Input_data) {
			std::vector<double> output_vecter(net_layer.back().Output.size());
			input_data(Input_data);
			for (int i = 1; i < number_of_layer(); i++) {
				std::vector<double> X = tisaMat::vector_multiply(net_layer[i - 1].Output, *net_layer[i].W);
				X = tisaMat::vector_add(X, net_layer[i].B);
				//�\�t�g�}�b�N�X�֐����g���Ƃ��͂܂��ő�l��S���������
				if (net_layer[i].Activation_f == SOFTMAX) {
					double max = *std::max_element(X.begin(), X.end());
					for (int X_count = 0; X_count < X.size(); X_count++) {
						X[X_count] -= max;
					}
				}

				for (int j = 0; j < X.size(); j++) {
					net_layer[i].Output[j] = (*Af[net_layer[i].Activation_f])(X[j]);
					if (isnan(net_layer[i].Output[j])) {
						bool nan_flug = 1;
					}
				}

				if (net_layer[i].Activation_f == SOFTMAX) {
					double sigma = 0.0;
					for (int node = 0; node < net_layer[i].Output.size(); node++) {
						sigma += net_layer[i].Output[node];
					}
					tisaMat::vector_multiscalar(net_layer[i].Output, 1.0 / sigma);
				}
			}
			output_vecter = net_layer.back().Output;
			return output_vecter;
		}

		tisaMat::matrix F_propagate(tisaMat::matrix& Input_data);
		tisaMat::matrix F_propagate(tisaMat::matrix& Input_data, std::vector<Trainer>& trainer);

		//�t�덷�`�d����
		void B_propagate(std::vector<std::vector<uint8_t>>& teacher, tisaMat::matrix& output, uint8_t error_func, std::vector<Trainer>& trainer,double lr, tisaMat::matrix& input_batch);

		//�l�b�g���[�N�̑w�̐������o��
		int number_of_layer();

		//���f���̏d�݂Ƃ�������������
		void initialize();

		//���f�����P������
		void train(double learning_rate, Data_set& train_data, Data_set& test_data, int epoc, int batch_size, uint8_t Error_func);

		//���f���̃t�@�C��(.tp)��ǂݍ���
		void load_model(const char* tp_file);

		//���f�����t�@�C���ɏo�͂���
		void save_model(const char* tp_file);

		//��������\��/��\���ɂ���
		void monitor_accuracy(bool monitor_accuracy);

		//�P�����̌덷���C�e���[�V�������ƂɋL�^����(csv�`����)
		void logging_error(const char* log_file);

	private:
		bool monitoring_accuracy = false;
		bool log_error = false;
		std::string log_filename;
		std::vector<layer> net_layer;
		double (*Ef[2])(std::vector<std::vector<uint8_t>>&, std::vector<std::vector<double>>&) = { mean_squared_error,cross_entropy_error };
		double (*Af[4])(double) = { sigmoid,ReLU,step,softmax };
		void m_a(std::vector<std::vector<double>>& output, std::vector<std::vector<uint8_t>>& answer, uint8_t error_func);
		void B_propagate2(std::vector<std::vector<uint8_t>>& teacher, tisaMat::matrix& output, uint8_t error_func, std::vector<Trainer>& trainer, double lr, tisaMat::matrix& input_batch);

	};

	void show_train_progress(int total_iteration,int now_iteration);
}
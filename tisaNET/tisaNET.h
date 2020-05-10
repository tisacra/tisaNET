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
#define INPUT 3

#define MEAN_SQUARED_ERROR 0
#define CROSS_ENTROPY 1

namespace tisaNET {

	struct Data_set {
		std::vector<std::vector<double>> data;
		std::vector<std::vector<double>> answer;
	};

	struct layer {
		uint8_t Activation_f = 0;
		uint8_t node = 0;
		tisaMat::matrix *W = nullptr;
		std::vector<double> B;
		std::vector<double> Output;
	};

	struct Trainer {
		tisaMat::matrix* dW = nullptr;
		std::vector<double> dB;
		std::vector<std::vector<double>> Y;
	};

	double step(double X);
	double sigmoid(double X);
	double ReLU(double X);

	//���l���o�C�i���ŕ\��
	bool print01(int bit, long Value);
	
	//���ϓ��덷�֐�
	std::vector<double> mean_squared_error(std::vector<std::vector<double>>&, std::vector<std::vector<double>>& output);

	//�����G���g���s�[�֐�
	std::vector<double> cross_entropy(std::vector<std::vector<double>>&, std::vector<std::vector<double>>& output);

	class Model {
	public:

		//�l�b�g���[�N�̈�Ԃ�����ɑw��������(init�͏d�݂�����������Ƃ��̒l�A�ȗ�����Ɨ���)
		void Create_Layer(int Outputs, uint8_t Activation);
		void Create_Layer(int nodes, uint8_t Activation, double init);

		//���͑w(�ŏ��̑w�̂���)�Ƀl�b�g���[�N�ւ̓��͂������
		void input_data(std::vector<double>& data);

		//���`�d����
		tisaMat::matrix F_propagate(tisaMat::matrix& Input_data);
		tisaMat::matrix F_propagate(tisaMat::matrix& Input_data, std::vector<Trainer>& trainer);

		//�t�덷�`�d����
		void B_propagate(std::vector<std::vector<double>>& teacher, tisaMat::matrix& output, uint8_t error_func, std::vector<Trainer>& trainer,double lr, tisaMat::matrix& input_batch);

		//�l�b�g���[�N�̑w�̐������o��
		int number_of_layer();

		//���f���̏d�݂Ƃ�������������
		void initialize();

		//���f�����P������
		void train(double learning_rate, Data_set& train_data, Data_set& test_data, int epoc, int iteration, uint8_t Error_func);

		//���f���̃t�@�C��(.tp)��ǂݍ���
		void load_model(const char* tp_file);

		//���f�����t�@�C���ɏo�͂���
		void save_model(const char* tp_file);

	private:
		std::vector<layer> net_layer;
		std::vector<double> (*Ef[2])(std::vector<std::vector<double>>&, std::vector<std::vector<double>>&) = { mean_squared_error,cross_entropy };
		double (*Af[3])(double) = { sigmoid,ReLU,step };
	};
}
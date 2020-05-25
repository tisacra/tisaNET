#include <tisaNET.h>

int main() {
	//�P���p�f�[�^(�V���b�t�����Ă��A���₵�Ă��f�[�^�Ɠ����̕��т��Ή����Ă����OK)
	//�񎟌��z��̌`�ɂȂ邱�Ƃɒ���! ���g�̔z��̗v�f�����l�b�g���[�N�ւ̓��͂ɂȂ�܂�(�����ł�2����)
	tisaNET::Data_set train_data;
	train_data.data = { {0,0},
						{0,1},
						{1,0},
						{1,1} };
	//�����o�͂���ꍇ�ɑΉ������邽�߁A�������񎟌��z��ɂ��܂�
	train_data.answer = { {0},
						  {1},
						  {1},
						  {0} };

	//�]���p�f�[�^
	tisaNET::Data_set test_data;
	test_data.data = { {0,0},{0,1},{1,0},{1,1} };
	test_data.answer = { {0},{1},{1},{0} };

	//��������l�b�g���[�N���u�g�ݗ��Ăāv�����܂�
	tisaNET::Model model;
	//�ŏ��̑w�͓��͂𕪔z��������̂Ƃ��Ďg���܂�
	//���͂���f�[�^�̃T�C�Y�ƍŏ��̑w�̃m�[�h���������Ă��Ȃ��ƃG���[�ɂȂ�܂�
	//(�m�[�h�̐�, �������֐��̎��)
	model.Create_Layer(2, INPUT);		//�n�߂̑w�͊������֐��̑����INPUT���w�肷��K�v������܂�
	model.Create_Layer(2, SIGMOID);		//�m�[�h2�A�������֐��̓V�O���C�h�֐��ł���w
	model.Create_Layer(1, STEP);		//�Ō�̑w�͏o�͑w�ɂȂ�܂��@����͂P�o�͂Ŋ������֐��̓X�e�b�v�֐�
										//ReLU�֐�,Softmax�֐����g���܂�(�g���Ƃ��͑S�đ啶��)

	//�d�݂Ȃǂ�Xaivier�̏����l�AHe�̏����l�ɂ��������ď��������܂�
	model.initialize();
	//�P�����ɂ��̃G�|�b�N���Ƃ̐��m����\�������܂�
	model.monitor_accuracy(true);

	//���悢��w�K���܂�
	//(�w�K��, �P���p�f�[�^, �]���p�f�[�^, �G�|�b�N(�S�̂����Z�b�g�w�K���邩), �o�b�`�T�C�Y(�S�̂������ɕ����Ċw�K���邩), �ړI(�덷)�֐�)
	//�ړI�֐��ɂ́A���ϓ��덷(MEAN_SQUARED_ERROR)�ƃN���X�G���g���s�[�덷(CROSS_ENTROPY_ERROR)��p�ӂ��Ă��܂�
	model.train(0.5,train_data,test_data,1000,4,MEAN_SQUARED_ERROR);//�P���f�[�^��4��1�G�|�b�N��4��`�d�Ȃ̂ŁA���̎��̓C�����C���w�K�ł�

	//���f���̃p�����[�^�[���t�@�C���ɕۑ��ł��܂�(�g���q�͎��R�ł�)
	model.save_model("2inXOR_model.tp");
	return 0;
}
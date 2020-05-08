#pragma once

#include <vector>

namespace tisaMat {
	class matrix {
	public:
		int mat_RC[2];
		std::vector<std::vector<double>> elements;

		//row�scolumn��̍s���p�ӂ���(�������͂��Ȃ�)
		matrix(int row, int column);

		//row�scolumn��̍s���p�ӂ���(init�ŏ�����)
		matrix(int row, int column,double init);

		//�Q�����z�񂩂�s������([�s][��]�Ƃ����`�ɂȂ�)
		matrix(std::vector<std::vector<double>> mat);

		//�s���\������
		void show();

		//mat���X�J���[�{����
		void multi_scalar(double scalar);

		//���ӂ̍s��ɉE�ӂ̍s�����
		void operator=(matrix mat) {
			this->mat_RC[0] = mat.mat_RC[0];
			this->mat_RC[1] = mat.mat_RC[1];
			this->elements = mat.elements;
		}
	};
	//mat1��mat2�𑫂�
	matrix* matrix_add(matrix& mat1, matrix& mat2);

	//vec1��vec2�𑫂�
	std::vector<double>* vector_add(std::vector<double>& vec1, std::vector<double>& vec2);

	//mat1����mat2������
	matrix* matrix_subtract(tisaMat::matrix& mat1, tisaMat::matrix& mat2);

	//vec1��vec2������
	std::vector<double>* vector_subtract(std::vector<double>& vec1, std::vector<double>& vec2);

	//mat1��mat2���|����i�s��̐ρj
	matrix* matrix_multiply(matrix& mat1, matrix& mat2);

	//mat1��mat2�̃A�_�}�[����
	matrix* matrix_Hadamard(matrix& mat1, matrix& mat2);

	//vec1��mat1���|����
	std::vector<double>* vector_multiply(std::vector<double> vec1,matrix& mat1);

	//mat��]�u����
	matrix* matrix_transpose(matrix& mat);

	//vec��1�s�̍s��ɂ���
	matrix* vector_to_matrix(std::vector<double>& vec);

	//vec���X�J���[�{����
	void vector_multiscalar(std::vector<double>& vec, double scalar);
}

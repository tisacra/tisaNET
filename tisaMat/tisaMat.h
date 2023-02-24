#pragma once
#include <iostream>
#include <vector>
#include <iomanip>

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
		//�^��double�ȊO�̎�
		template <typename T>
		matrix(std::vector<std::vector<T>> mat) {
			int R = mat.size(), C = mat[0].size();
			mat_RC[0] = R;
			mat_RC[1] = C;
			elements = std::vector<std::vector<double>>(R, std::vector<double>(C));
			for (int row = 0; row < R;row++) {
				for (int column = 0; column < C;column++) {
					elements[row][column] = (double)mat[row][column];
				}
			}
		}

		//�s���\������
		void show();

		//mat���X�J���[�{����
		void multi_scalar(double scalar);

		//���ς��v�Z����
		double average();

		//���U���v�Z����
		double variance();

		//���ӂ̍s��ɉE�ӂ̍s�����
		void operator=(matrix mat) {
			this->mat_RC[0] = mat.mat_RC[0];
			this->mat_RC[1] = mat.mat_RC[1];
			this->elements = mat.elements;
		}

		//�ő�l���擾
		double max();
		//�ŏ��l���擾
		double min();
	};
	//mat1��mat2�𑫂�
	matrix matrix_add(matrix& mat1, matrix& mat2);

	//vec1��vec2�𑫂�
	std::vector<double> vector_add(std::vector<double> vec1, std::vector<double> vec2);

	//mat1����mat2������
	matrix matrix_subtract(tisaMat::matrix& mat1, tisaMat::matrix& mat2);

	//vec1��vec2������
	//std::vector<double> vector_subtract(std::vector<double>& vec1, std::vector<double>& vec2);
	template <typename T>
	std::vector<T> vector_subtract(std::vector<T>& vec1, std::vector<T>& vec2) {
		if (vec1.size() != vec2.size()) {
			return std::vector<T>();
		}
		else {
			std::vector<T> tmp(vec1.size());
			for (int i = 0; i < tmp.size(); i++) {
				tmp[i] = vec1[i] - vec2[i];
			}
			return tmp;
		}
	}

	//mat1��mat2���|����i�s��̐ρj
	matrix matrix_multiply(matrix& mat1, matrix& mat2);

	//mat1��mat2�̃A�_�}�[����
	matrix Hadamard_product(matrix& mat1, matrix& mat2);

	//mat1��mat2�̃A�_�}�[�����Z
	matrix Hadamard_division(matrix& mat1, matrix& mat2);

	//vec1��mat1���|����
	std::vector<double> vector_multiply(std::vector<double>& vec1,matrix& mat1);

	//mat��]�u����
	matrix matrix_transpose(matrix& mat);

	//vec��1�s�̍s��ɂ���
	matrix vector_to_matrix(std::vector<double>& vec);

	//vec���X�J���[�{����
	void vector_multiscalar(std::vector<double>& vec, double scalar);

	//vec�𐳋K������
	void vector_normalization(std::vector<double>& vec);

	//vec��\������
	template <typename T>
	void vector_show(std::vector<T>& vec) {
		for (int i = 0; i < vec.size(); i++) {
			//printf("%lf ", vec[i]);
			std::cout << std::setw(6) << vec[i] << ' ';
		}
		printf("\n");
	}

	//vec���^�ϊ�����
	template <typename T,typename U>
	std::vector<T> vector_cast(std::vector<U>& vec) {
		int size = vec.size();
		std::vector<T> tmp(size);
		for (int i = 0; i < size;i++) {
			tmp[i] = (T)vec[i];
		}
		return tmp;
	}

	//�e���\������ő�l���擾
	double max(std::vector<tisaMat::matrix>& tensor);

	//�e���\������ŏ��l���擾
	double min(std::vector<tisaMat::matrix>& tensor);
}

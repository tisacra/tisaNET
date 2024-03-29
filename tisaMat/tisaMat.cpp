﻿#include "tisaMat.h"
#include <vector>
#include <iostream>
#include <iomanip>

namespace tisaMat {
	matrix::matrix(int row, int column) {
		mat_RC[0] = row;
		mat_RC[1] = column;
		std::vector<std::vector<double>> tmp(row);
		for (int i = 0; i < row; i++) {
			std::vector<double> tmpC(column);
			tmp[i] = tmpC;
		}
		elements = tmp;
	}
	matrix::matrix(int row, int column,double init) {
		mat_RC[0] = row;
		mat_RC[1] = column;
		std::vector<std::vector<double>> tmp(row);
		for (int i = 0; i < row; i++) {
			std::vector<double> tmpC(column,init);
			tmp[i] = tmpC;
		}
		elements = tmp;
	}
	matrix::matrix(std::vector<std::vector<double>> mat) {
		mat_RC[0] = mat.size();
		mat_RC[1] = mat[0].size();
		elements = mat;
	}
	//行列を表示する
	void matrix::show() {
		for (int row = 0; row < mat_RC[0]; row++) {
			for (int column = 0; column < mat_RC[1]; column++) {
				std::cout << std::setw(4) << elements[row][column] << ' ';
			}
			std::cout << "\n";
		}
	}
	//行列をスカラー倍する
	void matrix::multi_scalar(double scalar) {
		for (int row = 0; row < mat_RC[0]; row++) {
			for (int column = 0; column < mat_RC[1]; column++) {
				elements[row][column] *= scalar;
			}
		}
	}

	//mat1とmat2を足す
	matrix matrix_add(matrix& mat1, matrix& mat2) {
		matrix tmp(mat1.mat_RC[0], mat1.mat_RC[1]);
		if ((mat1.mat_RC[0] != mat2.mat_RC[0]) || (mat1.mat_RC[1] != mat2.mat_RC[1])) {
			printf("matrix_shape are diffarent!\n");//行列の形が違うので足せない
			return matrix(0, 0);
		}
		else {
			for (int row = 0; row < mat1.mat_RC[0]; row++) {
				for (int column = 0; column < mat1.mat_RC[1]; column++) {
					tmp.elements[row][column] = mat1.elements[row][column] + mat2.elements[row][column];
				}
			}
			return tmp;
		}
	}

	//vec1とvec2を足す
	std::vector<double> vector_add(std::vector<double> vec1, std::vector<double> vec2) {
		if (vec1.size() != vec2.size()) {
			printf("vector shape are diffarent! can not add each!\n");
			exit(EXIT_FAILURE);
		}
		else {
			std::vector<double> tmp(vec1.size());
			for (int i = 0; i < tmp.size(); i++) {
				tmp[i] = vec1[i] + vec2[i];
			}
			return tmp;
		}
	}
	//mat1からmat2を引く
	matrix matrix_subtract(matrix& mat1, matrix& mat2) {
		matrix tmp(mat1.mat_RC[0], mat1.mat_RC[1]);
		if ((mat1.mat_RC[0] != mat2.mat_RC[0]) || (mat1.mat_RC[1] != mat2.mat_RC[1])) {
			printf("matrix shape are diffarent!\n");//行列の形が違うので引けない
			return matrix(0, 0);
		}
		else {
			for (int row = 0; row < mat1.mat_RC[0]; row++) {
				for (int column = 0; column < mat1.mat_RC[1]; column++) {
					tmp.elements[row][column] = mat1.elements[row][column] - mat2.elements[row][column];
				}
			}
			return tmp;
		}
	}

	//vec1とvec2を引く
	/*std::vector<double> vector_subtract(std::vector<double>& vec1, std::vector<double>& vec2) {
		if (vec1.size() != vec2.size()) {
			return std::vector<double>();
		}
		else {
			std::vector<double> tmp(vec1.size());
			for (int i = 0; i < tmp.size(); i++) {
				tmp[i] = vec1[i] - vec2[i];
			}
			return tmp;
		}
	}*/

	//mat1とmat2を掛ける（行列の積）
	matrix matrix_multiply(matrix& mat1, matrix& mat2) {
		if ((mat1.mat_RC[1] != mat2.mat_RC[0]) && ((mat1.mat_RC[0] != mat2.mat_RC[1]))) {
			printf("can not multiply matrix!\n");
			return matrix(0,0);
		}
		else {
			matrix tmp(mat1.mat_RC[0], mat2.mat_RC[1]);
			for (int row = 0; row < (tmp.mat_RC[0]); row++) {
				for (int column = 0; column < (tmp.mat_RC[1]); column++) {
					double element = 0;
					for (int i = 0; i < (mat2.mat_RC[0]); i++) {
						element += mat1.elements[row][i] * mat2.elements[i][column];
					}
					tmp.elements[row][column] = element;
				}
			}
			return tmp;
		}
	}

	//mat1とmat2のアダマール積
	matrix Hadamard_product(matrix& mat1, matrix& mat2) {
		if ((mat1.mat_RC[0] != mat2.mat_RC[0]) || ((mat1.mat_RC[1] != mat2.mat_RC[1]))) {
			printf("matrix shape are different!\n");
			return matrix(0,0);
		}
		else {
			matrix tmp(mat1.mat_RC[0], mat2.mat_RC[1]);
			for (int row = 0; row < (tmp.mat_RC[0]); row++) {
				for (int column = 0; column < (tmp.mat_RC[1]); column++) {
					tmp.elements[row][column] = mat1.elements[row][column] * mat2.elements[row][column];
				}
			}
			return tmp;
		}
	}

	//mat1とmat2のアダマール除算
	matrix Hadamard_division(matrix& mat1, matrix& mat2) {
		if ((mat1.mat_RC[0] != mat2.mat_RC[0]) || ((mat1.mat_RC[1] != mat2.mat_RC[1]))) {
			printf("matrix shape are different!\n");
			return matrix(0, 0);
		}
		else {
			matrix tmp(mat1.mat_RC[0], mat2.mat_RC[1]);
			for (int row = 0; row < (tmp.mat_RC[0]); row++) {
				for (int column = 0; column < (tmp.mat_RC[1]); column++) {
					tmp.elements[row][column] = mat1.elements[row][column] / mat2.elements[row][column];
				}
			}
			return tmp;
		}
	}

	//vec1とmat1を掛ける
	std::vector<double> vector_multiply(std::vector<double>& vec1, matrix& mat1) {
		if (mat1.mat_RC[0] != vec1.size()) {
			printf("|!|can't multiply the vector and the matrix|!|\n");
			return std::vector<double>();
		}
		else {
			std::vector<double> tmp(mat1.mat_RC[1]);
			for (int column = 0; column < mat1.mat_RC[1]; column++) {
				double element = 0;
				for (int i = 0; i < (mat1.mat_RC[0]); i++) {
					element += vec1[i] * mat1.elements[i][column];
				}
				tmp[column] = element;
			}
			return tmp;
		}
	}

	//matを転置する
	matrix matrix_transpose(matrix& mat) {
		matrix tmp(mat.mat_RC[1], mat.mat_RC[0]);
		for (int row = 0; row < (tmp.mat_RC[0]); row++) {
			for (int column = 0; column < (tmp.mat_RC[1]); column++) {
				tmp.elements[row][column] = mat.elements[column][row];
			}
		}
		return tmp;
	}

	//vecを1行の行列にする
	matrix vector_to_matrix(std::vector<double>& vec) {
		matrix tmp(1, vec.size());
		for (int column = 0; column < (tmp.mat_RC[1]); column++) {
			tmp.elements[0][column] = vec[column];
		}
		return tmp;
	}

	//vecをスカラー倍する
	void vector_multiscalar(std::vector<double>& vec, double scalar) {
		for (int i = 0;i < vec.size();i++) {
			vec[i] *= scalar;
		}
	}

	double matrix::average() {
		uint16_t element_num = mat_RC[0] * mat_RC[1];
		double ave = 0.;
		for (int row = 0; row < mat_RC[0];row++) {
			for (int col = 0; col < mat_RC[1];col++) {
				ave += elements[row][col] / element_num;
			}
		}
		return ave;
	}

	double matrix:: variance() {
		uint16_t element_num = mat_RC[0] * mat_RC[1];
		double ave = average();
		double dist = 0.;
		for (int row = 0; row < mat_RC[0]; row++) {
			for (int col = 0; col < mat_RC[1]; col++) {
				dist += powf(elements[row][col] - ave,2) / (float)element_num;
			}
		}
		return dist;
	}
}

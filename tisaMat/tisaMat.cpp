#include "tisaMat.h"
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
				std::cout << std::setw(8) << elements[row][column] << ' ';
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
	matrix* matrix_add(matrix& mat1, matrix& mat2) {
		matrix* tmp = new matrix(mat1.mat_RC[0], mat1.mat_RC[1]);
		if (tmp == nullptr) {
			std::cout << "preparation faile" << "\n";
			return tmp;
		}
		else {
			if ((mat1.mat_RC[0] != mat2.mat_RC[0]) || (mat1.mat_RC[1] != mat2.mat_RC[1])) {
				return nullptr;//行列の形が違うので足せない
			}
			else {
				for (int row = 0; row < mat1.mat_RC[0]; row++) {
					for (int column = 0; column < mat1.mat_RC[1]; column++) {
						tmp->elements[row][column] = mat1.elements[row][column] + mat2.elements[row][column];
					}
				}
				return tmp;
			}
		}
	}
	//vec1とvec2を足す
	std::vector<double>* vector_add(std::vector<double>& vec1, std::vector<double>& vec2) {
		if (vec1.size() != vec2.size()) {
			return nullptr;
		}
		else {
			std::vector<double>* tmp = new std::vector<double>(vec1.size());
			for (int i = 0; i < (*tmp).size(); i++) {
				(*tmp)[i] = vec1[i] + vec2[i];
			}
			return tmp;
		}
	}
	//mat1からmat2を引く
	matrix* matrix_subtract(matrix& mat1, matrix& mat2) {
		matrix* tmp = new matrix(mat1.mat_RC[0], mat1.mat_RC[1]);
		if (tmp == nullptr) {
			std::cout << "preparation faile" << "\n";
			return tmp;
		}
		else {
			if ((mat1.mat_RC[0] != mat2.mat_RC[0]) || (mat1.mat_RC[1] != mat2.mat_RC[1])) {
				return nullptr;//行列の形が違うので足せない
			}
			else {
				for (int row = 0; row < mat1.mat_RC[0]; row++) {
					for (int column = 0; column < mat1.mat_RC[1]; column++) {
						tmp->elements[row][column] = mat1.elements[row][column] - mat2.elements[row][column];
					}
				}
				return tmp;
			}
		}
	}
	//vec1とvec2を引く
	std::vector<double>* vector_subtract(std::vector<double>& vec1, std::vector<double>& vec2) {
		if (vec1.size() != vec2.size()) {
			return nullptr;
		}
		else {
			std::vector<double>* tmp = new std::vector<double>(vec1.size());
			for (int i = 0; i < (*tmp).size(); i++) {
				(*tmp)[i] = vec1[i] - vec2[i];
			}
			return tmp;
		}
	}
	//mat1とmat2を掛ける（行列の積）
	matrix* matrix_multiply(matrix& mat1, matrix& mat2) {
		if ((mat1.mat_RC[1] != mat2.mat_RC[0]) && ((mat1.mat_RC[0] != mat2.mat_RC[1]))) {
			return nullptr;
		}
		else {
			matrix* tmp = new matrix(mat1.mat_RC[0], mat2.mat_RC[1]);
			if (tmp == nullptr) {
				return tmp;
			}
			else {
				for (int row = 0; row < (tmp->mat_RC[0]); row++) {
					for (int column = 0; column < (tmp->mat_RC[1]); column++) {
						double element = 0;
						for (int i = 0; i < (mat2.mat_RC[0]); i++) {
							element += mat1.elements[row][i] * mat2.elements[i][column];
						}
						tmp->elements[row][column] = element;
					}
				}
				return tmp;
			}
		}
	}

	//mat1とmat2のアダマール積
	matrix* matrix_Hadamard(matrix& mat1, matrix& mat2) {
		if ((mat1.mat_RC[0] != mat2.mat_RC[0]) || ((mat1.mat_RC[1] != mat2.mat_RC[1]))) {
			return nullptr;
		}
		else {
			matrix* tmp = new matrix(mat1.mat_RC[0], mat2.mat_RC[1]);
			if (tmp == nullptr) {
				return tmp;
			}
			else {
				for (int row = 0; row < (tmp->mat_RC[0]); row++) {
					for (int column = 0; column < (tmp->mat_RC[1]); column++) {
						tmp->elements[row][column] = mat1.elements[row][column] * mat2.elements[row][column];
					}
				}
				return tmp;
			}
		}
	}

	//vec1とmat1を掛ける
	std::vector<double>* vector_multiply(std::vector<double> vec1, matrix& mat1) {
		if (mat1.mat_RC[0] != vec1.size()) {
			printf("|!|can't multiply the vector and the matrix|!|\n");
			return nullptr;
		}
		else {
			std::vector<double>* tmp = new std::vector<double>(mat1.mat_RC[1]);
			for (int column = 0; column < mat1.mat_RC[1]; column++) {
				double element = 0;
				for (int i = 0; i < (mat1.mat_RC[0]); i++) {
					element += vec1[i] * mat1.elements[i][column];
				}
				(*tmp)[column] = element;
			}
			return tmp;
		}
	}

	//matを転置する
	matrix* matrix_transpose(matrix& const mat) {
		matrix* tmp = new matrix(mat.mat_RC[1], mat.mat_RC[0]);
		if (tmp == nullptr) {
			return tmp;
		}
		else {
			for (int row = 0; row < (tmp->mat_RC[0]); row++) {
				for (int column = 0; column < (tmp->mat_RC[1]); column++) {
					tmp->elements[row][column] = mat.elements[column][row];
				}
			}
			return tmp;
		}
	}

	//vecを1行の行列にする
	matrix* vector_to_matrix(std::vector<double>& vec) {
		matrix* tmp = new matrix(1, vec.size());
		if (tmp == nullptr) {
			return tmp;
		}
		else {
			for (int column = 0; column < (tmp->mat_RC[1]); column++) {
				tmp->elements[0][column] = vec[column];
			}
			return tmp;
		}
	}

	//vecをスカラー倍する
	void vector_multiscalar(std::vector<double>& vec, double scalar) {
		for (int i = 0;i < vec.size();i++) {
			vec[i] *= scalar;
		}
	}

	//vecを表示する
	void vector_show(std::vector<double>& vec) {
		for (int i = 0;i < vec.size();i++) {
			printf("%lf ", vec[i]);
		}
		printf("\n");
	}
}

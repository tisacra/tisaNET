#pragma once

#include <vector>

namespace tisaMat {
	class matrix {
	public:
		int mat_RC[2];
		std::vector<std::vector<double>> elements;

		//row行column列の行列を用意する(初期化はしない)
		matrix(int row, int column);

		//row行column列の行列を用意する(initで初期化)
		matrix(int row, int column,double init);

		//２次元配列から行列を作る([行][列]という形になる)
		matrix(std::vector<std::vector<double>> mat);

		//行列を表示する
		void show();

		//matをスカラー倍する
		void multi_scalar(double scalar);

		//左辺の行列に右辺の行列を代入
		void operator=(matrix mat) {
			this->mat_RC[0] = mat.mat_RC[0];
			this->mat_RC[1] = mat.mat_RC[1];
			this->elements = mat.elements;
		}
	};
	//mat1とmat2を足す
	matrix* matrix_add(matrix& mat1, matrix& mat2);

	//vec1とvec2を足す
	std::vector<double>* vector_add(std::vector<double>& vec1, std::vector<double>& vec2);

	//mat1からmat2を引く
	matrix* matrix_subtract(tisaMat::matrix& mat1, tisaMat::matrix& mat2);

	//vec1とvec2を引く
	std::vector<double>* vector_subtract(std::vector<double>& vec1, std::vector<double>& vec2);

	//mat1とmat2を掛ける（行列の積）
	matrix* matrix_multiply(matrix& mat1, matrix& mat2);

	//mat1とmat2のアダマール積
	matrix* matrix_Hadamard(matrix& mat1, matrix& mat2);

	//vec1とmat1を掛ける
	std::vector<double>* vector_multiply(std::vector<double> vec1,matrix& mat1);

	//matを転置する
	matrix* matrix_transpose(matrix& mat);

	//vecを1行の行列にする
	matrix* vector_to_matrix(std::vector<double>& vec);

	//vecをスカラー倍する
	void vector_multiscalar(std::vector<double>& vec, double scalar);
}

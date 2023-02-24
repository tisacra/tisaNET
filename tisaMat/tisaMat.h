#pragma once
#include <iostream>
#include <vector>
#include <iomanip>

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
		//型がdouble以外の時
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

		//行列を表示する
		void show();

		//matをスカラー倍する
		void multi_scalar(double scalar);

		//平均を計算する
		double average();

		//分散を計算する
		double variance();

		//左辺の行列に右辺の行列を代入
		void operator=(matrix mat) {
			this->mat_RC[0] = mat.mat_RC[0];
			this->mat_RC[1] = mat.mat_RC[1];
			this->elements = mat.elements;
		}

		//最大値を取得
		double max();
		//最小値を取得
		double min();
	};
	//mat1とmat2を足す
	matrix matrix_add(matrix& mat1, matrix& mat2);

	//vec1とvec2を足す
	std::vector<double> vector_add(std::vector<double> vec1, std::vector<double> vec2);

	//mat1からmat2を引く
	matrix matrix_subtract(tisaMat::matrix& mat1, tisaMat::matrix& mat2);

	//vec1とvec2を引く
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

	//mat1とmat2を掛ける（行列の積）
	matrix matrix_multiply(matrix& mat1, matrix& mat2);

	//mat1とmat2のアダマール積
	matrix Hadamard_product(matrix& mat1, matrix& mat2);

	//mat1とmat2のアダマール除算
	matrix Hadamard_division(matrix& mat1, matrix& mat2);

	//vec1とmat1を掛ける
	std::vector<double> vector_multiply(std::vector<double>& vec1,matrix& mat1);

	//matを転置する
	matrix matrix_transpose(matrix& mat);

	//vecを1行の行列にする
	matrix vector_to_matrix(std::vector<double>& vec);

	//vecをスカラー倍する
	void vector_multiscalar(std::vector<double>& vec, double scalar);

	//vecを正規化する
	void vector_normalization(std::vector<double>& vec);

	//vecを表示する
	template <typename T>
	void vector_show(std::vector<T>& vec) {
		for (int i = 0; i < vec.size(); i++) {
			//printf("%lf ", vec[i]);
			std::cout << std::setw(6) << vec[i] << ' ';
		}
		printf("\n");
	}

	//vecを型変換する
	template <typename T,typename U>
	std::vector<T> vector_cast(std::vector<U>& vec) {
		int size = vec.size();
		std::vector<T> tmp(size);
		for (int i = 0; i < size;i++) {
			tmp[i] = (T)vec[i];
		}
		return tmp;
	}

	//テンソルから最大値を取得
	double max(std::vector<tisaMat::matrix>& tensor);

	//テンソルから最小値を取得
	double min(std::vector<tisaMat::matrix>& tensor);
}

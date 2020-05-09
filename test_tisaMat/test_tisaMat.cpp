#include <vector>
#include <iostream>
#include <tisaMat.h>
#include <crtdbg.h>

using namespace tisaMat;

int main() {
	std::vector<std::vector<double>> d{ {1,2,3,4,5} };
	tisaMat::matrix* test1 = new tisaMat::matrix(d);

	tisaMat::matrix test2 = *test1;

	test2.show();

	tisaMat::vector_show(d[0]);
	_CrtDumpMemoryLeaks();
	return 0;
}
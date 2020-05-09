#include <random>

int main() {

	std::random_device seed_gen;
	std::default_random_engine rand_gen;

	std::normal_distribution<> dist(0.0,1.0);

	for (int i = 0; i < 50;i++) {
		printf("%lf\n",dist(rand_gen));
	}

	return 0;
}
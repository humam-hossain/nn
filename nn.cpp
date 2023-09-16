#define NN_IMPLEMENTATION

#include "nn.h"
#include <time.h>

float td_xor[] = {
	0, 0, 0,
	0, 1, 1,
	1, 0, 1,
	1, 1, 0,
};

float td_or[] = {
	0, 0, 0,
	0, 1, 1,
	1, 0, 1,
	1, 1, 1,
};

int main()
{
	srand(time(0));

	float* td = td_xor;
	size_t stride = 3;
	size_t n = 4;

	Mat ti;
	ti.rows = n;
	ti.cols = 2;
	ti.stride = stride;
	ti.es = td;

	Mat to;
	to.rows = n;
	to.cols = 1;
	to.stride = stride;
	to.es = td + 2;

	size_t arch[] = { 2, 2, 1 };
	NN nn = nn_alloc(arch, ARRAY_LEN(arch));
	NN g = nn_alloc(arch, ARRAY_LEN(arch));
	nn_rand(nn);

	float eps = 1e-1;
	float rate = 1e-1;

	for (size_t i = 0; i < 50 * 1000; ++i) {
		nn_finite_diff(nn, g, eps, ti, to);
		nn_learn(nn, g, rate);
	}

	NN_PRINT(nn, 4);
	for (size_t i = 0; i < 2; ++i) {
		for (size_t j = 0; j < 2; ++j) {
			MAT_AT(NN_INPUT(nn), 0, 0) = i;
			MAT_AT(NN_INPUT(nn), 0, 1) = j;
			nn_forward(nn);

			printf("%zu ^ %zu = %f\n", i, j, MAT_AT(NN_OUTPUT(nn), 0, 0));
		}
	}

	return 0;
}

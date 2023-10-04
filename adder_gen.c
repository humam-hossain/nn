#if 0
#define NN_IMPLEMENTATION
#include "nn.h"

#define BITS 4

int main()
{
	size_t n = (1 << BITS);
	size_t rows = n * n;
	size_t cols = 2 * BITS;
	
	Mat t = mat_alloc(rows, 2 * BITS + BITS + 1);
	
	Mat ti = mat_alloc(rows, 2 * BITS);
	ti.es = &MAT_AT(t, 0, 0);
	ti.rows = t.rows;
	ti.cols = 2 * BITS;
	ti.stride = t.stride;
	
	Mat to = mat_alloc(rows, BITS + 1);
	to.es = &MAT_AT(t, 0, 2 * BITS);
	to.rows = t.rows;
	to.cols = BITS + 1;
	to.stride = t.stride;

	for (size_t i = 0; i < rows; ++i) {
		size_t x = i / n;
		size_t y = i % n;
		size_t z = x + y;
		for (size_t j = 0; j < BITS; ++j) {
			MAT_AT(ti, i, j) = (x >> j) & 1;
			MAT_AT(ti, i, j + BITS) = (y >> j) & 1;
			MAT_AT(to, i, j) = (z >> j) & 1;
		}
		MAT_AT(to, i, BITS) = z >= n;
	}

	const char* out_file_path = "adder.matrix";
	FILE* out;
	errno_t err = fopen_s(&out, out_file_path, "wb");

	if (err != 0) {
		fprintf(stderr, "ERROR: could not open %s\n", out_file_path);
		return 1;
	}

	//fprintf(out, "Hello, World\n");
	mat_save(out, t);
	fclose(out);
	printf("Generated %s\n", out_file_path);


}

#endif
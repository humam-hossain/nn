#if 0
#define NN_IMPLEMENTATION
#include <errno.h>
#include "nn.h"

int main()
{
	Mat t = mat_alloc(4, 3);
	for (size_t i = 0; i < 2; ++i) {
		for (size_t j = 0; j < 2; ++j) {
			size_t row = i * 2 + j;
			MAT_AT(t, row, 0) = i;
			MAT_AT(t, row, 1) = j;
			MAT_AT(t, row, 2) = i^j;
		}
	}

	const char* out_file_path = "xor.matrix";
	FILE* out;
	errno_t err = fopen_s(&out, out_file_path, "wb");
	if (err != 0) {
		fprintf(stderr, "ERROR: could not open file %s\n", out_file_path);
	}
	mat_save(out, t);
	fclose(out);

	printf("Generated %s\n", out_file_path);

	return 0;
}

#endif
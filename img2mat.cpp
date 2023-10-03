#if 0
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <errno.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define NN_IMPLEMENTATION
#include "nn.h"

char* args_shift(int* argc, char*** argv)
{
	assert(*argc > 0);
	char* result = **argv;
	(*argc) -= 1;
	(*argv) += 1;

	return result;
}

int main(int argc, char** argv)
{
	const char* program = args_shift(&argc, &argv);

	if (argc <= 0) {
		fprintf(stderr, "USAGE: %s <input>\n", program);
		fprintf(stderr, "ERROR: no input file is provided\n");

		return 1;
	}
	const char* img_file_path = args_shift(&argc, &argv);

	int img_width, img_height, img_comp;
	uint8_t *img_pixels= (uint8_t*)stbi_load(img_file_path, &img_width, &img_height, &img_comp, 0);
	if (img_pixels == NULL) {
		fprintf(stderr, "ERROR: could not read image %s\n", img_file_path);
		return 1;
	}
	if (img_comp != 1) {
		fprintf(stderr, "ERROR: %s is %d bits image. Only 8 bit grayscale images are supported\n", img_file_path, img_comp * 8);
		
		return 1;
	}

	printf("%s size %dX%d %d bits\n", img_file_path, img_width, img_height, img_comp*8);

	Mat t = mat_alloc(img_width * img_height, 3);

	for (int y = 0; y < img_height; ++y) {
		for (int x = 0; x < img_width; ++x) {
			size_t i = y * img_width + x;

			MAT_AT(t, i, 0) = (float)x / (img_width - 1);
			MAT_AT(t, i, 1) = (float)y / (img_height - 1);
			MAT_AT(t, i, 2) = img_pixels[y * img_width + x] / 255.f;
		}
	}

	MAT_PRINT(t);

	const char* out_file_path = "img.matrix";
	FILE* out;
	errno_t err = fopen_s(&out, out_file_path, "wb");

	if (err != 0) {
		fprintf(stderr, "ERROR: could not open file %s\n", out_file_path);
		return 1;
	}

	mat_save(out, t);

	printf("Generated %s from %s\n", out_file_path, img_file_path);

	return 0;
}

#endif
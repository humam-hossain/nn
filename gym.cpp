#if 0

// Gym is a GUI app that trains your NN on the data you give it.
// The idea is that it will spit out a binary file that can be then loaded up with nn.h and used in your application.

#include <stdio.h>
#include <assert.h>

#include "raylib.h"
#define SV_IMPLEMENTATION
#include "sv.h"
#define NN_IMPLEMENTATION
#include "nn.h"

#define IMG_FACTOR 80
#define IMG_WIDTH (16 * IMG_FACTOR)
#define IMG_HEIGHT (9 * IMG_FACTOR)
#define DA_INIT_CAP 256

typedef struct {
	size_t* items;
	size_t count;
	size_t capacity;
} Arch;

#define da_append(da, item) \
	do { \
		if ((da)->count >= (da)->capacity) { \
			(da)->capacity = (da)->capacity == 0 ? DA_INIT_CAP : (da)->capacity * 2; \
			(da)->items = (size_t*)realloc((da)->items, (da)->capacity*sizeof(*(da)->items)); \
			assert((da)->items != NULL && "Buy more RAM lol"); \
		} \
		(da)->items[(da)->count++] = (item); \
	} while(0) \

int main(int argc, char **argv)
{
	unsigned int buffer_len = 0;
	unsigned char *buffer = LoadFileData("adder.arch", &buffer_len);

	String_View content = sv_from_parts((const char*)buffer, buffer_len);

	Arch arch = { 0 };

	content = sv_trim_left(content);
	while (content.count > 0 && isdigit(content.data[0])) {
		size_t x = sv_chop_u64(&content);
		da_append(&arch, x);
		content = sv_trim_left(content);
	}

	NN nn = nn_alloc(arch.items, arch.count);
	nn_rand(nn);
	NN_PRINT(nn);

	InitWindow(IMG_WIDTH, IMG_HEIGHT, "gym");
	SetTargetFPS(60);

	while (!WindowShouldClose()) {

	}

	return 0;
}

#endif
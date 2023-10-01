#if 1
// Gym is a GUI app that trains your NN on the data you give it.
// The idea is that it will spit out a binary file that can be then loaded up with nn.h and used in your application.

#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <float.h>

#include "raylib.h"
#define SV_IMPLEMENTATION
#include "sv.h"
#define NN_IMPLEMENTATION
#include "nn.h"

#define IMG_FACTOR 100
#define IMG_WIDTH (16 * IMG_FACTOR)
#define IMG_HEIGHT (9 * IMG_FACTOR)
#define DA_INIT_CAP 256

typedef struct {
	size_t* items;
	size_t count;
	size_t capacity;
} Arch;

typedef struct {
	float* items;
	size_t count;
	size_t capacity;
} Cost_Plot;

#define da_append(da, item, dtype) \
	do { \
		if ((da)->count >= (da)->capacity) { \
			(da)->capacity = (da)->capacity == 0 ? DA_INIT_CAP : (da)->capacity * 2; \
			(da)->items = (dtype*)realloc((da)->items, (da)->capacity*sizeof(*(da)->items)); \
			assert((da)->items != NULL && "Buy more RAM lol"); \
		} \
		(da)->items[(da)->count++] = (item); \
	} while(0) \

void nn_render_raylib(NN nn, int rx, int ry, int rw, int rh)
{
	Color low_color = { 0xFF, 0x00, 0xFF, 0xFF };
	Color high_color = { 0x00, 0xFF, 0x00, 0xFF };

	float neuron_radius = rh*0.03;
	int layer_border_hpad = 50;
	int layer_border_vpad = 50;
	int nn_width = rw - 2 * layer_border_hpad;
	int nn_height = rh - 2 * layer_border_vpad;
	int nn_x = rx + layer_border_hpad;
	int nn_y = ry + layer_border_vpad;
	size_t arch_count = nn.count + 1;
	int layer_hpad = nn_width / arch_count;

	for (size_t l = 0; l < arch_count; ++l) {
		int layer_vpad1 = nn_height / nn.as[l].cols;
		for (size_t i = 0; i < nn.as[l].cols; ++i) {
			int cx1 = nn_x + l * layer_hpad + layer_hpad / 2;
			int cy1 = nn_y + i * layer_vpad1 + layer_vpad1 / 2;

			// neuron weight connection color
			if (l + 1 < arch_count) {
				int layer_vpad2 = nn_height / nn.as[l + 1].cols;
				for (size_t j = 0; j < nn.as[l + 1].cols; ++j) {
					int cx2 = nn_x + (l + 1) * layer_hpad + layer_hpad / 2;
					int cy2 = nn_y + j * layer_vpad2 + layer_vpad2 / 2;
					float value = sigmoidf(MAT_AT(nn.ws[l], j, i));
					high_color.a = floorf(255.f * value);
					float thick = rh * 0.004;

					Vector2 start = {cx1, cy1};
					Vector2 end = {cx2, cy2};

					DrawLineEx(start, end, thick, ColorAlphaBlend(low_color, high_color, WHITE));
				}
			}
			// neuron bias color
			if (l > 0) {
				high_color.a = floorf(255.f * sigmoidf(MAT_AT(nn.ws[l], 0, i)));
				DrawCircle(cx1, cy1, neuron_radius, ColorAlphaBlend(low_color, high_color, WHITE));
			}
			else {
				DrawCircle(cx1, cy1, neuron_radius, GRAY);
			}
		}
	}
}

void cost_plot_minmax(Cost_Plot plot, float* min, float* max)
{
	*min = FLT_MAX;
	*max = FLT_MIN;

	for (size_t i = 0; i < plot.count; ++i) {
		if (*max < plot.items[i]) *max = plot.items[i];
		if (*min > plot.items[i]) *min = plot.items[i];
	}
}

void plot_cost_raylib(Cost_Plot plot, int rx, int ry, int rw, int rh)
{
	int layer_border_hpad = 50;
	int layer_border_vpad = 50;

	float min, max;
	cost_plot_minmax(plot, &min, &max);
	
	if (min > 0) min = 0;
	size_t n = plot.count;
	if (n < 1000) n = 1000;

	Vector2 origin, x_axis, y_axis;

	origin.x = rx + layer_border_hpad;
	origin.y = ry + layer_border_vpad + rh;
	x_axis.x = origin.x + rw;
	x_axis.y = origin.y;
	y_axis.x = origin.x;
	y_axis.y = origin.y - rh;

	DrawLineEx(origin, x_axis, rh * 0.005, BLUE);
	DrawLineEx(origin, y_axis, rh * 0.005, BLUE);

	char buffer[256];
	float text_font_size = rh * 0.03;

	snprintf(buffer, sizeof(buffer), "%.1f", min);
	DrawText(buffer, origin.x - rh * 0.06, origin.y + rh * 0.03, text_font_size, WHITE);
	snprintf(buffer, sizeof(buffer), "%.2f", max);
	DrawText(buffer, origin.x - rh * 0.06, y_axis.y, text_font_size, WHITE);
	snprintf(buffer, sizeof(buffer), "%zu", n);
	DrawText(buffer, x_axis.x, origin.y + rh * 0.03, text_font_size, WHITE);

	Vector2 start, end;
	start.x = 0;
	start.y = 0;
	end.x = 0;
	end.y = 0;

	for (size_t i = 0; i+1 < plot.count; ++i) {
		start.x = rx + layer_border_hpad + (float)rw / n * i;
		start.y = ry + layer_border_vpad + (1 - (plot.items[i] - min) / (max - min)) * rh;
		end.x = rx + layer_border_hpad + (float)rw / n * (i+1);
		end.y = ry + layer_border_vpad + (1 - (plot.items[i+1] - min) / (max - min)) * rh;
		
		DrawLineEx(start, end, rh * 0.005, RED);
	}

	snprintf(buffer, sizeof(buffer), "%f", plot.items[plot.count - 1]);
	DrawText(buffer, end.x, end.y, text_font_size, WHITE);
	
	DrawText("Cost", x_axis.x/2, origin.y + rh * 0.03, text_font_size, WHITE);

}

char* args_shift(int* argc, char*** argv)
{
	assert(*argc > 0);
	char* result = **argv;
	(*argc) -= 1;
	(*argv) += 1;

	return result;
}

int main(int argc, char **argv)
{
	srand(time(0));
	// parse files from arguments
	const char* program = args_shift(&argc, &argv);
	
	if (argc <= 0) {
		fprintf(stderr, "USAGE: %s <model.arch> <model.matrix>\n", program);
		fprintf(stderr, "ERROR: no architecture file was provided\n");
		return 1;
	}
	const char* arch_file_path = args_shift(&argc, &argv);
	
	if (argc <= 0) {
		fprintf(stderr, "USAGE: %s <model.arch> <model.matrix>\n", program);
		fprintf(stderr, "ERROR: no data file was provided\n");
		return 1;
	}
	const char* data_file_path = args_shift(&argc, &argv);

	// load and parse architecture file
	unsigned int buffer_len = 0;
	unsigned char *buffer = LoadFileData(arch_file_path, &buffer_len);
	if (buffer == NULL) {
		return 1;
	}

	String_View content = sv_from_parts((const char*)buffer, buffer_len);
	Arch arch = { 0 };

	content = sv_trim_left(content);
	while (content.count > 0 && isdigit(content.data[0])) {
		size_t x = sv_chop_u64(&content);
		da_append(&arch, x, size_t);
		content = sv_trim_left(content);
		printf("%zu\n", x);
	}

	FILE* in;
	errno_t err = fopen_s(&in, data_file_path, "rb");

	if (err != 0) {
		fprintf(stderr, "ERROR: could not open %s\n", data_file_path);
		return 1;
	}

	// load and parse training data file
	Mat t = mat_load(in);
	fclose(in);

	MAT_PRINT(t);

	// TODO: can we have NN with just input?
	NN_ASSERT(arch.count > 0);
	size_t in_size = arch.items[0];
	size_t out_size = arch.items[arch.count - 1];
	NN_ASSERT(t.cols == in_size + out_size);

	Mat ti = mat_alloc(t.rows, in_size);
	ti.es = &MAT_AT(t, 0, 0);
	ti.rows = t.rows;
	ti.cols = in_size;
	ti.stride = t.stride;

	Mat to = mat_alloc(t.rows, out_size);
	to.es = &MAT_AT(t, 0, in_size);
	to.rows = t.rows;
	to.cols = out_size;
	to.stride = t.stride;

	// initialize nural network
	NN nn = nn_alloc(arch.items, arch.count);
	NN g = nn_alloc(arch.items, arch.count);
	nn_rand(nn);
	NN_PRINT(nn);
	float rate = 0.5;

	// cost
	Cost_Plot plot = { 0 };
	
	// training neural network and visualization
	SetConfigFlags(FLAG_WINDOW_RESIZABLE);
	InitWindow(IMG_WIDTH, IMG_HEIGHT, "gym");
	SetTargetFPS(60);

	size_t epoch = 0;
	size_t max_epoch = 5000;
	while (!WindowShouldClose()) {
		for (size_t i = 0; i < 10 && epoch < max_epoch; ++i) {
			nn_backprop(nn, g, ti, to);
			nn_learn(nn, g, rate);
			++epoch;
			da_append(&plot, nn_cost(nn, ti, to), float);
		}

		BeginDrawing();
		Color background_color = { 0x18, 0x18, 0x18, 0xFF };
		ClearBackground(background_color);
		{
			int rw, rh, rx, ry;
			int w = GetRenderWidth();
			int h = GetRenderHeight();

			rw = w / 2;
			rh = h * 2 / 3;
			rx = 0;
			ry = h / 2 - rh / 2;
			plot_cost_raylib(plot, rx, ry, rw, rh);

			rw = w / 2;
			rh = h * 2 / 3;
			rx = w - rw;
			ry = h / 2 - rh/2;
			nn_render_raylib(nn, rx, ry, rw, rh);

			char buffer[256];
			snprintf(buffer, sizeof(buffer), "Epoch: %zu/%zu, Rate: %f", epoch, max_epoch, rate);
			DrawText(buffer, 0, 0, h * 0.04, WHITE);
		}
		EndDrawing();
	}

	return 0;
}

#endif
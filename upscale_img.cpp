#if 1
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <time.h>
#include <errno.h>
#include <float.h>

#include "raylib.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define NN_IMPLEMENTATION
#include "nn.h"

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

	float neuron_radius = rh * 0.03;
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

					Vector2 start = { cx1, cy1 };
					Vector2 end = { cx2, cy2 };

					DrawLineEx(start, end, thick, ColorAlphaBlend(low_color, high_color, WHITE));
				}
			}
			// neuron bias color
			if (l > 0) {
				high_color.a = floorf(255.f * sigmoidf(MAT_AT(nn.ws[l - 1], 0, i)));
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

	for (size_t i = 0; i + 1 < plot.count; ++i) {
		start.x = rx + layer_border_hpad + (float)rw / n * i;
		start.y = ry + layer_border_vpad + (1 - (plot.items[i] - min) / (max - min)) * rh;
		end.x = rx + layer_border_hpad + (float)rw / n * (i + 1);
		end.y = ry + layer_border_vpad + (1 - (plot.items[i + 1] - min) / (max - min)) * rh;

		DrawLineEx(start, end, rh * 0.005, RED);
	}

	snprintf(buffer, sizeof(buffer), "%f", plot.items[plot.count - 1]);
	DrawText(buffer, end.x, end.y, text_font_size, WHITE);

	DrawText("Cost", x_axis.x / 2, origin.y + rh * 0.03, text_font_size, WHITE);

}

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
	srand(time(0));

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

	size_t input = 2;
	size_t output = 1;
	size_t arch[] = { input, 7, 7, output };
	Mat t = mat_alloc(img_width * img_height, input + output);

	for (int y = 0; y < img_height; ++y) {
		for (int x = 0; x < img_width; ++x) {
			size_t i = y * img_width + x;

			MAT_AT(t, i, 0) = (float)x / (img_width - 1);
			MAT_AT(t, i, 1) = (float)y / (img_height - 1);
			MAT_AT(t, i, 2) = img_pixels[y * img_width + x] / 255.f;
		}
	}

	mat_shuffle_rows(t);

	NN nn = nn_alloc(arch, ARRAY_LEN(arch));
	NN g = nn_alloc(arch, ARRAY_LEN(arch));
	nn_rand(nn, -1.f, 1.f);

	clock_t start, end;
	float t_speed, est_time = 0;

	float rate = 0.5f;
	size_t max_epoch = 1e4;
	size_t epoch = 0;

	size_t batch_size = 28;
	size_t batch_count = (t.rows + batch_size - 1) / batch_size;
	size_t batches_per_frame = 100;
	size_t batch_begin = 0;

	float average_cost = 0.0f;

	// cost
	Cost_Plot plot = { 0 };

	// training neural network and visualization
	size_t IMG_FACTOR = 100;
	int IMG_WIDTH = (16 * IMG_FACTOR);
	int IMG_HEIGHT = (9 * IMG_FACTOR);

	SetConfigFlags(FLAG_WINDOW_RESIZABLE);

	char buffer[256];
	snprintf(buffer, sizeof(buffer), "%s Upscaling", img_file_path);
	InitWindow(IMG_WIDTH, IMG_HEIGHT, buffer);

	SetTargetFPS(60);

	// preview images
	int factor = 3;
	int prev_width = img_width*factor;
	int prev_height = img_height*factor;

	Image original_image = LoadImage(img_file_path);
	Texture2D original_texture = LoadTextureFromImage(original_image);
	Image preview_image = GenImageColor(prev_width, prev_height, BLACK);
	Texture2D preview_texture = LoadTextureFromImage(preview_image);

	bool paused = false;

	while (!WindowShouldClose()) {
		if (IsKeyPressed(KEY_SPACE)) {
			paused = !paused;
		}

		if (IsKeyPressed(KEY_R)) {
			epoch = 0;
			nn_rand(nn, -1.f, 1.f);
			plot.count = 0;
		}

		if (IsKeyPressed(KEY_C)) {
			max_epoch += max_epoch / 4;
		}

		if (!paused) {
			start = clock();
		}

		for (size_t i = 0; i < batches_per_frame && !paused && epoch < max_epoch; ++i) {
			size_t size = batch_size;

			if (batch_begin + batch_size >= t.rows) {
				size = t.rows - batch_begin;
			}

			Mat batch_ti;
			batch_ti.rows = size;
			batch_ti.cols = input;
			batch_ti.stride = t.stride;
			batch_ti.es = &MAT_AT(t, batch_begin, 0);


			Mat batch_to;
			batch_to.rows = size;
			batch_to.cols = output;
			batch_to.stride = t.stride;
			batch_to.es = &MAT_AT(t, batch_begin, batch_ti.cols);

			nn_backprop(nn, g, batch_ti, batch_to);
			nn_learn(nn, g, rate);
			average_cost += nn_cost(nn, batch_ti, batch_to);
			batch_begin += batch_size;

			if (batch_begin >= t.rows) {
				++epoch;
				da_append(&plot, average_cost/batch_count, float);
				average_cost = 0.0f;
				batch_begin = 0;
			}
		}

		BeginDrawing();
		Color background_color = { 0x18, 0x18, 0x18, 0xFF };
		ClearBackground(background_color);
		{
			int rw, rh, rx, ry;
			int w = GetRenderWidth();
			int h = GetRenderHeight();

			rw = w / 3;
			rh = h * 2 / 3;
			rx = 0;
			ry = h / 2 - rh / 2;
			plot_cost_raylib(plot, rx, ry, rw, rh);

			rx += rw;
			nn_render_raylib(nn, rx, ry, rw, rh);

			rx += rw;
			for (size_t y = 0; y < (size_t)prev_width; ++y) {
				for (size_t x = 0; x < (size_t)prev_height; ++x) {
					MAT_AT(NN_INPUT(nn), 0, 0) = (float)x / (prev_width - 1);
					MAT_AT(NN_INPUT(nn), 0, 1) = (float)y / (prev_height - 1);
					nn_forward(nn);
					uint8_t pixel = MAT_AT(NN_OUTPUT(nn), 0, 0) * 255.f;
					ImageDrawPixel(&preview_image, x, y, CLITERAL(Color){pixel, pixel, pixel, 255});
				}
			}

			int scale = 15;
			DrawTextureEx(original_texture, CLITERAL(Vector2) {(float)rx, (float)ry}, 0, scale, WHITE);
			UpdateTexture(preview_texture, preview_image.data);
			DrawTextureEx(preview_texture, CLITERAL(Vector2) {(float)rx, (float)(ry + prev_height*scale/factor)}, 0, scale/factor, WHITE);

			if (!paused) {
				end = clock();
				t_speed = (((float)end - (float)start) * batch_count/ (batches_per_frame * CLOCKS_PER_SEC));
				est_time = (max_epoch - epoch) * t_speed;
			}

			char buffer[256];
			snprintf(buffer, sizeof(buffer), "Epoch: %zu/%zu, Rate: %f, Est. Time: %f sec", epoch, max_epoch, rate, est_time);
			DrawText(buffer, 0, 0, h * 0.04, WHITE);
		}
		EndDrawing();
	}
	
	for (size_t y = 0; y < (size_t)img_height; ++y) {
		for (size_t x = 0; x < (size_t)img_width; ++x) {
			uint8_t pixel = img_pixels[y*img_width + x];
			if (pixel) printf("%3u ", pixel); else printf("    ");
		}
		printf("\n");
	}

	for (size_t y = 0; y < (size_t)img_height; ++y) {
		for (size_t x = 0; x < (size_t)img_width; ++x) {
			MAT_AT(NN_INPUT(nn), 0, 0) = (float)x / (img_width - 1);
			MAT_AT(NN_INPUT(nn), 0, 1) = (float)y / (img_height - 1);
			nn_forward(nn);
			uint8_t pixel = MAT_AT(NN_OUTPUT(nn), 0, 0) * 255.f;
			if (pixel) printf("%3u ", pixel); else printf("    ");
		}
		printf("\n");
	}

	size_t out_width = 512;
	size_t out_height = 512;
	uint8_t* out_pixels = (uint8_t*)malloc(sizeof(*out_pixels) * out_width * out_height);
	assert(out_pixels != NULL);

	for (size_t y = 0; y < out_height; ++y) {
		for (size_t x = 0; x < out_width; ++x) {
			MAT_AT(NN_INPUT(nn), 0, 0) = (float)x / (out_width - 1);
			MAT_AT(NN_INPUT(nn), 0, 1) = (float)y / (out_height - 1);
			nn_forward(nn);
			uint8_t pixel = MAT_AT(NN_OUTPUT(nn), 0, 0) * 255.f;
			out_pixels[y * out_width + x] = pixel;
		}
	}

	char out_file_path[256];
	snprintf(out_file_path, sizeof(out_file_path), "%s_upscaled.png", img_file_path);

	if (!stbi_write_png(out_file_path, out_width, out_height, 1, out_pixels, out_width * sizeof(*out_pixels))) {
		fprintf(stderr, "ERROR: could not save image %s\n", out_file_path);
		return 1;
	}

	printf("INFO: Generated %s from %s", out_file_path, img_file_path);

	return 0;
}

#endif
#if 1
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <time.h>
#include <errno.h>
#include <float.h>

#include "raylib.h"
#include "raymath.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define NN_IMPLEMENTATION
#include "nn.h"

#define DA_INIT_CAP 256
#define da_append(da, item, dtype) \
	do { \
		if ((da)->count >= (da)->capacity) { \
			(da)->capacity = (da)->capacity == 0 ? DA_INIT_CAP : (da)->capacity * 2; \
			(da)->items = (dtype*)realloc((da)->items, (da)->capacity*sizeof(*(da)->items)); \
			assert((da)->items != NULL && "Buy more RAM lol"); \
		} \
		(da)->items[(da)->count++] = (item); \
	} while(0) \

typedef struct {
	size_t* items;
	size_t count;
	size_t capacity;
} Arch;

typedef struct {
	float* items;
	size_t count;
	size_t capacity;
} Plot;

size_t arch[] = { 3, 14, 14, 1 };

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
				float value = sigmoidf(MAT_AT(nn.ws[l - 1], 0, i));
				high_color.a = floorf(255.f * value);
				Color circle_color = ColorAlphaBlend(low_color, high_color, WHITE);
				DrawCircle(cx1, cy1, neuron_radius, circle_color);
				char buffer[256];
				snprintf(buffer, sizeof(buffer), "%.2f", value);
				DrawText(buffer, cx1 - rh * 0.01, cy1 - rh * 0.005, rh * 0.01, BLACK);
			}
			else {
				DrawCircle(cx1, cy1, neuron_radius, GRAY);
			}
		}
	}
}

void plot_minmax(Plot plot, float* min, float* max)
{
	*min = FLT_MAX;
	*max = FLT_MIN;

	for (size_t i = 0; i < plot.count; ++i) {
		if (*max < plot.items[i]) *max = plot.items[i];
		if (*min > plot.items[i]) *min = plot.items[i];
	}
}

void plot_raylib(Plot plot, int rx, int ry, int rw, int rh)
{
	int layer_border_hpad = 50;
	int layer_border_vpad = 50;

	float min, max;
	plot_minmax(plot, &min, &max);

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
		fprintf(stderr, "USAGE: %s <image1> <image2>\n", program);
		fprintf(stderr, "ERROR: no image1 file is provided\n");

		return 1;
	}
	const char* img1_file_path = args_shift(&argc, &argv);

	if (argc <= 0) {
		fprintf(stderr, "USAGE: %s <image1> <image2>\n", program);
		fprintf(stderr, "ERROR: no image2 file is provided\n");

		return 1;
	}
	const char* img2_file_path = args_shift(&argc, &argv);

	int img1_width, img1_height, img1_comp;
	uint8_t *img1_pixels= (uint8_t*)stbi_load(img1_file_path, &img1_width, &img1_height, &img1_comp, 0);
	if (img1_pixels == NULL) {
		fprintf(stderr, "ERROR: could not read image %s\n", img1_file_path);
		return 1;
	}
	if (img1_comp != 1) {
		fprintf(stderr, "ERROR: %s is %d bits image. Only 8 bit grayscale images are supported\n", img1_file_path, img1_comp * 8);
		
		return 1;
	}

	printf("%s size %dX%d %d bits\n", img1_file_path, img1_width, img1_height, img1_comp * 8);

	int img2_width, img2_height, img2_comp;
	uint8_t *img2_pixels= (uint8_t*)stbi_load(img2_file_path, &img2_width, &img2_height, &img2_comp, 0);
	if (img2_pixels == NULL) {
		fprintf(stderr, "ERROR: could not read image %s\n", img2_file_path);
		return 1;
	}
	if (img2_comp != 1) {
		fprintf(stderr, "ERROR: %s is %d bits image. Only 8 bit grayscale images are supported\n", img1_file_path, img2_comp * 8);
		
		return 1;
	}

	printf("%s size %dX%d %d bits\n", img2_file_path, img2_width, img2_height, img2_comp*8);


	NN nn = nn_alloc(arch, ARRAY_LEN(arch));
	NN g = nn_alloc(arch, ARRAY_LEN(arch));

	Mat t = mat_alloc(img1_width * img1_height + img2_width * img2_height, NN_INPUT(nn).cols + NN_OUTPUT(nn).cols);

	for (int y = 0; y < img1_height; ++y) {
		for (int x = 0; x < img1_width; ++x) {
			size_t i = y * img1_width + x;

			MAT_AT(t, i, 0) = (float)x / (img1_width - 1);
			MAT_AT(t, i, 1) = (float)y / (img1_height - 1);
			MAT_AT(t, i, 2) = 0.0f;
			MAT_AT(t, i, 3) = img1_pixels[y * img1_width + x] / 255.f;
		}
	}

	for (int y = 0; y < img2_height; ++y) {
		for (int x = 0; x < img2_width; ++x) {
			size_t i = img1_width * img1_height +  y * img2_width + x;

			MAT_AT(t, i, 0) = (float)x / (img2_width - 1);
			MAT_AT(t, i, 1) = (float)y / (img2_height - 1);
			MAT_AT(t, i, 2) = 1.0f;
			MAT_AT(t, i, 3) = img2_pixels[y * img2_width + x] / 255.f;
		}
	}
	nn_rand(nn, -1.f, 1.f);

	clock_t start, end;
	float t_speed, est_time = 0;

	float rate = 1.0f;
	size_t max_epoch = 1e4;
	size_t epoch = 0;

	size_t batch_size = 28;
	size_t batch_count = (t.rows + batch_size - 1) / batch_size;
	size_t batches_per_frame = 100;
	size_t batch_begin = 0;

	float average_cost = 0.0f;

	// plots
	Plot cost_plot = { 0 };
	Plot time_plot = { 0 };

	// training neural network and visualization
	// creating window
	size_t WINDOW_FACTOR = 100;
	int WINDOW_WIDTH = (16 * WINDOW_FACTOR);
	int WINDOW_HEIGHT = (9 * WINDOW_FACTOR);

	SetConfigFlags(FLAG_WINDOW_RESIZABLE);
	InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Upscaling Image");
	SetTargetFPS(60);

	// preview images
	int factor = 3;
	int prev_width = img1_width*factor;
	int prev_height = img1_height*factor;

	Image original_image1 = LoadImage(img1_file_path);
	Texture2D original_texture1 = LoadTextureFromImage(original_image1);
	Image original_image2 = LoadImage(img2_file_path);
	Texture2D original_texture2 = LoadTextureFromImage(original_image2);
	
	Image preview_image1 = GenImageColor(prev_width, prev_height, BLACK);
	Texture2D preview_texture1 = LoadTextureFromImage(preview_image1);
	Image preview_image2 = GenImageColor(prev_width, prev_height, BLACK);
	Texture2D preview_texture2 = LoadTextureFromImage(preview_image2);
	
	Image preview_image = GenImageColor(prev_width, prev_height, BLACK);
	Texture2D preview_texture = LoadTextureFromImage(preview_image);

	bool paused = false;
	bool scroll_dragging = false;
	float scroll = 0.0f;

	while (!WindowShouldClose()) {
		if (IsKeyPressed(KEY_SPACE)) {
			paused = !paused;
		}

		if (IsKeyPressed(KEY_R)) {
			epoch = 0;
			nn_rand(nn, -1.f, 1.f);
			cost_plot.count = 0;
		}

		if (IsKeyPressed(KEY_I)) {
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
			batch_ti.cols = NN_INPUT(nn).cols;
			batch_ti.stride = t.stride;
			batch_ti.es = &MAT_AT(t, batch_begin, 0);


			Mat batch_to;
			batch_to.rows = size;
			batch_to.cols = NN_OUTPUT(nn).cols;
			batch_to.stride = t.stride;
			batch_to.es = &MAT_AT(t, batch_begin, batch_ti.cols);

			nn_backprop(nn, g, batch_ti, batch_to);
			nn_learn(nn, g, rate);
			average_cost += nn_cost(nn, batch_ti, batch_to);
			batch_begin += batch_size;

			if (batch_begin >= t.rows) {
				++epoch;
				da_append(&cost_plot, average_cost/batch_count, float);
				average_cost = 0.0f;
				batch_begin = 0;
				mat_shuffle_rows(t);
			}
		}

		BeginDrawing();
		Color background_color = { 0x18, 0x18, 0x18, 0xFF };
		ClearBackground(background_color);
		{
			char buffer[256];
			int rw, rh, rx, ry;
			int w = GetRenderWidth();
			int h = GetRenderHeight();

			// cost plot
			rw = w / 3;
			rh = h * 2 / 3;
			rx = 0;
			ry = h / 2 - rh / 2;
			plot_raylib(cost_plot, rx, ry, rw, rh);
			DrawText("Cost Plot", rx + 25 + rw * 0.5f, ry, rh * 0.03, WHITE);
			DrawText("Epoch -->", rx + 25 + rw * 0.5f, ry + 50 + rh + rh * 0.03, rh * 0.03, WHITE);

			// instructions render
			DrawText("Pause -> [SPACE]\t\t\t\t\tReload -> [R]\t\t\t\t\tIncrease Epoch -> [I]", rx + 25, h - 2 * h * 0.02, h * 0.02, WHITE);

			// nn render
			rx += rw;
			nn_render_raylib(nn, rx, ry, rw, rh);

			// preview image render
			rx += rw;
			for (size_t y = 0; y < (size_t)prev_width; ++y) {
				for (size_t x = 0; x < (size_t)prev_height; ++x) {
					MAT_AT(NN_INPUT(nn), 0, 0) = (float)x / (prev_width - 1);
					MAT_AT(NN_INPUT(nn), 0, 1) = (float)y / (prev_height - 1);
					MAT_AT(NN_INPUT(nn), 0, 2) = 0.0f;
					nn_forward(nn);
					uint8_t pixel = MAT_AT(NN_OUTPUT(nn), 0, 0) * 255.f;
					ImageDrawPixel(&preview_image1, x, y, CLITERAL(Color){pixel, pixel, pixel, 255});
				}
			}
			for (size_t y = 0; y < (size_t)prev_width; ++y) {
				for (size_t x = 0; x < (size_t)prev_height; ++x) {
					MAT_AT(NN_INPUT(nn), 0, 0) = (float)x / (prev_width - 1);
					MAT_AT(NN_INPUT(nn), 0, 1) = (float)y / (prev_height - 1);
					MAT_AT(NN_INPUT(nn), 0, 2) = 1.0f;
					nn_forward(nn);
					uint8_t pixel = MAT_AT(NN_OUTPUT(nn), 0, 0) * 255.f;
					ImageDrawPixel(&preview_image2, x, y, CLITERAL(Color){pixel, pixel, pixel, 255});
				}
			}
			for (size_t y = 0; y < (size_t)prev_width; ++y) {
				for (size_t x = 0; x < (size_t)prev_height; ++x) {
					MAT_AT(NN_INPUT(nn), 0, 0) = (float)x / (prev_width - 1);
					MAT_AT(NN_INPUT(nn), 0, 1) = (float)y / (prev_height - 1);
					MAT_AT(NN_INPUT(nn), 0, 2) = scroll;
					nn_forward(nn);
					uint8_t pixel = MAT_AT(NN_OUTPUT(nn), 0, 0) * 255.f;
					ImageDrawPixel(&preview_image, x, y, CLITERAL(Color){pixel, pixel, pixel, 255});
				}
			}

			float scale = rh * 0.013;

			snprintf(buffer, sizeof(buffer), "%s:", img1_file_path);
			DrawText(buffer, rx, ry - h * 0.02, h * 0.02, WHITE);
			DrawTextureEx(original_texture1, CLITERAL(Vector2) {(float)rx, (float)ry}, 0, scale, WHITE);

			snprintf(buffer, sizeof(buffer), "%s:", img2_file_path);
			DrawText(buffer, rx + original_image1.width * scale, ry - h * 0.02, h * 0.02, WHITE);
			UpdateTexture(original_texture2, original_image2.data);
			DrawTextureEx(original_texture2, CLITERAL(Vector2) {(float)rx + original_image1.width * scale, (float)ry}, 0, scale, WHITE);

			UpdateTexture(preview_texture1, preview_image1.data);
			DrawTextureEx(preview_texture1, CLITERAL(Vector2) {(float)rx, (float)(ry + prev_height * scale / factor)}, 0, scale / factor, WHITE);
			UpdateTexture(preview_texture2, preview_image2.data);
			DrawTextureEx(preview_texture2, CLITERAL(Vector2) {(float)rx + prev_width * scale / factor, (float)(ry + prev_height * scale / factor)}, 0, scale / factor, WHITE);

			DrawText("output:", rx, ry + prev_height * 2 * scale / factor, h * 0.02, WHITE);
			UpdateTexture(preview_texture, preview_image.data);
			DrawTextureEx(preview_texture, CLITERAL(Vector2) {(float)rx, (float)(ry + prev_height * 2 * scale / factor + h * 0.02)}, 0, scale / factor, WHITE);

			// slider
			Vector2 position = { (float)rx, (float)(ry + prev_height * 3 * scale / factor + h * 0.04) };
			Vector2 size = { prev_width * scale / factor, rh * 0.004f };
			float knob_radius = rh * 0.01;
			Vector2 knob_position = { rx + size.x * scroll, position.y + size.y * 0.5f };
			
			DrawRectangleV(position, size, WHITE);
			DrawCircleV(knob_position , knob_radius, RED);
			snprintf(buffer, sizeof(buffer), "%.2f", scroll);
			DrawText(buffer, rx + size.x * scroll, position.y + size.y + h * 0.01f, h * 0.005f, WHITE);

			if (scroll_dragging) {
				float x = GetMousePosition().x;

				if (x < position.x) x = position.x;
				if (x > position.x + size.x) x = position.x + size.x;
				scroll = (x - position.x) / size.x;
			}

			if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
				Vector2 mouse_position = GetMousePosition();
				if (Vector2Distance(mouse_position, knob_position) <= knob_radius) {
					scroll_dragging = true;
					printf("[DRAG] True\n");
				}
			}

			if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
				scroll_dragging = false;
				printf("[DRAG] False\n");
			}
			
			// info render
			if (!paused) {
				end = clock();
				t_speed = (((float)end - (float)start) * batch_count/ (batches_per_frame * CLOCKS_PER_SEC));
				est_time = (max_epoch - epoch) * t_speed;
				da_append(&time_plot, est_time, float);
			}

			rx += prev_width * scale / factor;
			ry += prev_height * 2 * scale / factor;
			plot_raylib(time_plot, rx, ry, prev_width * scale / factor, prev_height * scale / factor);
			DrawText("Time Plot", rx + 25 + prev_width * 0.5f * scale / factor, ry + prev_height * scale / factor * 0.03, prev_height * scale / factor * 0.05, WHITE);
			DrawText("Epoch -->", rx + 25 + prev_width * 0.5f * scale / factor, ry + 50 + prev_height * scale / factor + prev_height * scale / factor * 0.03, prev_height * scale / factor * 0.03, WHITE);

			snprintf(buffer, sizeof(buffer), "Epoch: %zu/%zu, Rate: %f, Est. Time: %f sec", epoch, max_epoch, rate, est_time);
			DrawText(buffer, 0, 0, h * 0.04, WHITE);
		}
		EndDrawing();
	}
	
	// stdout
	for (size_t y = 0; y < (size_t)img1_height; ++y) {
		for (size_t x = 0; x < (size_t)img1_width; ++x) {
			uint8_t pixel = img1_pixels[y*img1_width + x];
			if (pixel) printf("%3u ", pixel); else printf("    ");
		}
		printf("\n");
	}

	for (size_t y = 0; y < (size_t)img1_height; ++y) {
		for (size_t x = 0; x < (size_t)img1_width; ++x) {
			MAT_AT(NN_INPUT(nn), 0, 0) = (float)x / (img1_width - 1);
			MAT_AT(NN_INPUT(nn), 0, 1) = (float)y / (img1_height - 1);
			MAT_AT(NN_INPUT(nn), 0, 2) = scroll;
			nn_forward(nn);
			uint8_t pixel = MAT_AT(NN_OUTPUT(nn), 0, 0) * 255.f;
			if (pixel) printf("%3u ", pixel); else printf("    ");
		}
		printf("\n");
	}

	// generating upscaled output img
	size_t out_width = 512;
	size_t out_height = 512;
	uint8_t* out_pixels = (uint8_t*)malloc(sizeof(*out_pixels) * out_width * out_height);
	assert(out_pixels != NULL);

	for (size_t y = 0; y < out_height; ++y) {
		for (size_t x = 0; x < out_width; ++x) {
			MAT_AT(NN_INPUT(nn), 0, 0) = (float)x / (out_width - 1);
			MAT_AT(NN_INPUT(nn), 0, 1) = (float)y / (out_height - 1);
			MAT_AT(NN_INPUT(nn), 0, 2) = scroll;
			nn_forward(nn);
			uint8_t pixel = MAT_AT(NN_OUTPUT(nn), 0, 0) * 255.f;
			out_pixels[y * out_width + x] = pixel;
		}
	}

	char out_file_path[256] = "output/upscaled.png";

	if (!stbi_write_png(out_file_path, out_width, out_height, 1, out_pixels, out_width * sizeof(*out_pixels))) {
		fprintf(stderr, "ERROR: could not save image %s\n", out_file_path);
		return 1;
	}

	printf("INFO: Generated %s from %s,%s", out_file_path, img1_file_path, img2_file_path);

	return 0;
}

#endif
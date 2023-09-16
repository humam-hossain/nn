#ifndef NN_H_
#define NN_H_

#define ARRAY_LEN(xs) sizeof((xs))/sizeof((xs)[0])
#define MAT_AT(m, i, j) (m).es[(i)*(m).stride + (j)]

#include <stdio.h>
#include <math.h>

#ifndef NN_MALLOC
#include <stdlib.h>
#define NN_MALLOC malloc
#endif // NN_MALLOC

#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT assert
#endif

float rand_float();
float sigmoidf(float x);

// Matrices
typedef struct {
	size_t rows;
	size_t cols;
	size_t stride;
	float* es;
} Mat;

Mat mat_alloc(size_t rows, size_t cols);
void mat_fill(Mat a, float x);
void mat_rand(Mat m, float low = 0, float high = 1);
Mat mat_row(Mat m, size_t  row);
void mat_copy(Mat dst, Mat src);
void mat_dot(Mat dst, Mat a, Mat b);
void mat_sum(Mat dst, Mat a);
void mat_sig(Mat m);
void mat_print(Mat m, const char* name, size_t padding);
#define MAT_PRINT(m) mat_print(m, #m, 0)

// Nural Network
typedef struct {
	size_t count;
	Mat* as;
	Mat* ws;
	Mat* bs;
} NN;

#define NN_INPUT(nn) (nn).as[0]
#define NN_OUTPUT(nn) (nn).as[(nn).count]

NN nn_alloc(size_t* arch, size_t arch_count);
void nn_rand(NN nn, float low, float high);
void nn_forward(NN nn);
float nn_cost(NN nn, Mat ti, Mat to);
void nn_finite_diff(NN nn, NN g, float eps, Mat ti, Mat to);
void nn_learn(NN nn, NN g, float rate);
void nn_print(NN nn, const char* name, size_t padding);
#define NN_PRINT(nn, padding) nn_print(nn, #nn, padding);

#endif // NN_H_

#ifdef NN_IMPLEMENTATION

float rand_float()
{
	return (float)rand() / (float)RAND_MAX;
}

float sigmoidf(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

// Matrices
Mat mat_alloc(size_t rows, size_t cols)
{
	Mat m;
	m.rows = rows;
	m.cols = cols;
	m.stride = cols;
	m.es = (float*)NN_MALLOC(sizeof(*m.es) * rows * cols);
	NN_ASSERT(m.es != nullptr);
	return m;
}

void mat_fill(Mat m, float x)
{
	for (size_t i = 0; i < m.rows; ++i) {
		for (size_t j = 0; j < m.cols; ++j) {
			MAT_AT(m, i, j) = x;
		}
	}
}

void mat_rand(Mat m, float low, float high)
{
	for (size_t i = 0; i < m.rows; ++i) {
		for (size_t j = 0; j < m.cols; ++j) {
			MAT_AT(m, i, j) = rand_float() * (high - low) + low;
		}
	}
}

Mat mat_row(Mat m, size_t  row)
{
	Mat r;
	r.rows = 1;
	r.cols = m.cols;
	r.stride = m.stride;
	r.es = &MAT_AT(m, row, 0);

	return r;
}

void mat_copy(Mat dst, Mat src)
{
	NN_ASSERT(dst.rows == src.rows);
	NN_ASSERT(dst.cols == src.cols);

	for (size_t i = 0; i < dst.rows; ++i) {
		for (size_t j = 0; j < dst.cols; ++j) {
			MAT_AT(dst, i, j) = MAT_AT(src, i, j);
		}
	}
}

void mat_dot(Mat dst, Mat a, Mat b)
{
	NN_ASSERT(a.cols == b.rows);
	NN_ASSERT(dst.rows == a.rows);
	NN_ASSERT(dst.cols == b.cols);

	for (size_t i = 0; i < dst.rows; ++i) {
		for (size_t j = 0; j < dst.cols; ++j) {
			MAT_AT(dst, i, j) = 0;
			for (size_t k = 0; k < a.cols; ++k) {
				MAT_AT(dst, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
			}
		}
	}
}

void mat_sum(Mat dst, Mat a)
{
	NN_ASSERT(dst.rows == a.rows);
	NN_ASSERT(dst.cols == a.cols);

	for (size_t i = 0; i < dst.rows; ++i) {
		for (size_t j = 0; j < dst.cols; ++j) {
			MAT_AT(dst, i, j) += MAT_AT(a, i, j);
		}
	}
}

void mat_sig(Mat m)
{
	for (size_t i = 0; i < m.rows; ++i) {
		for (size_t j = 0; j < m.cols; ++j) {
			MAT_AT(m, i, j) = sigmoidf(MAT_AT(m, i, j));
		}
	}
}

void mat_print(Mat m, const char* name, size_t padding)
{
	printf("%*s%s = [\n", (int)padding, "", name);
	for (size_t i = 0; i < m.rows; ++i) {
		printf("%*s\t", (int)padding, "");
		for (size_t j = 0; j < m.cols; ++j) {
			printf("%f  ", MAT_AT(m, i, j));
		}
		printf("\n");
	}
	printf("%*s]\n", (int)padding, "");
}

// Nural Network
NN nn_alloc(size_t* arch, size_t arch_count)
{
	NN_ASSERT(arch_count > 0);

	NN nn;
	nn.count = arch_count - 1;

	nn.ws = (Mat*)NN_MALLOC(sizeof(*nn.ws) * nn.count);
	NN_ASSERT(nn.ws != NULL);

	nn.bs = (Mat*)NN_MALLOC(sizeof(*nn.bs) * nn.count);
	NN_ASSERT(nn.bs != NULL);

	nn.as = (Mat*)NN_MALLOC(sizeof(*nn.as) * (nn.count + 1));
	NN_ASSERT(nn.as != NULL);

	nn.as[0] = mat_alloc(1, arch[0]);

	for (size_t i = 1; i < arch_count; ++i) {
		nn.ws[i - 1] = mat_alloc(nn.as[i - 1].cols, arch[i]);
		nn.bs[i - 1] = mat_alloc(nn.as[i - 1].rows, nn.ws[i - 1].cols);
		nn.as[i] = mat_alloc(nn.as[i - 1].rows, nn.ws[i - 1].cols);
	}

	return nn;
}

void nn_rand(NN nn, float low = 0, float high = 1)
{
	for (size_t i = 0; i < nn.count; ++i) {
		mat_rand(nn.ws[i], low, high);
		mat_rand(nn.bs[i], low, high);
	}
}

void nn_forward(NN nn)
{
	for (size_t i = 0; i < nn.count; ++i) {
		mat_dot(nn.as[i + 1], nn.as[i], nn.ws[i]);
		mat_sum(nn.as[i + 1], nn.bs[i]);
		mat_sig(nn.as[i + 1]);
	}
}

float nn_cost(NN nn, Mat ti, Mat to)
{
	NN_ASSERT(ti.rows == to.rows);
	NN_ASSERT(to.cols == NN_OUTPUT(nn).cols);

	size_t n = ti.rows;

	float c = 0;
	for (size_t i = 0; i < n; ++i) {
		Mat x = mat_row(ti, i);
		Mat y = mat_row(to, i);

		mat_copy(NN_INPUT(nn), x);
		nn_forward(nn);

		size_t q = to.cols;
		for (size_t j = 0; j < q; ++j) {
			float d = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(y, 0, j);
			c += d * d;
		}
	}

	return c / n;
}

void nn_finite_diff(NN nn, NN g, float eps, Mat ti, Mat to)
{
	float saved;
	float c = nn_cost(nn, ti, to);

	for (size_t i = 0; i < nn.count; ++i) {
		for (size_t j = 0; j < nn.ws[i].rows; ++j) {
			for (size_t k = 0; k < nn.ws[i].cols; ++k) {
				saved = MAT_AT(nn.ws[i], j, k);
				MAT_AT(nn.ws[i], j, k) += eps;
				MAT_AT(g.ws[i], j, k) = (nn_cost(nn, ti, to) - c) / eps;
				MAT_AT(nn.ws[i], j, k) = saved;
			}
		}

		for (size_t j = 0; j < nn.bs[i].rows; ++j) {
			for (size_t k = 0; k < nn.bs[i].cols; ++k) {
				saved = MAT_AT(nn.bs[i], j, k);
				MAT_AT(nn.bs[i], j, k) += eps;
				MAT_AT(g.bs[i], j, k) = (nn_cost(nn, ti, to) - c) / eps;
				MAT_AT(nn.bs[i], j, k) = saved;
			}
		}
	}
}

void nn_learn(NN nn, NN g, float rate)
{
	for (size_t i = 0; i < nn.count; ++i) {
		for (size_t j = 0; j < nn.ws[i].rows; ++j) {
			for (size_t k = 0; k < nn.ws[i].cols; ++k) {
				MAT_AT(nn.ws[i], j, k) -= rate * MAT_AT(g.ws[i], j, k);
			}
		}

		for (size_t j = 0; j < nn.bs[i].rows; ++j) {
			for (size_t k = 0; k < nn.bs[i].cols; ++k) {
				MAT_AT(nn.bs[i], j, k) -= rate * MAT_AT(g.bs[i], j, k);
			}
		}
	}
}

void nn_print(NN nn, const char* name, size_t padding)
{
	char buf[256];
	printf("%s = [\n", name);
	for (size_t i = 0; i < nn.count; ++i) {
		snprintf(buf, sizeof(buf), "ws%zu", i);
		mat_print(nn.ws[i], buf, padding);
		snprintf(buf, sizeof(buf), "bs%zu", i);
		mat_print(nn.bs[i], buf, padding);
	}
	printf("]\n");
}


#endif // NN_IMPLEMENTATION
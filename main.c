#include <stdio.h>
#include <cblas.h>
#include <assert.h>
#include <math.h>
#include <memory.h>
#include <stdlib.h>

#define array_foreach(p_element, array, count)  \
    for (typeof(*array) *p_element = (array); p_element != (array) + (count); p_element++)

#define typeof_first(first, ...)        typeof(first)
#define array_literal(...)              ((typeof_first(__VA_ARGS__)[]) {__VA_ARGS__})
#define array_literal_type(type, ...)   ((type[]) {__VA_ARGS__})
#define array_length(arr)               ((size_t) (sizeof(arr) / sizeof(arr[0])))
#define args_count(...)                 (sizeof(array_literal(__VA_ARGS__)) / sizeof(typeof_first(__VA_ARGS__)))

float random_float(float min, float max) {
    const float scale = (float) rand() / (float) RAND_MAX;  /* [0, 1.0] */ // NOLINT
    return min + scale * (max - min);                       /* [min, max] */
}

typedef struct Mat {
    float *data;
    size_t rows;
    size_t cols;
} Mat;

Mat mat_new(size_t rows, size_t cols) {
    float *data = (float *) calloc(rows * cols, sizeof(*data));
    if (NULL == data) {
        perror("mat_new");
        exit(EXIT_FAILURE);
    }

    return (Mat) {
            data,
            .rows = rows,
            .cols = cols,
    };
}

void mat_free(Mat *m) {
    free(m->data);
    m->data = NULL;
    m->cols = m->rows = 0;
}

float *mat_at(Mat m, size_t row, size_t col) {
    assert(0 <= row && row < m.rows);
    assert(0 <= col && col < m.cols);

    return m.data + (row * m.cols + col);
}

typedef struct Vec {
    float *data;
    size_t size;
} Vec;

Vec vec_new(size_t size) {
    float *data = (float *) calloc(size, sizeof(*data));
    if (NULL == data) {
        perror("vec_new");
        exit(EXIT_FAILURE);
    }

    return (Vec) {
            data,
            .size = size,
    };
}

#define vec_of(...)                                     \
({                                                      \
    const float values[] = {__VA_ARGS__};               \
    const size_t count = array_length(values);          \
    Vec tmp = vec_new(count);                           \
    for (size_t i = 0; i < count; i++) {                \
        tmp.data[i] = values[i];                        \
    }                                                   \
    tmp;                                                \
})

void vec_free(Vec *v) {
    free(v->data);
    v->data = NULL;
    v->size = 0;
}

void mat_dot(Mat a, Mat b, Mat c) {
    assert(a.rows == c.rows);
    assert(a.cols == b.rows);
    assert(b.cols == c.cols);

    cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans, CblasNoTrans,
            (int) a.rows, (int) b.cols, (int) a.cols,
            1.0f, a.data, (int) /*a.stride*/ a.cols,
            b.data, (int) /*b.stride*/ b.cols, 0.0f,
            c.data, (int) /*c.stride*/ c.cols
    );
}

Mat vec_as_mat_row(Vec v) {
    return (Mat) {
            v.data,
            .rows = 1,
            .cols = v.size,
//            .stride = v.size
    };
}

Mat vec_as_mat_col(Vec v) {
    return (Mat) {
            v.data,
            .rows = v.size,
            .cols = 1,
//            .stride = 1
    };
}

Vec mat_as_vec(Mat m) {
    return (Vec) {
            m.data,
            .size = m.rows * m.cols
    };
}

void mat_vec_dot(Mat m, Vec v, Vec out) {
    mat_dot(m, vec_as_mat_col(v), vec_as_mat_col(out));
}

__attribute__((unused)) float vec_dot(Vec a, Vec b) {
    assert(a.size == b.size);
    return cblas_sdot((int) a.size, a.data, 1, b.data, 1);
}

void vec_add(Vec out, Vec a) {
    assert(a.size == out.size);
    cblas_saxpy((int) a.size, 1.0f, a.data, 1, out.data, 1);
}

__attribute__((unused)) void vec_copy(Vec from, Vec to) {
    assert(from.size == to.size);

    memcpy(to.data, from.data, from.size * sizeof(from.data[0]));
}

void mat_print(Mat m) {
    for (size_t row = 0; row < m.rows; row++) {
        printf("%s[", 0 == row ? "[" : " ");
        for (size_t col = 0; col + 1 < m.cols; col++) {
            printf("%.2f\t", *mat_at(m, row, col));
        }

        printf("%.2f]%s\n", *mat_at(m, row, m.cols - 1), m.rows - 1 == row ? "]" : "");
    }
}

typedef void (*VecFunction)(Vec);

typedef struct Linear {
    Mat weights;
    Mat weight_grad;
    Vec biases;
    Vec bias_grad;

    Vec output;

    VecFunction activation_function;
} Linear;

Linear linear_new(size_t inputs_count, size_t outputs_count, VecFunction act) {
    assert(inputs_count > 0);
    assert(outputs_count > 0);
    assert(NULL != act);

    return (Linear) {
            .weights = mat_new(outputs_count, inputs_count),
            .weight_grad = mat_new(outputs_count, inputs_count),
            .biases = vec_new(outputs_count),
            .bias_grad = vec_new(outputs_count),
            .output = vec_new(outputs_count),
            .activation_function = act,
    };
}

void linear_free(Linear *layer) {
    mat_free(&layer->weights);
    mat_free(&layer->weight_grad);

    vec_free(&layer->biases);
    vec_free(&layer->bias_grad);

    vec_free(&layer->output);

    layer->activation_function = NULL;
}

void linear_init_weights(Linear layer) {
    Vec layer_weights = mat_as_vec(layer.weights);
    for (size_t i = 0; i < layer_weights.size; i++) {
        layer_weights.data[i] = random_float(-1.0f, 1.0f);
    }

    for (size_t i = 0; i < layer.biases.size; i++) {
        layer.biases.data[i] = random_float(-1.0f, 1.0f);
    }
}

void linear_forward(Linear layer, Vec x) {
    mat_vec_dot(layer.weights, x, layer.output);
    vec_add(layer.output, layer.biases);

    layer.activation_function(layer.output);
}

void identity(Vec in) { (void) in; }

void sigmoid(Vec in) {
    for (size_t i = 0; i < in.size; i++) {
        in.data[i] = 1.0f / (1.0f + expf(-in.data[i]));
    }
}

typedef struct MLPRegressor {
    Linear *layers;
    size_t layers_count;
} MLPRegressor;

MLPRegressor mlp_new(
        const size_t *sizes,
        size_t sizes_count
) {
    assert(sizes_count > 1);
    if (NULL == sizes) {
        fprintf(stderr, "Argument is NULL");
        exit(EXIT_FAILURE);
    }

    const size_t layers_count = sizes_count - 1;

    Linear *layers = calloc(layers_count, sizeof(Linear));
    if (NULL == layers) {
        perror("mlp_new");
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < layers_count; i++) {
        layers[i] = linear_new(
                sizes[i],
                sizes[i + 1],
                i + 1 == layers_count ? identity : sigmoid
        );
    }

    return (MLPRegressor) {
            .layers = layers,
            .layers_count = layers_count
    };
}

void mlp_free(MLPRegressor *model) {
    array_foreach(layer, model->layers, model->layers_count) {
        linear_free(layer);
    }
    free(model->layers);
    model->layers = NULL;
    model->layers_count = 0;
}

void mlp_init_weights(MLPRegressor model) {
    array_foreach(layer, model.layers, model.layers_count) {
        linear_init_weights(*layer);
    }
}

Vec mlp_forward(MLPRegressor model, Vec x) {
    assert(model.layers_count > 0);

    Vec layer_in = x;

    array_foreach(layer, model.layers, model.layers_count) {
        linear_forward(*layer, layer_in);
        layer_in = layer->output;
    }

    return model.layers[model.layers_count - 1].output;
}

#define EPS 1e-5f

void mlp_approx_grads(MLPRegressor model, Vec x, Vec y_true, float (*cost_fn)(Vec, Vec)) {
    array_foreach(layer, model.layers, model.layers_count) {
        const float cost_initial = cost_fn(y_true, mlp_forward(model, x));

        Vec flat_weights = mat_as_vec(layer->weights);
        Vec flat_weight_grad = mat_as_vec(layer->weight_grad);

        for (size_t i = 0; i < flat_weights.size; i++) {
            const float initial_value = flat_weights.data[i];
            flat_weights.data[i] += EPS;

            const float cost_adjusted = cost_fn(y_true, mlp_forward(model, x));

            flat_weights.data[i] = initial_value;
            flat_weight_grad.data[i] = (cost_adjusted - cost_initial) / EPS;
        }

        for (size_t i = 0; i < layer->biases.size; i++) {
            const float initial_value = layer->biases.data[i];
            layer->biases.data[i] += EPS;

            const float cost_adjusted = cost_fn(y_true, mlp_forward(model, x));

            layer->biases.data[i] = initial_value;
            layer->bias_grad.data[i] = (cost_adjusted - cost_initial) / EPS;
        }
    }
}

void mlp_sgd_step(MLPRegressor model, float lr) {
    array_foreach(layer, model.layers, model.layers_count) {
        Vec flat_weights = mat_as_vec(layer->weights);
        Vec flat_weight_grad = mat_as_vec(layer->weight_grad);

        for (size_t i = 0; i < flat_weights.size; i++) {
            flat_weights.data[i] -= lr * flat_weight_grad.data[i];
        }

        for (size_t i = 0; i < layer->biases.size; i++) {
            layer->biases.data[i] -= lr * layer->bias_grad.data[i];
        }
    }
}

#define mlp_with_layers(inputs_count, ...)           \
    mlp_new(                                        \
        array_literal_type(                         \
            size_t,                                 \
            (inputs_count), __VA_ARGS__             \
        ),                                          \
        args_count((inputs_count), __VA_ARGS__)    \
    )

float mse(Vec a, Vec b) {
    assert(a.size == b.size);

    float squares_sum = 0;
    for (size_t i = 0; i < a.size; i++) {
        const float diff = a.data[i] - b.data[i];
        squares_sum += diff * diff;
    }

    return squares_sum / (float) a.size;
}

int main() {
    srand(69);  // NOLINT
    MLPRegressor model = mlp_with_layers(2, 2, 1);
    mlp_init_weights(model);

    Vec x_train[] = {
            vec_of(0.0f, 0.0f),
            vec_of(0.0f, 1.0f),
            vec_of(1.0f, 0.0f),
            vec_of(1.0f, 1.0f),
    };
    Vec y_train[] = {
            vec_of(0.0f),
            vec_of(1.0f),
            vec_of(1.0f),
            vec_of(0.0f),
    };
    assert(array_length(x_train) == array_length(y_train));

    const float samples_count = (float) array_length(x_train);
    const size_t epochs = 1000;
    const float lr = 0.1f;

    for (size_t epoch = 1; epoch <= epochs; epoch++) {
        for (size_t i = 0; i < array_length(x_train); i++) {
            mlp_approx_grads(model, x_train[i], y_train[i], mse);
            mlp_sgd_step(model, lr);
        }

        float mse_sum = 0.0f;
        for (size_t i = 0; i < array_length(x_train); i++) {
            mse_sum += mse(y_train[i], mlp_forward(model, x_train[i]));
        }
        printf("[epoch = %d] avg. mse = %.2f\n", (int) epoch, mse_sum / samples_count);
    }

    array_foreach(x, x_train, array_length(x_train)) {
        printf("x_train=");
        mat_print(vec_as_mat_row(*x));
        printf("y_pred=");
        mat_print(vec_as_mat_row(mlp_forward(model, *x)));
    }

    array_foreach(v, x_train, array_length(x_train)) { vec_free(v); }
    array_foreach(v, y_train, array_length(y_train)) { vec_free(v); }
    mlp_free(&model);

    return EXIT_SUCCESS;
}

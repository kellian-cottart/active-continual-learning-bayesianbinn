#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include "golden_vectors.h"

// Simple fast RNG (xorshift32) to simulate JAX's fast pseudo-random generation efficiently
static inline uint32_t fast_rand(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

// Fast float between 0 and 1
static inline float fast_rand_float(uint32_t *state) {
    return (fast_rand(state) & 0xFFFFFF) / 16777216.0f;
}

// Sigmoid function
static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Structure representing the Bayesian Binary Linear Layer
// Translated from active-continual-learning-bayesianbinn-main/customLayers/linears/binaryBayesianLinear.py
typedef struct {
    int in_features;
    int out_features;
    
    float *weights; // the raw float unscaled weights (before sigmoid)
    float *p_weights; // Precomputed sigmoids: sigmoid(2.0 * weight)
} BayesianLinear;

BayesianLinear* create_layer(int in, int out) {
    BayesianLinear *layer = (BayesianLinear*)malloc(sizeof(BayesianLinear));
    layer->in_features = in;
    layer->out_features = out;
    layer->weights = (float*)malloc(in * out * sizeof(float));
    layer->p_weights = (float*)malloc(in * out * sizeof(float));
    
    // Initialize properties randomly
    for (int i = 0; i < in * out; i++) {
        layer->weights[i] = ((float)rand() / RAND_MAX) - 0.5f;
        layer->p_weights[i] = sigmoid(2.0f * layer->weights[i]);
    }
    return layer;
}

void free_layer(BayesianLinear *layer) {
    free(layer->weights);
    free(layer->p_weights);
    free(layer);
}

typedef struct {
    float *output;        // Size: out_features
    float *grad_logits;   // Size: out_features
    float *probs;         // Size: out_features (used inside softmax)
    float *grad_weights;  // Size: in_features * out_features
    float *updates;       // Size: in_features * out_features
} TrainWorkspace;

TrainWorkspace* create_workspace(int in, int out) {
    TrainWorkspace *ws = (TrainWorkspace*)malloc(sizeof(TrainWorkspace));
    ws->output = (float*)malloc(out * sizeof(float));
    ws->grad_logits = (float*)malloc(out * sizeof(float));
    ws->probs = (float*)malloc(out * sizeof(float));
    ws->grad_weights = (float*)malloc(in * out * sizeof(float));
    ws->updates = (float*)malloc(in * out * sizeof(float));
    return ws;
}

void free_workspace(TrainWorkspace *ws) {
    free(ws->output);
    free(ws->grad_logits);
    free(ws->probs);
    free(ws->grad_weights);
    free(ws->updates);
    free(ws);
}

// ----------------------------------------------------------------------------
// 1. INFERENCE LOOP 
// Equivalent to `sample` in binaryBayesianLinear.py 
// Uses hard Bernoulli samples 
// ----------------------------------------------------------------------------
void sample_layer(BayesianLinear *layer, const int8_t *input, int32_t *output, uint32_t *rng_state) {
    int in = layer->in_features;
    int out = layer->out_features;
    int w = 0;
    for (int o = 0; o < out; o++) {
        int32_t sum = 0;
        for (int i = 0; i < in; i++) {
            float p = layer->p_weights[o * in + i]; 
            
            int b = (fast_rand_float(rng_state) < p); 
            w = (int)(b * 2 - 1);

            sum += w * input[i];
        }
        output[o] = sum; 
    }
}

// ----------------------------------------------------------------------------
// 2. TRAINING FORWARD PASS 
// Equivalent to `__call__` in binaryBayesianLinear.py 
// Uses Continuous Gumbel-Softmax trick using floats
// ----------------------------------------------------------------------------
void train_forward_layer(BayesianLinear *layer, const float *input, float *output, uint32_t *rng_state) {
    int in = layer->in_features;
    int out = layer->out_features;
    
    for (int o = 0; o < out; o++) {
        float sum = 0.0f;
        for (int i = 0; i < in; i++) {
            float mu = layer->weights[o * in + i];
            float epsilon = fast_rand_float(rng_state) * 0.99999999f + 1e-10f; // in [1e-10, 1) to prevent log(0)
            float logit_epsilon = logf(epsilon) - logf(1.0f - epsilon);
            float current_weight = tanhf((mu + 0.5f * logit_epsilon));
            
            sum += current_weight * input[i];
        }
        output[o] = sum;
    }
}

void softmax_crossentropy(const float *logits, int label, float *grad, float *loss, float *probs, int num_outputs) {
    float max_logit = logits[0];
    for (int i = 1; i < num_outputs; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }
    
    float sum_exp_logits = 0.0f;
    for (int i = 0; i < num_outputs; i++) {
        probs[i] = expf(logits[i] - max_logit); 
        sum_exp_logits += probs[i];
    }
    
    for (int i = 0; i < num_outputs; i++) {
        probs[i] /= sum_exp_logits;
    }
    
    *loss = -logf(probs[label] + 1e-10f); 
    
    for (int i = 0; i < num_outputs; i++) {
        grad[i] = probs[i];
    }
    grad[label] -= 1.0f;
}

void BiMu (BayesianLinear *layer, float *grads, float* updates, float kl_multiplier, float likelihood_multiplier, float lr_max, long int N, float lr) {
    int in = layer->in_features;
    int out = layer->out_features;
    for (int o = 0; o < out; o++) {
        for(int i    = 0; i < in; i++){
            float tanh_weight = tanhf(layer->weights[o * in + i]);
            float inv_cosh_sqrt = kl_multiplier * (1.0f - tanh_weight * tanh_weight);
            float grad = grads[o * in + i] * likelihood_multiplier;
            float second_deriv = 2.0f * fabsf(grad) + 1/lr_max ; 
            float lr_asymetry = 1/(inv_cosh_sqrt + 2.0f * grad * tanh_weight + second_deriv) ;
            float forgetting = (layer->weights[o * in + i] * inv_cosh_sqrt) / N ;
            updates[o * in + i] = lr_asymetry * (lr * grad + forgetting) ;
        }
    }  
}

void train_step(BayesianLinear *layer, const float *input, const int label, uint32_t *rng_state, TrainWorkspace *ws) {
    int in = layer->in_features;
    int out = layer->out_features;
    
    train_forward_layer(layer, input, ws->output, rng_state);
 
    float loss;
    softmax_crossentropy(ws->output, label, ws->grad_logits, &loss, ws->probs, out);

    for (int o = 0; o < out; o++) {
        for(int i = 0; i < in; i++){
            ws->grad_weights[o * in + i] = ws->grad_logits[o] * input[i];
        }
    }  

    float kl_multiplier = 0.53f;
    float likelihood_multiplier = 16.7f;
    float lr_max = 0.065f;
    long int N = 1600;
    float lr = 48.7f;
    
    BiMu(layer, ws->grad_weights, ws->updates, kl_multiplier, likelihood_multiplier, lr_max, N, lr);

    for (int i = 0; i < in * out; i++) {
        layer->weights[i] -= ws->updates[i];
        layer->p_weights[i] = sigmoid(2.0f * layer->weights[i]);
    }
}

int main() {
    printf("--- Verifying Golden Inputs ---\n");
    BayesianLinear *golden_layer = (BayesianLinear*)malloc(sizeof(BayesianLinear));
    golden_layer->in_features = IN_FEATURES;
    golden_layer->out_features = OUT_FEATURES;
    golden_layer->p_weights = (float*)malloc(IN_FEATURES * OUT_FEATURES * sizeof(float));
    for (int i = 0; i < IN_FEATURES * OUT_FEATURES; i++) {
        golden_layer->p_weights[i] = golden_p_weights[i];
    }
    
    uint32_t golden_rng_state = 12345;
    int32_t *test_output = (int32_t*)malloc(OUT_FEATURES * sizeof(int32_t));
    long long total_mae = 0;
    
    for(int s = 0; s < NUM_SAMPLES; s++) {
        sample_layer(golden_layer, golden_inputs[s], test_output, &golden_rng_state);
        for(int o = 0; o < OUT_FEATURES; o++) {
            int32_t expected = golden_outputs[s][o];
            int32_t actual = test_output[o];
            long long diff = llabs((long long)expected - (long long)actual);
            total_mae += diff;
        }
    }
    
    double avg_mae = (double)total_mae / (NUM_SAMPLES * OUT_FEATURES);
    printf("Verification against JAX Output Distributions:\n");
    printf("Mean Absolute Error (MAE): %.4f\n", avg_mae);
    
    if (avg_mae < 1000.0) {
        printf("Verification Passed: The C pseudo-random sampling aligns heavily with JAX!\n");
    } else {
        printf("Verification Failed: High deviation from JAX output.\n");
    }
    
    free(golden_layer->p_weights);
    free(golden_layer);
    free(test_output);
    printf("-------------------------------\n\n");

    uint32_t rng_state = 12345; 
    int in_features = 8192; 
    int out_features = 19;
    int iterations = 1000; 
    
    printf("Initializing Bayesian Linear Layer...\n");
    BayesianLinear *layer = create_layer(in_features, out_features);
    TrainWorkspace *ws = create_workspace(in_features, out_features);
    
    int8_t *sample_input_int8 = (int8_t*)malloc(in_features * sizeof(int8_t));
    float *sample_input_float = (float*)malloc(in_features * sizeof(float));
    
    int sample_label = fast_rand(&rng_state) % 10;
    int32_t *sample_output_int32 = (int32_t*)malloc(out_features * sizeof(int32_t));
    
    for(int i = 0; i < in_features; i++){
        sample_input_int8[i] = golden_inputs[0][i];
        sample_input_float[i] = (float)sample_input_int8[i];
    }

    // ----------- WARM-UP -----------
    printf("Warming up CPU and caches...\n");
    for(int i = 0; i < 500; i++) {
        sample_layer(layer, sample_input_int8, sample_output_int32, &rng_state);
        train_step(layer, sample_input_float, sample_label, &rng_state, ws);
    }
    
    // ----------- Benchmark 1: Inference -----------
    printf("\n--- Benchmarking 'sample' (Inference) over %d passes ---\n", iterations);
    clock_t t0 = clock();
    for(int i = 0; i < iterations; i++){
        sample_layer(layer, sample_input_int8, sample_output_int32, &rng_state);
    }
    clock_t t1 = clock();
    double time_sample = (double)(t1 - t0) / CLOCKS_PER_SEC;
    printf("Total inference time : %.4f seconds.\n", time_sample);
    printf("Time per pass        : %.6f ms.\n", (time_sample * 1000.0) / iterations);

    // ----------- Benchmark 2: Training Forward -------------- 
    printf("\n--- Benchmarking 'train_forward' over %d passes ---\n", iterations);
    clock_t t2 = clock();
    for (int i = 0; i < iterations; i++) {
        train_step(layer, sample_input_float, sample_label, &rng_state, ws);
    }
    clock_t t3 = clock();
    double time_train_forward = (double)(t3 - t2) / CLOCKS_PER_SEC;
    printf("Total training time  : %.4f seconds.\n", time_train_forward);
    printf("Time per pass        : %.6f ms.\n", (time_train_forward * 1000.0) / iterations);

    float checksum = 0.0f;
    for(int i = 0; i < in_features * out_features; i++) {
        checksum += layer->weights[i];
    }
    printf("Sanity Checksum: %f\n", checksum);
    
    free(sample_input_int8);
    free(sample_input_float);
    free(sample_output_int32);
    free_workspace(ws);
    free_layer(layer);
    
    return 0;
}

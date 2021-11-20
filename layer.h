#ifndef LAYER_H_
#define LAYER_H_


#include<fcntl.h>
#include<stdio.h>
#include<stdlib.h>
#include<iostream>
#include<unistd.h>
#include<string>
#include<math.h>
#include<sys/time.h>

class Layer {
	float init_limit;
	int n_count;
	int w_count;
	float lf;
	float *deltas;
	float *inputs;
	int arr_size;
    void init_weights();
    void init_inputs();
    void init_outputs();
    int arr_pos(int, int);
    void save_inputs(float*);
    float ac_f(float);
    float ac_f1(float);
    float combined_signal(int);
    float dropout(float);
    int rand_int();
    float rand_float();
public:
	Layer(int, int, float);
	~Layer();
	float *outputs;
	float *weights;
	void process(float*);
	void calculate_weight_deltas(float*);
	void apply_weight_deltas();
	int get_w_size();
	int get_n_size();
};

Layer::Layer(int n_count, int w_count, float lf) {
	this->init_limit = 0.05;
	this->n_count = n_count;
	this->w_count = w_count;
	this->lf = lf;
	arr_size = n_count * w_count;
	srandom((unsigned)time(0));
    init_weights();
    init_inputs();
    init_outputs();
}

Layer::~Layer() {
	delete[] deltas;
	delete[] inputs;
	delete[] outputs;
	delete[] weights;
}

int Layer::arr_pos(int n, int w) {
	return (n * w_count) + w;
}

void Layer::init_weights() {
    weights = new float[arr_size];
    deltas = new float[arr_size];
    for (int n = 0; n < n_count; n++) {
        for (int w = 0; w < w_count; w++) {
            weights[arr_pos(n, w)] = rand_float() * init_limit;
            deltas[arr_pos(n, w)] = 0;
        }
    }
}

void Layer::init_inputs() {
	inputs = new float[w_count];
}

void Layer::init_outputs() {
	outputs = new float[n_count];
}

void Layer::process(float *signal) {
	save_inputs(signal);
    for (int n = 0; n < n_count; n++) {
        outputs[n] = 0;
        outputs[n] = ac_f(combined_signal(n));
    }
}

void Layer::save_inputs(float *signal) {
    for (int w = 0; w < w_count; w++) {
        inputs[w] = signal[w];
    }
}

float Layer::ac_f(float x) {
	return x / (1 + abs(x));
}

float Layer::ac_f1(float x) {
	return 1 / pow(1 + abs(x), 2);
}

float Layer::combined_signal(int n) {
    float o = 0;
    for (int w = 0; w < w_count; w++) {
        o += weights[arr_pos(n, w)] * inputs[w];
    }
    return o;
}

void Layer::calculate_weight_deltas(float *output_diff) {
    double f1Val;
    for (int n = 0; n < n_count; n++) {
        f1Val = ac_f1(combined_signal(n));
        for (int w = 0; w < w_count; w++) {
            deltas[arr_pos(n, w)] = dropout(lf * output_diff[n] * f1Val * inputs[w]);
        }
    }
}

void Layer::apply_weight_deltas() {
    for (int n = 0; n < n_count; n++) {
        for (int w = 0; w < n_count; w++) {
            weights[arr_pos(n, w)] -= deltas[arr_pos(n, w)];
            deltas[arr_pos(n, w)] = 0;
        }
    }
}

float Layer::dropout(float x) {
	if((rand_int() & 1) == 1)
		return x;
	return 0;
}

int Layer::rand_int() {
	return random();
}

float Layer::rand_float() {
	return (float)random() / (float)(RAND_MAX);
}

int Layer::get_w_size() {
	return w_count;
}

int Layer::get_n_size() {
	return n_count;
}

#endif /* LAYER_H_ */

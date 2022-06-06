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
	float lf;
	float *deltas;
	float *inputs;
    float *cs;
	int arr_size;
    void init_weights();
    void init_inputs();
    void init_outputs();
    void init_cs();
    float ac_f(float);
    float ac_f1(float);
    float combined_signal(int);
    int rand_int();
    float rand_float();
public:
	Layer(int, int, float);
	~Layer();
	float *outputs;
	float *weights;
    int n_count;
	int w_count;
	void process(float*);
	void calculate_weight_deltas(float*);
	void apply_weight_deltas();
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
    init_cs();
}

Layer::~Layer() {
	delete[] deltas;
	delete[] inputs;
	delete[] outputs;
	delete[] weights;
    delete[] cs;
}

void Layer::init_weights() {
    weights = new float[arr_size];
    deltas = new float[arr_size];
    int index = 0;
    for (int n = 0; n < n_count; n++) {
        for (int w = 0; w < w_count; w++) {
            weights[index] = rand_float() * init_limit;
            deltas[index] = 0;
            index++;
        }
    }
}

void Layer::init_inputs() {
	inputs = new float[w_count];
}

void Layer::init_outputs() {
	outputs = new float[n_count];
}

void Layer::init_cs() {
    cs = new float[n_count];
}

float Layer::rand_float() {
	return (float)random() / (float)(RAND_MAX);
}

void Layer::process(float *signal) {
	for (int w = 0; w < w_count; w++) 
        inputs[w] = signal[w];
    
    for (int n = 0; n < n_count; n++) {
        cs[n] = combined_signal(n);
        outputs[n] = ac_f(cs[n]);
    }
}

float Layer::ac_f(float x) {
	return x / (1 + abs(x));
}

float Layer::ac_f1(float x) {
	return 1 / (1 + abs(2 * x) + (x * x));
}

float Layer::combined_signal(int n) {
    float o = 0;
    int index = n * w_count;
    for (int w = 0; w < w_count; w++) {
        o += weights[index] * inputs[w];
        index++;
    }
    return o;
}

void Layer::calculate_weight_deltas(float *output_diff) {
    double f1Val;
    int index = 0;
    for (int n = 0; n < n_count; n++) {
        f1Val = lf * output_diff[n] * ac_f1(cs[n]);
        for (int w = 0; w < w_count; w++) {            
            deltas[index] = (random() & 1) * f1Val * inputs[w];        
            index++;
        }
    }
}

void Layer::apply_weight_deltas() {
    for (int i = 0; i < arr_size; i++)
        weights[i] -= deltas[i];
}

#endif /* LAYER_H_ */

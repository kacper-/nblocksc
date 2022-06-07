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
    int mem_size;
    void init_weights();
    void init_inputs();
    void init_outputs();
    void init_cs();
    float combined_signal(int);
    int rand_int();
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
    mem_size = sizeof(inputs) * w_count;
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
            weights[index] = init_limit * random() / (float)(RAND_MAX);;
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

void Layer::process(float *signal) {
    for(int i=0;i<w_count;i++) 
        inputs[i] = signal[i];

    for(int i=0;i<n_count;i++) 
        cs[i] = 0;

    int index = 0;
    for (int n = 0; n < n_count; n++) {
        for (int w = 0; w < w_count; w++) {
            cs[n] += weights[index] * inputs[w];
            index++;
        }     
        outputs[n] = cs[n] / (1 + abs(cs[n]));   
    }
}

void Layer::calculate_weight_deltas(float *output_diff) {
    double f1Val;
    int index = 0;
    long r;

    for (int n = 0; n < n_count; n++) {
        r = random();
        f1Val = lf * output_diff[n] / ((1 + abs(2 * cs[n]) + (cs[n] * cs[n])));
        for (int w = 0; w < w_count; w++) {            
            deltas[index] = ((r >> (w % 16)) & 1) * f1Val * inputs[w];        
            index++;
        }
    }
}

void Layer::apply_weight_deltas() {
    for (int i = 0; i < arr_size; i++)
        weights[i] -= deltas[i];
}

#endif /* LAYER_H_ */

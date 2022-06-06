/*
 * net.h
 *
 *  Created on: Feb 18, 2021
 *      Author: kacper
 */

#ifndef NET_H_
#define NET_H_


#include"layer.h"


class Net {
	Layer *front, *back, *middle, *middle2;
	int size;
	void calculate_back_error(float*, float*, int, float*);
	void calculate_error(float*, float*, int, int, float*);
public:
	Net(int, int, float);
	~Net();
	int get_size();
	void process(float*, float*);
	void teach(float*, float*);
	int get_input_size();
	int get_output_size();
};

Net::Net(int size, int size_last, float lf) {
	front = new Layer(size, size, lf);
	back = new Layer(size, size, lf);
	middle = new Layer(size, size, lf);
	middle2 = new Layer(size_last, size, lf);
	this->size = size;
}

Net::~Net() {
	delete front;
	delete back;
	delete middle;
	delete middle2;
}

int Net::get_size() {
	return size;
}

void Net::process(float *signal, float *result) {
	front->process(signal);
	middle->process(front->outputs);
	middle2->process(middle->outputs);
	back->process(middle2->outputs);
    
	for (int i = 0; i < size; i++) 
    	result[i] = back->outputs[i];
}

void Net::teach(float *signal, float *expected) {
	front->process(signal);
	middle->process(front->outputs);
	middle2->process(middle->outputs);
	back->process(middle2->outputs);

	float back_error[back->n_count];
	calculate_back_error(back->outputs, expected, back->n_count, back_error);

	float middle2_error[middle2->w_count];
	calculate_error(back->weights, back_error, middle2->n_count, middle2->w_count, middle2_error);

	float middle_error[middle->w_count];
	calculate_error(middle2->weights, middle2_error, middle->n_count, middle->w_count, middle_error);

	float front_error[front->w_count];
	calculate_error(middle->weights, middle_error, front->n_count, front->w_count, front_error);

    back->calculate_weight_deltas(back_error);
    middle2->calculate_weight_deltas(middle2_error);
	middle->calculate_weight_deltas(middle_error);
	front->calculate_weight_deltas(front_error);

    front->apply_weight_deltas();
    middle->apply_weight_deltas();
    middle2->apply_weight_deltas();
    back->apply_weight_deltas();
}

void Net::calculate_back_error(float *result, float *expected, int size, float *error) {
    for (int i = 0; i < size; i++) 
        error[i] = result[i] - expected[i];
}

void Net::calculate_error(float *weights, float *error, int n_size, int w_size, float *result) {
    for (int w = 0; w < w_size; w++) {
    	result[w] = 0;
        for (int n = 0; n < n_size; n++)
            result[w] += weights[(n * w_size) + w] * error[n];
    }
}

int Net::get_input_size() {
	return front->w_count;
}

int Net::get_output_size() {
	return back->n_count;
}

#endif /* NET_H_ */

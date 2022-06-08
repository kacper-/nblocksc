/*
 * net.h
 *
 *  Created on: Feb 18, 2021
 *      Author: kacper
 */

#ifndef NET_H_
#define NET_H_

#include<fcntl.h>
#include<stdio.h>
#include<stdlib.h>
#include<iostream>
#include<unistd.h>
#include<string>
#include<math.h>
#include<sys/time.h>

const int SIZE = 64;
const int ARR_SIZE = SIZE * SIZE;
float const LF = 0.1;
int const REPS = 25000;
float const INIT_LIMIT = 0.05;
int const nbytes = sizeof(float) * SIZE;
int const nbytes4 = sizeof(float) * SIZE * 4;

struct combined_signal {
	float back[SIZE];
	float middle2[SIZE];
	float middle[SIZE];
	float front[SIZE];
};

struct combined_signal cs;

float back_error[SIZE];
float middle2_error[SIZE];
float middle_error[SIZE];
float front_error[SIZE];

float back_outputs[SIZE];
float middle2_outputs[SIZE];
float middle_outputs[SIZE];
float front_outputs[SIZE];

float back_deltas[ARR_SIZE];
float middle2_deltas[ARR_SIZE];
float middle_deltas[ARR_SIZE];
float front_deltas[ARR_SIZE];

float back_weights[ARR_SIZE];
float middle2_weights[ARR_SIZE];
float middle_weights[ARR_SIZE];
float front_weights[ARR_SIZE];

void layer_init(float *const weights, float *const deltas) {	
    for (int i = 0; i < ARR_SIZE; i++) {        
		weights[i] = INIT_LIMIT * random() / (float)(RAND_MAX);;
		deltas[i] = 0;        
    }
}

void layer_process(float *const cs, float *const outputs, float *const deltas, float *const weights, float *const signal) {
    int index = 0;
    for (int n = 0; n < SIZE; n++) {
        for (int w = 0; w < SIZE; w++) {
            cs[n] += weights[index] * signal[w];
            index++;
        }     
        outputs[n] = cs[n] / (1 + abs(cs[n]));   
    }
}

void calculate_weight_deltas(float *const cs, float *const deltas, float *const output_diff, float *const signal) {
    double f1Val;
    int index = 0;
    long r;

    for (int n = 0; n < SIZE; n++) {
        r = random();
        f1Val = LF * output_diff[n] / ((1 + abs(2 * cs[n]) + (cs[n] * cs[n])));
		index = n * SIZE;  
        for (int w = 0; w < SIZE; w++)          
            deltas[index + w] = ((r >> (w & 15)) & 1) * f1Val * signal[w];        
    }
}

void process(float *const signal, float *const result) 
	{
	memset(&cs, 0, nbytes4);

	layer_process(cs.front, front_outputs, front_deltas, front_weights, signal);
	layer_process(cs.middle, middle_outputs, middle_deltas, middle_weights, front_outputs);
	layer_process(cs.middle2, middle2_outputs, middle2_deltas, middle2_weights, middle_outputs);
	layer_process(cs.back, back_outputs, back_deltas, back_weights, middle2_outputs);
    
	for (int i = 0; i < SIZE; i++) 
    	result[i] = back_outputs[i];
}

void teach(float *const signal, float *const expected) 
	{
	memset(middle2_error, 0, nbytes);
	memset(middle_error, 0, nbytes);
	memset(front_error, 0, nbytes);
	memset(&cs, 0, nbytes4);

	layer_process(cs.front, front_outputs, front_deltas, front_weights, signal);
	layer_process(cs.middle, middle_outputs, middle_deltas, middle_weights, front_outputs);
	layer_process(cs.middle2, middle2_outputs, middle2_deltas, middle2_weights, middle_outputs);
	layer_process(cs.back, back_outputs, back_deltas, back_weights, middle2_outputs);

	auto index = 0;
    for (auto n = 0; n < SIZE; n++) {
		back_error[n] = back_outputs[n] - expected[n];
		index = n * SIZE;    	
        for (auto w = 0; w < SIZE; w++)
            middle2_error[w] += back_weights[index + w] * back_error[n];
    }
    for (auto n = 0; n < SIZE; n++) {
		index = n * SIZE;    	
        for (auto w = 0; w < SIZE; w++)
            middle_error[w] += middle2_weights[index + w] * middle2_error[n];
    }
    for (auto n = 0; n < SIZE; n++) {
		index = n * SIZE;    	
        for (auto w = 0; w < SIZE; w++)
            front_error[w] += middle_weights[index + w] * middle_error[n];
    }	

    calculate_weight_deltas(cs.back, back_deltas, back_error, middle2_outputs);
    calculate_weight_deltas(cs.middle2, middle2_deltas, middle2_error, middle_outputs);
	calculate_weight_deltas(cs.middle, middle_deltas, middle_error, front_outputs);
	calculate_weight_deltas(cs.front, front_deltas, front_error, signal);

	for (auto i = 0; i < ARR_SIZE; i++) {
        back_weights[i] -= back_deltas[i];
		middle2_weights[i] -= middle2_deltas[i];	
		middle_weights[i] -= middle_deltas[i];	
		front_weights[i] -= front_deltas[i];
	}
}

void train(float *const signal, float *const expected, int count) {	
	srandom((unsigned)time(0));

	layer_init(front_weights, front_deltas);
	layer_init(back_weights, back_deltas);
	layer_init(middle_weights, middle_deltas);
	layer_init(middle2_weights, middle2_deltas);	

	int j;
    for (int i = 0; i < REPS; i++) {
        j = (random() % count) * SIZE;
        teach(signal + j, expected + j);
    }
}

#endif /* NET_H_ */

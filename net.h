/*
 * net.h
 *
 *  Created on: Feb 18, 2021
 *      Author: kacper
 */

#ifndef NET_H_
#define NET_H_

#define SIZE 64
#define ARR_SIZE SIZE * SIZE
#define NBYTES 256
#define NBYTES4 1024
#define LF 0.07
#define PD 0.14
#define MD 0.07
#define INIT_LIMIT 0.05

#include<fcntl.h>
#include<stdio.h>
#include<stdlib.h>
#include<iostream>
#include<unistd.h>
#include<string>
#include<cstring>
#include<math.h>
#include<sys/time.h>

int const REPS = 30000;

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

inline void layer_process(float *const outputs, float *const deltas, float *const weights, float *const signal) {
    int i = - SIZE, w, n;
    for (n = 0; n < SIZE; n++) {
		i += SIZE;  
		outputs[n] = 0;
        for (w = 0; w < SIZE; w++) 
            outputs[n] += weights[i + w] * signal[w];
		outputs[n] = outputs[n] / (1 + abs(outputs[n])); 
    } 
}

inline void calculate_weight_deltas(float *const deltas, float *const output_diff, float *const signal, float *const weights) {
    float f1Val, f1Vals;
    int i = - SIZE, w, n;
    long r;

    for (n = 0; n < SIZE; n++) {
        r = random();
		f1Val = LF * output_diff[n];
		i += SIZE;  
        for (w = 0; w < SIZE; w++)          
            deltas[i + w] = ((r >> (w & 31)) & 1) * f1Val * signal[w];          
    }
}

inline void calculate_error(float *const error, float *const weights, float *const up_error) {
	int i = - SIZE, w, n;

	for (n = 0; n < SIZE; n++) {
		i += SIZE;  		
        for (w = 0; w < SIZE; w++) 
            error[w] += weights[i + w] * up_error[n];
    }

}

void process(float *const signal, float *const result) 
	{
	layer_process(front_outputs, front_deltas, front_weights, signal);
	layer_process(middle_outputs, middle_deltas, middle_weights, front_outputs);
	layer_process(middle2_outputs, middle2_deltas, middle2_weights, middle_outputs);
	layer_process(back_outputs, back_deltas, back_weights, middle2_outputs);
    
	for (int i = 0; i < SIZE; i++) 
    	result[i] = back_outputs[i];
}

void teach(float *const signal, float *const expected) 
	{
	memset(middle2_error, 0, NBYTES);
	memset(middle_error, 0, NBYTES);
	memset(front_error, 0, NBYTES);

	layer_process(front_outputs, front_deltas, front_weights, signal);
	layer_process(middle_outputs, middle_deltas, middle_weights, front_outputs);
	layer_process(middle2_outputs, middle2_deltas, middle2_weights, middle_outputs);
	layer_process(back_outputs, back_deltas, back_weights, middle2_outputs);

	for (int n = 0; n < SIZE; n++)
		back_error[n] = back_outputs[n] - expected[n];

	calculate_error(middle2_error, back_weights, back_error);
	calculate_error(middle_error, middle2_weights, middle2_error);
	calculate_error(front_error, middle_weights, middle_error);

    calculate_weight_deltas(back_deltas, back_error, middle2_outputs, back_weights);
    calculate_weight_deltas(middle2_deltas, middle2_error, middle_outputs, middle2_weights);
	calculate_weight_deltas(middle_deltas, middle_error, front_outputs, middle_weights);
	calculate_weight_deltas(front_deltas, front_error, signal, front_weights);

	for (int i = 0; i < ARR_SIZE; i++) {
        back_weights[i] -= back_deltas[i];
		middle2_weights[i] -= middle2_deltas[i];	
		middle_weights[i] -= middle_deltas[i];	
		front_weights[i] -= front_deltas[i];
	}
}

void train(float *const signal, float *const expected, int count) {	
	srandom((unsigned)time(0));
	
	int i, j;

	for (i = 0; i < ARR_SIZE; i++) {        
		front_weights[i] = INIT_LIMIT * random() / (float)(RAND_MAX);
		middle_weights[i] = front_weights[i];
		middle2_weights[i] = front_weights[i];
		back_weights[i] = front_weights[i];
    }

    for (i = 0; i < REPS; i++) {
        j = (random() % count) * SIZE;
        teach(signal + j, expected + j);
    }
}

#endif /* NET_H_ */

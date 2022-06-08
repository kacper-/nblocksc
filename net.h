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

float back_error[SIZE];
float middle2_error[SIZE];
float middle_error[SIZE];
float front_error[SIZE];

float back_cs[SIZE];
float middle2_cs[SIZE];
float middle_cs[SIZE];
float front_cs[SIZE];

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
        for (int w = 0; w < SIZE; w++) {            
            deltas[index] = ((r >> (w & 15)) & 1) * f1Val * signal[w];        
            index++;
        }
    }
}

inline void calculate_error(float *const weights, float *const error, float *const result) {
	int index;
    for (int n = 0; n < SIZE; n++) {
		index = n * SIZE;    	
        for (int w = 0; w < SIZE; w++)
            result[w] += weights[index + w] * error[n];
    }
}

void process(float *const signal, float *const result) 
	{
	for (int i = 0; i < SIZE; i++) {
		back_cs[i] = 0;
		middle2_cs[i] = 0;
		middle_cs[i] = 0;
		front_cs[i] = 0;
	}		
	layer_process(front_cs, front_outputs, front_deltas, front_weights, signal);
	layer_process(middle_cs, middle_outputs, middle_deltas, middle_weights, front_outputs);
	layer_process(middle2_cs, middle2_outputs, middle2_deltas, middle2_weights, middle_outputs);
	layer_process(back_cs, back_outputs, back_deltas, back_weights, middle2_outputs);
    
	for (int i = 0; i < SIZE; i++) 
    	result[i] = back_outputs[i];
}

void teach(float *const signal, float *const expected) 
	{
	
	for (int i = 0; i < SIZE; i++) {
    	middle2_error[i] = 0;
		middle_error[i] = 0;
		front_error[i] = 0;
		back_cs[i] = 0;
		middle2_cs[i] = 0;
		middle_cs[i] = 0;
		front_cs[i] = 0;
	}

	layer_process(front_cs, front_outputs, front_deltas, front_weights, signal);
	layer_process(middle_cs, middle_outputs, middle_deltas, middle_weights, front_outputs);
	layer_process(middle2_cs, middle2_outputs, middle2_deltas, middle2_weights, middle_outputs);
	layer_process(back_cs, back_outputs, back_deltas, back_weights, middle2_outputs);

	for (int i = 0; i < SIZE; i++) 
        back_error[i] = back_outputs[i] - expected[i];

	calculate_error(back_weights, back_error, middle2_error);
	calculate_error(middle2_weights, middle2_error, middle_error);
	calculate_error(middle_weights, middle_error, front_error);

    calculate_weight_deltas(back_cs, back_deltas, back_error, middle2_outputs);
    calculate_weight_deltas(middle2_cs, middle2_deltas, middle2_error, middle_outputs);
	calculate_weight_deltas(middle_cs, middle_deltas, middle_error, front_outputs);
	calculate_weight_deltas(front_cs, front_deltas, front_error, signal);

	for (int i = 0; i < ARR_SIZE; i++) {
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

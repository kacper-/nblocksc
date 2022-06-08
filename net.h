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

struct layer {
    float cs[SIZE];
	float outputs[SIZE];
	float deltas[ARR_SIZE];
	float weights[ARR_SIZE];
};

void layer_init(float weights[], float deltas[]);
void layer_process(float cs[], float outputs[], float deltas[], float weights[], float signal[]);
void calculate_weight_deltas(float cs[], float deltas[], float output_diff[], float signal[]);
void apply_weight_deltas(float weights[], float deltas[]);

void layer_init(float weights[], float deltas[]) {	
    for (int i = 0; i < ARR_SIZE; i++) {        
		weights[i] = INIT_LIMIT * random() / (float)(RAND_MAX);;
		deltas[i] = 0;        
    }
}

void layer_process(float cs[], float outputs[], float deltas[], float weights[], float signal[]) {
    for(int i=0;i<SIZE;i++) 
        cs[i] = 0;

    int index = 0;
    for (int n = 0; n < SIZE; n++) {
        for (int w = 0; w < SIZE; w++) {
            cs[n] += weights[index] * signal[w];
            index++;
        }     
        outputs[n] = cs[n] / (1 + abs(cs[n]));   
    }
}

void calculate_weight_deltas(float cs[], float deltas[], float output_diff[], float signal[]) {
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

void apply_weight_deltas(float weights[], float deltas[]) {
    for (int i = 0; i < ARR_SIZE; i++)
        weights[i] -= deltas[i];
}


struct net {
	struct layer front;
	struct layer back;
	struct layer middle;
	struct layer middle2;
};

void calculate_back_error(float result[], float expected[], float error[]);
void calculate_error(float weights[], float error[], float result[]);
void process(	float front_cs[], float front_outputs[], float front_deltas[], float front_weights[],
	float middle_cs[], float middle_outputs[], float middle_deltas[], float middle_weights[],
	float middle2_cs[], float middle2_outputs[], float middle2_deltas[], float middle2_weights[],
	float back_cs[], float back_outputs[], float back_deltas[], float back_weights[],
 float signal[], float result[]);
void teach(	float front_cs[], float front_outputs[], float front_deltas[], float front_weights[],
	float middle_cs[], float middle_outputs[], float middle_deltas[], float middle_weights[],
	float middle2_cs[], float middle2_outputs[], float middle2_deltas[], float middle2_weights[],
	float back_cs[], float back_outputs[], float back_deltas[], float back_weights[], float signal[], float expected[]);

void process(	
	float front_cs[], float front_outputs[], float front_deltas[], float front_weights[],
	float middle_cs[], float middle_outputs[], float middle_deltas[], float middle_weights[],
	float middle2_cs[], float middle2_outputs[], float middle2_deltas[], float middle2_weights[],
	float back_cs[], float back_outputs[], float back_deltas[], float back_weights[],
	 float signal[], float result[]) {
	layer_process(front_cs, front_outputs, front_deltas, front_weights, signal);
	layer_process(middle_cs, middle_outputs, middle_deltas, middle_weights, front_outputs);
	layer_process(middle2_cs, middle2_outputs, middle2_deltas, middle2_weights, middle_outputs);
	layer_process(back_cs, back_outputs, back_deltas, back_weights, middle2_outputs);
    
	for (int i = 0; i < SIZE; i++) 
    	result[i] = back_outputs[i];
}

void teach(
	float front_cs[], float front_outputs[], float front_deltas[], float front_weights[],
	float middle_cs[], float middle_outputs[], float middle_deltas[], float middle_weights[],
	float middle2_cs[], float middle2_outputs[], float middle2_deltas[], float middle2_weights[],
	float back_cs[], float back_outputs[], float back_deltas[], float back_weights[],
	float signal[], float expected[]) 
	{
	layer_process(front_cs, front_outputs, front_deltas, front_weights, signal);
	layer_process(middle_cs, middle_outputs, middle_deltas, middle_weights, front_outputs);
	layer_process(middle2_cs, middle2_outputs, middle2_deltas, middle2_weights, middle_outputs);
	layer_process(back_cs, back_outputs, back_deltas, back_weights, middle2_outputs);

	float back_error[SIZE];
	calculate_back_error(back_outputs, expected, back_error);

	float middle2_error[SIZE];
	calculate_error(back_weights, back_error, middle2_error);

	float middle_error[SIZE];
	calculate_error(middle2_weights, middle2_error, middle_error);

	float front_error[SIZE];
	calculate_error(middle_weights, middle_error, front_error);

    calculate_weight_deltas(back_cs, back_deltas, back_error, middle2_outputs);
    calculate_weight_deltas(middle2_cs, middle2_deltas, middle2_error, middle_outputs);
	calculate_weight_deltas(middle_cs, middle_deltas, middle_error, front_outputs);
	calculate_weight_deltas(front_cs, front_deltas, front_error, signal);

    apply_weight_deltas(back_weights, back_deltas);
    apply_weight_deltas(middle2_weights, middle2_deltas);
    apply_weight_deltas(middle_weights, middle_deltas);
    apply_weight_deltas(front_weights, front_deltas);
}

void calculate_back_error(float result[], float expected[], float error[]) {
    for (int i = 0; i < SIZE; i++) 
        error[i] = result[i] - expected[i];
}

void calculate_error(float weights[], float error[], float result[]) {
    for (int w = 0; w < SIZE; w++) {
    	result[w] = 0;
	}

	int index;
    for (int n = 0; n < SIZE; n++) {
		index = n * SIZE;    	
        for (int w = 0; w < SIZE; w++)
            result[w] += weights[index + w] * error[n];
    }
}

struct net train(float signal[], float expected[], int count) {	
	struct net ann;
	
	srandom((unsigned)time(0));

	layer_init(ann.front.weights, ann.front.deltas);
	layer_init(ann.back.weights, ann.back.deltas);
	layer_init(ann.middle.weights, ann.middle.deltas);
	layer_init(ann.middle2.weights, ann.middle2.deltas);	

	int j;
    for (int i = 0; i < REPS; i++) {
        j = (random() % count) * SIZE;
        teach(	
			ann.front.cs, ann.front.outputs, ann.front.deltas, ann.front.weights,
			ann.middle.cs, ann.middle.outputs, ann.middle.deltas, ann.middle.weights,
			ann.middle2.cs, ann.middle2.outputs, ann.middle2.deltas, ann.middle2.weights,
			ann.back.cs, ann.back.outputs, ann.back.deltas, ann.back.weights, 
			signal + j, expected + j);
    }

	return ann;
}

#endif /* NET_H_ */

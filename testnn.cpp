/*
 * testnn.cpp
 *
 *  Created on: Feb 19, 2021
 *      Author: kacper
 */

#include"net.h"
#include"sys/time.h"
#include"testdata.h"

void print_vector(float *s, float *result);
long get_millis();

int main(int argc, char *argv[]) {
	float result[SIZE];
	float *s;

	std::cout << "training..." << std::endl;
	
	long start = get_millis();
	train(input_s, input_e, COUNT);
	long stop = get_millis();

	std::cout << "results..." << std::endl;
	for(int i=0;i<COUNT;i++) {
		s = input_s + (i * SIZE);
		process(s, result);
		print_vector(s, result);
	}

	std::cout << std::endl << "finished in " << stop-start << " msec" << std::endl;

    return 0;
}

long get_millis() {
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return tp.tv_sec * 1000 + tp.tv_usec / 1000;
}

void print_vector(float *s, float *result) {
	int index = 0, index2 = 0;
	float max = 0, max2 = 0;
	for(int i=0;i<SIZE;i++) {
		if(*(result + i)>max) {
			max2 = max;
			index2 = index;
			max = *(result + i);
			index = i;
		} else {
			if(*(result + i)>max2) {
				max2 = *(result + i);
				index2 = i;
			}
		}
	}

	char a, a2;
	int ind = index % 36;
	int ind2 = index2 % 36;
	if(ind>25)
		a = ind+22;
	else
		a = ind+65;
	if(ind2>25)
		a2 = ind2+22;
	else
		a2 = ind2+65;

	for(int i=0;i<SIZE;i++) {
		if(i % 8 == 0) {
			if(i==24)
				printf("\n%c %2d (%.2f) ", a, ind, max);
			else {
				if(i==32)
					printf("\n%c %2d (%.2f) ", a2, ind2, max2);
			 	else 
					printf("\n            ");
			}
		}
		if(*(s + i) > 0.5)
			printf("X");
		else
			printf(" ");
	}
	printf("\n");
}

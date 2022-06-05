/*
 * testnn.cpp
 *
 *  Created on: Feb 19, 2021
 *      Author: kacper
 */

#include"net_util.h"
#include"sys/time.h"
#include"testdata.h"

float const LF = 0.1;

void print_vector(float *s, float *result);
long get_millis();

int main(int argc, char *argv[]) {
	float result[SIZE];
	float *s;
	std::cout << "creating net..." << std::endl;
	Net net(SIZE, SIZE, LF);

	std::cout << "training..." << std::endl;
	
	long start = get_millis();
	train(&net, input_s, input_e, COUNT, 25000);
	long stop = get_millis();

	std::cout << "results..." << std::endl;
	for(int i=0;i<COUNT;i++) {
		s = input_s + (i * SIZE);
		net.process(s, result);
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
	if(index>25)
		a = index+22;
	else
		a = index+65;
	if(index2>25)
		a2 = index2+22;
	else
		a2 = index2+65;

	for(int i=0;i<SIZE;i++) {
		if(i % 8 == 0) {
			if(i==24)
				printf("\n%c (%.2f) ", a, max);
			else {
				if(i==32)
					printf("\n%c (%.2f) ", a2, max2);
			 	else 
					printf("\n         ");
			}
		}
		if(*(s + i) > 0.5)
			printf("X");
		else
			printf(" ");
	}
	printf("\n");
}

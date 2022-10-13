/*
 * testnn.cpp
 *
 *  Created on: Feb 19, 2021
 *      Author: kacper
 */

#include"net.h"
#include"sys/time.h"
#include"testdata.h"

void print_vector(float *s, float *result, int c_int);
long get_millis();
int accuracy = 0;

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
		print_vector(s, result, i);
	}

	std::cout << std::endl << "accuracy " << accuracy << " / " << COUNT << std::endl;
	std::cout << "finished in " << stop-start << " msec" << std::endl;

    return 0;
}

long get_millis() {
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return tp.tv_sec * 1000 + tp.tv_usec / 1000;
}

void print_vector(float *s, float *result, int c_int) {
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
	if(index < 36) {
		if(index>25)
			a = index+22;
		else
			a = index+65;
	} else {
		a = 32;
	}
	if(index2 < 36) {
		if(index2>25)
			a2 = index2+22;
		else
			a2 = index2+65;
	} else {
		a2 = 32;
	}
	for(int i=0;i<SIZE;i++) {
		if(i % 8 == 0) {
			if(i==24)
				printf("\n%c %2d (%.2f) ", a, index, max);
			else {
				if(i==32)
					printf("\n%c %2d (%.2f) ", a2, index2, max2);
			 	else 
					printf("\n            ");
			}
		}
		if(*(s + i) > 0.5)
			printf("X");
		else
			printf(" ");
	}

	if((c_int % 36) == index)
		accuracy++;

	printf("\n");
}

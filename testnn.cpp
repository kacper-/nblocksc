/*
 * testnn.cpp
 *
 *  Created on: Feb 19, 2021
 *      Author: kacper
 */

#include"net_util.h"
#include"testdata.h"

float const LF = 0.1;

void print_vector(float *s, float *result);

int main(int argc, char *argv[]) {
	float result[SIZE];
	float *s;
	std::cout << "creating net..." << std::endl;
	Net net(SIZE, SIZE, LF);

	std::cout << "training..." << std::endl;

	train(&net, input_s, input_e, COUNT, 25000);

	std::cout << "results..." << std::endl;
	for(int i=0;i<COUNT;i++) {
		s = input_s + (i * SIZE);
		net.process(s, result);
		print_vector(s, result);
	}

	std::cout << "finished." << std::endl;

    return 0;
}

void print_vector(float *s, float *result) {
	int index = 0;
	float max = 0;
	for(int i=0;i<SIZE;i++) {
		if(*(result + i)>max) {
			max = *(result + i);
			index = i;
		}
	}

	char a = index + 65;

	for(int i=0;i<SIZE;i++) {

		if(i % 8 == 0) {
			if(i==32)
				printf("\n%c (%.2f) ", a, max);
			else
				printf("\n         ");
		}
		if(*(s + i) > 0.5)
			printf("X");
		else
			printf(" ");
	}
	printf("\n");
}

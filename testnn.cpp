/*
 * testnn.cpp
 *
 *  Created on: Feb 19, 2021
 *      Author: kacper
 */

#include"net_util.h"

int const COUNT = 10;
int const SIZE = 16;
float const LF = 0.08;

void print_vector(float *result);

int main(int argc, char *argv[]) {
	float result[SIZE];
	float *s;
	std::cout << "creating net..." << std::endl;
	Net net(SIZE, SIZE, LF);

	std::cout << "training..." << std::endl;
	float signal[] = {
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    	1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    	0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    	0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    	0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    	0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    	0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
    	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
    	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
    	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1
	};

	float expected[] = {
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    	1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1
	};

	train(&net, signal, expected, COUNT, 100000);

	std::cout << "results..." << std::endl;
	for(int i=0;i<COUNT;i++) {
		s = signal + (i * SIZE);
		net.process(s, result);
		float diff[SIZE];
		for(int j=0;j<SIZE;j++)
			diff[j] = fabs(*(s+j) - *(result+j));
		print_vector(diff);
	}

	std::cout << "finished." << std::endl;

    return 0;
}

void print_vector(float *result) {
	for(int i=0;i<SIZE;i++) {
		printf("\t%.2f ", *(result + i));
	}
	std::cout << std::endl;
}

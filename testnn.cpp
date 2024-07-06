/*
 * testnn.cpp
 *
 *  Created on: Feb 19, 2021
 *      Author: kacper
 */

#include"net.h"
#include"sys/time.h"
#include"fstream"
#include"sstream"
#include"string"
#include"vector"

void usage();
void load_config(char *arg);
void load_data(char *arg, int train);
void print_vector(float *s, float *result, int c_int, float *e);
void train();
void run();
long get_millis();
int accuracy = 0;
int COUNT = 0;
int real_count;
float LF;
float INIT_LIMIT;
int REPS;
float *input_s;
float *input_e;

int main(int argc, char *argv[]) {

	//if(argc==5 && strcmp("train", argv[1])==0) {
		//load_config(argv[2]);
		//load_data(argv[3], 1);
		load_config("/Users/kacper/repo/nblocksc/config.txt");
		load_data("/Users/kacper/repo/nblocksc/data.txt", 1);
		train();
		save(argv[4]);
		return 0;
	//}
	//if(argc==5 && strcmp("run", argv[1])==0) {
	//	load_config(argv[2]);
	//	load_data(argv[3], 0);
	//	load(argv[4]);
	//	run();
	//	return 0;
	//}

	//usage();
    //return 0;
}

void load_config(char *arg) {
	std::ifstream infile(arg);
	std::string line;
	while (std::getline(infile, line))
	{
		std::string segment;
		std::vector<std::string> seglist;
		std::stringstream entry(line);
		while(std::getline(entry, segment, '='))
		{
   			seglist.push_back(segment);
		}

		if(seglist.size()==2) {
			if(seglist[0].compare("data_set_count")==0)
				COUNT = stoi(seglist[1]);

			if(seglist[0].compare("lf")==0)
				LF = stof(seglist[1]);

			if(seglist[0].compare("init_limit")==0)
				INIT_LIMIT = stof(seglist[1]);

			if(seglist[0].compare("reps")==0)
				REPS = stoi(seglist[1]);
		}
	}
	real_count = COUNT * SIZE;
	input_s = new float[real_count];
	input_e = new float[real_count];
}

void load_data(char *arg, int train) {
	std::ifstream infile(arg);
	std::string line;
	int i; 
	int index = 0;
	int index2 = 0;
	while (std::getline(infile, line))
	{
		if(index<real_count) {
			if(line[0]=='0' || line[0]=='1') {
				for(i=0;i<line.size();i++) {
					input_s[index]=line[i]-48;
					index++;
				}
			}
		} else {
			if(train==1 && index2<real_count) {
				if(line[0]=='0' || line[0]=='1') {
					for(i=0;i<line.size();i++) {
						input_e[index2]=line[i]-48;
						index2++;
					}
				}
			}
		}
	}
}

void usage() {
	printf("error - not enough or unknow parameters\n");
	printf("usage:\n");
	printf("	train path_to_config path_to_data path_to_save\n");
	printf("	run path_to_config path_to_data path_to_save\n");
}

void train() {
	float result[SIZE];
	float *s;
	float *e;

	std::cout << "training..." << std::endl;
	
	long start = get_millis();
	train(input_s, input_e, COUNT, INIT_LIMIT, REPS, LF);
	long stop = get_millis();

	std::cout << "results..." << std::endl;
	for(int i=0;i<COUNT;i++) {
		s = input_s + (i * SIZE);
		e = input_e + (i * SIZE);
		process(s, result);
		print_vector(s, result, i, e);
	}

	std::cout << std::endl << "accuracy " << accuracy << " / " << COUNT << std::endl;
	std::cout << "finished in " << stop-start << " msec" << std::endl;
}

void run() {
	// TODO implement
}

long get_millis() {
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return tp.tv_sec * 1000 + tp.tv_usec / 1000;
}

void print_vector(float *s, float *result, int c_int, float *e) {
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

	for(int i=0;i<SIZE;i++) {
		if(*(e + i) > 0.5)
			printf("X");
		else
			printf(" ");
	}

	if((c_int % 36) == index)
		accuracy++;

	printf("\n");
}

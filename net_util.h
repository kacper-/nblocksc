/*
 * net_util.h
 *
 *  Created on: 18 lut 2021
 *      Author: kacpermarczewski
 */

#ifndef NET_UTIL_H_
#define NET_UTIL_H_

#include"net.h"

void train(Net *net, float *signal, float *expected, int size, int rep) {
	int j;
    for (int i = 0; i < rep; i++) {
        j = random() % size;
        net->teach(signal + (j * net->get_input_size()), expected + (j * net->get_output_size()));
    }
}


#endif /* NET_UTIL_H_ */

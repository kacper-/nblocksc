/*
 * testnn.cpp
 *
 *  Created on: Feb 19, 2021
 *      Author: kacper
 */

#include"net_util.h"

int main(int argc, char *argv[]) {
    Net net(16, 16, 0.05);
    std::cout << "OK" << std::endl;
    return 0;
}

#!/bin/sh
set -xe

gcc -o ./bin/adder_gen adder_gen.c -lm
gcc -o ./bin/img2mat img2mat.c -lm
gcc `pkg-config --cflags raylib` -o ./bin/gym gym.c `pkg-config --libs raylib` -lm -lpthread
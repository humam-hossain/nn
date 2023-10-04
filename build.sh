#!/bin/sh
set -xe

gcc `pkg-config --cflags raylib` -o ./bin/gym gym.c `pkg-config --libs raylib` -lm -lpthread
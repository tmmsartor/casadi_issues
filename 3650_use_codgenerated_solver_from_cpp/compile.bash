#! /bin/bash
g++ generate.cpp -o generate.out -lcasadi
./generate.out
g++ main.c -o main.out
g++ main_with_mem.c -o main_with_mem.out

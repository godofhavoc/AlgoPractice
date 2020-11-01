#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the missingNumbers function below.


def missingNumbers(arr, brr):


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    m = int(input())

    brr = list(map(int, input().rstrip().split()))

    result = missingNumbers(arr, brr)

    fptr.write(' '.join(map(str, result)))
    fptr.write('\n')

    fptr.close()


# >>>>>> backtracking psuedocode <<<<<<<
'''
    P: problem tree
    c: current node
'''
def backtrack(c):
    if reject(P, c):
        return
    if accept(P, c):
        output.append(c)
    
    s  = c.next
    while s != None:
        backtrack(s)
        s = s.next

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
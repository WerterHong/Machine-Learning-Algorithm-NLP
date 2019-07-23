
# -*- coding: utf-8 -*-
"""
Aim：实现隐马尔科夫模型的维特比算法(Viterbi Algorithm)
"""
import numpy as np
from numpy import *
import math

def viterbi(A, B, PI, O):
    N = shape(A)[0]
    I = mat(zeros((N, 1)))
    T = N
    sigma = mat(zeros((N, N)))
    omiga = mat(ones((N, N)))
    index = 0

    for i in range(N):
        if(O[0, 0] == 0):
            index = 0
        else:
            index = 1
        sigma[0, i] = PI[i, 0] * B[i, index]

    t = 1
    while(t < T):
        for i in range(N):
            sigma_temp = mat(zeros((N, 1)))
            for j in range(N):
                sigma_temp[j, 0] = sigma[t - 1, j] * A[j, i]
            max_value = sigma_temp.max(axis = 0)
            if(O[t, 0] == 0):
                index = 0
            else:
                index = 1
            sigma[t, i] = max_value[0, 0] * B[i, index]
            omiga[t, i] = sigma_temp.argmax() + 1
        t += 1
    P = sigma[N - 1, :].max()
    I[T -1, 0] = sigma[N - 1, :].argmax() + 1
    t = T - 2

    print(omiga)
    while(t >= 0):
        index = int(I[t + 1, 0] - 1)
        I[t, 0] = omiga[t + 1, index]
        t -= 1
    return I

if __name__ == "__main__":
    A = mat([[0.5, 0.2, 0.3],
             [0.3, 0.5, 0.2],
             [0.2, 0.3, 0.5]])
    B = mat([[0.5, 0.5],
             [0.4, 0.6],
             [0.7, 0.3]])
    PI = mat([[0.2],
              [0.4],
              [0.4]])
    O = mat([[0],
             [1],
             [0]])
    I = viterbi(A, B, PI, O)
    print(I)

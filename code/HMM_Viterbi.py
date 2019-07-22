
# -*- coding: utf-8 -*-
"""
Github: https://github.com/WerterHong/Machine-Learning-Algorithm-NLP/tree/master/code/HMM_Viterbi.py
Aim：实现隐马尔科夫模型的维特比算法(Viterbi Algorithm)
"""
import numpy as np
def viterbi_algorithm(A, B, pai, O):
    N = np.shape(A)[0] #隐马尔科夫模型状态个数
    T = np.shape(O)[0] #观测序列的观测个数，即时刻个数
    delta = np.zeros((T,N))#每个时刻每个状态对应的局部最优状态序列的概率数组
    psi = np.zeros((T,N))#每个时刻每个状态对应的局部最优状态序列的前导状态索引数组
    #(1)viterbi algorithm
    for t in range(T):#[0,1,...,T-1]
        if 0 == t:#计算初值
            delta[t] = np.multiply(pai.reshape((1, N)), np.array(B[:,O[t]]).reshape((1, N)))
            continue
        for i in range(N):
            delta_t_i = np.multiply(np.multiply(delta[t-1], A[:,i]), B[i, O[t]])
            delta[t,i] = max(delta_t_i)
            psi[t][i] = np.argmax(delta_t_i)
    states = np.zeros((T,))
    t_range = -1 * np.array(sorted(-1*np.arange(T)))
    for t in t_range:
        if T-1 == t:
            states[t] = np.argmax(delta[t])
        else:
            states[t] = psi[t+1, int(states[t+1])]
    print('局部最优状态的概率分布图:\n', delta)
    print('局部最优状态的前时刻状态索引图:\n', psi)
    print('最优状态序列:', states)
    return states

def HMM_Viterbi():
    #隐马尔可夫模型λ=(A, B, pai)
    #A是状态转移概率分布，状态集合Q的大小N=np.shape(A)[0]
    #从下给定A可知：Q={盒1, 盒2, 盒3}, N=3
    A = np.array([[0.5, 0.2, 0.3],
                  [0.3, 0.5, 0.2],
                  [0.2, 0.3, 0.5]])
    #B是观测概率分布，观测集合V的大小T=np.shape(B)[1]
    #从下面给定的B可知：V={红，白}，T=2
    B = np.array([[0.5, 0.5],
                  [0.4, 0.6],
                  [0.7, 0.3]])
    #pai是初始状态概率分布，初始状态个数=np.shape(pai)[0]
    pai = np.array([[0.2],
                    [0.4],
                    [0.4]])

    #观测序列
    O = np.array([[0],
                  [1],
                  [0]]) #0表示红色，1表示白，就是(红，白，红)观测序列
    viterbi_algorithm(A,B,pai,O)

if __name__=='__main__':
    HMM_Viterbi()

## 隐马尔科夫模型 HMM

- NLP-机器学习笔试面试题解析 [Github链接](https://github.com/WerterHong/Machine-Learning-Algorithm-NLP/)
- **隐马尔科夫模型 HMM** (若公式显示错误，请点击此链接) [有道云笔记](http://note.youdao.com/noteshare?id=c09f0762df3de876b945941fa1b991e5&sub=237FE09F086F4DC08ED4A3AD19C7F8DD)

### 1. 隐马尔科夫模型简述

隐马尔科夫模型(`hidden Markov model, HMM`)是关于时序的概率模型，描述由一个**隐藏的马尔可夫链**随机生成**不可观测状态随机序列**，再由各个状态生成一个观测而产生**观测随机序列**的过程。

`HMM`是一个**生成模型**，表示状态序列和观测序列的联合分布，但是状态序列是隐藏的，不可观测的。

<p align="center">
<img src="https://note.youdao.com/yws/public/resource/6ebba528b845b6d13a37d8d0f4b21f4f/A164ADF6682C4E82A2CE10BF6D16B12A?ynotemdtimestamp=1565971900792" />
<br/>
<strong>Fig</strong>. HMM 模型
</p>

> 隐马尔科夫模型`HMM`由**初始状态概率向量**`π`、**状态转移概率矩阵**`A`以及**观测概率矩阵**`B`确定。`λ =(A,B,π)`

设`Q`是所有可能状态的集合，`V`是对应的观测序列，即：

```math
Q=\{q_1,q_2,...,q_N\} \quad V=\{v_1,v_2,...,v_M\}
```

其中，`N`是可能的状态数，`M`是可能观测的数。另外设`I`是长度为`T`的状态序列，`O`是对应的观测序列：

```math
I=\{i_1,i_2,...,i_T\} \quad O=\{o_1,o_2,...,o_T\}
```

**状态转移概率矩阵**`A`为：

```math
A=[a_{ij}]_{N \times N}
```

其中,

```math
a_{ij}=p(i_{t+1}=q_j|i_t=q_i),i=1,2,...,N;j=1,2,...,N
```

是在时刻`t`处于状态`$q_i$`的条件下在`t+1`转移到状态`$q_j$`的概率。

**观测概率矩阵**`B`为：

```math
B=[b_j(k)]_{N \times M}
```

其中，

```math
b_j(k)=p(o_{t}=v_k|i_t=q_j),k=1,2,...,M;j=1,2,...,N
```

是在时刻`t`处于状态`$q_i$`的条件下生成观测`$v_k$`的概率。

**初始状态概率向量**`$\pi$`为：

```math
\pi=(\pi_i)
```

其中，

```math
\pi_i=P(i_1=q_i),i=1,2,...,N
```

是时刻`t=1`处于状态`$q_i$`的概率。

- ##### 隐马尔科夫模型基本假设：
1. **齐次马尔可夫性假设**，即假设隐藏的马尔可夫链在任意时刻`t`的状态只依赖于前一刻的状态，与其他时刻的状态及观测无关，也与时刻`t`无关。

```math
P\left(i_{t} | i_{t-1},o_{t-1} ,\ldots , i_{1},o_1\right)=P\left(i_{t} | i_{t-1}\right), t=1,2,...,T
```
2. **观测独立性假设**，即假设任意时刻的观测只依赖于该时刻的马尔可夫链的状态，与其他观测及状态无关。

```math
P\left(o_t|i_T,o_T,i_{T-1},o_{T-1},...,i_{i+1},o_{i+1},i_t,o_t,...,i_1,o_1\right)=p(o_t|i_t)
```
##### > 举个栗子 <

（**盒子和球模型**）假设有4个盒子，每个盒子里都装有红白两种颜色的球，盒子里的红白球数由下表给出：
<p align="center">
<table>
  <tr>
    <th>盒子</th>
    <th>1</th>
    <th>2</th>
    <th>3</th>
    <th>4</th>
  </tr>
  <tr>
    <td>红球数</td>
    <td>5</td>
    <td>3</td>
    <td>6</td>
    <td>8</td>
  </tr>
  <tr>
    <td>白球数</td>
    <td>5</td>
    <td>7</td>
    <td>4</td>
    <td>2</td>
  </tr>
</table>
</p>

抽球方式：开始，从4个盒子里以等概率随机选取1个盒子，从这个盒子随机抽出1个球，记录其颜色后，放回；然后，从当前盒子随即转移到下个盒子，转移规则如下(即状态转移矩阵`A`)：

```math
A=\left[ \begin{matrix} 0 & 1 & 0 & 0 \\ 0.4 & 0 & 0.6 & 0 \\ 0 & 0.4 & 0 & 0.6 \\ 0 & 0 & 0.5 & 0.5 \end{matrix} \right]
```

确定转移的盒子后，再从这个盒子里随机抽出1个球，记录其颜色，放回；如此重复5次，得到一个球的颜色的观测序列为`O={红,红,白,白,红}`

> 两个随机序列：盒子的序列（状态序列）、球的颜色的序列（观测序列）。前者式隐藏的，只有后者是可以观测的。可以明确：状态集合、观测集合以及模型的三要素。

盒子对应状态，状态的集合是`Q={盒子1,盒子2,盒子3,盒子4}, N=4`

球的颜色对应观测，观测的集合是`V={红,白}, V=2`

状态序列和观测序列长度`T=5`。初始概率分布为

```math
\pi = (0.25,0.25,0.25,0.25)^T
```

观测概率分布为：

```math
B=\left[ \begin{matrix} 0.5 & 0.5 \\ 0.3 & 0.7 \\ 0.6 & 0.4 \\ 0.8 & 0.2 \end{matrix} \right]
```

#### 观测序列的生成过程

输入：隐马尔科夫模型`λ =(A,B,π)`，观测序列长度`T`；

输出：观测序列`$O=\left\{o_{1}, o_{2}, \ldots, o_{T}\right\}$`。

1. 按照初始状态分布`π`产生状态`$i_1$`
2. 令`t=1`
3. 按照状态`$i_t$`的观测概率分布`$b_{i_t}(k)$`生成`$o_t$`
4. 按照状态`$i_t$`的状态转移概率分布`$\{a_{i_ti_{t+1}}\}$`产生状态`$i_{t+1}, i_{t+1}=1,2,...,N$`
5. 令`t=t+1`；如果`t<T`，转步(3)；否则，终止

### 2. 概率计算

#### 前向-后向算法

##### >举个栗子<

(**前向算法**)考虑盒子和球模型`λ=(A,B,π)`，状态集合`Q={1,2,3}`，观测集合`V={红,白}`，

```math
A=\left[ \begin{matrix} 0.5 & 0.2 & 0.3 \\ 0.3 & 0.5 & 0.2 \\ 0.2 & 0.3 & 0.5 \end{matrix} \right], B=\left[ \begin{matrix} 0.5 & 0.5 \\ 0.4 & 0.6 \\ 0.7 & 0.3 \end{matrix} \right], \pi=(0.2,0.2,0.4)^T
```

设`T=3`，`O={红,白,红}`，试用前向算法计算`P(O|λ)`。

**解**

(1) 计算初值

```math
\alpha_1(1)=\pi_1b_1(o_1)=0.10

\alpha_1(2)=\pi_2b_2(o_1)=0.16

\alpha_1(3)=\pi_3b_3(o_1)=0.28
```

(2) 递推计算
```math
\alpha_2(1)=[\sum_{i=1}^3 \alpha_1(i)a_{i1}]b_1(o_2)=0.154 \times 0.5 = 0.077

\alpha_2(2)=[\sum_{i=1}^3 \alpha_1(i)a_{i2}]b_2(o_2)=0.184 \times 0.6 = 0.1104

\alpha_2(3)=[\sum_{i=1}^3 \alpha_1(i)a_{i3}]b_3(o_2)=0.202 \times 0.3 = 0.0606

\alpha_3(1)=[\sum_{i=1}^3 \alpha_2(i)a_{i1}]b_1(o_3)= 0.04187

\alpha_3(2)=[\sum_{i=1}^3 \alpha_2(i)a_{i2}]b_2(o_3)= 0.03551

\alpha_3(3)=[\sum_{i=1}^3 \alpha_2(i)a_{i3}]b_3(o_3)= 0.05284
```
(3) 终止

```math
P(O|\lambda)=\sum_{i=1}^3 \alpha_3(i)= 0.13022
```

### 3. 预测算法

#### Viterbi 算法

**维特比算法**是用**动态规划**求概率最大路径（最优路径），一条路径对应一个状态序列。

根据动态规划原理，最优路径具有这样的特性：如果最优路径在时刻`t`通过节点`$i^*_t$`，那么这一路径从节点`$i^*_t$`到终点`$i^*_T$`的部分路径，对于从`$i^*_t$`到`$i^*_T$`的所有可能的部分路径来说，必须是最优的。

首先导入两个变量`δ(delta)`和`ψ(psi)`，在时刻`t`状态为`i`的所有单个路径`$(i_1, i_2, ..., i_t)$`中概率最大值为:

```math
\delta_{t}(i)=\max _{i_1,i_2,...,i_{t-1}}P(i_t=i,i_{t-1},...,i_1,o_t,...,o_1|\lambda),i=1,2,...,N
```
由定义可得变量`δ`的递推公式：

```math
\begin{aligned}
\delta_{t+1}(i) &= \max _{i_1,i_2,...,i_{t}}P(i_{t+1}=i,i_{t},...,i_1,o_{t+1},...,o_1|\lambda) \\ &= \max_{1 \leq j \leq N}\left[\delta_{t}(j) a_{j i}\right] b_{i}\left(o_{t+1}\right) ,i=1,2,...,N;t=1,2,...,T-1
\end{aligned}
```
定义在时刻`t`状态为`i`的所有单个路径`$(i_1, t_2,...,i_{t-1},i)$`中概率最大的路径的第`t-1`个节点为：

```math
\psi_{t}(i)=\arg \max _{1 \leq j \leq N}\left[\delta_{t-1}(j) a_{j i}\right], \quad i=1,2,...,N
```

##### >举个栗子<

(**维特比算法**)考虑盒子和球模型`λ=(A,B,π)`，状态集合`Q={1,2,3}`，观测集合`V={红,白}`，

```math
A=\left[ \begin{matrix} 0.5 & 0.2 & 0.3 \\ 0.3 & 0.5 & 0.2 \\ 0.2 & 0.3 & 0.5 \end{matrix} \right], B=\left[ \begin{matrix} 0.5 & 0.5 \\ 0.4 & 0.6 \\ 0.7 & 0.3 \end{matrix} \right], \pi=(0.2,0.2,0.4)^T
```
设`T=3`，`O={红,白,红}`，试求最优状态序列，及最优路径`$I^*=\{i^*_1, i^*_2, i^*_3\}$`.

**解**

(1) 初始化。在`t=1`时，对每个状态`i`，`i=1,2,3`，求状态为`i`观测`$o_1$`为红色的概率`$δ_1(i)$`:

```math
\delta_1(1)=\pi_ib_i(o_i)=\pi_ib_i(红), i=1,2,3
```

代入实际数据

```math
\delta_{1}(1)=0.10,\delta_{1}(2)=0.16,\delta_{1}(3)=0.28
```

记

```math
\psi_{1}(i)=0,i=1,2,3
```

(2) 递推。在`t=2`时，对每个状态`i`，`i=1,2,3`，求在`t=1`时状态为`j`观测为红并在`t=2`时状态为`i`观测`$o_2$`为白的路径的最大概率`$δ_2(i)$`：

```math
\delta_{2}(i)=\max _{1 \leq j \leq 3}\left[\delta_{1}(j) a_{j i}\right] b_{i}\left(o_{2}\right)
```

同时，对每个状态`i`，`i=1,2,3`，记录概率最大路径的前一个状态`j`：

```math
\psi_{2}(i)=\arg \max _{1 \leq j \leq 3}\left[\delta_{1}(j) a_{j i}\right], \quad i=1,2,3
```

计算：

```math
\begin{aligned} \delta_{2}(1) &=\max _{1 \leq j \leq 3}\left[\delta_{1}(j) a_{j 1}\right] b_{1}\left(o_{2}\right) \\ &=\max _{j}\{0.10 \times 0.5,0.16 \times 0.3,0.28 \times 0.2\} \times 0.5 \\ &=0.028 \\ \psi_{2}(1) &=3 \\ \delta_{2}(2) &=0.0504, \quad \psi_{2}(2)=3 \\ \delta_{2}(3) &=0.042, \quad \psi_{2}(3)=3 \end{aligned}
```

同样，在`t=3`时，

```math
\begin{aligned} \delta_{3}(i) &=\max _{1 \leq j \leq 3}\left[\delta_{2}(j) a_{j i}\right] b_{i}\left(o_{3}\right) \\ \psi_{3}(i) &=\arg \max _{1 \leq j \leq 3}\left[\delta_{2}(j) a_{j i}\right] \\ \delta_{3}(1) &=0.00756, \quad \psi_{3}(1)=2 \\ \delta_{3}(2) &=0.01008, \quad \psi_{3}(2)=2 \\ \delta_{3}(3) &=0.0147, \quad \psi_{3}(3)=3 \end{aligned}
```

(3) 终止。以`P^*`表示最优路径的概率，则：

```math
P^{*}=\max _{1 \leq i \leq 3} \delta_{3}(i)=0.0147
```

最优路径的终点是`$i^*_3$`:

```math
i_{3}^{*}=\arg \max _{i}\left[\delta_{3}(i)\right]=3
```

(4) 最优路径回溯。由最优路径的终点是`$i^*_3$`，逆向找到`$i^*_2，i^*_1$`：

```math
\begin{aligned} 在t=1时, i_{2}^{*} &=\psi_{3}\left(i_{3}^{*}\right)=\psi_{3}(3)=3 \\ 在t=2时, i_{1}^{*} &=\psi_{2}\left(i_{2}^{*}\right)=\psi_{2}(3)=3 \end{aligned}
```

于是求得最优路径，即最优状态序列`$I^*=(i^*_1,i^*_2,i^*_3)=(3,3,3)$`。

### Q1：隐马尔科夫模型适用范围？

1. 我们的问题是基于序列的，比如时间序列，或者状态序列。
2. 我们的问题中有两类数据，一类序列数据是可以观测到的，即观测序列；而另一类数据是不能观察到的，即隐藏状态序列，简称状态序列。

### Q2：Viterbi算法 编程

[源码见Github](https://github.com/WerterHong/Machine-Learning-Algorithm-NLP/blob/master/code/HMM_Viterbi.py)。

### Q3：隐马尔科夫模型的三个重要假设

1. 马尔可夫假设（构成1阶马尔可夫链）

```math
P\left(X_{t} | X_{t-1} \ldots X_{1}\right)=P\left(X_{t} | X_{t-1}\right), t=1,2,...,T
```

2. 观测独立性假设（状态与具体时间无关）

```math
P\left(X_{i+1}=S_j | X_{i}=S_i\right)=P\left(X_{j+1}=S_j | X_{j}=S_i\right), \quad \forall i, j
```

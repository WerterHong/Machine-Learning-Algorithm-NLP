## 隐马尔科夫模型 HMM

[NLP-机器学习笔试面试题解析]Github链接(https://github.com/WerterHong/Machine-Learning-Algorithm-NLP/tree/master/机器学习算法/)

### 1. 隐马尔科夫模型简述

隐马尔科夫模型(`hidden Markov model, HMM`)是关于时序的概率模型，描述由一个**隐藏的马尔可夫链**随机生成**不可观测状态随机序列**，再由各个状态生成一个观测而产生**观测随机序列**的过程。

`HMM`是一个**生成模型**表示状态序列和观测序列的联合分布，但是状态序列是隐藏的，不可观测的。

> 隐马尔科夫模型`HMM`由**初始状态概率向量**`π`、**状态转移概率矩阵**`A`以及**观测概率矩阵**`B`确定。`λ =(A,B,π)`

设`Q`是所有可能状态的集合，`V`是对应的观测序列，即：

<p align="center">
<img src="../img/HMM/HMM-18.png" />
</p>

其中，`N`是可能的状态数，`M`是可能观测的数。另外设`I`是长度为`T`的状态序列，`O`是对应的观测序列：

<p align="center">
<img src="../img/HMM/HMM-17.png" />
</p>

**状态转移概率矩阵**`A`为：

<p align="center">
<img src="../img/HMM/HMM-16.png" />
</p>

其中,

<p align="center">
<img src="../img/HMM/HMM-15.png" />
</p>

是在时刻`t`处于状态`$q_i$`的条件下在`t+1`转移到状态`$q_j$`的概率。

**观测概率矩阵**`B`为：

<p align="center">
<img src="../img/HMM/HMM-14.png" />
</p>

其中，

<p align="center">
<img src="../img/HMM/HMM-13.png" />
</p>

是在时刻`t`处于状态`$q_i$`的条件下生成观测`$v_k$`的概率。

**初始状态概率向量**`$\pi$`为：

<p align="center">
<img src="../img/HMM/HMM-12.png" />
</p>

其中，

<p align="center">
<img src="../img/HMM/HMM-11.png" />
</p>

是时刻`t=1`处于状态`$q_i$`的概率。

- ##### 隐马尔科夫模型基本假设：
1. **齐次马尔可夫性假设**，即假设隐藏的马尔可夫链在任意时刻`t`的状态只依赖于前一刻的状态，与其他时刻的状态及观测无关，也与时刻`t`无关。

<p align="center">
<img src="../img/HMM/HMM-10.png" />
</p>

2. **观测独立性假设**，即假设任意时刻的观测只依赖于该时刻的马尔可夫链的状态，与其他观测及状态无关。

<p align="center">
<img src="../img/HMM/HMM-9.png" />
</p>

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

<p align="center">
<img src="../img/HMM/HMM-8.png" />
</p>

确定转移的盒子后，再从这个盒子里随机抽出1个球，记录其颜色，放回；如此重复5次，得到一个球的颜色的观测序列为`O={红,红,白,白,红}`

> 两个随机序列：盒子的序列（状态序列）、球的颜色的序列（观测序列）。前者式隐藏的，只有后者是可以观测的。可以明确：状态集合、观测集合以及模型的三要素。

盒子对应状态，状态的集合是`Q={盒子1,盒子2,盒子3,盒子4}, N=4`

球的颜色对应观测，观测的集合是`V={红,白}, V=2`

状态序列和观测序列长度`T=5`。初始概率分布为

<p align="center">
<img src="../img/HMM/HMM-7.png" />
</p>

观测概率分布为：

<p align="center">
<img src="../img/HMM/HMM-6.png" />
</p>

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

<p align="center">
<img src="../img/HMM/HMM-5.png" />
</p>

设`T=3`，`O={红,白,红}`，试用前向算法计算`P(O|λ)`。

**解**

(1) 计算初值

<p align="center">
<img src="../img/HMM/HMM-4.png" />
</p>

(2) 递推计算

<p align="center">
<img src="../img/HMM/HMM-3.png" />
</p>

(3) 终止

<p align="center">
<img src="../img/HMM/HMM-2.png" />
</p>

### 3. 预测算法

#### Viterbi 算法

**维特比算法**是用**动态规划**求概率最大路径（最优路径），一条路径对应一个状态序列。

根据动态规划原理，最优路径具有这样的特性：如果最优路径在时刻`t`通过节点`$i^*_t$`，那么这一路径从节点`$i^*_t$`到终点`$i^*_T$`的部分路径，对于从`$i^*_t$`到`$i^*_T$`的所有可能的部分路径来说，必须是最优的。

首先导入两个变量`δ(delta)`和`ψ(psi)`，在时刻`t`状态为`i`的所有单个路径`$(i_1, i_2, ..., i_t)$`中概率最大值为:

<p align="center">
<img src="../img/HMM/HMM-1.png" />
</p>

由定义可得变量`δ`的递推公式：

<p align="center">
<img src="../img/HMM/HMM-19.png" />
</p>

定义在时刻`t`状态为`i`的所有单个路径`$(i_1, t_2,...,i_{t-1},i)$`中概率最大的路径的第`t-1`个节点为：

<p align="center">
<img src="../img/HMM/HMM-20.png" />
</p>

##### >举个栗子<

(**维特比算法**)考虑盒子和球模型`λ=(A,B,π)`，状态集合`Q={1,2,3}`，观测集合`V={红,白}`，

<p align="center">
<img src="../img/HMM/HMM-5.png" />
</p>

设`T=3`，`O={红,白,红}`，试求最优状态序列，及最优路径`$I^*=\{i^*_1, i^*_2, i^*_3\}$`.

**解**

(1) 初始化。在`t=1`时，对每个状态`i`，`i=1,2,3`，求状态为`i`观测`$o_1$`为红色的概率`$δ_1(i)$`:

<p align="center">
<img src="../img/HMM/HMM-21.png" />
</p>

代入实际数据

<p align="center">
<img src="../img/HMM/HMM-22.png" />
</p>

记

<p align="center">
<img src="../img/HMM/HMM-23.png" />
</p>

(2) 递推。在`t=2`时，对每个状态`i`，`i=1,2,3`，求在`t=1`时状态为`j`观测为红并在`t=2`时状态为`i`观测`$o_2$`为白的路径的最大概率`$δ_2(i)$`：

<p align="center">
<img src="../img/HMM/HMM-24.png" />
</p>

同时，对每个状态`i`，`i=1,2,3`，记录概率最大路径的前一个状态`j`：

<p align="center">
<img src="../img/HMM/HMM-25.png" />
</p>

计算：

<p align="center">
<img src="../img/HMM/HMM-26.png" />
</p>

同样，在`t=3`时，

<p align="center">
<img src="../img/HMM/HMM-27.png" />
</p>

(3) 终止。以`P^*`表示最优路径的概率，则：

<p align="center">
<img src="../img/HMM/HMM-28.png" />
</p>

最优路径的终点是`$i^*_3$`:

<p align="center">
<img src="../img/HMM/HMM-29.png" />
</p>

(4) 最优路径回溯。由最优路径的终点是`$i^*_3$`，逆向找到`$i^*_2，i^*_1$`：

<p align="center">
<img src="../img/HMM/HMM-30.png" />
</p>

于是求得最优路径，即最优状态序列`$I^*=(i^*_1,i^*_2,i^*_3)=(3,3,3)$`。

### Q1：隐马尔科夫模型适用范围？

1. 我们的问题是基于序列的，比如时间序列，或者状态序列。
2. 我们的问题中有两类数据，一类序列数据是可以观测到的，即观测序列；而另一类数据是不能观察到的，即隐藏状态序列，简称状态序列。

### Q2：Viterbi算法 编程

[源码见Github.](https://github.com/WerterHong/Machine-Learning-Algorithm-NLP/blob/master/code/HMM_Viterbi.py)

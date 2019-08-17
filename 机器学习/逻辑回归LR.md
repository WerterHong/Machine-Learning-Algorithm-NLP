## 逻辑回归 LR

- NLP-机器学习笔试面试题解析 [Github链接](https://github.com/WerterHong/Machine-Learning-Algorithm-NLP/)
- **逻辑回归 LR** (若公式显示错误，请点击此链接) [有道云笔记](http://note.youdao.com/noteshare?id=f13ce3b593a4a8b55909d1eee8fff4c8&sub=8A27C7DCB0224437AD99C7068D5D0313)

### 1. Logistic Regression 简述

Logistic regression 用来解决二分类问题，它假设数据服从伯努利分布，即输出为**正 负**两种情况，概率分别为`p`和 `1-p`，**目标函数** `$h_θ(x;θ)$` 是对`p`的模拟，`p`是个概率，这里用了 `p＝sigmoid` 函数，所以目标函数为：

```math
h_{\theta}(x ; \theta)=\frac{1}{1+e^{-\theta^{T} x}}
```

为什么用 sigmoid 函数？请看：[Logistic regression 为什么用 sigmoid](https://www.jianshu.com/p/5fd6a6740989)

损失函数是由极大似然得到，记：

```math
\begin{array}{l}{P(y=1 | x ; \theta)=h_{\theta}(x)} \\ {P(y=0 | x ; \theta)=1-h_{\theta}(x)}\end{array}
```

则可统一写成：

```math
p(y | x ; \theta)=\left(h_{\theta}(x)\right)^{y}\left(1-h_{\theta}(x)\right)^{1-y}
```

似然函数为：

```math
L(\theta)=\prod_{i=1}^mp(y^{(i)}|x^{(i)};\theta)=\prod_{i=1}^m\left(h_{\theta}(x^{(i)})\right)^{y^{(i)}}\left(1-h_{\theta}(x^{(i)})\right)^{1-y^{(i)}}
```

取对数：

```math
\begin{aligned} \ell(\theta) &=\log L(\theta) \\ &=\sum_{i=1}^{m} y^{(i)} \log h_{\theta}\left(x^{(i)}\right)+\left(1-y^{(i)}\right) \log \left(1-h_{\theta}\left(x^{(i)}\right)\right) \end{aligned}
```

**求解参数可以用梯度上升**：

先求偏导：

```math
\begin{aligned} \frac{\partial}{\partial \theta_{j}} \ell(\theta) &=\left(y \frac{1}{g\left(\theta^{T} x\right)}-(1-y) \frac{1}{1-g\left(\theta^{T} x\right)}\right) \frac{\partial}{\partial \theta_{j}} g\left(\theta^{T} x\right) \\ &=\left(y \frac{1}{g\left(\theta^{T} x\right)}-(1-y) \frac{1}{1-g\left(\theta^{T} x\right)}\right) g\left(\theta^{T} x\right)\left(1-g\left(\theta^{T} x\right)\right) \frac{\partial}{\partial \theta_{j}} \theta^{T} x\\ &=\left(y\left(1-g\left(\theta^{T} x\right)\right)-(1-y) g\left(\theta^{T} x\right)\right) x_{j} \\ &=\left(y-h_{\theta}(x)\right) x_{j} \end{aligned}
```
其中

```math
h_{\theta}(x)=g\left(\theta^{T} \mathbf{x}\right)

\frac{\partial}{\partial x}\left(\frac{1}{1+e^{-x}}\right)=\frac{e^{-x}}{\left(1+e^{-x}\right)^{2}}
```

再梯度更新：

```math
\theta_{j} :=\theta_{j}+\alpha\left(y^{(i)}-h_{\theta}\left(x^{(i)}\right)\right) x_{j}^{(i)}
```

常用的是**梯度下降最小化负的似然函数**。

### 2. 常用的损失函数

|  损失函数  |         举例         |                作用                | 公式 |
|:----------:|:--------------------:|:----------------------------------:|:----:|
|   0-1损失  | 用于分类，例如感知机 |  预测值和目标值不相等为1，否则为0  |   `$L(y, f(x)) = \begin{cases} 1, &  {y \neq f(x) } \\ 0, & {y = f(x)} \end{cases}$`   |
| 绝对值损失 |                      |    取绝对值，差距不会被平方缩放    |   `$L(y, f(x)) = | y -f(x) |$`   |
|  平方损失  |   Linear Regression  |  使得所有点到回归直线的距离和最小  |   `$L(y, f(x)) = (y - f(x))^2$`   |
|  对数损失  |  Logistic Regression | 常用于模型输出为每一类概率的分类器 |   `$L(y, p(y|x)) = - \log p(y|x)$`   |
|  Hinge损失 |         SVM          |          用于最大间隔分类          |   `$L(w,b) = max \{0, 1-yf(x) \}$`   |
|  指数损失  |       AdaBoost       |          主要用于Adaboost          |   `$L(Y,f(X))=\exp(-Yf(X))$`   |

几种损失函数的曲线：

<p align="center">
<img src="https://note.youdao.com/yws/public/resource/eb07e5c345811d5ce8374097a63f820d/B1DFA02B5A8142BDB89BBB87ECAF9CCF?ynotemdtimestamp=1564217457782" />
</p>

- 黑色：`0-1`损失函数
- 绿色：合页损失函数`Hinge Loss`中，当`yf(x)>1`时，其损失`=0`，当 `yf(x)<1`时，其损失呈线性增长（正好符合`svm`的需求）
- 红色为逻辑损失`Log`、蓝色为指数损失 `Exponential`： 在 合页`Hinge`的左侧都是凸函数，并且`0-1`损失为它们的下界

要求最大似然时(即概率最大化)，使用`Log Loss`最合适，一般会加上负号，变为求最小。损失函数的凸性及有界很重要，有时需要使用代理函数来满足这两个条件。

### Q1. LR损失函数为什么用极大似然函数

- 因为我们想要让每一个样本的预测都要得到最大的概率，即将所有的样本预测后的概率进行相乘都最大，也就是极大似然函数；
- 对极大似然函数取对数以后相当于对数损失函数，由上面梯度更新的公式可以看出，对数损失函数的训练求解参数的速度是比较快的，而且更新速度只和x，y有关，比较的稳定。**相乘转换成对数中的相加？**
- 为什么不用平方损失函数。如果使用平方损失函数，梯度更新的速度会和sigmod函数的梯度相关，sigmod函数在定义域内的梯度都不大于0.25，导致训练速度会非常慢。而且平方损失会导致损失函数是theta 的非凸函数，不利于求解，因为非凸函数存在很多局部最优解。

### Q2. 逻辑回归是线性模型？

[Why is logistic regression considered a linear model?](https://www.quora.com/Why-is-logistic-regression-considered-a-linear-model)

#### 解释 1
Logistic regression is considered a generalized linear model because **the outcome always depends on the sum of the inputs and parameters**.

逻辑回归的输出取决于输入和权重参数间的点乘的和，并非取决于参数之间的乘积。

逻辑回归主要解决二分类问题。逻辑回归的输出是一个概率，它代表了一个样本属于类别`1`或者类别`0`。我们的目标函数是最小化逻辑函数`$\Phi()$`（sigmoid函数）：

<p align="center">
<img src="https://note.youdao.com/yws/public/resource/eb07e5c345811d5ce8374097a63f820d/97E25FE01F7C438C8DD246F071DC7946?ynotemdtimestamp=1564217457782" />
</p>

虽然逻辑回归的结果是一个线性决策平面（将样本分为类别`1`或者类别`0`，如下图），但是逻辑（激活）函数看起来并不是线性的？

<p align="center">
<img src="https://note.youdao.com/yws/public/resource/eb07e5c345811d5ce8374097a63f820d/BEA65B6F2DBB48A3857EE4E83890AA2E?ynotemdtimestamp=1564217457782" />
</p>

在神经网络中，网络的输入函数是输入特征`$x_i$`和对应的模型系数`$w_i$`之间的点乘（dot product）：

<p align="center">
<img src="https://note.youdao.com/yws/public/resource/eb07e5c345811d5ce8374097a63f820d/39DF272936A2428B8DCE2C7DA8FE3672?ynotemdtimestamp=1564217457782" />
</p>

```math
z=w_0x_0+w_1x_1+...+w_mx_m=\sum_{i=1}^mw_ix_i=w^Tx
```
其中`$x_0$`指的是偏置单元的权重，它总是等于`1`。

**例**：假设我们有4个特征的样本训练点`x=[1,2,3,4]`，权重向量`w=[0.5,0.5,0.5,0.5]`，则`z`的值为

```math
z=w^Tx=1*0.5+2*0.5+3*0.5+4*0.5=5
```

**重点来了**！！样本属于类别`1`的概率为

```math
\Phi(z=5)=\frac{1}{1+e^{-5}}=0.993
```
逻辑回归产生线性决策边界的原因是参数（输入特征和模型系数/权重）的可加性：**结果`z`取决于参数的可加性**，例如:

```math
z=w_1x_1+w_2x_2+...
```
参数权重之间没有相互作用，没有像`$w_1*x_1*w_2*x_2$`那样，这样的模型是**非线性**的！

#### 解释 2
It's because the **decision boundary** is linear in x.

具体地说，在这种情况下的**决策边界**（超平面）由`$w^Tx=0$`给出（类似于支持向量机（SVM）在正例和反例之间的函数/几何边界）。

如果生成的是非线性决策边界，则为逻辑回归的非线性实例（类似于线性SVMs和非线性SVMs）。换句话说，我们需要先确定原始数据`X`是不是线性可分的并使用特征转换函数`h(X)`来代替`X`。

例如，使用二次特征转换函数`$h(x)$`对二维数据进行转换：

```math
h^T(x)=\{{{x_1}, {x_2}, {x_1^2}, {x_2^2}, {x_1x_2}}\}
```
现在逻辑模型`$y=f(w^Th(x))$`的决策边界`$w^Th(x)=0$`(原始数据空间中的非线性二次曲线)。

### Q3. 多项逻辑回归模型

将逻辑斯谛回归模型推广为多项逻辑斯谛回归模型，用于多分类问题。假设离散型随机变量`Y`的取值集合是`{1,2,...,K}`，则多项逻辑斯谛回归模型(softmax函数)为

```math
P(Y=k|x)=\frac{\exp(w_k·x)}{1+\sum_{k=1}^K\exp(w_k·x)},k=1,2,...,K-1

P(Y=K|x)=\frac{1}{1+\sum_{k=1}^K\exp(w_k·x)}
```
其中，`$x \in R^{n+1}$`,`$w_k \in R^{n+1}$`.

考虑`$N$`个类别`$\{C_1,C_2,…,C_N\}$`，**多分类学习**的基本思想是“拆解法”，即将多酚类任务拆分为若干个二分类任务求解。最经典的拆分策略有三种：
- “一对一”（One vs. One，简称OvO）
- “一对其余”（One vs. Rest，简称OvR）
- “多对多”（Many vs. Many，简称MvM）。

给定数据集`$D=\{(x_1,y_1),(x_2,y_2),...(x_my_m)\}$`，`$y_i \in \{C_1,C_2,...,C_N\}$`,
- `OvO`将这`N`个类别两两配对，从而产生`N(N-1)/2`个二分类任务,例如`OvO`将为区分类别`$C_i$`和`$C_j$`训练一个分类器，该分类器把`D`中的`$C_i$`类样例作为正例，`$C_j$`类样例作为反例。在测试阶段，新样本将同时提交给所有分类器，于是我们将得到`N(N-1)/2`个分类结果，最终结果可通过投票产生：即把被预测得最多的类别作为最终分类结果。
- `OvR`则是每次将一个类的样例作为正例，所有其他类的样例作为反例来训练`N`个分类器，在测试时若仅有一个分类器预测为正类，则对应的类别标记作为最终分类结果。

<p align="center">
<img src="https://images2018.cnblogs.com/blog/1159663/201803/1159663-20180304193607996-1146250910.png" width="600" />
</p>

> OvO的优点是，在类别很多时，训练时间要比OvR少。缺点是，分类器个数多。
>
> OvR的优点是，分类器个数少，存储开销和测试时间比OvO少。缺点是，类别很多时，训练时间长。

- `MvM`是每次将若干个类作为正类，若干个其他类作为反类。显然，`OvO`和`OvR`是`MvM`的特例。MvM的正、反类构造必须有特殊的设计，不能随意选取。这里我们介绍一种最常用的`MvM`技术"纠错输出码" (Error Correcting Output Codes，简称`ECOC`)。

> `ECOC`是将编码的思想引入类别拆分，并尽可能在解码的过程中具有容错性。`ECOC`工作过程主要分为两步：
>
> 1. 编码：对`N`个类别做`M`次划分，每次划分将一部分类别划为正类，一部分划为反类，从而形成一个二分类训练集，这样一共产生`M`个训练集，可训练出`M`个训练器。
>
> 2. 解码：`M`个分类器分别对测试样本进行预测，这些预测标记组成一个编码。将这个与此编码与每个类别各自的编码进行比较，返回其中距离最小的类别最为最终预测结果。

<p align="center">
<img src="https://note.youdao.com/yws/public/resource/eb07e5c345811d5ce8374097a63f820d/4AC198B7EE0E4B74BC62B968BAD376F1?ynotemdtimestamp=1564217457782" width="600" />
</p>

### Q4. 逻辑回归模型推导

[见Logistic Regression简述](https://github.com/WerterHong/Machine-Learning-Algorithm-NLP/tree/master/机器学习算法/逻辑回归LR.md)

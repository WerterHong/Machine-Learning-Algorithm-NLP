## BP算法

- NLP-机器学习笔试面试题解析 [Github链接](https://github.com/WerterHong/Machine-Learning-Algorithm-NLP/)
- **BP算法** (若公式显示错误，请点击此链接) [有道云笔记](http://note.youdao.com/noteshare?id=a2b90803e7396be4bd69bda62ba77562&sub=0EC7B5D646CA4C83B09AB6553D4B8B77)

### 1. BP算法

==误差逆传播算法==（`error BackPropagation`，`BP`算法）基于**梯度下降**（`gradient descent`）策略，以**目标的负梯度方向**对参数进行调整。

<p align="center">
    <img src="https://note.youdao.com/yws/public/resource/a2b90803e7396be4bd69bda62ba77562/820AD60EC2EC4B69870D301E37504CAB?ynotemdtimestamp=1565679038132" width=650 />
    <br/>
    <strong>Fig</strong>. 前馈 与 反向
</p>

> 标准`BP`算法 vs. 累积`BP`算法
>
> 标准`BP`算法：每次更新只针对单个样例，参数更新频繁，需要更多次数的迭代
>
> 累积`BP`算法：直接针对累积误差最小化，参数更新频率低，累积误差下降到一定程度后，进一步下降会非常缓慢

`BP`神经网络解决==过拟合==问题：

1. **早停**（`early stopping`）：将数据分为训练集和验证集，训练集用来计算梯度、更新连接权和阈值，验证集用来估计误差，若训练集误差降低但验证集误差升高，则停止训练，同时返回具有最小验证集误差的连接权和阈值。
2. **正则化**（`regularization`）：在误差目标函数中增加描述网络复杂度的项，例如连接权和阈值的平方和

> **局部极小与全局最小**

<p align="center">
    <img src="https://note.youdao.com/yws/public/resource/a2b90803e7396be4bd69bda62ba77562/F35F8FAADA0F406A9B5C23878B8E65A6?ynotemdtimestamp=1565679038132" width=500 />
    <br/>
    <strong>Fig</strong>. 局部极小与全局最小
</p>

> ==基于梯度的搜索==：从初始解出发，迭代寻找最优参数值。
> 
> 每次迭代中，先计算误差函数在当前的梯度，根据梯度确定搜索方向（由于负梯度时函数值下降最快的方向，因此**梯度下降就是沿着负梯度方向搜索最优解**）。若误差函数在当前的梯度为零。则以达到局部极小，更新量将为零，迭代的参数更新就此停止。
>
> `Solutions` : 
> - 以**多组不同的参数初始化多个神经网络**，只能找标准方法训练后，取其中误差最小的解作为最终参数（陷入不同的局部极小，从中选择有可能最接近全局最小的结果）
> - **模拟退火**（`simulated annealing`）技术，在每一步都以一定的概率接受比当前解更差的结果，从而有助于“跳出”局部极小。迭代过程中，接受“次优解”的概率要随着时间的推移逐渐降低，从而保证算法的稳定性
> - 使用**随机梯度下降**。即使陷入局部极小点，它计算出的梯度仍可能不为零，就有机会“跳出”局部极小值继续搜索

<p align="center">
    <img src="https://note.youdao.com/yws/public/resource/a2b90803e7396be4bd69bda62ba77562/A42053CA697541A6BD077D6D8E9F9381?ynotemdtimestamp=1565679038132" width=500 />
    <br/>
    <strong>Fig</strong>. 梯度下降 与 随机梯度下降
</p>

> 此外，**遗传算法**（`genetic algorithms`）等也可以用来训练神经网络以更好的逼近全局最小。
## Word2vec

- NLP-机器学习笔试面试题解析 [Github链接](https://github.com/WerterHong/Machine-Learning-Algorithm-NLP/)
- **word2vec** (若公式显示错误，请点击此链接) [有道云笔记](http://note.youdao.com/noteshare?id=d3385e2e4599150c59a78cfd34abdfb6&sub=7516C8DB76FE4331B749885749BFE467)

`word2vec`工具主要包含两个模型：==跳字模型==（`skip-gram`）和==连续词袋模型==（`CBOW`），以及两种高效训练的方法：==负采样==（`negative sampling`）和==层序==`softmax`（`hierarchical softmax`）

### 1. 预备知识

- [sigmiod函数](http://note.youdao.com/noteshare?id=61e22de3ed9ed9bb6d733fedf7245dbc&sub=81109D23317D485CA1FC1511699E9B6E)
- [逻辑回归](http://note.youdao.com/noteshare?id=eb07e5c345811d5ce8374097a63f820d&sub=22565B3DE3F74ADCA35D71663998C79D)
- [Bayes 公式](http://note.youdao.com/noteshare?id=0693c471d86002cc0bae2403a96e6632&sub=B89A3590BCC74A21BAA0183AE3418CDD)
- Huffman树

### 2. 参考知识

#### [2.1 统计语言模型 `SLM`]

统计语言模型是用来计算一个句子的**概率模型**，它通常基于一个语料库来构建。

假设`$W = w _ { 1 } ^ { T } : = ( w _ { 1 } , w _ { 2 } , ... , w _ { T } )$`表示由`T`个词`$w_{1}, w_{2}, ..., w_{T}$`按顺序构成一个句子，则`$w_{1}, w_{2}, \cdots, w_{T}$`的联合概率
```math
p(W)=p\left(w_{1}^{T}\right)=p\left(w_{1}, w_{2}, \cdots, w_{T}\right)
```
就是这个句子的概率。利用**Bayes公式**，上式可以被链式分解为：

```math
p\left(w_{1}^{T}\right)=p(w_{1}) \cdot p(w_{2} | w_{1}) \cdot p(w_{3} | w_{1}^{2}) \cdot p(w_{T} | w_{1}^{T-1})
```
其中(条件)概率`$p(w_{1}) , p(w_{2} | w_{1}) , p(w_{3} | w_{1}^{2}) , p(w_{T} | w_{1}^{T-1})$`就是`SLM`**的参数**。

#### [2.2 n-gram模型]

考虑`$p(w_k|w_1^{k-1})(k>1)$`的近似计算，利用`Bayes`公式，有

```math
p\left(w_{k} | w_{1}^{k-1}\right)=\frac{p\left(w_{1}^{k}\right)}{p\left(w_{1}^{k-1}\right)}
```
根据大数定理，当语料库足够大时，`$p(w_k|w_1^{k-1})(k>1)$`可近似地表示为：

```math
p\left(w_{k} | w_{1}^{k-1}\right) \approx \frac{\operatorname{count}\left(w_{1}^{k}\right)}{\operatorname{count}\left(w_{1}^{k-1}\right)}
```
其中`$count(w_1^k)$`和`$count(w_1^{k-1})$`分别表示词串`$w_1^k$`和`$w_1^{k-1}$`在语料中出现的次数。**`k`很大时，`count`统计将会很耗时**。

`n-gram`**模型**作了一个`n-1`阶的`Markov`**假设**，认为一个词出现的概率就只与它前面的`n-1`个词相关，即：

```math
p\left(w_{k} | w_{1}^{k-1}\right) \approx p\left(w_{k} | w_{k-n+1}^{k-1}\right)
```
有

```math
p\left(w_{k} | w_{1}^{k-1}\right) \approx \frac{\operatorname{count}\left(w_{k-n+1}^{k}\right)}{\operatorname{count}\left(w_{k-n+1}^{k-1}\right)}
```
以`n=2`为例，就有

```math
p\left(w_{k} | w_{1}^{k-1}\right) \approx \frac{\operatorname{count}\left(w_{k-1}, w_{k}\right)}{\operatorname{count}\left(w_{k-1}\right)}
```
这样使得但各参数的统计变得容易(统计时**需要匹配的词串更短**)，也使得**参数的总数变少了**。

<p align="center">
模型参数数量与n的关系
<table>
    <tr>
        <td>n</td>
        <td>模型参数数量</td>
    </tr>
    <tr>
        <td>1(unigram)</td>
        <td>2*10^5</td>
    </tr>
    <tr>
        <td>2(bigram)</td>
        <td>4*10^10</td>
    </tr>
    <tr>
        <td>3(trigram)</td>
        <td>8*10^15</td>
    </tr>
    <tr>
        <td>4(4-gram)</td>
        <td>16*10^20</td>
    </tr>
</table>
</p>

在`n-gram`模型中还有一个叫做**平滑化**的重要环节，考虑
1. 若`$count(w_{k-n+1}^{k})=0$`，能否认为`$p(w_k|w_1^{k-1})$`就等于`0`呢？
2. 若`$count(w_{k-n+1}^{k})=0=count(w_{k-n+1}^{k-1})=0$`，能否认为`$p(w_k|w_1^{k-1})$`就等于`0`呢？

**不能**。[具体可参考《数学之美》 吴军著]

`n-gram`模型时在语料**中统计词串出现的次数**以及**平滑处理**。概率值计算好之后就存储起来，下次需要计算一个句子的概率时，只需找到相关的相关参数，将它们连乘起来。

对统计语言模型而言，利用**最大对数似然**，可把目标函数设为：
```math
\mathcal{L}=\sum_{w \in C} \log p(w | \operatorname{Context}(w))
```
其中，`C`表示语料，`Context(w)`表示词`w`的上下文，即`w`周边的的词的集合。对于`n-gram`，有`$Context(w_i)=w_{i-n+1}^{i-1}$`。

#### [2.3 神经概率语言模型]

相较于`n-gram`的优势：
1. **词语之间的相似性**可以通过**词向量**来体现
2. 基于词向量的模型**自带平滑化功能**，不需要像`n-gram`那样进行额外处理。

<p align="center">
<img src="https://note.youdao.com/yws/public/resource/d3385e2e4599150c59a78cfd34abdfb6/FD01B07137484F5DA4FC126A47C3989D?ynotemdtimestamp=1564373669137" />
</p>

<p align="center">
<strong>Fig</strong>. 神经概率语言模型
</p>

#### [2.4 词向量]

##### 独热表示(One-Hot)

`One-Hot`编码使用`N`位状态寄存器来对`N`个状态进行编码，每个状态都有它独立的寄存器位，并且在任意时候，其中只有一位有效。

例：

考虑以下的三个特征：
`["male", "female"]、
["from Europe", "from US", "from Asia"]、
["uses Firefox", "uses Chrome", "uses Safari", "uses Internet Explorer"]`

将它换成独热编码后，应该是：
`feature1=[01,10]、
feature2=[001,010,100]、
feature3=[0001,0010,0100,1000]`

- **优点**
1. 解决了分类器不好处理离散数据的问题
2. 在一定程度上也起到了扩充特征的作用
- **缺点**
1. 任意两个词之间都是==孤立==的，根本无法表示出在==语义层面上词语词之间的相关信息==，而这一点是致命的。
2. 我们的词汇表一般都非常大，比如达到百万级别，这样每个词都用百万维的向量来表示简直是==内存的灾难==。

##### 分布式表示(distribution representation)

**分布式表示**的思路是通过训练，将每个词都映射成一个固定长素的短向量。所有的这些词向量就构成了**向量空间**，在这个空间引入“距离”，就可以根据词之间的距离来判断它们之间的(词法、语义上的)相似性了。

**分布式表示模型**用来==对抗维数灾难==，通过训练语句对指数级语义相关的句子进行建模，该模型同时**学习每个单词的分布式表示以及单词序列的概率函数**。

---

> `word2vec`就是采用分布式表示的词向量。`word2vec`有两个重要模型`CBOW`和`skip-gram`。对于`CBOW`和`skip-gram`，`word2vec`给出了两套框架，它们分别基于层序`softmax`(`Hierarchiral Softmax`)和`Negative Sampling`。

### 3. 基于Hierarchiral softmax的模型

#### 3.1 基本思想

<p align="center">
<img src="https://note.youdao.com/yws/public/resource/d3385e2e4599150c59a78cfd34abdfb6/703BF6D9FD7C4E998EF0B936093BF127?ynotemdtimestamp=1564373669137" width=500 />
</p>

<p align="center">
<strong>Fig</strong>. Hierarchiral softmax
</p>

**约定**：==将一个节点进行分类时，分到左边就是负类，分到右边就是正类==。

一个节点被分为正类的概率为

```math
\sigma\left(\mathbf{x}_{w}^{\top} \theta\right)=\frac{1}{1+e^{-\mathbf{x}_{w}^{\top} \theta}}
```
被分为负类的概率就等于

```math
1-\sigma\left(\mathbf{x}_{w}^{\top} \theta\right)
```
对于从根节点出发到达“足球”这个叶子节点所经历的4次分类，将分类结果的概率写出来就是：
1. 第一次
```math
p\left(d_{2}^{w} | \mathbf{x}_{w}, \theta_{1}^{w}\right)=1-\sigma\left(\mathbf{x}_{w}^{\top} \theta_{1}^{w}\right)
```
2. 第二次
```math
p\left(d_{3}^{w} | \mathbf{x}_{w}, \theta_{2}^{w}\right)=\sigma\left(\mathbf{x}_{w}^{\top} \theta_{2}^{w}\right)
```
3. 第三次
```math
p\left(d_{4}^{w} | \mathbf{x}_{w}, \theta_{3}^{w}\right)=\sigma\left(\mathbf{x}_{w}^{\top} \theta_{3}^{w}\right) )
```
4. 第四次
```math
p\left(d_{5}^{w} | \mathbf{x}_{w}, \theta_{4}^{w}\right)=1-\sigma\left(\mathbf{x}_{w}^{\top} \theta_{4}^{w}\right)
```
所以要求的`p(足球|context(足球))`为
```math
p(足球|context(足球))=\prod_{j=2}^{5} p\left(d_{j}^{w} | \mathbf{x}_{w}, \theta_{j-1}^{w}\right)
```

这就是`Hierarchiral softmax`的基本思想。

由图可见，两个模型都包括三层：**输入层**、**投影层**和**输出层**，前者是在已知前词`$w_t$`的上下文`$context:\{w_{t-2},w_{t-1},w_{t+1}.w_{w+2}\}$`的前提下预测当前词；而后者是在一只当前词`$w_t$`的前提下，预测其上下文`$context:\{w_{t-2},w_{t-1},w_{t+1}.w_{w+2}\}$`。

<p align="center">
<img src="https://note.youdao.com/yws/public/resource/d3385e2e4599150c59a78cfd34abdfb6/65E3B7479C14418784B624098AE2A4CD?ynotemdtimestamp=1564373669137" width=600 />
</p>

<p align="center">
<strong>Fig</strong>. CBOW和skip-gram
</p>

#### 3.2 CBOW 模型

`CBOW`模型的网络模型分为三层：输出层、投影层和输出层，以样本`(ontext(w),w)`为例（这里 `context(w)`由`w`前后各`c`个词组成）：

1. **输出层**：包含`context(w)`中`2c`个词的词向量`$v(context(w)_1),v(context(w)_2),...,v(context(w)_{2c}) \in R^m$`，这里`m`表示词向量的长度
2. **投影层**：将输出层的`2c`个向量做==求和累加==，

```math
\mathrm{x}_{w}=\sum_{i=1}^{2 c} \mathrm{v}\left(\text { Context }(w)_{i}\right) \in \mathbb{R}^{m}
```

3. **输出层**：输出层对应一棵二叉树，它是语料库中出现过的词当叶节点，以==各词在语料中出现的次数当权值==构造出来的==Huffman树==。在这棵`Huffman`树，叶子节点共`N`个，分别对应词典中的词，非叶子节点`N-1`个。

<p align="center">
<img src="https://note.youdao.com/yws/public/resource/d3385e2e4599150c59a78cfd34abdfb6/70AFCF2AEEE047D6A5C116163EB8BC19?ynotemdtimestamp=1564373669137" width=500 />
</p>

<p align="center">
<strong>Fig</strong>. CBOW 模型
</p>


- `CBOW`模型 VS 神经概率语言模型

1. `CBOW`模型==无隐藏层==
2. `CBOW`输出层使用==树形结构==（`Huffman`树）
3. 从输入层到投影层，前者通过==累加求和==，后者通过==拼接==

神经概率语言模型大部分计算集中在==隐藏层和输出层之间的矩阵向量运算==，以及输出层上的`softmax`==归一化向量运算==，而`CBOW`模型针对这些计算复杂度高的地方做了改变。

**梯度计算**详情见 [逻辑回归LR](http://note.youdao.com/noteshare?id=eb07e5c345811d5ce8374097a63f820d&sub=22565B3DE3F74ADCA35D71663998C79D)

> Q: 采用平均贡献会不会更合理？即使用公式：

```math
\mathrm{v}(\widetilde{w}) :=\mathrm{v}(\widetilde{w})+\frac{\eta}{|C o n t e x t(w)|} \sum_{j=2}^{l^{w}} \frac{\partial \mathcal{L}(w, j)}{\partial \mathbf{x}_{w}}, \quad \widetilde{w} \in \operatorname{Context}(w)
```

> Why ?

#### 3.3 Skip-gram 模型

`Skip-gram`模型的网络结构同`CBOW`模型的网络结构一样：

1. **输入层**：只含当前样本的中心词`w`的词向量`$v(m) \in R^m$`
2. **投影层**: ==恒等投影==，将`$v(m) \in R^m$`投影到`$v(m) \in R^m$`，**这个投影层其实是多余的**，方便与`CBOW`模型的网络结构作比较
3. **输出层**：和`CBOW`模型一样，输出层也是一棵`Huffman`树

<p align="center">
<img src="https://note.youdao.com/yws/public/resource/d3385e2e4599150c59a78cfd34abdfb6/C77EB504CFDD4A8A9880649D3E055B9A?ynotemdtimestamp=1564373669137" width=500 />
</p>

<p align="center">
<strong>Fig</strong>. Skip-gram 模型
</p>

==梯度计算也与`CBOW`模型完全类似==。

### 4. 基于Negative Sampling的模型

`Negative Sampling`目的是==提高训练速度==并==改善所得词向量的质量==，不使用`Huffman`树，而是蚕蛹相对简单的==随机负采样==。

#### 4.1 负采样算法

`Negative Sampling`本质是一个==带权采样问题==，高频词被选为负样本的概率就应该比较大，反之，低频词被选中的概率就应该比较小。

> 设词典`D`中的每一个词`w`对应的一个线段`l(w)`，长度为：
```math
\operatorname{len}(w)=\frac{\operatorname{counter}(w)}{\sum_{u \in \mathcal{D}} \operatorname{counter}(u)}
```
> 其中`counter(·)`表示一个词在语料中出现的次数，将这些线段首尾连接在一起，形成长度为1的线段。

<p align="center">
<img src="https://note.youdao.com/yws/public/resource/d3385e2e4599150c59a78cfd34abdfb6/8725606C47464A4383D0658275ECB998?ynotemdtimestamp=1564373669137" width=500 />
</p>

<p align="center">
<strong>Fig</strong>. 带权采样
</p>

`word2vec`在`Negative Sampling`==非等距剖分==的基础上引入区间`[0,1]`上的一个==等距离剖分==，剖分点为`$\{m_j\}_{j=0}^M$`，见下图：

<p align="center">
<img src="https://note.youdao.com/yws/public/resource/d3385e2e4599150c59a78cfd34abdfb6/FF7B4D20DE654CD2A47E83462BA8498F?ynotemdtimestamp=1564373669137" width=500 />
</p>

<p align="center">
<strong>Fig</strong>. Table(·) 映射示意图
</p>

将内部剖分点`$\{m_j\}_{j=1}^{M-1}$`==投影到非等距剖分==上，则可建立`$\{m_j\}_{j=1}^{M-1}$`与区间`$\{I_j\}_{j=1}^{N}$`（`$\{w_j\}_{j=1}^{N}$`）的==映射==关系

```math
I_{i}=\left(l_{i-1}, l_{i}\right], i=1,2, \cdots, N

Table(i)=w_{k}, \quad \text { where } \quad m_{i} \in I_{k}, \quad i=1,2, \cdots, M-1
```

每次生成一个`[1,M-1]`间的随机整数`r`，`Table(r)`就是一个样本。

> Q: 对`w_i`进行负采样时，如果碰巧选到`w_i`自己，怎么办？
>
> A: 跳过去呗！

`word2vec`源码中为词典中的词设置权值时，不是直接使用`counter(w)`，==而是对其作了`α`次幂==，其中`α=3/4`，即

```math
\operatorname{len}(w)=\frac{[\operatorname{counter}(w)]^{\frac{3}{4}}}{\sum_{u \in \mathcal{D}}[\operatorname{counter}(u)]^{\frac{3}{4}}}
```

#### 4.2 CBOW 模型

对于给定的`context(w)`，词`w`就是一个==正样本==，其他词就是==负样本==。负样本选取采用==带权采样==。假定负样本子集`$N E G(w) \neq \emptyset$`，对`$\forall \widetilde{w} \in \mathcal{D}$`，定义

```math
L^{w}(\widetilde{w})=\left\{\begin{array}{ll}{1,} & {\widetilde{w}=w} \\ {0,} & {\widetilde{w} \neq w}\end{array}\right.
```
表示词`$\widetilde{w}$`的标签，即正样本的标签为1，负样本的标签为0.

对于一个给定的正样本`(context(w),w)`，我们希望最大化(式. 1)

```math
g(w)=\prod_{u \in\{w\} \cup N E G(w)} p(u | \operatorname{Context}(w))
```
其中

```math
p(u | \text { Context }(w))=\left\{\begin{array}{ll}{\sigma\left(\mathbf{x}_{w}^{\top} \theta^{u}\right),} & {L^{w}(u)=1} \\ {1-\sigma\left(\mathbf{x}_{w}^{\top} \theta^{u}\right),} & {L^{w}(u)=0}\end{array}\right.
```
或者写成整体表达式(式. 2)

```math
p(u | \text {Context}(w))=\left[\sigma\left(\mathbf{x}_{w}^{\top} \theta^{u}\right)\right]^{L^{w}(u)} \cdot\left[1-\sigma\left(\mathbf{x}_{w}^{\top} \theta^{u}\right)\right]^{1-L^{w}(u)}
```
> Q: 为什么要最大化`g(w)`呢？

> A: 将(式. 2)代入到(式. 1)得到

```math
g(w)=\sigma\left(\mathbf{x}_{w}^{\top} \theta^{w}\right) \prod_{u \in N E G(w)}\left[1-\sigma\left(\mathbf{x}_{w}^{\top} \theta^{u}\right)\right]
```
> 其中`$\sigma(\mathbf{x}_{w}^{\top} \theta^{w})$`表示上下文为`context(w)`时，预测中心词为`w`的概率，而`$\sigma(\mathbf{x}_{w}^{\top} \theta^{u})$`表示上下文为`context(w)`时，预测中心词为`u`的概率，从形式上看，最大化`g(w)`就相当于最大化`$\sigma(\mathbf{x}_{w}^{\top} \theta^{w})$`，同时最小化`$\sigma(\mathbf{x}_{w}^{\top} \theta^{u}),u \in N E G(w)$`。相当于==增大正样本的概率的同时降低负样本的概率==。

对于一个给定的语料库`C`，函数

```math
G=\prod_{w \in \mathcal{C}} g(w)
```
就是整体优化的目标。为方便计算，对`G`取对数，然后使用==随机梯度上升法==进行优化。

#### 4.3 Skip-gram 模型

由`CBOW`模型过渡到`Skip-gram`模型的推到经验，将优化目标函数改写为

```math
G=\prod_{w \in \mathcal{C}} g(w) \Rightarrow G=\prod_{w \in \mathcal{C}} \prod_{u \in C o n t e x t(w)} g(u)
```
其中`$\prod_{u \in C o n t e x t(w)} g(u)$`表示对于一个给定的样本`(w,context(w))`，我们希望最大化的量，`g(u)`的定义为

```math
g(u)=\prod_{z \in\{u\} \cup N E G(u)} p(z | w)
```
其中`NEG(u)`表示处理词`u`时生成的负样本子集，条件概率为

```math
p(z | w)=\left\{\begin{array}{ll}{\sigma\left(\mathbf{v}(w)^{\top} \theta^{z}\right),} & {L^{u}(z)=1} \\ {1-\sigma\left(\mathbf{v}(w)^{\top} \theta^{z}\right),} & {L^{u}(z)=0}\end{array}\right.
```
或者写成整体表达式

```math
p(z | w)=\left[\sigma\left(\mathbf{v}(w)^{\top} \theta^{z}\right)\right]^{L^{u}(z)} \cdot\left[1-\sigma\left(\mathbf{v}(w)^{\top} \theta^{z}\right)\right]^{1-L^{u}(z)}
```
接下来的梯度计算与参数更新与`CBOW`模型一致。

## Word2vec

- NLP-机器学习笔试面试题解析 [Github链接](https://github.com/WerterHong/Machine-Learning-Algorithm-NLP/)
- **word2vec** (若公式显示错误，请点击此链接) [有道云笔记](http://note.youdao.com/noteshare?id=d3385e2e4599150c59a78cfd34abdfb6&sub=7516C8DB76FE4331B749885749BFE467)

`word2vec`工具主要包含两个模型：跳字模型（`skip-gram`）和连续词袋模型（`CBOW`），以及两种高效训练的方法：负采样（`negative sampling`）和层序`softmax`（`hierarchical softmax`）

### 1. 预备知识

- [sigmiod函数](http://note.youdao.com/noteshare?id=61e22de3ed9ed9bb6d733fedf7245dbc&sub=81109D23317D485CA1FC1511699E9B6E)
- [逻辑回归](http://note.youdao.com/noteshare?id=eb07e5c345811d5ce8374097a63f820d&sub=22565B3DE3F74ADCA35D71663998C79D)
- [Bayes 公式](http://note.youdao.com/noteshare?id=0693c471d86002cc0bae2403a96e6632&sub=B89A3590BCC74A21BAA0183AE3418CDD)
- Huffman树

### 2. 参考知识

#### [1.1 统计语言模型 `SLM`]

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

#### [1.2 n-gram模型]

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

#### [1.3 神经概率语言模型]

相较于`n-gram`的优势：
1. **词语之间的相似性**可以通过**词向量**来体现
2. 基于词向量的模型**自带平滑化功能**，不需要像`n-gram`那样进行额外处理。

#### [1.4 词向量]

- ##### 独热表示(One-Hot)

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
1. 任意两个词之间都是孤立的，根本无法表示出在语义层面上词语词之间的相关信息，而这一点是致命的。
2. 我们的词汇表一般都非常大，比如达到百万级别，这样每个词都用百万维的向量来表示简直是内存的灾难。

- ##### 分布式表示(distribution representation)

**分布式表示**的思路是通过训练，将每个词都映射成一个固定长素的短向量。所有的这些词向量就构成了**向量空间**，在这个空间引入“距离”，就可以根据词之间的距离来判断它们之间的(词法、语义上的)相似性了。

---

> `word2vec`就是采用分布式表示的词向量。`word2vec`有两个重要模型`CBOW`和`skip-gram`。对于`CBOW`和`skip-gram`，`word2vec`给出了两套框架，它们分别基于层序`softmax`(`Hierarchiral Softmax`)和`Negative Sampling`。

### 3. 基于层序softmax的模型

由图可见，两个模型都包括三层：**输入层**、**投影层**和**输出层**，前者是在已知前词`$w_t$`的上下文`$context:\{w_{t-2},w_{t-1},w_{t+1}.w_{w+2}\}$`的前提下预测当前词；而后者是在一只当前词`$w_t$`的前提下，预测其上下文`$context:\{w_{t-2},w_{t-1},w_{t+1}.w_{w+2}\}$`。

<p align="center">
<img src="https://note.youdao.com/yws/public/resource/d3385e2e4599150c59a78cfd34abdfb6/65E3B7479C14418784B624098AE2A4CD?ynotemdtimestamp=1564216514845" width=600 />
</p>

<p align="center">
<strong>Fig</strong>. CBOW和skip-gram
</p>


#### 4.1 CBOW模型



#### 4.2 Skip-gram模型

## 精确率(precision)和召回率(recall)

[人工智能-机器学习笔试面试题解析]Github链接(https://github.com/WerterHong/machine-learning-interview-qa/)

混淆矩阵

- True Positive(真正例, TP)：将正类预测为正类数.
- False Negative(假负例, FN)：将正类预测为负类数 → 漏报 (Type II error).
- False Positive(假正例, FP)：将负类预测为正类数 → 误报 (Type I error).
- True Negative(真反例, TN)：将负类预测为负类数.

<table>
  <tr>
    <th rowspan="2"><br>真实情况</th>
    <th colspan="2">预测结果</th>
  </tr>
  <tr>
    <td>正例 Positive</td>
    <td>反例 Negative</td>
  </tr>
  <tr>
    <td>正例 True</td>
    <td>True Positive (TP)</td>
    <td>True Negative (TN)</td>
  </tr>
  <tr>
    <td>反例 False</td>
    <td>False Positive (FP)</td>
    <td>False Negative (FN)</td>
  </tr>
</table>

精确率(precision)定义为：

```math
Precision=\frac{TP}{TP+FP}
```

需要注意的是精确率(precision)和准确率(accuracy)是不一样的，

```math
Accuracy=\frac{TP}{TP+TN+FP+FN}
```
其中`$TP+TN+FP+FN=样本总例$`。

在正负样本不平衡的情况下，准确率这个评价指标有很大的缺陷。比如在互联网广告里面，点击的数量是很少的，一般只有千分之几，如果用acc，即使全部预测成负类（不点击）acc 也有 99% 以上，没有意义。

召回率(recall, sensitivity, true positive rate)定义为：

```math
Recall=\frac{TP}{TP+FN}
```

此外，还有`$F1$`值，是精确率和召回率的调和均值，

```math
\frac{2}{F_1}=\frac{1}{Precision}+\frac{1}{Recall}

F_1=\frac{2*Precision*Recall}{Precision+Recall}=\frac{2*TP}{2*TP+FP+FN}=\frac{2*TP}{样本总例+TP-TN}
```

精确率和准确率都高的情况下，`F1`值也会高。
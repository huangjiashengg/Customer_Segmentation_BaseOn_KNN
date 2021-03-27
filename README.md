# 天猫客户群分类—基于KNN算法的购物决策分类算法

## 前言

本实验是基于KNN分类算法的构建，以对客户购物决策进行分类预测。实验数据采用薛薇出版教材《R语言与数据挖掘方法与应用》中自带的训练数据集和测试数据集。KNN分类算法适用于输入变量维度较低的数据集，尤其是当参数K/N无限趋近于0，其预测值逼近真实值。尤其针对分类边界不规则的分类预测，KNN的处理显得得心应手，只需要增大参数K即可使边界变得平滑。

## 结论

1，KNN对于维度较低的数据集表现优秀。本实验采用的数据集只有4个维度，采用袋外观测评判模型预测精准度时，训练出来的模型预测精准度高达99%；

2，K参数对于预测精准度有较大影响。当K取值为1时，模型预测精准度非常高。但是单纯使K取值为1风险较大，不利于模型的泛化能力(即稳健性较低，易波动)，因此本实验通过旁置法最终确定的k参数为3；

3，本实验输入数据有4个维度，分别由生产环境产生的基础数据派发出的各种转换率。由实验结果可清楚看到，这几个变量对与模型预测的精准度都起到了重大作用。

## 数据预览

**数据字段：**

![image-20210327122934235](C:\Users\DELL\PycharmProjects\数据结构与算法\机器学习算法手写与实践\KNN\README.assets\image-20210327122934235.png)

**数据说明：**

1，BuyorNot: 根据订单成交数是否大于0给每个顾客定义一个成交标签变量（1表示有成交，0表示无成交）；

2，BuyDNactDN: 有成交天数/活跃天数

3，ActDNTotalDN: 活跃天数/研究周期天数

4，BuyBBrand: 成交品牌数量/浏览的品牌数量

5，BuyHit: 订单成交数/商品点击次数

其中，BuyorNot作为分类标签，其他字段作为输入变量。

## 数据预处理

此阶段包含：数据汇总与派生、数据导入与格式转换、训练集数据分布与测试集数据分布展示。

1，原始数据为业务明细数据，不利于分析。因此，针对原始数据，以顾客为基本单位汇总得到一段时间（前3个月和后一个月）的业务数据，包含：浏览的品牌个数，成交品牌数量，活跃天数，有成交的天数，商品点击次数，订单成交数，收藏数，存入购物车数。并进一步汇总成数据预览中的5个字段；

2，数据导入与格式转换（只展示训练数据集部分，测试集做同样处理）：

数据导入，去掉表头并存放在列表中：

```python
input_file = "C:/Users/DELL/Desktop/机器学习/R语言与数据挖掘电子资料/天猫_Train_1.txt"
with open(input_file, encoding='utf-8') as f:
    sample_list = []
    for line in f:
        sample = []
        for l in line.strip().split(','):
            sample.append(l)
        sample_list.append(sample)
sample_list = sample_list[1:]
sample_list = np.array(sample_list)
```

数据标签分割与格式转换：

```python
X = []
y = []
for i in range(len(sample_list)):
    y.append(float(sample_list[i][0]))
    x = []
    for j in range(1, len(sample_list[0])):
        x.append(float(sample_list[i][j]))
    X.append(x)
```

训练集数据分布展示：

```python
plt.figure()
plt.scatter(range(1, len(y)+1), y)
plt.suptitle('Training_Distribution')
plt.show()
```

![Figure_1](C:\Users\DELL\PycharmProjects\数据结构与算法\机器学习算法手写与实践\KNN\README.assets\Figure_1.png)

测试集数据分布展示：

![Figure_2](C:\Users\DELL\PycharmProjects\数据结构与算法\机器学习算法手写与实践\KNN\README.assets\Figure_2.png)

训练集与测试集分布一致，可放心做下一步的数据训练。

## 模型训练

### 导入Sklearn用于K近邻分析的包

```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
```

### 开始数据训练

建立分类器，将k参数设置为3：

```python
neigh = KNeighborsClassifier(n_neighbors=3)
```

模型拟合：

```python
neigh.fit(X, y)
```

## 模型评估

针对模型评估阶段，采用单值评估，以模型预测的精准度作为单一的评价指标。

因此，为了评估模型，我们需要先引入测试集数据，做数据格式转换处理，导入模型进行测试集数据预测，输出混淆矩阵及模型预测准确率。

引入测试集数据：

```python
input_file_test = 'C:/Users/DELL/Desktop/机器学习/R语言与数据挖掘电子资料/天猫_Test_1.txt'
with open(input_file, encoding='utf-8') as f:
    sample_list_test = []
    for line in f:
        sample_test = []
        for l in line.strip().split(','):
            sample_test.append(l)
        sample_list_test.append(sample)
sample_list_test = sample_list[1:]
sample_list_test = np.array(sample_list_test)
```

格式转换：

```python
X_test = []
y_test = []
for i in range(len(sample_list_test)):
    y_test.append(float(sample_list_test[i][0]))
    x_test = []
    for j in range(1, len(sample_list_test[0])):
        x_test.append(float(sample_list_test[i][j]))
    X_test.append(x_test)
```

数据预测与混淆矩阵输出：

```python
predictions = neigh.predict(X_test)
C = confusion_matrix(y_test, predictions)
print('混淆矩阵为：\n', C)
score = (neigh.score(X_test, y_test)) * 100
print('模型预测精准度为：%.6f' % score, '%')
```

输出：

混淆矩阵为：
 [[177   2]
 [  0 675]]
模型预测精准度为：99.765808 %

## 模型参数调优

KNN模型参数相对简单，我们针对K参数进行调优

```python
# 将每个k值对应输出的准确率放在score_list里边
score_list = []
for k in range(1, 15):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X, y)

    input_file_test = 'C:/Users/DELL/Desktop/机器学习/R语言与数据挖掘电子资料/天猫_Test_1.txt'
    with open(input_file, encoding='utf-8') as f:
        sample_list_test = []
        for line in f:
            sample_test = []
            for l in line.strip().split(','):
                sample_test.append(l)
            sample_list_test.append(sample)
    sample_list_test = sample_list[1:]
    sample_list_test = np.array(sample_list_test)
    # print(sample_list_test)

    X_test = []
    y_test = []
    for i in range(len(sample_list_test)):
        y_test.append(float(sample_list_test[i][0]))
        x_test = []
        for j in range(1, len(sample_list_test[0])):
            x_test.append(float(sample_list_test[i][j]))
        X_test.append(x_test)
    # print(X_test)
    # print(y_test)
    # plt.figure()
    # plt.scatter(range(1, len(y_test)+1), y_test)
    # plt.suptitle('Testing_Distribution')
    # plt.show()

    predictions = neigh.predict(X_test)
    C = confusion_matrix(y_test, predictions)
    print('混淆矩阵为：\n', C)
    score = (neigh.score(X_test, y_test)) * 100
    print('模型预测精准度为：%.6f' % score, '%')
    score_list.append(score)
# 展示
plt.figure()
plt.plot(range(1, 15), score_list, 'bo', range(1, 15), score_list, 'k')
for a, b in zip(range(1, 15), score_list):
    plt.text(a+0.01, b+0.02, '%.2f' % b, ha='center', va='bottom', fontsize=7)
plt.suptitle('Score_Neighbors')
plt.xlabel('Neighbors')
plt.ylabel('Score%')
plt.show()
```

部分输出结果：

混淆矩阵为：
 [[179   0]
 [  0 675]]
模型预测精准度为：100.000000 %
混淆矩阵为：
 [[179   0]
 [  0 675]]
模型预测精准度为：100.000000 %
混淆矩阵为：
 [[177   2]
 [  0 675]]
模型预测精准度为：99.765808 %
混淆矩阵为：
 [[178   1]
 [  0 675]]
模型预测精准度为：99.882904 %
混淆矩阵为：
 [[176   3]
 [  0 675]]................

**图表输出：**

![Figure_3](C:\Users\DELL\PycharmProjects\数据结构与算法\机器学习算法手写与实践\KNN\README.assets\Figure_3.png)

## 模型缺陷

1，适用于维度较低的分类数据集；

2，测试数据集与验证数据集按1:1划分，有一定的不合理之处。

## 交流

问题可联系2393946194@qq.com，欢迎互相学习与交流。


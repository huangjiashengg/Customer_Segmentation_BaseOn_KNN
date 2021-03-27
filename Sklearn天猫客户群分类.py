# 加载数据
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
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
X = []
y = []
for i in range(len(sample_list)):
    y.append(float(sample_list[i][0]))
    x = []
    for j in range(1, len(sample_list[0])):
        x.append(float(sample_list[i][j]))
    X.append(x)

plt.figure()
plt.scatter(range(1, len(y)+1), y)
plt.suptitle('Training_Distribution')
plt.show()
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

plt.figure()
plt.plot(range(1, 15), score_list, 'bo', range(1, 15), score_list, 'k')
for a, b in zip(range(1, 15), score_list):
    plt.text(a+0.01, b+0.02, '%.2f' % b, ha='center', va='bottom', fontsize=7)
plt.suptitle('Score_Neighbors')
plt.xlabel('Neighbors')
plt.ylabel('Score%')
plt.show()








# neigh = KNeighborsClassifier(n_neighbors=3)
# neigh.fit(X, y)
#
# input_file_test = 'C:/Users/DELL/Desktop/机器学习/R语言与数据挖掘电子资料/天猫_Test_1.txt'
# with open(input_file, encoding='utf-8') as f:
#     sample_list_test = []
#     for line in f:
#         sample_test = []
#         for l in line.strip().split(','):
#             sample_test.append(l)
#         sample_list_test.append(sample)
# sample_list_test = sample_list[1:]
# sample_list_test = np.array(sample_list_test)
# # print(sample_list_test)
#
# X_test = []
# y_test = []
# for i in range(len(sample_list_test)):
#     y_test.append(float(sample_list_test[i][0]))
#     x_test = []
#     for j in range(1, len(sample_list_test[0])):
#         x_test.append(float(sample_list_test[i][j]))
#     X_test.append(x_test)
# print(X_test)
# print(y_test)
# plt.figure()
# plt.scatter(range(1, len(y_test)+1), y_test)
# plt.suptitle('Testing_Distribution')
# plt.show()
#
# predictions = neigh.predict(X_test)
# C = confusion_matrix(y_test, predictions)
# print('混淆矩阵为：\n', C)
# score = (neigh.score(X_test, y_test)) * 100
# print('模型预测精准度为：%.6f' % score, '%')






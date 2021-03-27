from 机器学习算法手写与实践.Dknn import Knn
import pandas as pd
train_data = pd.read_csv("C:/Users/DELL/Desktop/机器学习/R语言与数据挖掘电子资料/天猫_Train_1.txt", header=0)
trainx_data = train_data.iloc[:, 1:5]
trainx_min = trainx_data.rolling(trainx_data.shape[0]).min().iloc[trainx_data.shape[0] - 1, ]
trainx_max = trainx_data.rolling(trainx_data.shape[0]).max().iloc[trainx_data.shape[0] - 1, ]
trainx_data = (trainx_data - trainx_min)/(trainx_max-trainx_min)

trainy_data = train_data.iloc[:, 0]

test_data = pd.read_csv("C:/Users/DELL/Desktop/机器学习/R语言与数据挖掘电子资料/天猫_Test_1.txt", header=0)
testx_data = test_data.iloc[:, 1:5]
testx_min = testx_data.rolling(testx_data.shape[0]).min().iloc[testx_data.shape[0] - 1, ]
testx_max = testx_data.rolling(testx_data.shape[0]).max().iloc[testx_data.shape[0] - 1, ]
testx_data = (testx_data - testx_min)/(testx_max - testx_min)
testy_data = test_data.iloc[:, 0]

for k in range(1, 15):
    Model = Knn(trainx_data, trainy_data, testx_data, testy_data, k)
    Model.calculation()






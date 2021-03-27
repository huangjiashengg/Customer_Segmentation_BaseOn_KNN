import pandas as pd
import math
import numpy as np


class Knn:
    def __init__(self, trainx_dataframe, trainy_dataframe, testx_dataframe, testy_dataframe, k):
        self.trainX = trainx_dataframe
        self.trainY = trainy_dataframe
        self.testX = testx_dataframe
        self.testY = testy_dataframe
        self.k = k

    def calculation(self):
        global prediction_test
        train_nrow = self.trainX.shape[0]
        train_pcol = self.trainX.shape[1]
        test_nrow = self.testX.shape[0]

        listed = []
        distances = []
        predictions_test = []
        for k in range(test_nrow):   # k=820
            for i in range(train_nrow):  # i=855
                for j in range(train_pcol):  # j=4
                    sub = self.testX.iloc[k, j] - self.trainX.iloc[i, j]
                    x = pow(sub, 2)
                    listed.append(x)
                distance = math.sqrt(sum(listed))  # 执行i次
                listed = []
                distances.append(distance)

            self.trainX['distances'] = distances
            self.trainX['Y'] = self.trainY
            self.trainX = self.trainX.sort_values(["distances"], ascending=True)
            s = self.trainX['Y'].rolling(self.k).sum().iloc[self.k - 1, ]
            r = s/self.k
            if r > 0.5:
                prediction_test = 1
            else:
                prediction_test = 0
            predictions_test.append(prediction_test)

            distances = []
        self.testX["predictions"] = predictions_test
        self.testX["Y"] = self.testY
        ct = pd.crosstab(self.testX["predictions"], self.testX["Y"], rownames=["Prediction"], colnames=["Labels"])
        print("k值为：", self.k)
        print("混淆矩阵为：", ct)
        wrong_ratio = (1 - (sum(np.diag(ct))/test_nrow))*100
        print("错判率为：", wrong_ratio)

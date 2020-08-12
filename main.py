#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 23:37:08 2020

@author: przemek
"""
from Net import Net
import numpy as np
"""
x=np.array([[1,2],[8,16],[5,10],[7,14]])
y=np.array([[0,1]])
test=Net()
test.add(4,1,'sigmoid')
#test.addo(3)
#test.addo(1,'sigmoid')
tt=np.copy(test.layers[0].weights)
test.fit(x,y)
print('________predicted___________')
print(test.predict(x))
print('________current___________')
print(test.layers[0].weights)
print('________first___________')
print(tt)
print('________diff___________')
print(test.layers[0].weights-tt)
"""
test = Net()
test.add(784, 600)
test.addo(300)
test.addo(150)
test.addo(10, 'sigmoid')
import pandas as pd

data = pd.read_csv('train.csv').to_numpy().T
xtr, xte = np.divide(data[1:, :40000], 255), np.divide(data[1:, 40000:], 255)
y = np.zeros((10, data.shape[1]))
for i in range(data.shape[1]):
    y[data[0, i], i] = 1
ytr, yte = y[:, :40000], y[:, 40000:]
test.fit(xtr, ytr)
o = test.predict(xte)
print('________predicted___________')
print(np.argmax(o, axis=0))
print('________score___________')
oo = np.argmax(yte, axis=0)

score = np.mean([x == y for x, y in zip(np.argmax(o, axis=0), oo)])
print(score)
for i in test.layers:
    print('min ', np.min(i.weights), ' max ', np.max(i.weights))
print(np.mean(np.argmax(o, axis=0)))
# import matplotlib.pyplot as plt
# plt.imshow(np.reshape(xte[:,0],(28,-1)))
# plt.show()
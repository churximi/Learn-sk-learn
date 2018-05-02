#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
功能：sk-learn学习
时间：2018年05月02日11:13:48
"""
import pickle
from sklearn import datasets
from sklearn import svm
from sklearn.externals import joblib


# 加载数据
iris = datasets.load_iris()  # (150, 4)
digits = datasets.load_digits()  # (1797, 64)

print(iris.data.shape)
print(digits.data.shape)

print(digits.target_names)  # 标签
print(dir(digits))

# SVM分类
clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(digits.data[:-1], digits.target[:-1])  # 训练
pred = clf.predict(digits.data[-1:])
print("预测值：{}，实际值：{}".format(pred, digits.target[-1:]))

# 保存模型
clf2 = svm.SVC()
X, y = iris.data, iris.target
clf2.fit(X, y)

with open("测试.pkl", "wb") as fout:
    s = pickle.dump(clf2, fout)

with open("测试.pkl", "rb") as f:
    clf3 = pickle.load(f)
pred = clf3.predict(X[0:1])
print("预测值：{}，实际值：{}".format(pred, y[0:1]))

# 或者用joblib保存
# joblib.dump(clf, 'filename.pkl')
# clf = joblib.load('filename.pkl')

# 约定


if __name__ == "__main__":
    pass

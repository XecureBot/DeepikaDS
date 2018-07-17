# import numpy as np
# import pandas as pd
# import tkinter
# import matplotlib.pyplot as plt
# from scipy import sparse
# from sklearn import datasets
# from sklearn.datasets import load_boston
# from sklearn import preprocessing
# import urllib
# iris=datasets.load_iris()
# print(iris)
# print(iris.data.shape)
# print(iris)
# plt.show()
# iris_df=pd.DataFrame(iris.data,columns=iris.feature_names)
# print(iris_df)
# for class_number in np.unique(iris.target):
# 	plt.figure(1)
# 	iris_df['sepal length (cm)'].iloc[np.where(iris.target==class_number)[0]].hist(bins=30)
# boston=load_boston()
# X,y=boston.data,boston.target.reshape(-1,1)
# new_target=preprocessing.binarize(y,threshold=boston.target.mean())
# print(new_target[:6])
# print((y[:,]> y.mean()).astype(int))
# print((y[:6]> y.mean()).astype(int))
# binar=preprocessing.Binarizer(y.mean())
# new_target=binar.fit_transform(y)
# print(new_target[:5])
# url=" http://insight.dev.schoolwires.com/HelpAssets/C2Assets/C2Files/HalfHourParentTeacherConferenceSampleImportFile.csv"
# set1=urllib.request.Request(url)
# iris_p=urllib.request.urlopen(set1)
# iris_other=pd.read_csv(iris_p,sep=',',decimal='.',header=None,names=['sepal_length','sepal_width','petal_length','petal_width','target'])
# print(iris_other.head())



# import numpy as np
# from sklearn import metrics
# y = np.array([1, 1, 2, 2])
# scores = np.array([0.1, 0.4, 0.35, 0.8])
# fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
# print(fpr)
# print(tpr)
# print(thresholds)



# import numpy as np
# from sklearn.metrics import roc_auc_score
# y_true = np.array([0, 0, 1, 1])
# y_scores = np.array([0.1, 0.4, 0.35, 0.8])
# print(roc_auc_score(y_true, y_scores))
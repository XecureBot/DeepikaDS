# import pandas as pd
# import numpy as np
# a=np.array([[1,2,3],[2,3,4]])
# print(type(a))
# print(a.shape)
# print(a)
# print(a[0][1])


import numpy as np
import pandas as pd
import tkinter
import matplotlib.pyplot as plt

# For preprocessing the data
from sklearn.preprocessing import Imputer
# To split the dataset into train and test datasets
from sklearn.cross_validation import train_test_split
# To model the Gaussian Navie Bayes classifier
from sklearn.naive_bayes import GaussianNB
# To calculate the accuracy score of the model
from sklearn.metrics import accuracy_score
from scipy import sparse
from sklearn import datasets
from sklearn.datasets import load_boston
from sklearn import preprocessing
import urllib
iris=datasets.load_iris()
print(iris)
# print(iris.data.shape)
# print(iris)
# plt.show()
iris_df=pd.DataFrame(iris.data,columns=iris.feature_names)
print(iris_df)
iris_df_rev = iris_df
iris_df_rev.describe(include= 'all')
features = iris_df.values[:,:8]
target = iris_df.values[:,2]
features_train, features_test, target_train, target_test = train_test_split(features,
	target, test_size = 0.33, random_state = 10)
clf = GaussianNB()
clf.fit(features_train, target_train)
target_pred = clf.predict(features_test)

accuracy_score(target_test, target_pred, normalize = True)
1
	


# from sklearn.metrics import confusion_matrix
# y_true = [2, 0, 2, 2, 0, 1]
# y_pred = [0, 0, 2, 2, 0, 2]
# b=confusion_matrix(y_true, y_pred)
# print(b)
# plt.plot(y_true, y_pred)
# plt.show()






# import pandas as pd
# y_actu = pd.Series([2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2], name='Actual')
# y_pred = pd.Series([0, 0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 2], name='Predicted')
# df_confusion = pd.crosstab(y_actu, y_pred)
# print(df_confusion)
# plt.plot(y_actu, y_pred)
# plt.show()


# # import matplotlib.pyplot as plt
# # import matplotlib.mlab as mlab
# # import tkinter 
# # array_1=np.arange(10)
# # print(array_1)
# # print(array_1.shape)
# # print(array_1.ndim)
# print(array_1.reshape((5,2)))
# array_2=array_1*array_1
# print(array_2.reshape(5,2))
# array_1=array_1+1
# print(array_1.reshape(5,2))
# array_3=array_1+array_2
# print(array_3.reshape(5,2))
# A = np.array( [[1,1],[0,1]] )
# B = np.array( [[2,0],[3,4]] )
# print(A*B)
# print(A.dot(B))
# print(np.arange(10000))



# data = np.array(['a','b','c','d'])
# s = pd.Series(data)
# print (s)
# import pandas as pd
# data = [['Alex',10],['Bob',12],['Clarke',13]]
# df = pd.DataFrame(data,columns=['Name','Age'])
# print (df)
# import pandas as pd
# data = [['Alex',10],['Bob',12],['Clarke',13]]
# df = pd.DataFrame(data,columns=['Name','Age'],dtype=float)
# print (df)
# import pandas as pd
# data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'],'Age':[28,34,29,42]}
# df = pd.DataFrame(data)
# print (df)

# l1=[]
# l2=[]
# l3=[]
# l4=[]
# l5=[]
# f=[4,44,36,28,6]
# for i in range(0,20):
# 	l1.append(i)
# print(l1)#first interval
# for i in range(20,40):
# 	l2.append(i)
# print(l2)#second interval
# for i in range(40,50):
# 	l3.append(i)
# print(l3)#third interval
# for i in range(50,70):
# 	l4.append(i)
# print(l4)#fourth interval
# for i in range(70,100):
# 	l5.append(i)
# print(l5)#fifth interval
# w1=len(l1)
# print("width of first interval=",w1)
# w2=len(l2)
# print("width of second interval=",w2)
# w3=len(l3)
# print("width of third interval=",w3)
# w4=len(l4)
# print("width of fourth interval=",w4)
# w5=len(l5)
# print("width of fifth interval=",w5)
# fd1=f[0]/w1
# print("freq density of first interval=",fd1)
# fd2=f[1]/w2
# print("freq density of second interval=",fd2)
# fd3=f[2]/w3
# print("freq density of third interval=",fd3)
# fd4=f[3]/w4
# print("freq density of fourth interval=",fd4)
# fd5=f[4]/w5
# print("freq density of fifth interval=",fd5)
# frequency=[fd1,fd2,fd3,fd4,fd5]
# interval=[w1,w2,w3,w4,w5]

# plt.plot(frequency,interval)


# plt.show()
# import numpy as np
# import matplotlib.pyplot as plt
# import tkinter 

# # evenly sampled time at 200ms intervals
# t = np.arange(0., 5., 0.2)

# # red dashes, blue squares and green triangles
# plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
# plt.show()
# print(t)

# z = np.array([[1, 2, 3, 4],
#          [5, 6, 7, 8],
#          [9, 10, 11, 12]])
# print(z.shape)
# print(z.reshape(-1))
# print(z.reshape(-1,1))


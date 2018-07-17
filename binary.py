# d={'name':'deepa',"age":18}
# print(d)
# print(list(d))

# print(d.values())
# d.clear()
# print(d)
# d.pop('age')
# print(d)

# d.popitem()
# print(d)
# d1={"id":1 }
# d.update(d1)
# print(d)

# d={'a','e','i','o','u'}

# b=dict.fromkeys(d)
# print(b)



# for i in reversed(range(10,1,2)):
# 	print(i)
# for i in reversed(range(1, 10, 2)):
# 	print(i)
# import numpy as np

# from sklearn.metrics import confusion_matrix
# y_true = [2, 0, 2, 2, 0, 1]
# y_pred = [0, 0, 2, 2, 0, 2]
# b=confusion_matrix(y_true, y_pred)
# print(b)




# import pandas as pd
# y_actu = pd.Series([2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2], name='Actual')
# y_pred = pd.Series([0, 0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 2], name='Predicted')
# df_confusion = pd.crosstab(y_actu, y_pred)
# print(df_confusion)
# plot_confusion_matrix(df_confusion)


# a=2
# b=3
# d=complex(a,b)
# print(d)


# l1=[[1,2],[3,4]]
# print(len(l1))
# l2=[[4,5],[6,7]]
# c=[[0,0],[0,0]]
# for i in range(0,len(l1)):
# 	for j in range(0,len(l1[0])):
# 		c[i][j]=l1[i][j]+l2[i][j]
# print(c)




def fib(n):
	a=0
	b=1
	while a < n:
		print(a, end=' ')
		a,b=b,a+b
		
	print( )
fib(5)
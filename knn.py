# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.neighbors import NearestNeighbors

# # Input data
# X = np.array([[2.1, 1.3], [1.3, 3.2], [2.9, 2.5], [2.7, 5.4], [3.8, 0.9], 
#         [7.3, 2.1], [4.2, 6.5], [3.8, 3.7], [2.5, 4.1], [3.4, 1.9],
#         [5.7, 3.5], [6.1, 4.3], [5.1, 2.2], [6.2, 1.1]])

# # Number of nearest neighbors
# k = 5

# # Test datapoint 
# test_datapoint = [4.3, 2.7]

# # Plot input data 
# plt.figure()
# plt.title('Input data')
# plt.scatter(X[:,0], X[:,1], marker='o', s=75, color='black')

# # Build K Nearest Neighbors model
# knn_model = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X)
# distances, indices = knn_model.kneighbors([test_datapoint])

# # Print the 'k' nearest neighbors
# print("\nK Nearest Neighbors:")
# for rank, index in enumerate(indices[0][:k], start=1):
#     print(str(rank) + " ==>", X[index])

# Visualize the nearest neighbors along with the test datapoint 
# plt.figure()
# plt.title('Nearest neighbors')
# plt.scatter(X[:, 0], X[:, 1], marker='o', s=75, color='k')
# plt.scatter(X[indices][0][:][:, 0], X[indices][0][:][:, 1], 
#         marker='o', s=250, color='k', facecolors='none')
# plt.scatter(test_datapoint[0], test_datapoint[1],
#         marker='x', s=75, color='k')

# plt.show()




# from sklearn.metrics import classification_report

# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.utils import shuffle
# from sklearn.datasets import fetch_mldata
# from sklearn.cross_validation import train_test_split
# mnist=fetch_mldata("MNIST original")
# mnist.data,mnist.target=shuffle(mnist.data,mnist.target)
# mnist.data=mnist.data[:1000]
# mnist.target=mnist.target[:1000]
# x_train,x_test,y_train,y_test=train_test_split(mnist.data,mnist.target,test_size=0.8,random_state=0)

# clf=KNeighborsClassifier(3)
# clf.fit(x_train,y_train)
# y_pred=clf.predict(x_test)
# print(classification_report(y_test,y_pred))


import numpy as np
from sklearn.metrics import log_loss
def cross_entropy(predictions, targets):
    N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions))/N
    return ce

predictions = np.array([[0.25,0.25,0.25,0.25],
                        [0.01,0.01,0.01,0.97]])
targets = np.array([[1,0,0,0],
                   [0,0,0,1]])

x = cross_entropy(predictions, targets)
print(log_loss(targets, predictions))





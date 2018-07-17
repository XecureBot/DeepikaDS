import pandas as pd
import numpy as np
# For preprocessing the data
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
# To split the dataset into train and test datasets
from sklearn.cross_validation import train_test_split
# To model the Gaussian Navie Bayes classifier
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
# To calculate the accuracy score of the model
from sklearn.metrics import accuracy_score
import urllib
file = r'/home/deepa/Downloads/adult.csv'
# df = pd.read_csv(file)
# print(df)
# comma delimited is the default
adult_df = pd.read_csv(file,
                       header = None, delimiter=' *, *', engine='python')
adult_df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
                    'marital_status', 'occupation', 'relationship',
                    'race', 'sex', 'capital_gain', 'capital_loss',
                    'hours_per_week', 'native_country', 'income']
adult_df.isnull().sum()
for value in ['workclass', 'education',
          'marital_status', 'occupation',
          'relationship','race', 'sex',
          'native_country', 'income']:
        print (value,":", sum(adult_df[value] == '?'))
#data preprocessing 	
adult_df_rev = adult_df
adult_df_rev.describe(include= 'all')
# for value in ['workclass', 'education',
#           'marital_status', 'occupation',
#           'relationship','race', 'sex',
#           'native_country', 'income']:
#     adult_df_rev[value].replace(['?'], [adult_df_rev.describe(include='all')[value][2]],
#                                 inplace='True')

num_features = ['age', 'workclass_cat', 'fnlwgt', 'education_cat', 'education_num',
                'marital_cat', 'occupation_cat', 'relationship_cat', 'race_cat',
                'sex_cat', 'capital_gain', 'capital_loss', 'hours_per_week',
                'native_country_cat']
 
scaled_features = {}
for each in num_features:
    mean, std = adult_df_rev[each].mean(), adult_df_rev[each].std()
    scaled_features[each] = [mean, std]
    adult_df_rev.loc[:, each] = (adult_df_rev[each] - mean)/std
rev.values[:,:14]
target = adult_df_rev.values[:,14]
features_train, features_test, target_train, target_test = train_test_split(features,
                                                                            target, test_size = 0.33, random_state = 10)
clf = GaussianNB()
clf.fit(features_train, target_train)
target_pred = clf.predict(features_test)
d=accuracy_score(target_test, target_pred, normalize = True)
print(d)


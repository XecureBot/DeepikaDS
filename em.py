import os
import numpy as np
from collections import Counter
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def make_Dictionary(root_dir):
   all_words = []
   emails = '/home/deepa/Downloads'
   for mail in emails:
      with open(mail) as m:
        for line in m:
          words = line.split()
          all_words += words
          dictionary = Counter(all_words)
          list_to_remove = dictionary.keys()
          for item in list_to_remove:
            if item.isalpha() == False:
              del dictionary[item]
            elif len(item) == 1:
              del dictionary[item]
              dictionary = dictionary.most_common(3000)
              return dictionary

def extract_features(mail_dir):
  files = '/home/deepa/Downloads'
  features_matrix = np.zeros((len(files),3000))
  train_labels = np.zeros(len(files))
  count = 0;
  docID = 0;
  for fil in files:
    with open(fil) as fi:
      for i,line in enumerate(fi):
        if i == 2:
          words = line.split()
          for word in words:
            wordID = 0
            for i,d in enumerate(dictionary):
              if d[0] == word:
                wordID = i
                features_matrix[docID,wordID] = words.count(word)
      train_labels[docID] = 0;
      filepathTokens = fil.split()
      lastToken = filepathTokens[len(filepathTokens) - 1]
      if lastToken.startswith("spmsg"):
          train_labels[docID] = 1;
          count = count + 1
      docID = docID + 1
  return features_matrix, train_labels

TRAIN_DIR = "email/train-mails"
TEST_DIR = "email/test-mails"

dictionary = make_Dictionary(TRAIN_DIR)

print ("reading and processing emails from file.")
features_matrix, labels = extract_features(TRAIN_DIR)
test_feature_matrix, test_labels = extract_features(TEST_DIR)

model = GaussianNB()

print ("Training model.")
#train model
model.fit(features_matrix, labels)

predicted_labels = model.predict(test_feature_matrix)

print ("FINISHED classifying. accuracy score : ")
print (accuracy_score(test_labels, predicted_labels))
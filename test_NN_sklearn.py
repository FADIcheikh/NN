# coding: utf-8
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier




df = pd.read_table("D:\Data_Minig\seance9_neuralNetwork\Performance_entrep.txt",sep ='\t',header = 0)
#split dataset into explicatives vars and target
explicative =df.drop(['PERF'],axis=1)
names = explicative.columns
target =df['PERF']
#set 25% for test
X_train, X_test, y_train, y_test = train_test_split(explicative,target, test_size=0.25, random_state=0)
#Centrage et reduction
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
#check for NA :there is no NA
for index, row in explicative.iterrows():
    for j in  explicative.columns:
      if(np.isnan(row[j])):
        print index
        print j+ " "+str(row[j])
#train
clf = MLPClassifier(hidden_layer_sizes=(100,20), max_iter=100, alpha=1e-4,
                    solver='lbfgs', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)
clf.fit(X_train, y_train)
#score
#test
print clf.predict(X_test)
print y_test
print("Training set score: %f" % clf.score(X_train, y_train))
print("Test set score: %f" % clf.score(X_test, y_test))

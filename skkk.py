import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

# cancer_data= load_breast_cancer()
# #print(cancer_data['data'].shape)
# df = pd.DataFrame(cancer_data['data'], columns=cancer_data['feature_names'])
# df['target'] = cancer_data['target']
# X = df[cancer_data.feature_names].values
# y = df['target']
# model = LogisticRegression(solver='liblinear')
# model.fit(X, y)
# model.predict([X[0]])
# print(type(X))
# print(model.score(X,y))

n = int(input())

n = 6
X = []
for i in range(n):
   X.append([float(x) for x in input().split()])
y = [int(x) for x in input().split()]
datapoint = [float(x) for x in input().split()]
model = LogisticRegression()
model.fit(X, y)
#print(type(datapoint))
print(model.predict([datapoint])[0])
import pandas as pd
import matplotlib.pyplot as plt

#pd.options.display.max_columns=6
df= pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
col = df[['Pclass', 'Fare', 'Age']]
col2 = df[['Pclass', 'Fare', 'Age']].values
#mask = col[:,2] < 18
#print(mask)
#print(col[mask])
#print((col2[:,2] < 18).sum())
#print(col2[col2[:,2] < 18].sum())
#arr=col[:2].shape
#print(arr)
#arr2= col['Age'].shape
#print(arr2)
#print(col['Age'])
#print(col2[:,2])

#plt.scatter(df['Age'], df['Fare'], c=df['Pclass'])
#plt.xlabel('Age')
#plt.ylabel('Fare')
#plt.plot([0, 80], [85, 5])
plt.plot([10, 0], [100, 3])
plt.show()


# both pd and np accept loolean list to keep lines with true values
import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer_data = load_breast_cancer()
df = pd.DataFrame(cancer_data['data'], columns=cancer_data['feature_names'])
df['target'] = cancer_data['target']
term = cancer_data.feature_names
terms = cancer_data['feature_names']
print(term.shape)
print(type(cancer_data))
print(df.target)
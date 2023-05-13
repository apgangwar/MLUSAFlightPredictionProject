import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle
#Load the csv file
df = pd.read_csv("transformed_dataset.csv")

print(df.head())
## Getting train and test data ready for modelling
X = df.loc[:, df.columns != 'fare']
y = df['fare']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.decomposition import PCA
pca = PCA(n_components=0.99).fit(X_train)
X_train_pca = pca.transform(X_train)

X_test_pca = pca.transform(X_test)

#instantiate the model
model_rf = RandomForestRegressor()

model_rf.fit(X_train_pca, y_train)

# Make pickle file of our model
pickle.dump(model_rf, open("C:\\ML  model deployment\\model.pkl", "wb"))
import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data = []
folder_name = 'DataFrames'
df_names = os.listdir(folder_name + '/')

for name in df_names:
    data.append(pd.read_csv(folder_name + '/' + name, index_col=0))

df_full = pd.concat(data, axis=0)
df_full = df_full.sample(frac=1, ignore_index=True)

X = df_full.drop('Sign name', axis=1)
y = df_full['Sign name']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2021)

K_neigh = KNeighborsClassifier()
K_neigh.fit(X_train, y_train)

y_hat = K_neigh.predict(X_test)
accuracy = accuracy_score(y_test, y_hat)

print('Accuracy of this model is: ' + accuracy)

# Save the model
filename = 'hand_language_prediction_model.sav'
pickle.dump(K_neigh, open(filename, 'wb'))
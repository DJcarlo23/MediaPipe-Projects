import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Import dataframes and connect them
data = []
folder_name = 'HandsTracking/Hand Language Signs Recognizer/DataFrames'
df_names = os.listdir(folder_name + '/')

for name in df_names:
    data.append(pd.read_csv(folder_name + '/' + name, index_col=0))

df_full = pd.concat(data, axis=0)
df_full = df_full.sample(frac=1, ignore_index=True)

# Divide dataset into train and test set
X = df_full.drop('Sign name', axis=1)
y = df_full['Sign name']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2021)

# Create simple k-nearest neighbors model
K_neigh = KNeighborsClassifier()
K_neigh.fit(X_train, y_train)

# Make test of accuracy of the model
y_hat = K_neigh.predict(X_test)
accuracy = accuracy_score(y_test, y_hat)
print('Accuracy of this model is: ' + str(accuracy))

# Save the model to file
filename = 'HandsTracking/Hand Language Signs Recognizer/hand_language_prediction_model.sav'
pickle.dump(K_neigh, open(filename, 'wb'))
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("tic-tac-toe.data")
print(data)

le = preprocessing.LabelEncoder()
topLeftSquare = le.fit_transform(list(data["tls"]))
topMS = le.fit_transform(list(data["topmidsquare"]))
topRightSquare = le.fit_transform(list(data["trs"]))
middleLeftSquare = le.fit_transform(list(data["mls"]))
middleMiddleSquare = le.fit_transform(list(data["mms"]))
middleRightSquare = le.fit_transform(list(data["mrs"]))
bottomLeftSquare = le.fit_transform(list(data["bls"]))
bottomMiddleSquare = le.fit_transform(list(data["bms"]))
bottomRightSquare = le.fit_transform(list(data["brs"]))
cls = le.fit_transform(list(data["class"]))

predict = "class"  # optional

X = list(zip(topLeftSquare, topMS, topRightSquare, middleLeftSquare, middleMiddleSquare, middleRightSquare,
             bottomLeftSquare, bottomMiddleSquare, bottomRightSquare))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

print(X_train, y_test)

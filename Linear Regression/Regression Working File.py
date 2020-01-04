import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("SkillCraft1_Dataset.csv", sep=",", nrows=300)

# data = data.set_index("LeagueIndex")
# data = data.drop(8)

data = data[["LeagueIndex", "Age", "HoursPerWeek"]]

predict = "HoursPerWeek"

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

"""best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print(acc)

if acc > best:
    best = acc
    with open("SkillCraftmodel.pickle", "wb") as f:
        pickle.dump(linear, f)"""

pickle_in = open("SkillCraftmodel.pickle", "rb")
linear = pickle.load(pickle_in)

print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

p = 'LeagueIndex'
style.use("ggplot")
pyplot.scatter(data[p], data["HoursPerWeek"])
pyplot.xlabel(p)
pyplot.ylabel("HPW")
pyplot.show()
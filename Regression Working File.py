import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv("SkillCraft1_Dataset.csv", sep=",")

print(data.head())

data = data[["age", "HoursPerWeek"]]




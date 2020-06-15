import numpy as np
import pandas as pd
from pandas.plotting import andrews_curves


def embarked_to_int(x):
    if x == "S":
        return 0
    if x == "C":
        return 1
    if x == "P":
        return 2
    if x == "Q":
        return 3
    else:
        return 4


def analyze_training_data():
    data = pd.read_csv('Data/train.csv')
    data['Sex'] = data['Sex'].map(lambda x: int(x == 'male'))
    data['Embarked'] = data['Embarked'].map(embarked_to_int)
    correlation = data.corr(method='pearson')
    print(correlation['Survived'])


# this method reads in the specified dataset and returns a pandas dataframe
# The Sex and Embarked information is mapped to a numeric value and the following columns are dropped as they do not
# contain usable Information: Name, Ticket, Cabin, PassengerId
def read_data(name):
    data = pd.read_csv('Data/' + name + '.csv')
    data['Sex'] = data['Sex'].map(lambda x: int(x == 'male'))
    data['Embarked'] = data['Embarked'].map(embarked_to_int)
    data = data.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'])
    # replace nan values by the average value of that column
    for name in ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]:
        mean = data[name].mean()
        data[name] = data[name].fillna(mean)
    return data

# analyze_training_data()

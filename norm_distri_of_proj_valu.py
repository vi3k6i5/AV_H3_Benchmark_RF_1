import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import math

x_train = pd.read_csv("Train.csv")
x_test = pd.read_csv("Test.csv")

def log_method(x):
    if x == 0:
        return 0
    return math.log(x,2)

test = x_train["Project_Valuation"].order()

test = test.apply(lambda x: log_method(x))

mean = sum(test) / len(test)
varience = sum((average - value) ** 2 for value in test) / len(test)
sigma = math.sqrt(variance)
plt.plot(test,mlab.normpdf(test,mean,sigma))

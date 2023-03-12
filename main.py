from statistics import LinearRegression, linear_regression
import pandas as pd
import sqlite3
import plotly.express as px
import matplotlib.pyplot as plt
from newdf import newDf
# from linearreg import multipleLinearReg
from linearRegression import multipleLinearRegression
import numpy as np

file = "C:\\xampp\\htdocs\\T13\\py_proj\\SIT_INF1002_Python_Analysis\\data_files\\export2.xlsx"
df = newDf(file)
model = multipleLinearRegression()
# df.scattermapbox()
# df.lease_scattermapbox()
df.cleandata()
df.encodeCol()
# df.correl()
# df.encodeRemainingLease()
# df.export()
# model.actualvspred(Y_test, test_pred)
X_train, Y_train, X_test, Y_test = df.split_data()
trained_weight = model.trainModel(X_train, Y_train)
# test_pred, test_loss = model.testModel(X_test, Y_test, trained_weight)
model.pred_all(trained_weight, X_test, Y_test)
model.plotLoss()

# model.actual_to_pred_graph(Y_test, test_pred)

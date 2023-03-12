from unicodedata import digit
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import sqlite3


class newDf:
    # https://royleekiat.com/2020/10/22/how-to-predict-hdb-resale-prices-using-3-different-machine-learning-models-linear-regression-neural-networks-k-nearest-neighbours/
    def __init__(self, file):
        self.df = pd.read_excel(file)
        self.con = sqlite3.connect('resaleFlat.db')
        self.cur = self.con.cursor()

    def print_head(self):
        print(self.df.head())
        print(self.df.info())

    def correl(self):
        print(self.df.describe)
        corrMatrix = self.df.corr()
        print(corrMatrix)

    def cleandata(self):
        self.df.dropna()
        collist = [x.lower().replace(" ", "").replace("?.", "").replace("-", "_").replace(r"/", "_").replace(".", "") \
                        .replace("\\", "_").replace("%", "_").replace(")", "").replace(r"(", "").replace("$", "").replace("#", "") for x in self.df.columns]
        self.df.columns = collist

    # export df to file
    def export(self):
        self.df.to_excel(r'C:\xampp\htdocs\T13\py_proj\data_files\export.xlsx', index = False, header=True)

    # label encode columns in to_be_coded
    def encodeCol(self):
        # label encoding town, storeyrange, flatype
        le = LabelEncoder()
        to_be_coded = ['town', 'storey_range', 'flat_type', 'lease_commence_date']
        for i in to_be_coded:
            self.df[i] = le.fit_transform(self.df[i])
        
    def encodeFloorareasqm(self):
        range_list = [40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250]
        for i in self.df['floor_area_sqm']:
            count = 1
            for j in range_list:
                if i < j:
                    self.df['floor_area_sqm'] = self.df['floor_area_sqm'].replace([i],count)
                else:
                    count+=1
            
    # splits the given dataset into trainset and testset
    def split_data(self, split=0.8,random_state=0):
        """
        X = dependent
        Y = independent
        Training = 80% of data
        Test = 20% of data
        """
        Y = self.df[['resale_price']].to_numpy()
        X = self.df[['town', 'storey_range', 'floor_area_sqm', 'flat_type']].to_numpy()
        # X = self.df[['town', 'storey_range']].to_numpy()
    
        np.random.seed(random_state)                  #seed(0) to generate same set of random indices everytime
        indices = np.random.permutation(len(X))       #shuffle array of indices
        split_size = int(X.shape[0] * split)          #n% of data = n/100*data. eg.if 80%, split=0.8

        #Separate the X, Y into the Train and Test Set
        train_indices = indices[:split_size]      #take first n% array of indices
        test_indices = indices[split_size:]       #take last remainder % array of indices
        X_train = X[train_indices]                    
        Y_train = Y[train_indices]
        X_test = X[test_indices]
        Y_test = Y[test_indices]
        return X_train, Y_train, X_test, Y_test

    def con_db(self):
        self.cur.execute(""" CREATE TABLE IF NOT EXISTS resaleFlat(
                         id INTEGER PRIMARY KEY AUTOINCREMENT,
                         month DATE NOT NULL,
                         town TEXT NOT NULL,
                         flat_type TEXT NOT NULL,
                         block TEXT NOT NULL, 
                         street_name TEXT NOT NULL, 
                         storey_range TEXT NOT NULL,
                         floor_area_sqm FLOAT NOT NULL,
                         flat_model TEXT NOT NULL,
                         lease_commence_date INTEGER NOT NULL,
                         resale_price INTEGER NOT NULL,
                         remaining_lease INTEGER NOT NULL)
                     """)
        # send df to sqlite
        self.df.to_sql('resaleFlat', self.con, if_exists='replace')

    def get_data_from_db(self):
        self.df = pd.read_sql("SELECT * FROM resaleFlat", self.con)
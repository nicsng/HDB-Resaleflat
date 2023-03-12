import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import sqlite3
import requests
import math
import plotly.express as px
import plotly.graph_objects as go
from geopy.distance import geodesic as gd

class newDf:
    def __init__(self, file):
        self.df = pd.read_excel(file)
        self.con = sqlite3.connect('resaleFlat.db')
        self.cur = self.con.cursor()

    def print_head(self):
        # print(self.df.head())
        print(self.df.info())

    # correlation matrix of variables
    def correl(self):
        corrMatrix = self.df.corr()
        print(corrMatrix)
        fig = go.Figure()
        fig.add_trace(go.Heatmap(x = corrMatrix.columns, y = corrMatrix.index, z = np.array(corrMatrix), text=corrMatrix.values,
            colorscale='Blues',texttemplate='%{text:.2f}'))
        fig.show()

    # scattermapbox chart for resale price based on address
    def scattermapbox(self):
        fig = px.scatter_mapbox(self.df, lat = "lat", lon = "long", color = "resale_price", color_continuous_scale = px.colors.cyclical.IceFire, 
            size_max = 20, zoom = 10, mapbox_style = "carto-positron")
        fig.show()

    def lease_scattermapbox(self):
        fig = px.scatter_mapbox(self.df, lat = "lat", lon = "long", color = "remaining_lease", color_continuous_scale = px.colors.sequential.Blues, 
            size_max = 20, zoom = 10, mapbox_style = "carto-positron")
        fig.show()

    # clean special characters in column names
    def cleandata(self):
        self.df.dropna()
        self.df.drop_duplicates()
        collist = [x.lower().replace(" ", "").replace("?.", "").replace("-", "_").replace(r"/", "_").replace(".", "") \
                        .replace("\\", "_").replace("%", "_").replace(")", "").replace(r"(", "").replace("$", "").replace("#", "") for x in self.df.columns]
        self.df.columns = collist

        # extract year column from month column
        self.df['year'] = pd.DatetimeIndex(self.df['month']).year

    # export df to file
    def export(self):
        self.df.to_excel(r'C:\xampp\htdocs\T13\py_proj\SIT_INF1002_Python_Analysis\data_files\export2.xlsx', index = False, header=True)

    # label encode columns in to_be_coded list
    def encodeCol(self):
        le = LabelEncoder()
        to_be_coded = ['town', 'storey_range', 'flat_type', 'lease_commence_date', 'flat_model', 'month', 'year']
        for i in to_be_coded:
            self.df[i] = le.fit_transform(self.df[i])
        
    # encode floor_area_sqm according to range_list
    def encodeFloorareasqm(self):
        range_list = [40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250]
        for i in self.df['floor_area_sqm']:
            count = 1
            for j in range_list:
                if i < j:
                    self.df['floor_area_sqm'] = self.df['floor_area_sqm'].replace([i], count)
                else:
                    count+=1

    # encode remaining_lease according to range_list
    def encodeRemainingLease(self):
        range_list = [45,50,55,60,65,70,75,80,85,90,95,100]
        totaldone = 1
        for i in self.df['remaining_lease']:
            count = 1
            for j in range_list:
                if i < j:
                    self.df['remaining_lease'] = self.df['remaining_lease'].replace([i], count)
                    totaldone += 1
                else:
                    count+=1
            print(totaldone, "out of", len(self.df))
            if totaldone == 135195:
                break

    # splits the given dataset into trainset and testset
    def split_data(self, split = 0.8, random_state = 0):
        """
        X = dependent variable
        Y = independent variable
        Training = 80% of data
        Test = 20% of data
        """
        Y = self.df[['resale_price']].to_numpy()
        X = self.df[['storey_range', 'floor_area_sqm', 'flat_type', 'remaining_lease', 'year', 'town']].to_numpy()
    
        np.random.seed(random_state)                  # seed(0) to generate same set of random indices everytime
        indices = np.random.permutation(len(X))       # shuffle array of indices
        split_size = int(X.shape[0] * split)          # n% of data = n/100*data. eg.if 80%, split=0.8

        #Separate the X, Y into the Train and Test Set
        train_indices = indices[:split_size]      #take first n% array of indices
        test_indices = indices[split_size:]       #take last remainder % array of indices
        X_train = X[train_indices]                    
        Y_train = Y[train_indices]
        X_test = X[test_indices]
        Y_test = Y[test_indices]
        return X_train, Y_train, X_test, Y_test

    # connect to db if need to use for displaying dataset
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

    # get data from db if need to use for displaying dataset
    def get_data_from_db(self):
        self.df = pd.read_sql("SELECT * FROM resaleFlat", self.con)
        self.con.close()

    # use oneMap API to get long lat of address
    def useAPI(self, address):
        req = requests.get('https://developers.onemap.sg/commonapi/search?searchVal='+address+'&returnGeom=Y&getAddrDetails=Y&pageNum=1')
        resultsdict = eval(req.text)
        if len(resultsdict['results'])>0:
            return resultsdict['results'][0]['LATITUDE'], resultsdict['results'][0]['LONGITUDE']
        else:
            pass

    # pass df with long lat of addresses into excel sheet 
    def getcoord(self):
        self.df['Address'] = self.df['block'].apply(str) + " " + self.df['street_name'] # add block + street_name
        addresslist = list(self.df['Address'])                                          # add address column in df to list
        coordinateslist= []
        for address in addresslist:
            try:
                if len(self.useAPI(address))>0:                                         # if API returns a result, add to coordinateslist, else none
                    coordinateslist.append(self.useAPI(address))
            except:
                coordinateslist.append(None)    

        coordlist = []                                                                       
        for i in coordinateslist:                                                       # remove None from list
            if i is not None:
                coordlist.append(list(i))
            else:
                coordlist.append((None,None))
        
        df_coordinates = pd.DataFrame(coordlist)
        df_coordinates.columns = ['lat','long']                                         # change column names to lat, long
        combineddf = self.df.join(df_coordinates)                                       # join longlat df to main df
        combineddf.dropna(inplace=True)                                                 # drop columns with NA
        combineddf.to_excel('2021-2022.xlsx', index = False)

    # convert degree to radians
    def degree2rad(self, deg):
        return deg * (math.pi/180)

    # use pythagoras theorem to calculate dist between HDB and raffles place 
    def distFromCity(self):
        distlist = []
        count = 0
        radius = 6371                                                                           # radius of the earth in km
        for i in range(len(self.df)):
            count = count + 1
            print('Extracting',count,'out of',len(self.df),'addresses')
            lat2 = self.df['lat'][i]
            long2 = self.df['long'][i]
            dLat = self.degree2rad((lat2 - 1.2837))                                             # latlong of raffles place mrt: 1.2837, 103.8509
            dLon = self.degree2rad((long2 - 103.8509))
            a = math.sin(dLat/2)**2 + math.cos(1.2837) * math.cos(lat2) * math.sin(dLon/2)**2   # pythagoras theorem
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            dist = radius * c
            distlist.append(dist)
        df_dist = pd.DataFrame(distlist)
        df_dist.columns = ['dist_from_city_center']
        combineddf = self.df.join(df_dist)                                                      # join longlat df to main df
        combineddf.dropna(inplace=True)                                                         # drop columns with NA
        combineddf.to_excel('test2.xlsx', index = False)


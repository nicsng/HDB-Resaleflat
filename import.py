import pandas as pd
import sqlite3
import sys
import plotly.express as px

# create connection to sqlite database
# con = sqlite3.connect(':memory:')
con = sqlite3.connect('resaleFlat.db')
cur = con.cursor()
try:
    if sys.argv[1] == 'drop':
        # cur.execute("DROP TABLE IF EXIST resaleFlat")

        #create table if doesnt exist
        cur.execute(""" CREATE TABLE IF NOT EXISTS resaleFlat(
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

        df = pd.read_excel(r"resale flat price.xlsx")
        file = 'resale flat price'

        #clean table names -> lower case letters, remove all white spaces and $, replace -, /, \\, $ with _
        clean_table_name = file.lower().replace(" ", "").replace("?", "").replace("-", "_") \
                            .replace(r"/", "_").replace("\\", "_").replace("%", "_").replace(")", "").replace("$", "")

        #clean column names
        df.columns = [x.lower().replace(" ", "").replace("?", "").replace("-", "_") \
                            .replace(r"/", "_").replace("\\", "_").replace("%", "_").replace(")", "").replace("$", "") for x in df.columns]

        # df['month'] = pd.to_datetime(df['month'].dt.date)
        # send df to sqlite
        df.to_sql('resaleFlat', con, if_exists='replace', index=False)
    
except:
    df = pd.read_sql("SELECT * FROM resaleFlat", con)
    print(df.head())

con.close()
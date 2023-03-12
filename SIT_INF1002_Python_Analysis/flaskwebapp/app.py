from base64 import encode
from msilib.schema import tables
from turtle import title
from xml.etree.ElementInclude import DEFAULT_MAX_INCLUSION_DEPTH
from flask import Flask, render_template
import pandas as pd
import json
import plotly
import plotly.figure_factory as ff
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import os
import csv
from newdf import newDf
from sklearn.preprocessing import LabelEncoder
#from pathlib import Path

app = Flask(__name__)

def encodeCol(df):
    le = LabelEncoder()
    to_be_coded = ['town', 'storey_range', 'flat_type', 'lease_commence_date', 'flat_model', 'month']
    for i in to_be_coded:
        df[i] = le.fit_transform(df[i])

@app.route('/')

def notdash():
   ################     Graph 1     ################
   #excelPath = os.path.join(os.getcwd(), "/resaleflatpricelonglat.xlsx" )
   print(os.getcwd())
   df123 = pd.read_excel(os.getcwd() + "/datasets" + "/resaleflatpricelonglat.xlsx", sheet_name="Sheet1")

   flattype_count = df123['storey_range'].value_counts()
   count_results = pd.DataFrame(flattype_count)
   count_results = count_results.reset_index()  
   count_results.columns = ['storey_ranges', 'resale_sold']

   flattype2_count = df123['flat_type'].value_counts()
   count_results2 = pd.DataFrame(flattype2_count)
   count_results2 = count_results2.reset_index()  
   count_results2.columns = ['flat_type', 'resale_sold']

   labels = count_results['storey_ranges']
   labels2= count_results2['flat_type']
   values = count_results['resale_sold']
   values2 = count_results2['resale_sold']

   fig = go.Figure(data=[go.Pie(labels=labels, values=values, name='Flat type', textinfo='label+percent', textposition='inside'),
                      go.Pie(labels=labels2, values=values2, name='Storey range', textinfo='label+percent',visible= False,textposition='inside')])
   fig.update_layout(
    updatemenus=[
        dict(
            active=0,
            buttons=list([
                dict(label="Storey range",
                     method="update",
                     args=[{"visible": [True, False]},
                           {"title": "Number of flats sold by storey range",
                            }]),
                dict(label="Flat type",
                     method="update",
                     args=[{"visible": [False, True]},
                           {"title": "Number of flats sold by type",
                            }]),
            ]),
        )
    ])
    #Number of flats sold by type
    #Number of flats sold by storey range
    #adding title
   fig.update_layout(title_text='Number of flats sold', title_x=0.5)
   
   graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

   ################     Graph 2 massive scatter plot colorfk    ################

   df2 = pd.read_excel(os.getcwd() + "/datasets" + "/resaleflatpricelonglat.xlsx", sheet_name="Sheet1")
   df2['storey_range'] = df2['storey_range'].apply(lambda storey_range:storey_range[:2])
   df2['storey_range'] = df2['storey_range'].astype(float)
   fig2 = px.scatter(df2, x="floor_area_sqm", y="resale_price", color="flat_type")
   
   # add title
   fig2.update_layout(title_text='Flat type and Floor area VS resale price', title_x=0.5)
   
   graph2JSON = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

   ################     Graph 3 dist plot    ################

   df3 = pd.read_excel(os.getcwd() + "/datasets" +"/resaleflatpricelonglat.xlsx", sheet_name="Sheet1")
   fig3 = px.histogram(df3, x="resale_price")

   # add title
   fig3.update_layout(title_text='Distribution of housing prices', title_x=0.5)

   graph3JSON = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)


   ################     Graph 4   pink scatter  ################


   df4 = pd.read_excel(os.getcwd() + "/datasets" + "/resaleflatpricelonglat.xlsx", sheet_name="Sheet1")
   fig4 = px.scatter(df4, x="floor_area_sqm", y="resale_price", color_discrete_sequence=['pink'], trendline="ols",trendline_color_override="red")

   # add title
   fig4.update_layout(title_text='Floor Area vs Resale prices', title_x=0.5, font=dict())

   graph4JSON = json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder)

   ################     Graph 5   Prices over time line graph   ################
   df11 = pd.read_excel(os.getcwd() + "/datasets" + "/resaleflatpricelonglat.xlsx", sheet_name="Sheet1")

   #group by averages etc
   mean_price_lease=df11.groupby('lease_commence_date', as_index=False)['resale_price'].mean()
   median_price_lease=df11.groupby('lease_commence_date', as_index=False)['resale_price'].median()
   max_price_lease=df11.groupby('lease_commence_date', as_index=False)['resale_price'].max()
   min_price_lease=df11.groupby('lease_commence_date', as_index=False)['resale_price'].min()

   #convert groupby to dataframe
   dfmean = pd.DataFrame(mean_price_lease)
   dfmedian = pd.DataFrame(median_price_lease)
   dfmax = pd.DataFrame(max_price_lease)
   dfmin = pd.DataFrame(min_price_lease)

   #convert dataframe to numpy array
   s_mean = dfmean['resale_price'].to_numpy()
   s_median = dfmedian['resale_price'].to_numpy()
   s_max = dfmax['resale_price'].to_numpy()
   s_min = dfmin['resale_price'].to_numpy()
   s_year = dfmin['lease_commence_date'].to_numpy()

      # Create graph
   fig5 = go.Figure()

   fig5.add_trace(go.Scatter(x=s_year, y=s_mean,
                     mode='lines',
                     name='Mean Price'))
   fig5.add_trace(go.Scatter(x=s_year, y=s_median,
                     mode='lines',
                     name='Median Price'))
   fig5.add_trace(go.Scatter(x=s_year, y=s_max,
                     mode='lines',
                     name='Max Price'))
   fig5.add_trace(go.Scatter(x=s_year, y=s_min,
                     mode='lines',
                     name='Min Price'))

   # Add dropdown
   fig5.update_layout(
      updatemenus=[
         dict(
               active=0,
               buttons=list([
                  dict(label="All",
                        method="update",
                        args=[{"visible": [True]},
                              {"title": "Resale Price over time",
                              }]),
                  dict(label="Mean Price",
                        method="update",
                        args=[{"visible": [True, False, False, False]},
                              {"title": "Mean Price over time",
                              }]),
                  dict(label="Median Price",
                        method="update",
                        args=[{"visible": [False, True, False, False]},
                              {"title": "Median Price over time",
                              }]),
                  dict(label="Max Price",
                        method="update",
                        args=[{"visible": [False,False, True, False]},
                              {"title": "Max Price over time",
                              }]),
                  dict(label="Min Price",
                        method="update",
                        args=[{"visible": [False, False, False,True]},
                              {"title": "Min Price over time",
                              }]),
                  
               ]),
         )
      ])

   fig5.update_layout(title_text='Resale prices over time', title_x=0.5, font=dict())

   graph5JSON = json.dumps(fig5, cls=plotly.utils.PlotlyJSONEncoder)

   ################     Graph 6 Scatter Mapbox  ################
   
   dataframe6 = pd.read_excel(os.getcwd() + "/datasets" + "/resaleflatpricelonglat.xlsx", sheet_name="Sheet1")
   #encodeCol(dataframe6)
   year_count = dataframe6['month'].value_counts()
   count_results6 = pd.DataFrame(year_count)
   count_results6 = count_results6.reset_index()
   count_results6.columns = ['month_and_year', 'resale_sold']
   fig6 = px.bar(count_results6, x='month_and_year', y='resale_sold', title='Number/Volume of units sold over time', color='resale_sold')
   
   # add title
   fig6.update_layout(title_text='Number/Volume of units sold over time', title_x=0.5, font=dict())

   graph6JSON = json.dumps(fig6, cls=plotly.utils.PlotlyJSONEncoder)

    ################     Graph  Heatmap  ################

   dfmap = pd.read_excel(os.getcwd() + "/datasets" + "/resaleflatpricelonglat.xlsx", sheet_name="Sheet1")
   encodeCol(dfmap)
   figmap = px.scatter_mapbox(dfmap, lat = "lat", lon = "long", color = "resale_price", color_continuous_scale = px.colors.cyclical.IceFire, 
            size_max = 20, zoom = 10, mapbox_style = "carto-positron")
   # add title
   figmap.update_layout(title_text='Resale price of all flats', title_x=0.5, font=dict())

   graphmapJSON = json.dumps(figmap, cls=plotly.utils.PlotlyJSONEncoder)
   
   
   ################     Graph 7 Correlation Matrix  ################

   df7 = pd.read_excel(os.getcwd() + "/datasets" + "/resaleflatpricelonglat.xlsx", sheet_name="Sheet1")
   corrMatrix = df7.corr()
   fig7 = go.Figure()
   fig7.add_trace(go.Heatmap(x = corrMatrix.columns, y = corrMatrix.index, z = np.array(corrMatrix), text=corrMatrix.values,
            colorscale='Blues',texttemplate='%{text:.2f}'))

    # add title
   fig7.update_layout(title_text='Resale price of all flats', title_x=0.5, font=dict())

   graph7JSON = json.dumps(fig7, cls=plotly.utils.PlotlyJSONEncoder)

   ################     Graph X Filter     ################

   dfx = pd.read_excel(os.getcwd() + "/datasets" + "/resaleflatpricelonglat.xlsx", sheet_name="Sheet1")

   average_price_town=dfx.groupby('town', as_index=False)['resale_price'].median()
   average_price_type=dfx.groupby('flat_type', as_index=False)['resale_price'].median()

   df55 = pd.DataFrame(average_price_town)
   df66 = pd.DataFrame(average_price_type)

   s_town = df55['town'].to_numpy()
   s_price = df55['resale_price'].to_numpy()

   s_type = df66['flat_type'].to_numpy()
   s_price2 = df66['resale_price'].to_numpy()

   plotx = go.Figure(data=[go.Bar(
      name='Town',
      x=s_town,
      y=s_price
   ),
      go.Bar(
      name='flat Type',
      x=s_type,
      y=s_price2, visible=False
   )])


   # Add dropdown
   plotx.update_layout(
      updatemenus=[
         dict(
               active=0,
               buttons=list([
                  dict(label="Town",
                        method="update",  

                        args=[{"visible": [True, False]},
                              {"title": "Town",
                              }]),
                  dict(label="Flat Type",
                        method="update",
                        args=[{"visible": [False, True]},
                              {"title": "Flat Type",
                              }]),
               ]),
         )
      ])

   plotx.update_layout(title_text='Average price of a HDB flat', title_x=0.5, font=dict())   

   graphxJSON = json.dumps(plotx, cls=plotly.utils.PlotlyJSONEncoder)

   

   ################     end of graphs     ################
   #htmlPath = os.path.join(os.getcwd() + '/views/index.html')
   #return render_template(htmlPath, graphJSON=graphJSON, graph2JSON=graph2JSON,graphxJSON=graphxJSON,graph3JSON=graph3JSON,graph4JSON=graph4JSON,graph5JSON=graph5JSON)
   return render_template('index.html', graphJSON=graphJSON, graph2JSON=graph2JSON, graphxJSON=graphxJSON,graph3JSON=graph3JSON,graph4JSON=graph4JSON,graph5JSON=graph5JSON,graph6JSON=graph6JSON, graph7JSON=graph7JSON, graphmapJSON=graphmapJSON)


@app.route('/datasets')

def view_data_page():
    results = []
    #f = open (os.getcwd() + "/datasets" +'/resale-flat-prices-based-on-registration-date-from-jan-2017-onwards.csv', 'r')
    f = open (os.getcwd() + "/datasets" + "/resaleflatpricelonglat.csv", 'r')
    with f:
        reader = csv.DictReader(f)
        fieldnames = list()
        for row in reader:
            row['resale_price']= "${:,.2f}".format(float(row['resale_price']))
            results.append(dict(row))
                
        results = sorted(results, key=lambda x:x['ï»¿month'], reverse=True)
        fieldnames = [key for key in results[0].keys()] #column references for data table
        headers = [key for key in results[0].keys()] #column names for header
      
        '''Removing '_' in header names'''
        n = 0
        for k in headers:
            for check in k:
                if check == '_':
                    new_key = k.replace('_',' ')
                    headers[n] = new_key

                elif check == "ï":
                    new_key = k.replace('ï','')
                    headers[n] = new_key

                elif check == "»":
                    new_key = k.replace('»','')
                    headers[n] = new_key

                elif check == "¿":
                    new_key = k.replace('¿','')
                    headers[n] = new_key    
                else: 
                    continue
            n +=1

        results = results[:10000]
        return render_template('viewtable.html', results=results, fieldnames=fieldnames, len=len, headers=headers)

@app.route('/custom')

def view_custom_chart():
    ################     Graph XXL Custom graph   ################

   dfXXL = pd.read_excel(os.getcwd() + "/datasets" + "/resaleflatpricelonglat.xlsx", sheet_name="Sheet1")

   average_price_town=dfXXL.groupby('town', as_index=False)['resale_price'].median()
   average_price_type=dfXXL.groupby('flat_type', as_index=False)['resale_price'].median()
   average_price_month=dfXXL.groupby('month', as_index=False)['resale_price'].median()
   average_price_storey=dfXXL.groupby('storey_range', as_index=False)['resale_price'].median()
   average_price_areasq=dfXXL.groupby('floor_area_sqm', as_index=False)['resale_price'].median()
   average_price_model=dfXXL.groupby('flat_model', as_index=False)['resale_price'].median()
   average_price_lease=dfXXL.groupby('remaining_lease', as_index=False)['resale_price'].median()


   df55 = pd.DataFrame(average_price_town)
   df66 = pd.DataFrame(average_price_type)
   df77 = pd.DataFrame(average_price_month)
   df88 = pd.DataFrame(average_price_storey)
   df99 = pd.DataFrame(average_price_areasq)
   df100 = pd.DataFrame(average_price_model)
   df101 = pd.DataFrame(average_price_lease)


   s_town = df55['town'].to_numpy()
   s_price = df55['resale_price'].to_numpy()

   s_type = df66['flat_type'].to_numpy()
   s_price2 = df66['resale_price'].to_numpy()

   s_month = df77['month'].to_numpy()
   s_price3 = df77['resale_price'].to_numpy()

   s_storey = df88['storey_range'].to_numpy()
   s_price4 = df88['resale_price'].to_numpy()

   s_areasq = df99['floor_area_sqm'].to_numpy()
   s_price5 = df99['resale_price'].to_numpy()

   s_model = df100['flat_model'].to_numpy()
   s_price6 = df100['resale_price'].to_numpy()

   s_lease = df101['remaining_lease'].to_numpy()
   s_price7 = df101['resale_price'].to_numpy()


   figXXL = go.Figure(data=[go.Bar(
       name='Town',
       x=s_town,
       y=s_price
   ),
       go.Bar(
       name='month',
       x=s_month,
       y=s_price3, visible=False
   ),
       go.Bar(
       name='storey_range',
       x=s_storey,
       y=s_price4, visible=False
   ),
       go.Bar(
       name='flat Type',
       x=s_type,
       y=s_price2, visible=False
   ),
       go.Bar(
       name='floor area sqm',
       x=s_areasq,
       y=s_price5, visible=False
   ),
       go.Bar(
       name='Flat Model',
       x=s_model,
       y=s_price6, visible=False
   ),
       go.Bar(
       name='Remaining Lease',
       x=s_lease,
       y=s_price7, visible=False
   )])


# Add dropdown
   figXXL.update_layout(
       updatemenus=[
           dict(
               buttons=list([
                   dict(
                       args=["type", "bar"],
                       label="Bar Chart",
                       method="restyle"
                   ),
                   dict(
                       args=[{"mode": "markers", "type": "scatter"}],
                       label="Scatter Plot",
                       method="restyle"
                   ),
                   dict(
                       args=[{"mode": "lines", "type": "pie", "textinfo":"label+percent", "textposition":"inside"}],
                       label="Pie Chart",
                       method="restyle"
                   ),
                   dict(
                       args=[{"mode": "lines", "type": "scatter"}],
                       label="Line Chart",
                       method="restyle"
                   )
               ]),
               direction="down",
               pad={"r": 10, "t": 10},
               showactive=True,
               x=0.1,
               xanchor="left",
               y=1.5,
               yanchor="top"
           ),
           
        
           dict(
               active=0,
               buttons=list([
                   dict(label="Town",
                        method="update",
                        args=[{"visible": [True, False, False, False, False, False, False]},
                              {"title": "Flat Type",
                               }]),
                   dict(label="Month",
                        method="update",
                        args=[{"visible": [False, True, False, False, False, False, False]},
                              {"title": "Month",
                               }]),
                   dict(label="Storey",
                        method="update",
                        args=[{"visible": [False, False, True, False, False, False, False]},
                              {"title": "Storey",
                               }]),
                   dict(label="Flat Type",
                        method="update",
                        args=[{"visible": [False, False, False, True, False, False, False]},
                              {"title": "Flat Type",
                               }]),
                   dict(label="Floor area sqm",
                        method="update",
                        args=[{"visible": [False, False, False, False, True, False, False]},
                              {"title": "Flat Type",
                               }]),
                   dict(label="Flat Model",
                        method="update",
                        args=[{"visible": [False, False, False, False, False, True, False]},
                              {"title": "Flat Model",
                               }]),
                   dict(label="Remaining Lease",
                        method="update",
                        args=[{"visible": [False, False, False, False, False, False, True]},
                              {"title": "Remaining Lease",
                               }]),
                
                
                
               ]),
               direction="down",
               pad={"r": 10, "t": 10},
               showactive=True,
               x=0.3,
               xanchor="left",
               y=1.5,
               yanchor="top"
           )])
      #adding title 
   figXXL.update_layout(title_text='Number/Volume of units sold over time', title_x=0.5, font=dict())

   graphXXLJSON = json.dumps(figXXL, cls=plotly.utils.PlotlyJSONEncoder)


   ################     end of graphs     ################
   #htmlPath = os.path.join(os.getcwd() + '/views/index.html')
   #return render_template(htmlPath, graphJSON=graphJSON, graph2JSON=graph2JSON,graphxJSON=graphxJSON,graph3JSON=graph3JSON,graph4JSON=graph4JSON,graph5JSON=graph5JSON)
   return render_template('custom.html', graphXXLJSON=graphXXLJSON)


if __name__ != '__main__':
    app.run(debug=False)
else:
   app.run(debug=True)




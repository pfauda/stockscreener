#IMPORTING ALL THE NECESSARY PACKAGES.
import matplotlib.pyplot as plt                #Importing matplotlib to plot and analyse data.
import pandas as pd
import plotly.express as px   
from prophet import Prophet                  #Importing prophet (prediction and forecasting library.)
from prophet.plot import plot_plotly, plot_components_plotly
import yfinance as yf
import streamlit as st

def load_data(symbol: str):

    df = yf.download(symbol, '2022-01-01', auto_adjust=True)
    df = df.reset_index()
    df = df.reset_index()
    df[['ds','y']] = df[['Date','Close']]
    return df

col1, col2 = st.columns(2) #st.columns(3)
#c1, c2 = st.beta_columns([1, 4])

sn = col1.text_input("Activo", key="symbol_name", max_chars=10)
if col2.button('Analizar'):
    df = load_data(sn)
    st.dataframe(df.sample(10))


def get_holidays():

    playoffs = pd.DataFrame({
      'holiday': 'playoff',
      'ds': pd.to_datetime(['2008-01-13', '2009-01-03', '2010-01-16',
                            '2010-01-24', '2010-02-07', '2011-01-08',
                            '2013-01-12', '2014-01-12', '2014-01-19',
                            '2014-02-02', '2015-01-11', '2016-01-17',
                            '2016-01-24', '2016-02-07']),
      'lower_window': 0,
      'upper_window': 1,
    })
    superbowls = pd.DataFrame({
      'holiday': 'superbowl',
      'ds': pd.to_datetime(['2010-02-07', '2014-02-02', '2016-02-07']),
      'lower_window': 0,
      'upper_window': 1,
    })
    holidays = pd.concat((playoffs, superbowls))

    lockdowns = pd.DataFrame([
        {'holiday': 'lockdown_1', 'ds': '2020-03-21', 'lower_window': 0, 'ds_upper': '2020-06-06'},
        {'holiday': 'lockdown_2', 'ds': '2020-07-09', 'lower_window': 0, 'ds_upper': '2020-10-27'},
        {'holiday': 'lockdown_3', 'ds': '2021-02-13', 'lower_window': 0, 'ds_upper': '2021-02-17'},
        {'holiday': 'lockdown_4', 'ds': '2021-05-28', 'lower_window': 0, 'ds_upper': '2021-06-10'},
    ])
    for t_col in ['ds', 'ds_upper']:
        lockdowns[t_col] = pd.to_datetime(lockdowns[t_col])
    lockdowns['upper_window'] = (lockdowns['ds_upper'] - lockdowns['ds']).dt.days

    return lockdowns


#df = load_data('HMY')
#fig = px.line(df, x='Date', y='Close')
#fig.update_xaxes(rangeslider_visible=True)
#fig.show()

#m = Prophet(holidays=get_holidays())
#m.add_country_holidays(country_name='US')
#m.fit(df)
#
#future = m.make_future_dataframe(periods=90)
#
#forecast = m.predict(future)
#forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
#
#fig1 = m.plot(forecast)
#fig2 = m.plot_components(forecast)
#
#plot_plotly(m, forecast, uncertainty=True, plot_cap=False, trend=True, changepoints=True)
#plot_components_plotly(m, forecast)
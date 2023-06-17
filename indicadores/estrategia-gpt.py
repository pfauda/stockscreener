# Importamos las bibliotecas de Pandas y Numpy
import pandas as pd
import numpy as np

import sqlite3
from sqlite3 import Error
import os
from datetime import date

import matplotlib.pyplot as plt


# create connection to database
def create_connection(db_file):
    ''' create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    '''
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)

    return conn

# get prices of stock
def get_stock_prices(conn, symbol, sdate, edate):

    sql = ''' SELECT Date, Open, High, Low, Close, Volume FROM stocks WHERE Name = ? AND Date BETWEEN ? AND ? '''
    cur = conn.cursor()
    try:
        cur.execute(sql, (symbol, sdate, edate, ))
        rows = cur.fetchall()
        df = pd.DataFrame(rows)
        col_map = {df.columns[0]: 'Date', df.columns[1]: 'Open', df.columns[2]: 'High', df.columns[3]: 'Low', df.columns[4]: 'Close', df.columns[5]: 'Volume'}
        df.rename(columns=col_map, inplace=True)
        df.set_index('Date', inplace=True)
        return df
    except Error as e:
        print(e)
        return pd.DataFrame()

# get symbols
def select_stocks_symbols(conn):

    sql = ''' SELECT * FROM symbols '''
    cur = conn.cursor()
    try:
        cur.execute(sql)
        return pd.DataFrame(cur.fetchall(), columns=['Name', 'Panel', 'Comprado', 'Vigilar'])
    except:
        return pd.DataFrame()

def main():

    conn = create_connection(os.path.realpath('stocks.db'))

    # Cargamos los datos del precio en un DataFrame de Pandas
    ticker='GGAL'
    startdate='2016-01-01'
    enddate=date.today()
    df = get_stock_prices(conn, ticker, startdate, enddate)

    # Definimos las constantes que se utilizarán en el código
    length = 20  # Longitud de las bandas
    stdDev = 2  # Desviación estándar de las bandas

    # Calculamos las bandas de Bollinger
    df['Upper'], df['Middle'], df['Lower'] = pd.Series.rolling(df['Close'], length).mean() + stdDev * pd.Series.rolling(df['Close'], length).std(), pd.Series.rolling(df['Close'], length).mean(), pd.Series.rolling(df['Close'], length).mean() - stdDev * pd.Series.rolling(df['Close'], length).std()

    # Creamos una columna de señales de trading
    df['Signal'] = np.where(df['Close'] > df['Upper'], 1, np.where(df['Close'] < df['Lower'], -1, 0))

    # Creamos una columna de posiciones de trading
    df['Position'] = df['Signal'].shift(1)

    # Calculamos el rendimiento de la estrategia
    df['Pips'] = df['Position'] * (df['Close'] - df['Close'].shift(1))
    df['Profit'] = df['Pips'] * 10000
    df['Cumulative'] = df['Profit'].cumsum()

    # Graficamos el rendimiento de la estrategia
    df[['Profit', 'Cumulative']].plot()

    plt.show()


if __name__ == '__main__':
    main()

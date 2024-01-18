import sqlite3
from sqlite3 import Error
from sqlite3.dbapi2 import Connection
import yfinance as yf
import datetime as dt
import numpy as np
import getpass
from tqdm import tqdm

from pyhomebroker import HomeBroker

def create_connection(db_file) -> Connection:
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        sqlite3.register_adapter(np.int64, int)
        sqlite3.register_adapter(np.int32, int)
        conn = sqlite3.connect(db_file, cached_statements=0)
    except Error as e:
        print(str(e))

    return conn  # type: ignore

def insert_stock_prices_from(conn, stocks_prices: np.recarray):

    if stocks_prices.size > 0:

        sql = ''' INSERT INTO stocks(Name, Date, Open, High, Low, Close, AdjClose, Volume)
                  VALUES(?, ?, ?, ?, ?, ?, ?, ?) '''
        cur = conn.cursor()
        cur.executemany(sql, stocks_prices)
        conn.commit()

def select_last_stock_price(conn, stock_name):

    sql = ''' SELECT MAX(Date) FROM stocks WHERE Name = ? '''
    cur = conn.cursor()
    cur.execute(sql, (stock_name,))
    last_date = cur.fetchone()[0]
    return "2010-01-01" if last_date is None else last_date

def select_stocks_symbols(conn):

    sql = ''' SELECT Name FROM symbols '''
    cur = conn.cursor()
    cur.execute(sql)
    return cur.fetchall()

def get_stock_prices_from_yf(conn, symbol) -> np.recarray:

    str_start_date = select_last_stock_price(conn, symbol)

    result = np.recarray((0, ), dtype=[("Name", str), 
                                       ("Date", str),
                                       ("Open", float),
                                       ("High", float),
                                       ("Low", float),
                                       ("Close", float),
                                       ("AdjClose", float),
                                       ("Volume", int)])

    if dt.datetime.strptime(str_start_date, "%Y-%m-%d").date() < dt.datetime.today().date():

        start_date = dt.datetime(int(str_start_date[0:4]), int(str_start_date[5:7]), int(str_start_date[8:10]))
        start_date += dt.timedelta(days=1)

        try:
            df_stocks_prices = yf.download(symbol,
                start_date,
                group_by='row',
                interval='1d',
                progress=False,
                threads=False,
                auto_adjust=False,
                rounding=True)

            df_stocks_prices.fillna(method='ffill', axis=0, inplace=True)

            df_stocks_prices.reset_index(inplace=True)
            df_stocks_prices.insert(0, 'Name', symbol)
            df_stocks_prices.drop(df_stocks_prices[df_stocks_prices.Date <= str_start_date].index, inplace=True)
            df_stocks_prices['Date'] = df_stocks_prices['Date'].apply(lambda x: dt.datetime.strftime(x, '%Y-%m-%d'))
            df_stocks_prices.rename(columns={'Adj Close': 'AdjClose'}, inplace=True)
            result = df_stocks_prices.to_records(index=False)

        except:  # noqa: E722
            pass
    
    return result

def get_stock_prices_from_hb(conn, hb, symbol) -> np.recarray:


    str_start_date = select_last_stock_price(conn, symbol + '.BA')
 
    result = np.recarray((0, ), dtype=[("Name", str), 
                                       ("Date", str),
                                       ("Open", float),
                                       ("High", float),
                                       ("Low", float),
                                       ("Close", float),
                                       ("AdjClose", float),
                                       ("Volume", int)])
    
    if dt.datetime.strptime(str_start_date, "%Y-%m-%d").date() < dt.datetime.today().date():

       start_date = dt.datetime(int(str_start_date[0:4]), int(str_start_date[5:7]), int(str_start_date[8:10]))
       start_date += dt.timedelta(days=1)

       df_stocks_prices = hb.history.get_daily_history(symbol, start_date.date(), dt.datetime.today().date())

       df_stocks_prices.insert(0, 'Name', symbol + '.BA')
       col_map = {df_stocks_prices.columns[1]: 'Date', 
                  df_stocks_prices.columns[2]: 'Open', 
                  df_stocks_prices.columns[3]: 'High', 
                  df_stocks_prices.columns[4]: 'Low', 
                  df_stocks_prices.columns[5]: 'Close', 
                  df_stocks_prices.columns[6]: 'Volume'}
       df_stocks_prices.rename(columns=col_map, inplace=True)
       df_stocks_prices.insert(6, 'AdjClose', df_stocks_prices['Close'])

       df_stocks_prices['Date'] = df_stocks_prices['Date'].apply(lambda x: dt.datetime.strftime(x, '%Y-%m-%d'))
       df_stocks_prices.drop(df_stocks_prices[df_stocks_prices.Date <= str_start_date].index, inplace=True)

       result = df_stocks_prices.to_records(index=False)

    return result

def main():

    database = "stocks.db"

    hb = HomeBroker(265)
    hb_dni = input('DNI:')
    hb_usr = getpass.getpass('Usuario:')
    hb_psw = getpass.getpass('Password:')
    hb.auth.login(dni=hb_dni, user=hb_usr, password=hb_psw, raise_exception=True)

    # create a database connection
    conn = create_connection(database)

    with conn:
        tickers = select_stocks_symbols(conn)

        for ticker in (pbar:=tqdm(tickers, "Obteniendo datos", colour="red")):
            pbar.set_postfix_str(ticker[0])
            try:
                if ticker[0][-3:] == '.BA':
                    insert_stock_prices_from(conn, get_stock_prices_from_hb(conn, hb, ticker[0][:-3]))
                else:
                    insert_stock_prices_from(conn, get_stock_prices_from_yf(conn, ticker[0]))
            except Error as e:
                pbar.write(str(e))

    print("\nProceso finalizado!")

if __name__ == '__main__':
    main()

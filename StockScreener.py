import datetime as dt
from numpy import inf, nan, sign
import pandas as pd
import pandas_ta as ta
from pandas_datareader import data as pdr
from pandas.core.frame import DataFrame
import numpy as np
from pandas_ta.momentum import rsi
import requests

import sqlite3
from sqlite3 import Error
from sqlite3.dbapi2 import Cursor

import os
from xlsxwriter.utility import xl_rowcol_to_cell

CONST_BUY = "Buy"
CONST_NEUTRAL = "Neutral"
CONST_SELL = "Sell"
CONST_ADJ = False

def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)

    return conn

def get_stock_prices(conn, symbol):

    sql = ''' SELECT Date, Open, High, Low, Close, Volume FROM stocks WHERE Name = ? '''
    cur = conn.cursor()
    cur.execute(sql, (symbol,))
    rows = cur.fetchall()
    df = pd.DataFrame(rows)
    col_map = {df.columns[0]: 'Date', df.columns[1]: 'Open', df.columns[2]: 'High', df.columns[3]: 'Low', df.columns[4]: 'Close', df.columns[5]: 'Volume'}
    df.rename(columns=col_map, inplace=True)
    df.set_index('Date', inplace=True)
    return df

def select_stocks_symbols(conn):

    sql = ''' SELECT * FROM symbols '''
    cur = conn.cursor()
    cur.execute(sql)
    return pd.DataFrame(cur.fetchall(), columns=['Name', 'Panel', 'Comprado', 'Vigilar'])

def ccl(conn, _start):

    tickers = ['GGAL', 'YPF', 'PAM', 'BMA', 'GGAL.BA', 'YPFD.BA', 'PAMP.BA', 'BMA.BA']

    df_tickers = {}
    for ticker in tickers:
        df_tickers[ticker] = get_stock_prices(conn, ticker)['Close']

    # Calcular el CCL ponderado
    # TV=BCBA:PGR/(0.2*BCBA:GGAL/NASDAQ:GGAL*10+0.2*BCBA:YPFD/NYSE:YPF+0.125*BCBA:PAMP/NYSE:PAM*25+0.125*BCBA:BMA/NYSE:BMA*10)/0.65
    df_ccl = (0.2   * df_tickers['GGAL.BA'] / df_tickers['GGAL'] * 10 +
              0.2   * df_tickers['YPFD.BA'] / df_tickers['YPF'] +
              0.125 * df_tickers['PAMP.BA'] / df_tickers['PAM'] * 25 +
              0.125 * df_tickers['BMA.BA']  / df_tickers['BMA'] * 10) / 0.65
    df_ccl.fillna(method='ffill', axis=0, inplace=True)
    return df_ccl


def pine_rma(src, lenght):
    # pine_rma(src, length) =>
    #	alpha = 1/length
    #	sum = 0.0
    #	sum := na(sum[1]) ? sma(src, length) : alpha * src + (1 - alpha) * nz(sum[1])

    #   RMA = ((RMA(t-1) * (n-1)) + Xt) / n
    #   n = The length of the Moving Average
    #   X = Price

    alpha = 1/lenght
    sum = src
    sum[0] = 0.0

    iter = enumerate(src)
    next(iter)
    for i, s in iter:
        if isinstance(sum[i-1], float):
            #sum[i] = alpha * src[i] + (1 - alpha) * sum[i-1]
            sum[i] = (sum[i-1]*(lenght-1)+src[i]) / lenght
        else:
            sum[i] = src[0:i].rolling(lenght)
    return sum


def rsi_tradingview(ohlc: pd.DataFrame, period: int = 14):
    """ Implements the RSI indicator as defined by TradingView on March 15, 2021.
        The TradingView code is as follows:
        //@version=4
        study(title="Relative Strength Index", shorttitle="RSI", format=format.price, precision=2, resolution="")
        len = input(14, minval=1, title="Length")
        src = input(close, "Source", type = input.source)
        up = rma(max(change(src), lenght=0), lenght=len)
        down = rma(-min(change(src), lenght=0), lenght=len)
        rsi = down == 0 ? 100 : up == 0 ? 0 : 100 - (100 / (1 + up / down))
        plot(rsi, "RSI", color=#8E1599)
        band1 = hline(70, "Upper Band", color=#C0C0C0)
        band0 = hline(30, "Lower Band", color=#C0C0C0)
        fill(band1, band0, color=#9915FF, transp=90, title="Background")
    :param ohlc:
    :param period:
    :param round_rsi:
    :return: an array with the RSI indicator values
    """

    delta = ohlc.diff()

    up = delta.copy()
    up[up < 0] = 0
    #up.fillna(0, inplace=True)
    #up = pd.Series.ewm(up, alpha=1/period, min_periods=0, adjust=False).mean()
    up = pine_rma(up, period)

    down = delta.copy()
    down[down > 0] = 0
    #down.fillna(0, inplace=True)
    down *= -1
    #down = pd.Series.ewm(down, alpha=1/period, min_periods=0, adjust=False).mean()
    down = pine_rma(down, period)

# pine_rma(src, length) =>
#	alpha = 1/length
#	sum = 0.0
#	sum := na(sum[1]) ? sma(src, length) : alpha * src + (1 - alpha) * nz(sum[1])

    rsi = np.where(up == 0, 0, np.where(
        down == 0, 100, 100 - (100 / (1 + up / down))))
    return rsi


def calc_bb_o(df, length=None, std=None, mamode=None, ddof=0, **kwargs):
    """Indicator: Bollinger Bands (BBANDS)"""
    # Validate arguments
    length = int(length) if length and length > 0 else 5
    std = float(std) if std and std > 0 else 2.0
    mamode = mamode if isinstance(mamode, str) else "sma"
    ddof = int(ddof) if ddof >= 0 and ddof < length else 0

    if df is None:
        return

    # Calculate Result
    standard_deviation = df.ta.stdev(length=length, ddof=ddof)
    deviations = std * standard_deviation

    mid = ta.ma(mamode, df['Close'], length=length, **kwargs)
    lower = mid - deviations
    upper = mid + deviations
    osc = (df['Close'] - lower) / (upper - lower)

    # Por si hay valores infinitos
    osc.replace([np.inf, -np.inf], 0, inplace=True) 

    osc.name = f"BBO_{length}_{std}"

    # Prepare DataFrame to return+
    data = {
        osc.name: osc
    }
    bbandsdf_o = DataFrame(data)
    bbandsdf_o.name = f"BBANDS_O_{length}_{std}"

    return bbandsdf_o


def get_sentiment_analysis(ticker):

    y_header = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'application/json'
    }

    yf_url_base = 'https://query2.finance.yahoo.com/v10/finance/quoteSummary/{0}'.format(ticker)

    try:
        yf_response = requests.get(yf_url_base+'?modules=recommendationTrend', headers=y_header)
        yf_json = yf_response.json()
    except:
        pass

    rec = 2.5
    if yf_json['quoteSummary']['result'] != None:

        yf_recomentacion = yf_json['quoteSummary']['result'][0]['recommendationTrend']['trend'][0]

        strong_buy = yf_recomentacion['strongBuy']
        buy = yf_recomentacion['buy']
        hold = yf_recomentacion['hold']
        underperform = yf_recomentacion['sell']
        sell = yf_recomentacion['strongSell']
        recommendation_num = strong_buy + buy + hold + underperform + sell
        if recommendation_num != 0:
            rec = (strong_buy + buy * 2 + hold * 3 + underperform * 4 + sell * 4) / recommendation_num

    return rec

start_date = dt.datetime(2011, 1, 1)

exportList = pd.DataFrame(columns=[
    'Stock',
    'Stock_Flag',
    'Open',
    'Close',
    'Change',
    '50 D. MA',
    '150 D. Ma',
    '200 D. MA',
    '52 W. Low',
    '52 W. High',
    'SMA S.',
    'RSI',
    'RSI S.',
    'StochK',
    'StochD',
    'Stoch S.',
    'Fast BB u2',
    'Fast BB u1',
    'Fast BB l1',
    'Fast BB l2',
    'Fast BB S.',
    'Slow BB u2',
    'Slow BB u1',
    'Slow BB l1',
    'Slow BB l2',
    'Slow BB S.',
    'Fast BBO',
    'Fast BBO S.',
    'Slow BBO',
    'Slow BBO S.',
    'MACD H.',
    'M.Pring V.',
    'M.Pring S.',
	'Sent.'])

conn = create_connection(r"stocks.db")

stocklist = select_stocks_symbols(conn)

df_ccl = ccl(conn, start_date)

start_date = df_ccl.index[0]
if (dt.datetime.today() -  dt.datetime.strptime(start_date, '%Y-%m-%d')).days > 3650:
	start_date = dt.datetime.today() - dt.timedelta(days=3650)
if dt.datetime.today().hour < 18:
    end_date = dt.datetime.today() - dt.timedelta(days=1)
else:
    end_date = dt.datetime.today()

for i in stocklist.index:

    stock = str(stocklist['Name'][i])
    stock_flag = ''
    if stocklist['Comprado'][i] != '':
        stock_flag = 'Comprado'

    if stocklist['Vigilar'][i] != '':
        stock_flag = 'Vigilar'

    df_sma = DataFrame()
    df_bb_fast = DataFrame()
    df_bb_slow = DataFrame()

    print(stock)

    try:
        df = get_stock_prices(conn, stock)
        df.fillna(method='ffill', axis=0, inplace=True)

        if stock[-3:] == '.BA':
            df = df.divide(df_ccl, axis=0)
        
        df.dropna(inplace=True)

        smaUsed = [50, 150, 200]
        for x in smaUsed:
            sma = x
            try:
                df_sma['SMA_' + str(sma)] = df.ta.sma(length=sma)
            except Exception as e:
                df_sma['SMA_' + str(sma)] = 0

        try:
            currentOpen = df['Open'][-1]
            currentClose = df['Close'][-1]
            percentageChange = df['Close'].pct_change()[-1]
            moving_average_50 = df_sma['SMA_50'][-1]
            moving_average_150 = df_sma['SMA_150'][-1]
            moving_average_200 = df_sma['SMA_200'][-1]
            low_of_52week = min(df['Close'][-260:])
            high_of_52week = max(df['Close'][-260:])
        except Exception as e:
            moving_average_50 = 0
            moving_average_150 = 0
            moving_average_200 = 0
            low_of_52week = 0
            high_of_52week = 0

        try:
            moving_average_200_20 = df_sma['SMA_200'][-20]
        except Exception as e:
            moving_average_200_20 = 0

        try:
            #v_rsis = rsi_tradingview(df)
            v_rsi = df.ta.rsi()[-1]
        except Exception as e:
            v_rsi = 0

        try:
            v_stochK = df.ta.stoch()['STOCHk_14_3_3'][-1]
            v_stochD = df.ta.stoch()['STOCHd_14_3_3'][-1]
        except Exception as e:
            v_stochK = 0
            v_stochD = 0

        try:
            v_bb_fast_u1 = df.ta.bbands(length=20, std=1)['BBU_20_1.0'][-1]
            v_bb_fast_l1 = df.ta.bbands(length=20, std=1)['BBL_20_1.0'][-1]
            v_bb_fast_u2 = df.ta.bbands(length=20, std=2)['BBU_20_2.0'][-1]
            v_bb_fast_l2 = df.ta.bbands(length=20, std=2)['BBL_20_2.0'][-1]
        except Exception as e:
            v_bb_fast_u1 = 0
            v_bb_fast_l1 = 0
            v_bb_fast_u2 = 0
            v_bb_fast_l2 = 0

        try:
            v_bb_slow_u1 = df.ta.bbands(length=84, std=1)['BBU_84_1.0'][-1]
            v_bb_slow_l1 = df.ta.bbands(length=84, std=1)['BBL_84_1.0'][-1]
            v_bb_slow_u2 = df.ta.bbands(length=84, std=2)['BBU_84_2.0'][-1]
            v_bb_slow_l2 = df.ta.bbands(length=84, std=2)['BBL_84_2.0'][-1]
        except Exception as e:
            v_bb_slow_u1 = 0
            v_bb_slow_l1 = 0
            v_bb_slow_u2 = 0
            v_bb_slow_l2 = 0

        if df['Close'][-1] >= v_bb_fast_u1:
            v_bb_fast_signal = CONST_SELL
        elif df['Close'][-1] <= v_bb_fast_l1:
            v_bb_fast_signal = CONST_BUY
        else:
            v_bb_fast_signal = CONST_NEUTRAL

        if df['Close'][-1] >= v_bb_slow_u1:
            v_bb_slow_signal = CONST_SELL
        elif df['Close'][-1] <= v_bb_slow_l1:
            v_bb_slow_signal = CONST_BUY
        else:
            v_bb_slow_signal = CONST_NEUTRAL

        #DBB %B
        try:
            v_bb_fast_o = calc_bb_o(df, length=20, std=2)['BBO_20_2.0'][-1]
            if v_bb_fast_o >= 1:
                v_bb_fast_o_signal = CONST_SELL
            elif v_bb_fast_o <= 0:
                v_bb_fast_o_signal = CONST_BUY
            else:
                v_bb_fast_o_signal = CONST_NEUTRAL
        except Exception as e:
            v_bb_fast_o = 0

        try:
            v_bb_slow_o = calc_bb_o(df, length=84, std=2)['BBO_84_2.0'][-1]
            if v_bb_fast_o >= 1:
                v_bb_slow_o_signal = CONST_SELL
            elif v_bb_slow_o <= 0:
                v_bb_slow_o_signal = CONST_BUY
            else:
                v_bb_slow_o_signal = CONST_NEUTRAL
        except Exception as e:
            v_bb_slow_o = 0

        #Martin Pring - Trend Analisys
        try:
            df_roc1 = df.ta.roc(length=10).rolling(10).mean() * 1
            df_roc2 = df.ta.roc(length=15).rolling(10).mean() * 2
            df_roc3 = df.ta.roc(length=20).rolling(10).mean() * 3
            df_roc4 = df.ta.roc(length=30).rolling(15).mean() * 4
            df_roc5 = df.ta.roc(length=40).rolling(50).mean() * 1
            df_roc6 = df.ta.roc(length=65).rolling(65).mean() * 2
            df_roc7 = df.ta.roc(length=75).rolling(75).mean() * 3
            df_roc8 = df.ta.roc(length=100).rolling(100).mean() * 4
            df_roc9 = df.ta.roc(length=195).rolling(130).mean() * 1
            df_roc10 = df.ta.roc(length=265).rolling(130).mean() * 2
            df_roc11 = df.ta.roc(length=390).rolling(130).mean() * 3
            df_roc12 = df.ta.roc(length=530).rolling(195).mean() * 4
            v_macd_h = df.ta.macd(9, 26)['MACDh_9_26_9'][-1]
            df_mp_osc = df_roc1 + df_roc2 + df_roc3 + df_roc4 + df_roc5 + df_roc6 + df_roc7 + df_roc8 + df_roc9 + df_roc10 + df_roc11 + df_roc12
            if df_mp_osc.isnull()[-1]:
                v_mp_osc = 0
                v_mp_slope = 0
            else:    
                v_mp_osc = df_mp_osc[-1]
                if df_mp_osc.isnull()[-2]:
                    v_mp_slope = 0
                else:
                    v_mp_slope = ( df_mp_osc[-1] - df_mp_osc[-2] ) / df_mp_osc[-1]
        except Exception as e:
            v_mp_osc = 0
            v_mp_slope = 0

        # Sell
        # Condition 1: Current Price > 150 SMA and > 200 SMA
        if(currentClose > moving_average_150 > moving_average_200):
            cond_Sell_1 = True
        else:
            cond_Sell_1 = False
        # Condition 2: 150 SMA and > 200 SMA
        if(moving_average_150 > moving_average_200):
            cond_Sell_2 = True
        else:
            cond_Sell_2 = False
        # Condition 3: 200 SMA trending up for at least 1 month (ideally 4-5 months)
        if(moving_average_200 > moving_average_200_20):
            cond_Sell_3 = True
        else:
            cond_Sell_3 = False
        # Condition 4: 50 SMA > 150 SMA and 50 SMA > 200 SMA
        if(moving_average_50 > moving_average_150 > moving_average_200):
            #print("Condition 4 met")
            cond_Sell_4 = True
        else:
            #print("Condition 4 not met")
            cond_Sell_4 = False
        # Condition 5: Current Price > 50 SMA
        if(currentClose > moving_average_50):
            cond_Sell_5 = True
        else:
            cond_Sell_5 = False
        # Condition 6: Current Price is at least 30% above 52 week low (Many of the best are up 100-300% before coming out of consolidation)
        if(currentClose >= (1.3*low_of_52week)):
            cond_Sell_6 = True
        else:
            cond_Sell_6 = False
        # Condition 7: Current Price is within 25% of 52 week high
        if(currentClose >= (.75*high_of_52week)):
            cond_Sell_7 = True
        else:
            cond_Sell_7 = False

        # Buy
        # Condition 1: Current Price < 150 SMA and < 200 SMA
        if(currentClose < moving_average_150 < moving_average_200):
            cond_Buy_1 = True
        else:
            cond_Buy_1 = False
        # Condition 2: 150 SMA and < 200 SMA
        if(moving_average_150 < moving_average_200):
            cond_Buy_2 = True
        else:
            cond_Buy_2 = False
        # Condition 3: 200 SMA trending up for at least 1 month (ideally 4-5 months)
        if(moving_average_200 < moving_average_200_20):
            cond_Buy_3 = True
        else:
            cond_Buy_3 = False
        # Condition 4: 50 SMA < 150 SMA and 50 SMA < 200 SMA
        if(moving_average_50 < moving_average_150 < moving_average_200):
            #print("Condition 4 met")
            cond_Buy_4 = True
        else:
            #print("Condition 4 not met")
            cond_Buy_4 = False
        # Condition 5: Current Price < 50 SMA
        if(currentClose < moving_average_50):
            cond_Buy_5 = True
        else:
            cond_Buy_5 = False
        # Condition 6: Current Price is at least 30% above 52 week low (Many of the best are up 100-300% before coming out of consolidation)
        if(currentClose <= (1.3 * low_of_52week)):
            cond_Buy_6 = True
        else:
            cond_Buy_6 = False
        # Condition 7: Current Price is within 25% of 52 week high
        if(currentClose <= (.75 * high_of_52week)):
            cond_Buy_7 = True
        else:
            cond_Buy_7 = False

        signal = CONST_NEUTRAL
        #worksheet.set_row(i, i, format_neutral)
        if(cond_Buy_1 and cond_Buy_2 and cond_Buy_3 and cond_Buy_4 and cond_Buy_5 and cond_Buy_6 and cond_Buy_7):
            signal = CONST_BUY
        if(cond_Sell_1 and cond_Sell_2 and cond_Sell_3 and cond_Sell_4 and cond_Sell_5 and cond_Sell_6 and cond_Sell_7):
            signal = CONST_SELL

        signal_rsi = CONST_NEUTRAL
        if v_rsi >= 70:
            signal_rsi = CONST_SELL
        elif v_rsi <= 30:
            signal_rsi = CONST_BUY

        signal_stoch = CONST_NEUTRAL
        if v_stochK >= 80:
            signal_stoch = CONST_SELL
        elif v_stochK <= 20:
            signal_stoch = CONST_BUY

        v_sentiment = get_sentiment_analysis(stock)

        exportRow = pd.DataFrame({
            'Stock': stock,
            'Stock_Flag': stock_flag,
            'Open': round(currentOpen, 4),
            'Close': round(currentClose, 4),
            'Change': round(percentageChange, 4),
            '52 W. Low': round(low_of_52week, 4),
            '52 W. High': round(high_of_52week, 4),
            '50 D. MA': round(moving_average_50, 4),
            '150 D. Ma': round(moving_average_150, 4),
            '200 D. MA': round(moving_average_200, 4),
            'SMA S.': signal,
            'RSI': round(v_rsi, 4),
            'RSI S.': signal_rsi,
            'StochK': round(v_stochK, 4),
            'StochD': round(v_stochD, 4),
            'Stoch S.': signal_stoch,
            'Fast BB u2': round(v_bb_fast_u2, 4),
            'Fast BB u1': round(v_bb_fast_u1, 4),
            'Fast BB l1': round(v_bb_fast_l1, 4),
            'Fast BB l2': round(v_bb_fast_l2, 4),
            'Fast BB S.': v_bb_fast_signal,
            'Slow BB u2': round(v_bb_slow_u2, 4),
            'Slow BB u1': round(v_bb_slow_u1, 4),
            'Slow BB l1': round(v_bb_slow_l1, 4),
            'Slow BB l2': round(v_bb_slow_l2, 4),
            'Slow BB S.': v_bb_slow_signal,
            'Fast BBO': round(v_bb_fast_o, 4),
            'Fast BBO S.': v_bb_fast_o_signal,
            'Slow BBO': round(v_bb_slow_o, 4),
            'Slow BBO S.': v_bb_slow_o_signal,
            'MACD H.': v_macd_h,
            'M.Pring V.': v_mp_osc,
            'M.Pring S.': v_mp_slope,
            'Sent.': round(v_sentiment, 4)
        }, index=[0])

        exportList = pd.concat([exportList, exportRow], ignore_index=True)

    except Exception as e:
        print(repr(e))
        print("No data on " + stock)

exportList.sort_values('RSI', inplace=True)
exportList = exportList.reset_index().drop('index', axis=1)

print(exportList)

dir = os.path.abspath('.\\')

# Create a Pandas Excel writer using XlsxWriter as the engine.
newFile = dir + r'\ScreenOutput_' + end_date.strftime('%Y%m%d') + '.xlsx'
writer = pd.ExcelWriter(newFile, engine='xlsxwriter')

#df_ccl.to_excel(writer, "CCL")

# Get the xlsxwriter workbook and worksheet objects.
workbook = writer.book
worksheet = workbook.add_worksheet("Screener ajustado por CCL")

# Get the dimensions of the dataframe.
(max_row, max_col) = exportList.shape
# Formato de columnas con 'Buy', 'Neutral', 'Sell'
str_ranges = [
    'K2:K' + str(max_row + 1),
    'M2:M' + str(max_row + 1),
    'P2:P' + str(max_row + 1),
    'U2:U' + str(max_row + 1),
    'Z2:Z' + str(max_row + 1),
    'AB2:AB' + str(max_row + 1),
    'AD2:AD' + str(max_row + 1)]

header_format_default = workbook.add_format(
    {
        'border': 1,
        'align': 'left',
        'font_size': 10,
        'bg_color': '#C4BD97',
        'bold': True
    })

header_format_comprado = workbook.add_format(
    {
        'border': 1,
        'align': 'left',
        'font_size': 10,
        'bg_color': '#C4BD97',
        'font_color': '#FF0000',
        'bold': True
    })

header_format_vigilar = workbook.add_format(
    {
        'border': 1,
        'align': 'left',
        'font_size': 10,
        'bg_color': '#C4BD97',
        'font_color': '#7030A0',
        'bold': True
    })

header_format_bullish = workbook.add_format(
    {
        'border': 1,
        'font_size': 10,
        'bg_color': '#9BBB59',
        'num_format': '0.000'
    })

header_format_bearish = workbook.add_format(
    {
        'border': 1,
        'font_size': 10,
        'bg_color': '#FC6252',
        'num_format': '0.000'
    })

header_format_neutral = workbook.add_format(
    {
        'border': 1,
        'font_size': 10,
        'bg_color': '#FFEB84',
        'num_format': '0.000'
    })

border_format = workbook.add_format(
    {
        'border': 1,
        'font_size': 10,
        #'bg_color': '#EEECE1',
        'bg_color': '#C4BD97',
        'num_format': '0.000'
    })

percentage_format = workbook.add_format(
    {
        'border': 1,
        'font_size': 10,
        'bg_color': '#C4BD97',
        'num_format': '0.00%'
    })

# Write the column headers with the defined format.
worksheet.set_column(0, max_col, 10)
for col_num, value in enumerate(exportList.columns.values):
    worksheet.write(0, col_num, value, header_format_default)

# Colorear las columnas de Stock, Open y Close
for row_num, row in exportList.iterrows():
    for col_num in range(row.shape[0]):
        cell_reference = xl_rowcol_to_cell(
            row_num + 1, col_num, row_abs=True, col_abs=True)
        if col_num == 0:
            if row.iloc[1] == 'Comprado':
                worksheet.write('%s' % (cell_reference),
                                row.iloc[0], header_format_comprado)
            elif row.iloc[1] == 'Vigilar':
                worksheet.write('%s' % (cell_reference),
                                row.iloc[0], header_format_vigilar)
            else:
                worksheet.write('%s' % (cell_reference),
                                row.iloc[0], header_format_default)
        elif col_num > 1 and col_num < 4:  #Open Close
            if row.iloc[2] < row.iloc[3]:
                worksheet.write('%s' % (cell_reference),
                                row.iloc[col_num], header_format_bullish)
            elif row.iloc[2] > row.iloc[3]:
                worksheet.write('%s' % (cell_reference),
                                row.iloc[col_num], header_format_bearish)
            else:
                worksheet.write('%s' % (cell_reference),
                                row.iloc[col_num], header_format_neutral)
        elif col_num == 4:
            worksheet.write('%s' % (cell_reference),
                            row.iloc[col_num], percentage_format)
        else:
            worksheet.write('%s' % (cell_reference),
                            row.iloc[col_num], border_format)

my_cond_formats_bns = {
    '"' + CONST_SELL + '"': '#FC6252',
    '"' + CONST_BUY + '"': '#9BBB59',
    '"' + CONST_NEUTRAL + '"': '#FFEB84'
}

my_cond_formats_change = {
    'type': '3_color_scale',
    'min_type': 'num',
    'min_value': '-0.1',
    'mid_type': 'num',
    'mid_value': '0',
    'max_type': 'num',
    'max_value': '0.1',
    'min_color': '#FC6252',
    'mid_color': '#FFEB84',
    'max_color': '#9BBB59'}

my_cond_formats_rsi = {
    'type': '3_color_scale',
    'min_type': 'num',
    'min_value': '30',
    'max_type': 'num',
    'max_value': '70',
    'min_color': '#9BBB59',
    'mid_color': '#FFEB84',
    'max_color': '#FC6252'}

my_cond_formats_macd = {
    'type': '3_color_scale',
    'mid_type': 'num',
    'mid_value': '0',
    'min_color': '#9BBB59',
    'mid_color': '#FFEB84',
    'max_color': '#FC6252'}

my_cond_formats_stoch = {
    'type': '3_color_scale',
    'min_type': 'num',
    'min_value': '20',
    'max_type': 'num',
    'max_value': '80',
    'min_color': '#9BBB59',
    'mid_color': '#FFEB84',
    'max_color': '#FC6252'}

my_cond_formats_bbo = {
    'type': '3_color_scale',
    'min_type': 'num',
    'min_value': '0',
    'max_type': 'num',
    'max_value': '1',
    'min_color': '#9BBB59',
    'mid_color': '#FFEB84',
    'max_color': '#FC6252'}

my_cond_formats_sentiment = {
    'type': '3_color_scale',
    'mid_type': 'num',
    'mid_value': '2.5',
    'min_color': '#9BBB59',
    'mid_color': '#FFEB84',
    'max_color': '#FC6252'}

my_cond_formats_mp = {
    'type': '3_color_scale',
    'mid_type': 'num',
    'mid_value': '0',
    'min_color': '#FC6252',
    'mid_color': '#FFEB84',
    'max_color': '#9BBB59'}

for str_range in str_ranges:
    for val, color in my_cond_formats_bns.items():
        fmt = workbook.add_format(
            {
                'bg_color': color
            }
        )
        worksheet.conditional_format(str_range,
                                        {
                                            'type': 'cell',
                                            'criteria': '=',
                                            'value': val,
                                            'format': fmt
                                        }
                                     )

# Columnas coloreadas con rangos
worksheet.conditional_format('E2:E' + str(max_row + 1), my_cond_formats_change)
worksheet.conditional_format('L2:L' + str(max_row + 1), my_cond_formats_rsi)
worksheet.conditional_format('N2:N' + str(max_row + 1), my_cond_formats_stoch)
worksheet.conditional_format('O2:O' + str(max_row + 1), my_cond_formats_stoch)
worksheet.conditional_format('AA2:AA' + str(max_row + 1), my_cond_formats_bbo)
worksheet.conditional_format('AC2:AC' + str(max_row + 1), my_cond_formats_bbo)
worksheet.conditional_format('AE2:AE' + str(max_row + 1), my_cond_formats_macd)
worksheet.conditional_format('AF2:AF' + str(max_row + 1), my_cond_formats_mp)
worksheet.conditional_format('AG2:AG' + str(max_row + 1), my_cond_formats_mp)
worksheet.conditional_format('AH2:AH' + str(max_row + 1), my_cond_formats_sentiment)

# Columns can be hidden explicitly. This doesn't increase the file size..
worksheet.set_column('B:B', None, None, {'hidden': True})
worksheet.set_column('F:J', None, None, {'hidden': True})
worksheet.set_column('Q:T', None, None, {'hidden': True})
worksheet.set_column('V:Y', None, None, {'hidden': True})

worksheet.freeze_panes(1, 0)
worksheet.autofilter('A1:AH' + str(max_row + 1))

writer.save()

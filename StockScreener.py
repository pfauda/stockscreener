import os
import datetime as dt
import pandas as pd
import pandas_ta as ta
import numpy as np
import requests

import sqlite3
from sqlite3 import Error

import xlsxwriter
from typing import Final
from tqdm import tqdm

class const_signal():
    BUY:           Final = "Buy"
    BUY_COLOR:     Final = "#9BBB59"
    NEUTRAL:       Final = "Neutral"
    NEUTRAL_COLOR: Final = "#FFEB84"
    SELL:          Final = "Sell"
    SELL_COLOR:    Final = "#FC6252"

class const_status():
    BOUGHT: Final = "Comprado"
    WATCH:  Final = "Vigilar"

class StockScreener:

    CONST_ADJ = False

    def __init__(self, db_file, start_date):

        self.database = db_file
        self.start_date = start_date

        self.exportList = pd.DataFrame(columns=[
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
            #'Fast BB u2',
            #'Fast BB u1',
            #'Fast BB l1',
            #'Fast BB l2',
            #'Fast BB S.',
            #'Slow BB u2',
            #'Slow BB u1',
            #'Slow BB l1',
            #'Slow BB l2',
            #'Slow BB S.',
            'Fast BBO',
            'Fast BBO S.',
            'Slow BBO',
            'Slow BBO S.',
            'MACD H.',
            'M.Pring V.',
            'M.Pring S.',
        	'Sent.'])

        self.__create_connection__()
        self.__select_stocks_symbols__()
        self.__ccl__()
        self.exportList = pd.DataFrame()

    def __create_connection__(self):
        """ create a database connection to the SQLite database
            specified by db_file
        :param db_file: database file
        :return: Connection object or None
        """
        try:
            self.conn = sqlite3.connect(self.database)
        except Error as e:
            print(e)

    def __select_stocks_symbols__(self):
        sql = ''' SELECT * FROM symbols '''
        cur = self.conn.cursor()
        cur.execute(sql)
        self.stocklist = pd.DataFrame(cur.fetchall(), columns=['Name', 'Panel', const_status.BOUGHT, const_status.WATCH])

    def __get_stock_prices__(self, symbol):

        sql = ''' SELECT Date, Open, High, Low, Close, Volume FROM stocks WHERE Name = ? '''
        cur = self.conn.cursor()
        cur.execute(sql, (symbol,))
        rows = cur.fetchall()
        df = pd.DataFrame(rows)
        col_map = {df.columns[0]: 'Date', df.columns[1]: 'Open', df.columns[2]: 'High', df.columns[3]: 'Low', df.columns[4]: 'Close', df.columns[5]: 'Volume'}
        df.rename(columns=col_map, inplace=True)
        df.set_index('Date', inplace=True)
        return df

    def __calc_bb_o__(self, df: pd.DataFrame, len: int, std: float, mamode="sma", ddof=0) -> float:

            """Indicator: Bollinger Bands (BBANDS)"""
            # Validate arguments
            length = int(len) if len and len > 0 else 5
            std = std if std and std > 0 else 2
            mamode = mamode if isinstance(mamode, str) else "sma"
            ddof = int(ddof) if ddof >= 0 and ddof < length else 0

            if df is None:
                return 0

            # Calculate Result
            standard_deviation = df.ta.stdev(length=len, ddof=ddof)
            deviations = std * standard_deviation

            mid = ta.ma(mamode, df['Close'], length=len)
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
            bbandsdf_o = pd.DataFrame(data)
            bbandsdf_o.name = f"BBANDS_O_{length}_{std}"

            return bbandsdf_o[f'BBO_{length}_{std}'][-1]

    def __get_sentiment_analysis__(self, ticker):

        y_header = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json'
        }

        yf_url_base = 'https://query2.finance.yahoo.com/v10/finance/quoteSummary/{0}'.format(ticker)

        rec = 2.5
        try:
            yf_response = requests.get(yf_url_base+'?modules=recommendationTrend', headers=y_header)
            yf_json = yf_response.json()
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
        except:
            pass

        return rec

    def __ccl__(self):

        tickers = ['GGAL', 'YPF', 'PAM', 'BMA', 'GGAL.BA', 'YPFD.BA', 'PAMP.BA', 'BMA.BA']

        df_tickers = {}
        for ticker in tickers:
            df_tickers[ticker] = self.__get_stock_prices__(ticker)['Close']

        # Calcular el CCL ponderado
        # TV=BCBA:PGR/(0.2*BCBA:GGAL/NASDAQ:GGAL*10+0.2*BCBA:YPFD/NYSE:YPF+0.125*BCBA:PAMP/NYSE:PAM*25+0.125*BCBA:BMA/NYSE:BMA*10)/0.65
        self.df_ccl = (0.2   * df_tickers['GGAL.BA'] / df_tickers['GGAL'] * 10 +
                       0.2   * df_tickers['YPFD.BA'] / df_tickers['YPF'] +
                       0.125 * df_tickers['PAMP.BA'] / df_tickers['PAM'] * 25 +
                       0.125 * df_tickers['BMA.BA']  / df_tickers['BMA'] * 10) / 0.65
        
        self.df_ccl.fillna(method='ffill', axis=0, inplace=True)
        self.start_date = self.df_ccl.index[0]

    def get_data(self) -> pd.DataFrame:

        for i in (pbar:=tqdm(self.stocklist.index, "Cargando datos", colour="green")):

            stock = str(self.stocklist['Name'][i])
            stock_flag = ''
            if self.stocklist[const_status.BOUGHT][i] != '':
                stock_flag = const_status.BOUGHT

            if self.stocklist[const_status.WATCH][i] != '':
                stock_flag = const_status.WATCH

            df_sma = pd.DataFrame()
            df_bb_fast = pd.DataFrame()
            df_bb_slow = pd.DataFrame()

            #print(stock)
            pbar.set_postfix_str(stock)

            try:
                df = self.__get_stock_prices__(stock)
                df.fillna(method='ffill', axis=0, inplace=True)

                if stock[-3:] == '.BA':
                    df = df.divide(self.df_ccl, axis=0)
                
                df.dropna(inplace=True)

                smaUsed = [50, 150, 200]
                for x in smaUsed:
                    sma = x
                    try:
                        df_sma['SMA_' + str(sma)] = df.ta.sma(length=sma)
                    except Exception as e:
                        df_sma['SMA_' + str(sma)] = 0

                currentOpen = 0
                currentClose = 0
                percentageChange = 0
                moving_average_50 = 0
                moving_average_150 = 0
                moving_average_200 = 0
                low_of_52week = 0
                high_of_52week = 0
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
                    pass

                try:
                    moving_average_200_20 = df_sma['SMA_200'][-20]
                except Exception as e:
                    moving_average_200_20 = 0

                try:
                    v_rsi = df.ta.rsi()[-1]
                except Exception as e:
                    v_rsi = 0

                try:
                    v_stochK = df.ta.stoch()['STOCHk_14_3_3'][-1]
                    v_stochD = df.ta.stoch()['STOCHd_14_3_3'][-1]
                except Exception as e:
                    v_stochK = 0
                    v_stochD = 0

                # Oscilador de bndas de bolinger - DBB %B
                v_bb_fast_o_signal = const_signal.NEUTRAL
                v_bb_fast_o = 0
                try:
                    v_bb_fast_o = self.__calc_bb_o__(df, len=20, std=2)
                    if v_bb_fast_o >= 1:
                        v_bb_fast_o_signal = const_signal.SELL
                    elif v_bb_fast_o <= 0:
                        v_bb_fast_o_signal = const_signal.BUY
                except Exception as e:
                    pass

                v_bb_slow_o_signal = const_signal.NEUTRAL
                v_bb_slow_o = 0
                try:
                    v_bb_slow_o = self.__calc_bb_o__(df, len=84, std=2)
                    if v_bb_fast_o >= 1:
                        v_bb_slow_o_signal = const_signal.SELL
                    elif v_bb_slow_o <= 0:
                        v_bb_slow_o_signal = const_signal.BUY
                except Exception as e:
                    pass

                v_macd_h = df.ta.macd(9, 26)['MACDh_9_26_9'][-1]

                # Martin Pring - Trend Analisys
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

                signal = const_signal.NEUTRAL
                # worksheet.set_row(i, i, format_neutral)
                if(cond_Buy_1 and cond_Buy_2 and cond_Buy_3 and cond_Buy_4 and cond_Buy_5 and cond_Buy_6 and cond_Buy_7):
                    signal = const_signal.BUY
                if(cond_Sell_1 and cond_Sell_2 and cond_Sell_3 and cond_Sell_4 and cond_Sell_5 and cond_Sell_6 and cond_Sell_7):
                    signal = const_signal.SELL

                signal_rsi = const_signal.NEUTRAL
                if v_rsi >= 70:
                    signal_rsi = const_signal.SELL
                elif v_rsi <= 30:
                    signal_rsi = const_signal.BUY

                signal_stoch = const_signal.NEUTRAL
                if v_stochK >= 80:
                    signal_stoch = const_signal.SELL
                elif v_stochK <= 20:
                    signal_stoch = const_signal.BUY

                v_sentiment = self.__get_sentiment_analysis__(stock)

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
                        'Fast BBO': round(v_bb_fast_o, 4),
                        'Fast BBO S.': v_bb_fast_o_signal,
                        'Slow BBO': round(v_bb_slow_o, 4),
                        'Slow BBO S.': v_bb_slow_o_signal,
                        'MACD H.': v_macd_h,
                        'M.Pring V.': v_mp_osc,
                        'M.Pring S.': v_mp_slope,
                        'Sent.': round(v_sentiment, 4)
                }, index=[0])

                self.exportList = pd.concat([self.exportList, exportRow], ignore_index=True)

            except Exception as e:
                print(repr(e))
                print("No data on " + stock)

        self.exportList = self.exportList.sort_values('RSI').reset_index().drop('index', axis=1)
        return self.exportList

    def to_excel(self, file: str):

        workbook = xlsxwriter.Workbook(file)
        worksheet = workbook.add_worksheet("Screener ajustado por CCL")

        # Get the dimensions of the dataframe.
        (max_row, max_col) = self.exportList.shape

        #Formatos de la primera columna de Tickers
        dic_header_format     =  { 'border': 1,
                                   'align': 'left',
                                   'font_size': 10,
                                   'bg_color': '#C4BD97',
                                   'bold': True }

        header_format_default = workbook.add_format(dic_header_format | {'font_color': '#000000'})
        header_format_bought  = workbook.add_format(dic_header_format | {'font_color': '#FF0000'})
        header_format_watch   = workbook.add_format(dic_header_format | {'font_color': '#7030A0'})

        # Formatos de las columnas "C" a "E"
        dic_header_format     =  { 'border': 1,
                                   'font_size': 10,
                                   'num_format': '0.000' }

        header_format_bullish = workbook.add_format(dic_header_format | {'bg_color': '#9BBB59'})
        header_format_bearish = workbook.add_format(dic_header_format | {'bg_color': '#FC6252'})
        header_format_neutral = workbook.add_format(dic_header_format | {'bg_color': '#FFEB84'})
        border_format         = workbook.add_format(dic_header_format | {'bg_color': '#C4BD97'})

        percentage_format     = workbook.add_format({ 'border': 1,
                                                      'font_size': 10,
                                                      'bg_color': '#C4BD97',
                                                      'num_format': '0.00%'})

        # Write the column headers with the defined format.
        worksheet.set_column(0, max_col, 10)
        for col_num, value in enumerate(self.exportList.columns.values):
            worksheet.write(0, col_num, value, header_format_default)

        # Colorear las columnas de Stock, Open y Close
        for row_num, row in self.exportList.iterrows():
            for col_num in range(row.shape[0]):

                format = border_format
                if col_num == 0:
                    if row.iloc[1] == const_status.BOUGHT:
                        format = header_format_bought
                    elif row.iloc[1] == const_status.WATCH:
                        format = header_format_watch
                    else:
                        format = header_format_default
                elif col_num > 1 and col_num < 4:  #Open Close
                    if row.iloc[2] < row.iloc[3]:
                        format = header_format_bullish
                    elif row.iloc[2] > row.iloc[3]:
                        format = header_format_bearish
                    else:
                        format = header_format_neutral
                elif col_num == 4:
                    format = percentage_format

                try:
                    worksheet.write(int(str(row_num))+1, col_num, row.iloc[col_num], format)
                except:
                    pass

        my_cond_formats_signal = {
            '"' + const_signal.SELL + '"':    const_signal.SELL_COLOR,
            '"' + const_signal.BUY + '"':     const_signal.BUY_COLOR,
            '"' + const_signal.NEUTRAL + '"': const_signal.NEUTRAL_COLOR
        }

        for col in [10, 12, 15, 17, 19]: #K, M, P, R, T
            for val, color in my_cond_formats_signal.items():
                fmt = workbook.add_format({'bg_color': color})
                worksheet.conditional_format(1,
                                             col, 
                                             max_row,
                                             col,
                                             {
                                                'type': 'cell',
                                                'criteria': '=',
                                                'value': val,
                                                'format': fmt
                                             }
                                            )

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

        my_cond_formats_martinpring = {
            'type': '3_color_scale',
            'mid_type': 'num',
            'mid_value': '0',
            'min_color': '#FC6252',
            'mid_color': '#FFEB84',
            'max_color': '#9BBB59'}

        # Columnas coloreadas en degrade por rangos
        worksheet.conditional_format(1,  4, max_row,  4, my_cond_formats_change)      #E2:En
        worksheet.conditional_format(1, 11, max_row, 11, my_cond_formats_rsi)         #L2:Ln
        worksheet.conditional_format(1, 13, max_row, 13, my_cond_formats_stoch)       #N2:Nn
        worksheet.conditional_format(1, 14, max_row, 14, my_cond_formats_stoch)       #O2:On
        worksheet.conditional_format(1, 16, max_row, 16, my_cond_formats_bbo)         #Q2:Qn
        worksheet.conditional_format(1, 18, max_row, 18, my_cond_formats_bbo)         #S2:Sn
        worksheet.conditional_format(1, 20, max_row, 20, my_cond_formats_macd)        #U2:Un
        worksheet.conditional_format(1, 21, max_row, 21, my_cond_formats_martinpring) #V2:Vn
        worksheet.conditional_format(1, 22, max_row, 22, my_cond_formats_martinpring) #W2:Wn
        worksheet.conditional_format(1, 23, max_row, 23, my_cond_formats_sentiment)   #X2:Xn

        # Columns can be hidden explicitly. This doesn't increase the file size..
        worksheet.set_column(1, 1, None, None, {'hidden': True})   #B:B
        worksheet.set_column(5, 9, None, None, {'hidden': True})   #F:J

        worksheet.freeze_panes(1, 0)
        worksheet.autofilter(0, 0, max_row, 33)

        workbook.close()

def main():
    start_date = dt.datetime(2011, 1, 1)

    if (dt.datetime.today() - start_date).days > 3650:
        start_date = dt.datetime.today() - dt.timedelta(days=3650) #Diez años
    if dt.datetime.today().hour < 17:
        end_date = dt.datetime.today() - dt.timedelta(days=1)
    else:
        end_date = dt.datetime.today()

    ss = StockScreener("stocks.db", dt.datetime.strftime(start_date, "%Y-%m-%d"))
    exportList = ss.get_data()
    print(exportList)

    dir = os.path.abspath(".\\")

    # Exportar datos a Excel
    newFile = dir + r"\ScreenOutput_" + end_date.strftime("%Y%m%d") + ".xlsx"
    ss.to_excel(newFile)

if __name__ == "__main__":
    main()
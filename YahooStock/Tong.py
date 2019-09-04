import bs4 as bs  # bs4
import datetime as dt
import os
import pandas as pd
import numpy as np
import pickle
import requests
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from matplotlib import style
from scipy import stats
import json
from collections import defaultdict
style.use('ggplot')


def save_sp500_tickers_sector():
    resp = requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find("table", {'class': 'wikitable sortable'})
    tickers_sectors = defaultdict(list)
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text.strip()
        sector = row.findAll('td')[3].text.strip().replace(' ', '_')
        tickers.append(ticker)
        tickers_sectors[sector].append(ticker)    
    with open('tickers_sectors.json', 'w') as fw:
        fw.write(json.dumps(tickers_sectors))
    with open('tickers.txt', 'w') as fw:
        fw.write('\t'.join(tickers) + '\n')
    return tickers, tickers_sectors
# save_sp500_tickers_sector()

def load_ticker_sectors():
    with open('tickers_sectors.json') as f:
        tickers_sectors = json.load(fp = f)
    with open('tickers.txt') as f:
        tickers = f.read().strip().split('\t')
    return tickers, tickers_sectors

def get_data_from_yahoo():
    if not os.path.exists('/data/tongxueqing/zhaox/stockExisted'):
        os.makedirs('/data/tongxueqing/zhaox/stockExisted')
    start = dt.datetime(2010, 1, 1)
    end = dt.datetime(2018, 12, 31)
    tickers, _ = load_ticker_sectors()
    for ticker in tickers:
        print(ticker)
        if not os.path.exists('/data/tongxueqing/zhaox/stockExisted/{}.csv'.format(ticker)):
            try:
                df = web.DataReader(ticker, 'yahoo', start, end)
                df.to_csv('/data/tongxueqing/zhaox/stockExisted/{}.csv'.format(ticker))
            except:
                print('Error: Not Found', ticker)
                continue
        else:
            print('Already have {}'.format(ticker))
    return tickers
get_data_from_yahoo()

def compile_data_together():
    tickers, tickers_sectors = load_ticker_sectors()
    main_total = pd.DataFrame()
    for sector, tickers in tickers_sectors.items():
        sector_file = pd.DataFrame()
        for ticker in tickers:
            try:
                f = pd.read_csv('/data/tongxueqing/zhaox/stockExisted/{}.csv'.format(ticker))
            except FileNotFoundError:
                continue
            f.set_index('Date', inplace = True)
            f = f[['Adj Close']]
            f.columns = [ticker]
            sector_file = f if sector_file.empty else sector_file.join(f)
        main_total = sector_file if main_total.empty else main_total.join(sector_file)
        sector_file.to_csv('/data/tongxueqing/zhaox/stockCompiled/sp500_%s_dj_close.csv' % sector)
    main_total.to_csv('/data/tongxueqing/zhaox/stockCompiled/sp500_total_close.csv')

compile_data_together()

# correlation...


def visualize_data():
    _, tickers_sectors = load_ticker_sectors()
    for sector in tickers_sectors:
        f = pd.read_csv('csvFiles/sp500_%s_dj_close.csv' % sector)
        f_corr = f.corr()
        plt.figure(figsize = (9, 9))
        plt.xticks(np.arange(len(f.columns) - 1) + 
                   0.5, f.columns[1:])
        plt.yticks(np.arange(len(f.columns) - 1) +
                   0.5, f.columns[1:])
        plt.xticks(rotation = 90)
        plt.imshow(f_corr.values, cmap = 'RdYlGn', origin = "lower")
        plt.clim(-1, 1)
        plt.colorbar()
        plt.tight_layout()
        plt.grid(False)
        # plt.show()
        plt.savefig('jpgFiles/%s1.jpg' % sector)
    # choose the strong correlation |r|>0.7 and marked?????

# visualize_data()

    

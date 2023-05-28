import pandas as pd
import holidays
import numpy as np
import requests
from bs4 import BeautifulSoup
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm.auto import tqdm
import pickle
import argparse
from collections import defaultdict

def get_args_parser():
    parser = argparse.ArgumentParser('Image segmentation', add_help=False)
    parser.add_argument('--data_path', default="terminal_data_hackathon v4.xlsx", type=str)
    parser.add_argument('--model_path', default="catboost.pkl", type=str)
    parser.add_argument('--tid_path', default="tid_mean.pkl", type=str)
    parser.add_argument('--months', default="['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']", type=str)
    parser.add_argument('--years', default="[2022]", type=str)
    parser.add_argument('--output_path', default="res.csv", type=str)
    return parser

def proccessing(data_path='terminal_data_hackathon v4.xlsx', model_path='catboost.pkl', tid_path='tid_mean.pkl',
                  months = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
              'August', 'September', 'October', 'November', 'December'], years = [2022], output_path='res.csv'):
    
    def create_sales_lag_feats(df, gpby_cols, target_col, lags):
        gpby = df.groupby(gpby_cols)
        for i in lags:
            df['_'.join([target_col, 'lag', str(i)])] = \
                    gpby[target_col].shift(i).values + np.random.normal(scale=1, size=(len(df),)) * 0
        return df

    # Creating sales rolling mean features
    def create_sales_rmean_feats(df, gpby_cols, target_col, windows, min_periods=2, 
                                 shift=1, win_type=None):
        gpby = df.groupby(gpby_cols)
        for w in windows:
            df['_'.join([target_col, 'rmean', str(w)])] = \
                gpby[target_col].shift(shift).rolling(window=w, 
                                                      min_periods=min_periods,
                                                      win_type=win_type).mean().values +\
                np.random.normal(scale=1, size=(len(df),)) * 0
        return df

    # Creating sales rolling median features
    def create_sales_rmed_feats(df, gpby_cols, target_col, windows, min_periods=2, 
                                shift=1, win_type=None):
        gpby = df.groupby(gpby_cols)
        for w in windows:
            df['_'.join([target_col, 'rmed', str(w)])] = \
                gpby[target_col].shift(shift).rolling(window=w, 
                                                      min_periods=min_periods,
                                                      win_type=win_type).median().values +\
                np.random.normal(scale=1, size=(len(df),)) * 0
        return df

    # Creating sales exponentially weighted mean features
    def create_sales_ewm_feats(df, gpby_cols, target_col, alpha=[0.9], shift=[1]):
        gpby = df.groupby(gpby_cols)
        for a in alpha:
            for s in shift:
                df['_'.join([target_col, 'lag', str(s), 'ewm', str(a)])] = \
                    gpby[target_col].shift(s).ewm(alpha=a).mean().values
        return df
    
    def parse_table(table):
        res = {'temp': [],
               'wet': [],
               'p': [],
               'wind': []}

        tags = table.findAll('td')
        k = 0
        for tag in tags:
            if tag.find('a') is not None:
                continue

            if k == 0:
                k += 1
                res['temp'].append(float(tag.text.replace('°C', '').replace('+','').replace('−','-')))
            elif k == 1:
                k += 1
                res['wet'].append(float(tag.text.replace('%','')))
            elif k == 2:
                k += 1
                res['p'].append(int(tag.text))
            else:
                k = 0
                res['wind'].append(int(tag.text.replace(' м/с', '')))
        return res

    def parse_url(url):
        page = requests.get(url)
        soup = BeautifulSoup(page.text, "html.parser")

        tables = soup.findAll('table', class_='smart')
        for table in tables:
            if 'Среднесуточная' in str(table):
                return parse_table(table)
    
    data = pd.read_excel(data_path, 'Incomes')
    terms = pd.read_excel(data_path)
    df_unpivot = pd.melt(data, id_vars='TID', value_vars=data.columns[2:])
    data = df_unpivot.sort_values(by=['TID', 'variable'])
    data = data.rename(columns={'TID': 'tid', 'variable': 'date', 'value': 'income'})
    data['date'] = pd.to_datetime(data['date'])
    ru_holidays = holidays.RU()
    data['is_holiday'] = data['date'].apply(lambda x: x in ru_holidays)
    data['dayofmonth'] = data.date.dt.day
    data['dayofweek'] = data.date.dt.dayofweek
    data['month'] = data.date.dt.month
    data['is_month_start'] = (data.date.dt.is_month_start).astype(int)
    data['is_month_end'] = (data.date.dt.is_month_end).astype(int)
    
    data['income'] = 0
    
    data = create_sales_lag_feats(data, gpby_cols=['tid'], target_col='income', 
                               lags=[1, 7, 14, 28])

    data = create_sales_rmean_feats(data, gpby_cols=['tid'], 
                                     target_col='income', windows=[1, 3, 7, 14, 28], 
                                     min_periods=1, win_type='triang')

    data = create_sales_rmed_feats(data, gpby_cols=['tid'], 
                                     target_col='income', windows=[2, 3, 7, 14, 28], 
                                     min_periods=2, win_type=None)

    data = create_sales_ewm_feats(data, gpby_cols=['tid'], 
                                   target_col='income', 
                                   alpha=[0.9, 0.7, 0.6], 
                                   shift=[3, 7, 14, 28])
    with open(tid_path, 'rb') as f:
        tid_mean = pickle.load(f)
        data = data.merge(tid_mean, how='left')
        
    url = 'http://weatherarchive.ru/Temperature/Moscow/{month}-{year}'
    
    stats = {}
    for year in years:
        stats[year] = {}
        for month in months:
            stats[year][month] = parse_url(url.format(month=month, year=year))

    weather = []
    for i, (month, v) in enumerate(stats[2022].items()):
        i = i + 1
        for j, (temp, wet, p, wind) in enumerate(zip(v['temp'], v['wet'], v['p'], v['wind'])):
            j = j + 1
            si = '0' + str(i) if i < 10 else str(i)
            sj = '0' + str(j) if j < 10 else str(j)

            weather.append({'date': '2022-{}-{}'.format(si, sj),
                            'temp': temp,
                            'wet': wet,
                            'p': p,
                            'wind': wind})
    weather = pd.DataFrame(weather)
    weather['date'] = pd.to_datetime(weather['date'])
    data = data.merge(weather, on='date', how='left')
        
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    res = pd.read_excel(data_path, 'Incomes')
    
    preds = [None for _ in range(len(data))]
    for i in range(len(data)//1630):
        data = create_sales_lag_feats(data, gpby_cols=['tid'], target_col='income', 
                               lags=[1, 7, 14, 28])

        data = create_sales_rmean_feats(data, gpby_cols=['tid'], 
                                         target_col='income', windows=[1, 3, 7, 14, 28], 
                                         min_periods=1, win_type='triang')

        data = create_sales_rmed_feats(data, gpby_cols=['tid'], 
                                         target_col='income', windows=[2, 3, 7, 14, 28], 
                                         min_periods=2, win_type=None)

        data = create_sales_ewm_feats(data, gpby_cols=['tid'], 
                                       target_col='income', 
                                       alpha=[0.9, 0.7, 0.6], 
                                       shift=[3, 7, 14, 28])
    
        msk = model.predict(data)
        for j in range(i, len(data), len(data)//1630):
            preds[j] = msk[j]
            data['income'].iloc[j] = msk[j]

    a = defaultdict(str)
    for i in range(len(data)):
        a[f"{data['tid'].iloc[i]}-{data['date'].iloc[i]}"] = preds[i]
    
    for dt in res.columns[2:]:
        for i in range(len(res)):
            res[dt].iloc[i] = a[f"{res['TID'].iloc[i]}-{dt}"]
            
    res.to_csv(output_path, index=False)
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Image segmentation', parents=[get_args_parser()])
    args = parser.parse_args()
    proccessing(args.data_path, args.model_path, args.tid_path, eval(args.months), eval(args.years), args.output_path)

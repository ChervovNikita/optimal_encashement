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
    parser.add_argument('--model_path', default="catboost_zero.pkl", type=str)
    parser.add_argument('--months', default="['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']", type=str)
    parser.add_argument('--years', default="[2022]", type=str)
    parser.add_argument('--output_path', default="res.csv", type=str)
    parser.add_argument('--next_days', default=30, type=int)
    parser.add_argument('--agg_path', default="zero_aggregation.pkl", type=str)
    return parser

def proccessing(data_path='terminal_data_hackathon v4.xlsx', model_path='catboost_zero.pkl',
                  months = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
              'August', 'September', 'October', 'November', 'December'], years = [2022, 2023, 2024], output_path='res.csv', next_days=30, agg_path='zero_aggregation.pkl'):
    
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
    dti = [str(x) for x in pd.date_range(data.columns[-1][:10], periods=next_days, freq="D")]
    for x in dti[1:]:
        data[x] = [0 for _ in range(len(data))]
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
    data['target'] = 0
    data = create_sales_lag_feats(data, gpby_cols=['tid'], target_col='target', 
                               lags=[1, 7, 14, 28])

    data = create_sales_rmean_feats(data, gpby_cols=['tid'], 
                                     target_col='target', windows=[1, 3, 7, 14, 28], 
                                     min_periods=1, win_type='triang')

    data = create_sales_rmed_feats(data, gpby_cols=['tid'], 
                                     target_col='target', windows=[2, 3, 7, 14, 28], 
                                     min_periods=2, win_type=None)

    data = create_sales_ewm_feats(data, gpby_cols=['tid'], 
                                   target_col='target', 
                                   alpha=[0.9, 0.7, 0.6], 
                                   shift=[3, 7, 14, 28])
    
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
   
    with open(agg_path, 'rb') as f:
        nw = pickle.load(f)
        data = data.merge(nw, on='tid', how='left')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    preds = [None for _ in range(len(data))]
    for i in range(len(data)//1630):
        
        data = create_sales_lag_feats(data, gpby_cols=['tid'], target_col='target', 
                               lags=[1, 7, 14, 28])

        data = create_sales_rmean_feats(data, gpby_cols=['tid'], 
                                         target_col='target', windows=[1, 3, 7, 14, 28], 
                                         min_periods=1, win_type='triang')

        data = create_sales_rmed_feats(data, gpby_cols=['tid'], 
                                         target_col='target', windows=[2, 3, 7, 14, 28], 
                                         min_periods=2, win_type=None)

        data = create_sales_ewm_feats(data, gpby_cols=['tid'], 
                                       target_col='target', 
                                       alpha=[0.9, 0.7, 0.6], 
                                       shift=[3, 7, 14, 28])
        
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
    
        msk = model.predict_proba(data.drop(columns=['income', 'target']))[:, 1]>0.357
        for j in range(i, len(data), len(data)//1630):
            preds[j] = msk[j]
            data['target'].iloc[j] = msk[j]
            if msk[j] == 1:
                data['income'].iloc[j] = 0
            else:
                data['income'].iloc[j] = nw['tid_mean_income'].iloc[j//(len(data)//1630)]
    preds = data['income']
    
    a = defaultdict(str)
    for i in range(len(data)):
        a[f"{data['tid'].iloc[i]}-{data['date'].iloc[i]}"] = preds[i]
    res = pd.read_excel(data_path, 'Incomes')
    dti = [str(x) for x in pd.date_range(res.columns[-1][:10], periods=next_days, freq="D")]
    for x in dti[1:]:
        res[x] = [0 for _ in range(len(res))]
    for dt in res.columns[2:]:
        for i in range(len(res)):
            res[dt].iloc[i] = a[f"{res['TID'].iloc[i]}-{dt}"]
            
    res.to_csv(output_path, index=False)
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser('incomes_solver', parents=[get_args_parser()])
    args = parser.parse_args()
    proccessing(args.data_path, args.model_path, eval(args.months), eval(args.years), args.output_path, args.next_days, args.agg_path)

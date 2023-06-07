import json
import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sklearn.preprocessing import LabelEncoder as LE
import argparse
from collections import defaultdict
from main import config


# get command line arguments
def get_args_parser():
    parser = argparse.ArgumentParser('Image segmentation', add_help=False)
    parser.add_argument('--report_json', default="raw_report_final.json", type=str)
    parser.add_argument('--income_path', default="terminal_data_hackathon v4.xlsx", type=str)
    parser.add_argument('--times_path', default="times v4.csv", type=str)
    parser.add_argument('--output_path', default="res.xlsx", type=str)
    return parser


# main function
def main():
    global args
    # import log json
    with open(args.report_json) as f:
        report = json.load(f)

    # load incomes
    incomes = pd.read_excel(args.income_path, 'Incomes')
    rep_df = incomes.drop('остаток на 31.08.2022 (входящий)', axis=1)
    rep_df[rep_df.columns[1]] = incomes['остаток на 31.08.2022 (входящий)']
    # init dataset for collection
    inc_df = rep_df.copy()
    for cl in inc_df.columns[1:]:
        inc_df[cl] = 0
    # calc collection dataset and remains dataset
    for i in range(len(report['logs'])):
        q = report['logs'][i]['visited']
        for j in range(len(q)):
            if q[j]:
                inc_df[inc_df.columns[i + 1]].iloc[j] = max(100, rep_df[rep_df.columns[i + 1]].iloc[j] * 0.01 * 0.01)
                rep_df[rep_df.columns[i + 1]].iloc[j] = 0
        if i == len(report['logs']) - 1:
            continue
        rep_df[rep_df.columns[i + 2]] = incomes[incomes.columns[i + 2]] + rep_df[rep_df.columns[i + 1]]

    # init funding dataset
    fond_df = rep_df.copy() * 0.02 / 365
    cols = ['статья расходов'] + [str(x) for x in pd.date_range("2022-09-01", periods=91, freq="D")]
    agg_df = pd.DataFrame(columns=cols)
    # aggregate datasets
    num_vehicles = 4
    agg_df.loc[0] = ['фондирование'] + fond_df.drop("TID", axis=1).sum().tolist()
    agg_df.loc[1] = ['инкассация'] + inc_df.drop("TID", axis=1).sum().tolist()
    agg_df.loc[2] = ['стоимость броневиков'] + [num_vehicles * config['armored_car_day_cost'] for _ in range(91)]
    agg_df.loc[3] = ['итого'] + (fond_df.drop("TID", axis=1).sum() + inc_df.drop("TID", axis=1).sum() + num_vehicles * config['armored_car_day_cost']).tolist()
    q = pd.read_csv(args.times_path)
    # calc distance matrix and get tid decoder
    terms = pd.read_excel(args.income_path)  # 'terminal_data_hackathon v4.xlsx'
    le = LE().fit(q['Origin_tid'])
    terms['tr'] = le.transform(terms['TID'])
    enc = {x: i for i, x in enumerate(terms)}
    t = [[0 for j in range(1630)] for i in range(1630)]
    q['or_tr'] = le.transform(q['Origin_tid'])
    q['de_tr'] = le.transform(q['Destination_tid'])
    for i in tqdm(range(len(q))):
        t[q['or_tr'].iloc[i]][q['de_tr'].iloc[i]] = q['Total_Time'].iloc[i]

    # init path dataset
    path_df = pd.DataFrame(columns=['порядкой номер броневика', 'устройство', 'дата-время прибытия', 'дата-время отъезда'])
    dates = pd.date_range("2022-09-01", periods=len(report['logs']), freq="D")
    dec = le.inverse_transform([i for i in range(1630)])
    for i in tqdm(range(len(report['logs']))):
        q = report['logs'][i]['paths']
        for j in range(len(q)):
            cur_time = dates[i] + pd.Timedelta(hours=8)
            for k, w in enumerate(q[j]):
                cur_row = ['', dec[w], str(cur_time), '']
                cur_time += pd.Timedelta(seconds=600)
                if k == 0:
                    cur_row[0] = str(j + 1)
                if k != len(q[j]) - 1:
                    cur_row[-1] = str(cur_time)
                    cur_time += pd.Timedelta(minutes=round(t[q[j][k]][q[j][k + 1]]))
                path_df.loc[len(path_df)] = cur_row
        # add null row for comfort
        path_df.loc[len(path_df)] = ['', '', '', '']

    text = pd.Series([
                         'Лист "остатки на конец дня" показывает сумму остатков в устройствах в разрезе дат на конец дня. Т.е. в случае, если устройство было инкассировано, в ячейке точно должен быть 0',
                         'Лист "стоимость фондирования" показывает суммы, которые получаются начислением процента на неинкассированные остатки в устройствах каждый день. В случае, если устройство не было инкассировано, банк платит процент за использование денег. ',
                         'Лист "стоимость инкасации" показывает стоимости за процедуру изымания денег из устройства на дату. Т.е. если устройство было обслужено, то услуга была оплачена.',
                         'Лист "маршруты" содержит информарцию об объездах каждого броневика точек, которые инкассируются в разрезе всего временого периода. Фиксируется время прибытия к устройству и время уезда от него.',
                         'Лист "итог" формирует суммарные издержки банка по дням (для простоты приведены формулы расчёта некоторых ячеек)'])
    # remove needless dates
    agg_df = agg_df[agg_df.columns[:len(report['logs']) + 1]]
    fond_df = fond_df[fond_df.columns[:len(report['logs']) + 1]]
    inc_df = inc_df[inc_df.columns[:len(report['logs']) + 1]]
    rep_df = rep_df[rep_df.columns[:len(report['logs']) + 1]]

    writer = pd.ExcelWriter(args.output_path, engine='xlsxwriter')
    text.to_excel(writer, sheet_name='пояснения по отчёту')
    rep_df.to_excel(writer, sheet_name='остатки на конец дня')
    fond_df.to_excel(writer, sheet_name='стоимость фондирования')
    inc_df.to_excel(writer, sheet_name='стоимость инкассирования')
    path_df.to_excel(writer, sheet_name='маршруты')
    agg_df.to_excel(writer, sheet_name='итог')
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('incomes_report', parents=[get_args_parser()])
    args = parser.parse_args()
    main()

    # python3 src/report.py --report_json=data/processed/raw_report_final.json --income_path="data/raw/terminal_data_hackathon v4.xlsx"
    # --times_path="data/raw/times v4.csv" --output_path="data/processed/report_4.xlsx"
import predictor
import pandas as pd
import numpy as np
from sklearn import preprocessing
import plotly.express as px
from tqdm import tqdm

import optuna
from optuna.samplers import RandomSampler
from optuna.samplers import TPESampler
import os

INF = int(1e9)


def read_dist(dist_path):
    dist = pd.read_csv(dist_path)
    le = preprocessing.LabelEncoder()
    le.fit(dist['Origin_tid'])
    dist['from_int'] = le.transform(dist['Origin_tid'])
    dist['to_int'] = le.transform(dist['Destination_tid'])
    return dist


config = {'cnt_terminals': 1630,
          'persent_day_income': 0.02 / 365,
          'terminal_service_cost': 100, #{'min': 100, 'persent': 0.01},
          'max_terminal_money': 1000000,
          'max_not_service_days': 14,
          'armored_car_day_cost': 20000,
          'work_time': 10 * 60,
          'service_time': 10,
          'left_days_coef': 0,
          'encashment_coef': 0}


def optimize_configs(trial):
    config['encashment_coef'] = trial.suggest_int('encashment_coef', 0, 30, step=5, log=False)
    # config['left_days_coef'] = trial.suggest_int('left_days_coef', 0, 100, step=5, log=False)
    model = predictor.Predictor(config, dist)
    try:
        hist, opt = model.build_ans(data[data.columns[1:]].values, num_days=31, num_vehicles=25,
                                    vrp_time_limit=30, verbose=False)
    except:
        return INF

    return sum(hist['loss'])


if __name__ == '__main__':
    dirname = os.path.dirname(os.path.abspath(__file__))
    dist = read_dist(os.path.join(dirname, '../data/raw/times v4.csv'))
    data = pd.read_excel(os.path.join(dirname, '../data/raw/terminal_data_hackathon v4.xlsx'), 'Incomes')

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(sampler=TPESampler(seed=42,
                                                   multivariate=True,
                                                   warn_independent_sampling=False, ), )

    study.optimize(optimize_configs,
                   n_trials=6,
                   n_jobs=6,
                   show_progress_bar=True, )

    print(study.best_value, study.best_params)


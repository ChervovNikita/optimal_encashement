import predictor


def read_dist():
    dist = pd.read_csv('../data/raw/times v4.csv')
    le = preprocessing.LabelEncoder()
    le.fit(dist['Origin_tid'])
    dist['from_int'] = le.transform(dist['Origin_tid'])
    dist['to_int'] = le.transform(dist['Destination_tid'])
    dist.head()


config = {'cnt_terminals': dist['from_int'].max() + 1,
          'persent_day_income': 0.02 / 365,
          'terminal_service_cost': 100, #{'min': 100, 'persent': 0.01},
          'max_terminal_money': 1000000,
          'max_not_service_days': 14,
          'armored_car_day_cost': 20000,
          'work_time': 10 * 60,
          'service_time': 10,
          'left_days_coef': 0,
          'encashment_coef': 0}
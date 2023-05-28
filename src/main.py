import pandas as pd
from sklearn import preprocessing
import vrpp
import predict
import optimal_mask
import importlib
importlib.reload(vrpp)
importlib.reload(predict)
importlib.reload(optimal_mask)
import warnings
import json
import argparse
import tqdm

warnings.simplefilter('ignore')


INF = 1e9
config = {'num_terminals': 1630,
          'persent_day_income': 0.02 / 365,
          'terminal_service_cost': 100,
          'terminal_service_persent': 0.01 * 0.01,
          'max_terminal_money': 1000000,
          'max_not_service_days': 14,
          'armored_car_day_cost': 20000,
          'work_time': 10 * 60,
          'service_time': 10,
          'left_days_coef': 0,
          'encashment_coef': 0,
          'num_vehicles': 5,
          'inverse_delta_loss': 300
}

class MainPredictor:
    def __init__(self, dist_path, incomes_path, predictor_path, zero_aggregation_path):
        self.predicted_data = predict.proccessing(incomes_path, predictor_path, agg_path=zero_aggregation_path).to_numpy()[:, 1:]
        self.real_data = pd.read_excel(incomes_path, 'Incomes')
        self.real_data = self.real_data[self.real_data.columns[1:]].values.copy()
        # self.predicted_data = self.real_data.copy()  # CHAAAAAAAAAAANGE THIS
        self.dist_path = dist_path
        self.incomes_path = incomes_path
        dist = pd.read_csv(dist_path)
        le = preprocessing.LabelEncoder()
        le.fit(dist['Origin_tid'])
        dist['from_int'] = le.transform(dist['Origin_tid'])
        dist['to_int'] = le.transform(dist['Destination_tid'])
        self.raw_results = None

        self.vrp = vrpp.VRPP(dist, 10, 10 * 60, config['num_vehicles'], solution_limit=100, time_limit=100, dead_loss=False)

    def get_cost(self, days_left, delta_loss):
        if days_left == 0:
            return INF
        return 2 ** (config['max_not_service_days'] - days_left) * config['inverse_delta_loss'] + delta_loss

    def simulate(self):
        days = self.real_data.shape[1]
        num_terminals = config['num_terminals']
        num_vehicles = config['num_vehicles']

        day_losses = []
        day_paths = []
        day_visited = []

        cur_cash = self.real_data[:, 0]
        time_until_force = [config['max_not_service_days'] for i in range(num_terminals)]

        for day in range(1, days):
            print(f"DAY {day}")
            mask = []
            cost = []
            to_counter = [0 for i in range(config['max_not_service_days'] + 1)]
            for i in range(num_terminals):
                force = time_until_force[i]
                if cur_cash[i] > config['max_terminal_money']:
                    force = 0
                elif cur_cash[i] + self.predicted_data[i, day] > config['max_terminal_money']:
                    force = 1

                force_for_show = time_until_force[i]
                current_money = cur_cash[i]
                for forecast in range(min(force_for_show, days - day)):
                    if current_money > config['max_terminal_money']:
                        force_for_show = forecast
                        break
                    current_money += self.predicted_data[i, day + forecast]

                to_counter[force_for_show] += 1
                mask.append(1)

                adds = [cur_cash]
                for forecast in range(30):
                    adds.append(self.predicted_data[i, day + forecast])

                cost.append(int(self.get_cost(force, optimal_mask.find_optimal(len(adds), time_until_force[i], adds))))

            print(to_counter)
            visited, paths = self.vrp.find_vrp(cost, mask)

            day_paths.append(paths)
            day_visited.append(visited)

            cur_loss = 0
            for i in range(num_terminals):
                if cur_cash[i] > config['max_terminal_money'] or time_until_force[i] == 0:
                    if not visited[i]:
                        cur_loss += INF

                if visited[i]:
                    cur_loss += max(config['terminal_service_cost'], cur_cash[i] * config['terminal_service_persent'])
                    cur_cash[i] = 0
                    time_until_force[i] = config['max_not_service_days']
                else:
                    time_until_force[i] -= 1

                cur_loss += cur_cash[i] * config['persent_day_income']
                cur_cash[i] += self.real_data[i, day]

            day_losses.append(cur_loss + config['armored_car_day_cost'] * num_vehicles)
            print(f"LOSS {day_losses[-1]}")

        self.raw_results = (day_losses, day_visited, day_paths)
        return day_losses, day_visited, day_paths

    def build_json(self, results_path):
        data = {'num_vehicles': config['num_vehicles'], 'logs': []}
        losss, visiteds, pathss = self.raw_results
        for i, (loss, visited, paths) in enumerate(zip(losss, visiteds, pathss)):
            day_log = {'loss': loss, 'visited': visited, 'paths': []}
            for path in paths:
                day_log['paths'].append(path)
            data['logs'].append(day_log)
        # json.dump(data, open(results_path, 'w'), indent=4)
        report = data    
        incomes = pd.read_excel(self.incomes_path, 'Incomes')
        rep_df = incomes.drop('остаток на 31.08.2022 (входящий)', axis=1)
        rep_df[rep_df.columns[1]] = incomes['остаток на 31.08.2022 (входящий)']
        for i in range(len(report['logs'])):
            q = report['logs'][i]['visited']
            for j in range(len(q)):
                if q[j]:
                    rep_df[rep_df.columns[i+1]].iloc[j] = 0
            if i == len(report['logs'])-1:
                continue
            rep_df[rep_df.columns[i+2]] += rep_df[rep_df.columns[i+1]]

        fond_df = rep_df.copy()
        inc_df = rep_df.copy()
        for cl in fond_df.columns[1:]:
            for i in range(len(fond_df)):
                fond_df[cl].iloc[i] *= 0.02 / 365
                if rep_df[cl].iloc[i] == 0:
                    inc_df[cl].iloc[i] = max(100, inc_df[cl].iloc[i] * 0.0001)
                else:
                    inc_df[cl].iloc[i] = 0
        cols = ['статья расходов'] + [str(x) for x in pd.date_range("2022-09-01", periods=91, freq="D")]
        agg_df = pd.DataFrame(columns = cols)
        agg_df.loc[0] = ['фондирование'] + fond_df.drop("TID", axis=1).sum().tolist()
        agg_df.loc[1] = ['инкассация'] + inc_df.drop("TID", axis=1).sum().tolist()
        agg_df.loc[2] = ['стоимость броневиков'] + [100000 for _ in range(91)]
        agg_df.loc[3] = ['итого'] + (fond_df.drop("TID", axis=1).sum() + inc_df.drop("TID", axis=1).sum() + 100000).tolist()
        q = pd.read_csv(self.dist_path)
        terms = pd.read_excel('terminal_data_hackathon v4.xlsx')['TID'].tolist()
        enc = {x: i for i,x in enumerate(terms)}
        t = [[0 for j in range(1630)] for i in range(1630)]
        for i in tqdm(range(len(q))):
            t[enc[q['Origin_tid'].iloc[i]]][enc[q['Destination_tid'].iloc[i]]] = q['Total_Time'].iloc[i]
        path_df = pd.DataFrame(columns = ['порядкой номер броневика','устройство', 'дата-время прибытия', 'дата-время отъезда'])
        dates = pd.date_range("2022-09-01", periods=len(report['logs']), freq="D")
        sm = 0
        for i in tqdm(range(len(report['logs']))):
            q = report['logs'][0]['paths']
            for j in range(len(q)):
                cur_time = dates[i]+pd.Timedelta(hours=9)
                for k, w in enumerate(q[j]):
                    cur_row = ['', terms[w], str(cur_time), '']
                    cur_time += pd.Timedelta(seconds=600)
                    if k == 0:
                        cur_row[0] = '1'
                    if k != len(q[j])-1:
                        cur_row[-1] = str(cur_time)
                        cur_time += pd.Timedelta(minutes=round(t[ q[j][k] ][ q[j][k+1] ]))
                        sm += t[ q[j][k] ][ q[j][k+1] ]
                    path_df.loc[len(path_df)] = cur_row
            path_df.loc[len(path_df)] = ['', '', '', '']
        text = pd.Series(['Лист "остатки на конец дня" показывает сумму остатков в устройствах в разрезе дат на конец дня. Т.е. в случае, если устройство было инкассировано, в ячейке точно должен быть 0',
                         'Лист "стоимость фондирования" показывает суммы, которые получаются начислением процента на неинкассированные остатки в устройствах каждый день. В случае, если устройство не было инкассировано, банк платит процент за использование денег. ',
                         'Лист "стоимость инкасации" показывает стоимости за процедуру изымания денег из устройства на дату. Т.е. если устройство было обслужено, то услуга была оплачена.',
                         'Лист "маршруты" содержит информарцию об объездах каждого броневика точек, которые инкассируются в разрезе всего временого периода. Фиксируется время прибытия к устройству и время уезда от него.',
                         'Лист "итог" формирует суммарные издержки банка по дням (для простоты приведены формулы расчёта некоторых ячеек)'])
        writer = pd.ExcelWriter(results_path, engine = 'xlsxwriter')
        text.to_excel(writer, sheet_name = 'пояснения по отчёту')
        rep_df.to_excel(writer, sheet_name = 'остатки на конец дня')
        fond_df.to_excel(writer, sheet_name = 'стоимость фондирования')
        inc_df.to_excel(writer, sheet_name = 'стоимость инкассирования')
        path_df.to_excel(writer, sheet_name = 'маршруты')
        agg_df.to_excel(writer, sheet_name = 'итог')
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Optimal encashment', add_help=False)
    parser.add_argument('--dist_path', default="data/raw/times v4.csv", type=str)
    parser.add_argument('--incomes_path', default="data/raw/terminal_data_hackathon v4.xlsx", type=str)
    parser.add_argument('--model_path', default="models/catboost_zero.pkl", type=str)
    parser.add_argument('--zero_aggregation_path', default="models/zero_aggregation.pkl", type=str)
    parser.add_argument('--output_path', default="data/processed/raw_report.json", type=str)

    args = parser.parse_args()

    predictor = MainPredictor(args.dist_path, args.incomes_path, args.model_path, args.zero_aggregation_path)
    day_losses, day_visited, day_paths = predictor.simulate()
    # for i, (loss, visited, paths) in enumerate(zip(day_losses, day_visited, day_paths)):
    #     print("=" * 50, f"DAY {i}")
    #     print(loss)
    #     print(visited)
    #     for path in paths:
    #         print(path)
    predictor.build_json(args.output_path)

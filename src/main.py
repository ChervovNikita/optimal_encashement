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
    def __init__(self, dist_path, incomes_path, predictor_path, tid_path):
        self.predicted_data = predict.proccessing(incomes_path, predictor_path, tid_path).to_numpy()[:, 1:]
        self.real_data = pd.read_excel(incomes_path, 'Incomes')
        self.real_data = self.real_data[self.real_data.columns[1:]].values.copy()
        # self.predicted_data = self.real_data.copy()  # CHAAAAAAAAAAANGE THIS

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
        json.dump(data, open(results_path, 'w'), indent=4)


if __name__ == '__main__':
    predictor = MainPredictor('data/raw/times v4.csv',
                              'data/raw/terminal_data_hackathon v4.xlsx',
                              'models/catboost_zero.pkl',
                              'models/tid_mean.pkl')
    day_losses, day_visited, day_paths = predictor.simulate()
    # for i, (loss, visited, paths) in enumerate(zip(day_losses, day_visited, day_paths)):
    #     print("=" * 50, f"DAY {i}")
    #     print(loss)
    #     print(visited)
    #     for path in paths:
    #         print(path)
    predictor.build_json('data/processed/raw_dp_report.json')
    print(sum(day_losses))

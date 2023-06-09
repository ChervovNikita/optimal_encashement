import os

import pandas as pd
from sklearn import preprocessing

import outdated.predictor_one_day
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
import random
from tqdm import tqdm

warnings.simplefilter('ignore')


# set config
INF = 1e9
config = {
    'num_terminals': 1630,
    'persent_day_income': 0.02 / 365,
    'terminal_service_cost': 100,
    'terminal_service_persent': 0.01 * 0.01,
    'max_terminal_money': 1000000,
    'max_not_service_days': 14,
    'armored_car_day_cost': 20000,
    'work_time': 12 * 60,
    'service_time': 10,
    'left_days_coef': 0,
    'encashment_coef': 0,
    'num_vehicles': 4,
    'inverse_delta_loss': 1000,
    'vrp_time_limit': 1000,
    'vrp_solution_limit': 1000,
    'days': None,
}

# main - dp4 threshold 5 = 10907529
# Local - dp4 threshold 4 = 10913919
# Local2 - dp4 threshold 3 = 10920628
# Local3 - 3 vehicles no dp = dead
# Local - dp4 threshold 3 with one day dp
# Local2 - dp4 no dp affect = 10910242.376712326
# Local - dp4 threshold 1


class MainPredictor:
    """
        main optimizer that combines all algorithms
    """

    def __init__(self, dist_path, incomes_path, predictor_path, zero_aggregation_path):
        """
            dist_path - path to matrix with distances
            incomes_path - csv file with real incomes over day
            predictor_path - model which is used for time series predictions forcasting
            zero_aggregation_path - additional file for forecasting

            setup timeseries predictions and vehicle routing problem sovler
        """

        self.predicted_data = predict.proccessing(incomes_path, predictor_path, agg_path=zero_aggregation_path).to_numpy()[:, 1:]
        self.real_data = pd.read_excel(incomes_path, 'Incomes')
        self.real_data = self.real_data[self.real_data.columns[1:]].values.copy()
        self.dist_path = dist_path
        self.incomes_path = incomes_path
        dist = pd.read_csv(dist_path)
        le = preprocessing.LabelEncoder()
        le.fit(dist['Origin_tid'])
        dist['from_int'] = le.transform(dist['Origin_tid'])
        dist['to_int'] = le.transform(dist['Destination_tid'])
        self.raw_results = None

        self.vrp = vrpp.VRPP(dist, config['service_time'], config['work_time'], config['num_vehicles'],
                             solution_limit=config['vrp_solution_limit'],
                             time_limit=config['vrp_time_limit'], dead_loss=False)

        self.one_day_predictor = outdated.predictor_one_day.Predictor(config, dist)

    def get_cost(self, days_left, delta_loss):
        """
            days_left - number of days until 14 days deadline
            delta_loss - difference between masks where we take point on the first day and where we don't

            return cost that is used for VRP solver
        """

        if days_left == 0:
            return INF
        return 2 ** (config['max_not_service_days'] - days_left) * config['inverse_delta_loss'] + delta_loss

    def simulate_one_day(self, cur_cash, time_until_force, day, days):
        cur_cash = cur_cash.copy()
        time_until_force = time_until_force.copy()

        num_vehicles = config['num_vehicles']
        num_terminals = config['num_terminals']

        hist = {'losses': [],
                'paths': [],
                'visited': [],
                'costs': [],
                'days_until_death': [],
                'dp': []}

        print(f"DAY {day}")
        mask = []
        cost = []
        dp = []
        to_counter = [0 for i in range(config['max_not_service_days'] + 1)]
        # iterate over terminals
        time_until_cash_limit = self.get_time_until_cash_limit(cur_cash, day, days)

        # opt = pd.DataFrame([self.one_day_predictor.find_optimal_day(el) for el in self.predicted_data[:, day:days]])
        days_until_death = []
        to_counter_not_zero_cost = [0 for i in range(config['max_not_service_days'] + 1)]
        for i in range(num_terminals):
            # calculate amount of days we have
            assert time_until_force[i] >= 0
            force = time_until_force[i]
            # also if it is money limit or limit will be reached soon (uses predicted data)
            # we say that it is important to process this terminal soon
            if time_until_cash_limit[i] <= 1:
                force = min(time_until_cash_limit[i], force)

            # this logs show how many day until deadline
            nearest_force = min(time_until_force[i], time_until_cash_limit[i])
            assert nearest_force >= 0
            to_counter[nearest_force] += 1
            mask.append(1)
            days_until_death.append(nearest_force)

            adds = [cur_cash[i]]
            for forecast in range(30):
                adds.append(self.predicted_data[i, day + forecast])

            # update costs using how many days left and optimal_mask.py (dynamic programming) predictions
            dp_res = optimal_mask.find_optimal(len(adds), time_until_force[i], adds)
            # dp_res *= -1
            # dp_res = -(opt.loc[i, 'daily_losses'][1] - opt.loc[i, 'daily_losses'][0])
            dp.append(dp_res)

            if day > 14 and nearest_force >= 2 and dp_res > 0:
                cost.append(0)
                # cost.append(int(self.get_cost(force, 0)))
            else:
                to_counter_not_zero_cost[nearest_force] += 1
                cost.append(int(self.get_cost(force, 0)))

        # run vehicle routing problem solver
        print('Force counter', to_counter)
        print('Not zero counter', to_counter_not_zero_cost, sum([el > 0 for el in dp]), sum([el < 0 for el in dp]), max(dp), min(dp), sum(dp))

        visited, paths = self.vrp.find_vrp(cost, mask)

        hist['days_until_death'].append(days_until_death)
        hist['costs'].append(cost)
        hist['dp'].append(dp)
        hist['paths'].append(paths)
        hist['visited'].append(visited)

        # read real data and update loss
        visited_counter = [0 for i in range(config['max_not_service_days'] + 1)]
        cur_loss = 0
        for i in range(num_terminals):
            if cur_cash[i] > config['max_terminal_money'] or time_until_force[i] == 0:
                if not visited[i]:
                    cur_loss += INF

            if visited[i]:
                visited_counter[days_until_death[i]] += 1
                cur_loss += max(config['terminal_service_cost'], cur_cash[i] * config['terminal_service_persent'])
                cur_cash[i] = 0
                time_until_force[i] = config['max_not_service_days']
            else:
                time_until_force[i] -= 1

            cur_loss += cur_cash[i] * config['persent_day_income']
            cur_cash[i] += self.real_data[i, day]

        print('Visited counter', visited_counter)
        hist['losses'].append(cur_loss + config['armored_car_day_cost'] * num_vehicles)
        return hist, cur_cash, time_until_force

    def get_time_until_cash_limit(self, cur_cash, day, days):
        res = [INF] * config['num_terminals']
        for i in range(config['num_terminals']):
            current_money = cur_cash[i]
            for forecast in range(days - day):
                if current_money > config['max_terminal_money']:
                    res[i] = forecast
                    break
                current_money += self.predicted_data[i, day + forecast]
        return res

    def simulate(self):
        """
            return oprimal paths for every day
        """

        # innitial parameters
        if config['days'] is None:
            days = self.real_data.shape[1]
        else:
            days = config['days']

        num_terminals = config['num_terminals']
        num_vehicles = config['num_vehicles']

        hist = {'losses': [],
                'paths': [],
                'visited': [],
                'costs': [],
                'days_until_death': [],
                'dp': [],}

        cur_cash = self.real_data[:, 0]
        time_until_force = [config['max_not_service_days'] for i in range(num_terminals)]

        # iterate over days
        for day in range(1, days):
            # time_until_cash_limit = self.get_time_until_cash_limit(cur_cash, day, days)
            cur_hist, cur_cash, time_until_force = self.simulate_one_day(cur_cash, time_until_force, day, days)
            for k, v in cur_hist.items():
                hist[k] += v
            print(f"LOSS {hist['losses'][-1]}, Sum loss {sum(hist['losses'])}")

        # save logs
        self.raw_results = hist
        return hist

    def build_json(self, results_path):
        """ save logs to convinient format, so we can create report """
        data = {'num_vehicles': config['num_vehicles'],
                'logs': []}

        for i in range(len(self.raw_results['losses'])):
            day_log = {}
            for k in self.raw_results.keys():
                day_log[k] = self.raw_results[k][i]
            data['logs'].append(day_log)

        json.dump(data, open(results_path, 'w'), indent=4)


if __name__ == '__main__':
    # argument parser
    # os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    id_log = random.randint(0, 1000)
    print('Default logs file: ', "raw_report_{}.json".format(id_log))
    parser = argparse.ArgumentParser('Optimal encashment', add_help=False)
    parser.add_argument('--dist_path', default="data/raw/times v4.csv", type=str)
    parser.add_argument('--incomes_path', default="data/raw/terminal_data_hackathon v4.xlsx", type=str)
    parser.add_argument('--model_path', default="models/catboost_zero.pkl", type=str)
    parser.add_argument('--zero_aggregation_path', default="models/zero_aggregation.pkl", type=str)
    parser.add_argument('--output_path', default="data/processed/raw_report_{}.json".format(id_log), type=str)

    args = parser.parse_args()

    # run main script
    predictor = MainPredictor(args.dist_path, args.incomes_path, args.model_path, args.zero_aggregation_path)
    hist = predictor.simulate()
    predictor.build_json(args.output_path)


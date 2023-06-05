import os

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

# Local - 5veh + 50inverse = 12544977.
# Local (3) - 4veh + 1000inverse = 2010909046.0972607

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

    def get_cost(self, days_left, delta_loss):
        """
            days_left - number of days until 14 days deadline
            delta_loss - difference between masks where we take point on the first day and where we don't

            return cost that is used for VRP solver
        """

        if days_left == 0:
            return INF
        return 2 ** (config['max_not_service_days'] - days_left) * config['inverse_delta_loss'] + delta_loss

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

        day_losses = []
        day_paths = []
        day_visited = []
        hist_costs = []
        hist_days_until_death = []

        cur_cash = self.real_data[:, 0]
        time_until_force = [config['max_not_service_days'] for i in range(num_terminals)]

        # iterate over days
        for day in range(1, days):
            print(f"DAY {day}")
            mask = []
            cost = []
            to_counter = [0 for i in range(config['max_not_service_days'] + 1)]
            # iterate over terminals
            force_for_show_all = []
            for i in range(num_terminals):
                # calculate amount of days we have
                assert time_until_force[i] >= 0
                force = time_until_force[i]
                # also if it is money limit or limit will be reached soon (uses predicted data)
                # we say that it is important to process this terminal soon
                if cur_cash[i] > config['max_terminal_money']:
                    force = 0
                elif cur_cash[i] + self.predicted_data[i, day] > config['max_terminal_money']:
                    force = min(1, force)

                # this logs show how many day until deadline
                force_for_show = time_until_force[i]
                current_money = cur_cash[i]
                for forecast in range(min(force_for_show, days - day)):
                    if current_money > config['max_terminal_money']:
                        force_for_show = forecast
                        break
                    current_money += self.predicted_data[i, day + forecast]

                assert force_for_show >= 0
                to_counter[force_for_show] += 1
                mask.append(1)
                force_for_show_all.append(force_for_show)

                adds = [cur_cash]
                for forecast in range(30):
                    adds.append(self.predicted_data[i, day + forecast])

                # update costs using how many days left and optimal_mask.py (dynamic programming) predictions
                cost.append(int(self.get_cost(force, optimal_mask.find_optimal(len(adds), time_until_force[i], adds))))

            # run vehicle routing problem solver
            print(to_counter)
            visited, paths = self.vrp.find_vrp(cost, mask)

            hist_days_until_death.append(force_for_show_all)
            hist_costs.append(cost)

            day_paths.append(paths)
            day_visited.append(visited)

            # read real data and update loss
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
            print(f"LOSS {day_losses[-1]}, Sum loss {sum(day_losses)}")

        # save logs
        self.raw_results = (day_losses, day_visited, day_paths, hist_costs, hist_days_until_death)
        return day_losses, day_visited, day_paths, hist_costs, hist_days_until_death

    def build_json(self, results_path):
        """ save logs to convinient format, so we can create report """
        data = {'num_vehicles': config['num_vehicles'], 'logs': []}
        losss, visiteds, pathss, costss, hist_days_until_deaths = self.raw_results
        for i, (loss, visited, paths, costs, hist_days_until_death) in enumerate(zip(losss, visiteds, pathss, costss, hist_days_until_deaths)):
            day_log = {'loss': loss,
                       'visited': visited,
                       'paths': [],
                       'costs': costs,
                       'dayes_until_death': hist_days_until_death}
            for path in paths:
                day_log['paths'].append(path)
            data['logs'].append(day_log)
        json.dump(data, open(results_path, 'w'), indent=4)


if __name__ == '__main__':
    # argument parser
    # os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser = argparse.ArgumentParser('Optimal encashment', add_help=False)
    parser.add_argument('--dist_path', default="data/raw/times v4.csv", type=str)
    parser.add_argument('--incomes_path', default="data/raw/terminal_data_hackathon v4.xlsx", type=str)
    parser.add_argument('--model_path', default="models/catboost_zero.pkl", type=str)
    parser.add_argument('--zero_aggregation_path', default="models/zero_aggregation.pkl", type=str)
    parser.add_argument('--output_path', default="data/processed/raw_report_4.json", type=str)

    args = parser.parse_args()

    # run main script
    predictor = MainPredictor(args.dist_path, args.incomes_path, args.model_path, args.zero_aggregation_path)
    day_losses, day_visited, day_paths, _, _ = predictor.simulate()
    predictor.build_json(args.output_path)


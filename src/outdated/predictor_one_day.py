import pandas as pd
import numpy as np
from tqdm import tqdm
import vrpp


INF = int(1e9)


class Predictor:
    def __init__(self, config, dist):
        self.config = config
        self.dist = dist
        self.hist = {}

    def count_term_loss(self, cash, idx):
        loss = 0
        sum_cash = 0
        for i in range(idx):
            sum_cash += cash[i]
            loss += sum_cash * self.config['persent_day_income']

        sum_cash = 0
        for i in range(idx, len(cash)):
            sum_cash += cash[i]
            loss += sum_cash * self.config['persent_day_income']

        return loss + idx * self.config['left_days_coef'] + (len(cash) - idx) * self.config['encashment_coef']

    def find_optimal_day(self, cash):
        force_id = self.config['max_not_service_days'] - 1
        sum_cash = 0
        for i in range(len(cash)):
            sum_cash += cash[i]
            if sum_cash >= self.config['max_terminal_money']:
                force_id = min(force_id, i + 1)
                break

        best = (INF, -1)
        daily_loss = []
        for i in range(force_id + 1):
            best = min(best, (self.count_term_loss(cash[:force_id + 1], i), i))
            daily_loss.append(self.count_term_loss(cash[:force_id + 1], i))
        daily_loss.append(INF)

        return {'best_loss': best[0], 'best_id': best[1], 'force_id': force_id, 'daily_losses': daily_loss}

    def build_ans(self, terminal_income, num_days=31, num_vehicles=25, vrp_time_limit=100, verbose=True):
        prev = []
        num_terminals = len(terminal_income)
        cur_cash = np.zeros(num_terminals)
        hist = {'visited': [], 'loss': [], 'opt': [], 'prev': []}
        self.hist = hist

        opt = pd.DataFrame([self.find_optimal_day(el) for el in terminal_income])

        if verbose:
            iterator = tqdm(range(num_days))
        else:
            iterator = range(num_days)
        
        for d in iterator:
            mask = [i in prev or opt.loc[i, 'best_id'] == d for i in range(num_terminals)]
            cost = []
            for i in range(num_terminals):
                if mask[i] == 0:
                    cost.append(0)
                    continue
                delta = opt.loc[i, 'daily_losses'][d + 1] - opt.loc[i, 'daily_losses'][d]
                cost.append(int(delta))

            myvrp = vrpp.VRPP(self.dist, 10, 10 * 60, num_vehicles, solution_limit=100, time_limit=vrp_time_limit, dead_loss=False)
            visited, paths = myvrp.find_vrp(cost, mask)
            hist['visited'].append(visited)

            prev = []
            cur_loss = 0
            for i in range(num_terminals):
                if visited[i]:
                    cur_loss += max(self.config['terminal_service_cost'], cur_cash[i] * self.config['terminal_service_persent'])
                    cur_cash[i] = 0
                    now = self.find_optimal_day(terminal_income[i][d:])
                    now['best_id'] += d
                    now['force_id'] += d
                    now['daily_losses'] = opt.loc[i, 'daily_losses'][:d] + now['daily_losses']
                    opt.loc[i] = now
                elif mask[i]:
                    prev.append(i)
                    assert opt.loc[i, 'daily_losses'][d + 1] != INF, f'Got infinity loss for day {d}, terminal {i}, opt.loc[i] = {opt.loc[i]}'

                cur_loss += cur_cash[i] * self.config['persent_day_income']
                cur_cash[i] += terminal_income[i][d]

            hist['loss'].append(cur_loss)
            hist['opt'].append(opt.copy())
            hist['prev'].append(prev)

        return hist

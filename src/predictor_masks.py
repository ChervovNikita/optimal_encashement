import pandas as pd
import numpy as np
from tqdm import tqdm
import vrpp
import optimal_mask

INF = int(1e9)


class Predictor:
    def __init__(self, config, dist):
        self.config = config
        self.dist = dist
        # self.config['horizont_loss_counting'] = 14

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

    def get_nikita_result(self, day, idx, left, cash, cur_cash):
        if day == 0:
            cp = np.concatenate([[0], cash[idx].copy()])
        else:
            cp = cash[idx, day - 1:].copy()
            cp[0] = cur_cash[idx]
        res = optimal_mask.find_optimal(14, left, cp)
        return res

    def build_ans(self, terminal_income, num_days=31, num_vehicles=25, vrp_time_limit=100, verbose=True):
        num_terminals = self.config['num_terminals']
        cur_cash = np.zeros(num_terminals)
        hist = {'visited': [], 'loss': [], 'costs': [], 'masks': []}
        time_until_force = [self.config['max_not_service_days'] - 1 for i in range(num_terminals)]

        if verbose:
            iterator = tqdm(range(num_days))
        else:
            iterator = range(num_days)

        for d in iterator:
            mask = []
            cost = []
            for i in range(num_terminals):
                (need, cur_cost) = self.get_nikita_result(d, i, time_until_force[i], terminal_income, cur_cash)
                mask.append(int(need))
                cost.append(int(cur_cost))

            myvrp = vrpp.VRPP(self.dist, 10, 10 * 60, num_vehicles, solution_limit=100, time_limit=vrp_time_limit, dead_loss=False)
            visited, paths = myvrp.find_vrp(cost, mask)

            cur_loss = 0
            for i in range(num_terminals):
                if cur_cash[i] > self.config['max_terminal_money'] or time_until_force[i] == 0:
                    if not visited[i]:
                        print('We dead')
                        assert False, f'Got infinity loss for day {d}, terminal {i}'

                if visited[i]:
                    cur_loss += max(self.config['terminal_service_cost'], cur_cash[i] * self.config['terminal_service_persent'])
                    cur_cash[i] = 0
                    time_until_force[i] = self.config['max_not_service_days']

                time_until_force[i] -= 1
                cur_loss += cur_cash[i] * self.config['persent_day_income']
                cur_cash[i] += terminal_income[i][d]

            hist['loss'].append(cur_loss)
            hist['visited'].append(visited)
            hist['costs'].append(cost)
            hist['masks'].append(mask)

        return hist

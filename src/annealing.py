from vrp import VehicleRoutingProblem
import pandas as pd
import random
from copy import deepcopy
from math import ceil
from sklearn import preprocessing
from first_day import current_tmp

class SimulateAnnealing:
    def __init__(self, dist, current, wait):
        self.current = current
        self.dist = dist
        self.wait = wait

        self.vehicles = 43
        self.full = 1000000
        self.interest = 2. / 100 / 365
        self.max_wait = 14
        # self.loss_func = lambda x: max(100, x * 0.01) - x * self.interest
        self.loss_func = lambda x: -x * self.interest
        self.total_points = 1630

        self.t = 90
        self.alpha = 0.99
        self.adj_count = 100
        self.steps = 1000

        self.free = {i for i in range(self.total_points) if (current[i] < self.full and wait[i] < self.max_wait)}
        self.mask = [int(i not in self.free) for i in range(self.total_points)]

    def loss(self, mask):
        vrp = VehicleRoutingProblem(self.dist, 10, 10 * 60, mask)
        if not vrp.find_vrp(43, False):
            return 1e9
        current_loss = 0
        for i in range(self.total_points):
            if mask[i]:
                current_loss += self.loss_func(self.current[i])
        return current_loss

    def adjust(self, current_mask):
        for _ in range(self.adj_count):
            mask = deepcopy(current_mask)
            points_number = 100000
            while points_number > len(self.free):
                points_number = ceil(1 / random.random())
            points = random.sample(self.free, points_number)
            for point in points:
                mask[point] = 1 - mask[point]
            yield mask

    def step(self, mask):
        # print(sum(mask))
        current_loss = self.loss(mask)
        # print(f'CURRENT {current_loss}')
        for adj in self.adjust(mask):
            adj_loss = self.loss(adj)
            # print(f'ADJ {adj_loss}')
            p = (current_loss - adj_loss) / self.t
            if random.random() <= p:
                return adj
        return mask

    def process(self):
        for i in range(self.steps):
            self.mask = self.step(self.mask)
            self.t *= self.alpha
            print(f'STEP {i}, LOSS {self.loss(self.mask)}, POINTS {sum(self.mask)}')
        return self.mask


if __name__ == '__main__':
    dist = pd.read_csv('../data/times v4.csv')
    le = preprocessing.LabelEncoder()
    le.fit(dist['Origin_tid'])
    dist['from_int'] = le.transform(dist['Origin_tid'])
    dist['to_int'] = le.transform(dist['Destination_tid'])

    current = current_tmp
    wait = [0 for i in range(1630)]

    annealing = SimulateAnnealing(dist, current, wait)
    mask = annealing.process()

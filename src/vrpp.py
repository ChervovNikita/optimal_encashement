import pandas as pd
import numpy as np
from sklearn import preprocessing
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from copy import deepcopy

INF = int(1e4)


class VRPP:
    def __init__(self, dist, service_time, work_time, num_vehicles):
        self.mask = None
        self.toid = None
        self.real_cnt = None
        self.dist = dist
        self.service_time = service_time
        self.work_time = 100 * work_time
        self.cnt_terminals = dist['from_int'].max() + 1
        self.num_vehicles = num_vehicles

    def print_solution(self, data, manager, routing, solution):
        """Prints solution on console."""
        print(f'Objective: {solution.ObjectiveValue()}')
        max_route_distance = 0
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
            route_distance = 0
            while not routing.IsEnd(index):
                plan_output += ' {} -> '.format([manager.IndexToNode(index)])
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id)
            plan_output += '{}\n'.format([manager.IndexToNode(index)])
            plan_output += 'Distance of the route: {}m\n'.format(route_distance)
            print(plan_output)
            max_route_distance = max(route_distance, max_route_distance)
        print('Maximum of the route distances: {}m'.format(max_route_distance))

    def solution(self, data, manager, routing, solution):
        visited = [0 for i in range(self.cnt_terminals)]
        paths = []
        for vehicle in range(self.num_vehicles):
            path = []
            i = routing.Start(vehicle)
            while not routing.IsEnd(i):
                i = solution.Value(routing.NextVar(i))
                if i > 0 and i <= self.real_cnt:
                    visited[self.toid[i]] = 1
                    path.append(self.toid[i])
            paths.append(path)
        return visited, paths

    def get_distance_matrix(self):
        cnt_terminals = self.cnt_terminals
        distance_matrix = np.ones((cnt_terminals + 2, cnt_terminals + 2)) * INF
        for i, j, w in zip(self.dist['from_int'], self.dist['to_int'], self.dist['Total_Time']):
            # if 0 <= i < 10:
            #     i += 10
            # if 10 <= i < 20:
            #     i -= 10
            # if 0 <= j < 10:
            #     j += 10
            # if 10 <= j < 20:
            #     j -= 10
            # if i < 20 and j < 20 and str(w) == '47.49':
            #     print(i, j)
            distance_matrix[i + 1, j + 1] = w + self.service_time

        for i in range(1, cnt_terminals + 1):
            distance_matrix[i, 0] = INF
            distance_matrix[0, i] = self.service_time
            distance_matrix[i, i] = 0
            distance_matrix[i, cnt_terminals + 1] = 0
            distance_matrix[cnt_terminals + 1, i] = INF

        distance_matrix[0, cnt_terminals + 1] = 0
        distance_matrix[cnt_terminals + 1, 0] = INF
        distance_matrix = (100 * distance_matrix).astype(int)

        # take = [0] + [i + 1 for i in self.mask] + [cnt_terminals + 1]
        take = [False for i in range(cnt_terminals + 2)]
        take[0] = True
        take[cnt_terminals + 1] = True
        for i in range(len(self.mask)):
            if self.mask[i]:
                take[i + 1] = True
        distance_matrix = distance_matrix[take, :][:, take]

        # distance_matrix[1:11, :], distance_matrix[11:21, :] = distance_matrix[11:21, :], distance_matrix[1:11, :]
        # distance_matrix[:, 1:11], distance_matrix[:, 11:21] = distance_matrix[:, 11:21], distance_matrix[:, 1:11]

        # with open('test2.txt', 'w') as f:
        #     for i in range(0, 22, 1):
        #         for j in range(0, 22, 1):
        #             i = min(i, len(distance_matrix) - 1)
        #             j = min(j, len(distance_matrix) - 1)
        #             f.write(" " * (10 - len(str(distance_matrix[i, j]))) + str(distance_matrix[i, j]) + ' ')
        #         f.write('\n')

        return distance_matrix

    def get_routing(self, vrp_data, cost):
        manager = pywrapcp.RoutingIndexManager(len(vrp_data['distance_matrix']),
                                               vrp_data['num_vehicles'],
                                               vrp_data['starts'],
                                               vrp_data['ends'])

        routing = pywrapcp.RoutingModel(manager)
        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return vrp_data['distance_matrix'][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        dimension_name = 'Distance'
        routing.AddDimension(
            transit_callback_index,
            0,
            self.work_time,
            True,
            dimension_name)

        for i in range(len(cost)):
            routing.AddDisjunction([manager.NodeToIndex(i + 1)], cost[i] * INF * 10)

        # for node in range(1, len(vrp_data['distance_matrix']) - 1):
        #     routing.AddDisjunction([manager.NodeToIndex(node)], cost[node] * INF * 10)

        distance_dimension = routing.GetDimensionOrDie(dimension_name)
        distance_dimension.SetGlobalSpanCostCoefficient(100)
        return routing, manager

    def get_search_parameters(self, solution_limit=100, time_limit=5):
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC)
        search_parameters.solution_limit = solution_limit
        search_parameters.time_limit.seconds = time_limit
        return search_parameters

    def find_vrp(self, cost, mask=None):
        if mask is None:
            mask = [1 for _ in range(self.cnt_terminals)]

        self.real_cnt = sum(mask)

        self.mask = mask
        tmp = [i for i in range(len(self.mask)) if self.mask[i]]
        self.toid = [*[-1], *[val for val in tmp], *[-2]]

        if len(cost) != sum(mask):
            r_cost = []
            for i in range(len(mask)):
                if mask[i]:
                    r_cost.append(cost[i])
            cost = deepcopy(r_cost)
        cost = [max(0, c) for c in cost]
        distance_matrix = self.get_distance_matrix()

        # print(distance_matrix[:10, :10])

        vrp_data = {'distance_matrix': distance_matrix,
                    'num_vehicles': self.num_vehicles,
                    'num_locations': self.real_cnt + 2}

        vrp_data['starts'] = [0] * vrp_data['num_vehicles']
        vrp_data['ends'] = [int(self.real_cnt + 1)] * vrp_data['num_vehicles']

        routing, manager = self.get_routing(vrp_data, cost)
        search_parameters = self.get_search_parameters()

        solution = routing.SolveWithParameters(search_parameters)
        # if solution:
        #     self.print_solution(vrp_data, manager, routing, solution)
        # else:
        #     print('what the hell')
        return self.solution(vrp_data, manager, routing, solution)


class GetLoss:
    def __init__(self, a, b, c, t):
        self.a = a
        self.b = b
        self.c = c
        self.t = t

    def __call__(self, money, max_money, day, max_day):
        if money >= max_money or day >= max_day:
            return INF
        return self.a * (money * 2 / 100 / 365) + self.b * (max(100, money / 1e4)) + self.c + self.t * (max_day - day)


if __name__ == '__main__':
    dist = pd.read_csv('../data/times v4.csv')
    le = preprocessing.LabelEncoder()
    le.fit(dist['Origin_tid'])
    dist['from_int'] = le.transform(dist['Origin_tid'])
    dist['to_int'] = le.transform(dist['Destination_tid'])

    # cost = [(1630 - i) // 100 for i in range(1630)]
    cost = [0 for i in range(1630)]
    for i in range(10):
        cost[i] = 1000
    mask = [int(i % 10 < 5) for i in range(1630)]
    myvrp = VRPP(dist, 10, 10 * 60, 20)
    visited, paths = myvrp.find_vrp(cost, mask)
    print(sum(visited))
    print(visited)
    for path in paths:
        print(path)
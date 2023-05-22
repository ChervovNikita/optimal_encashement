import pandas as pd
import numpy as np
from sklearn import preprocessing
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

INF = int(1e4)
class VehicleRoutingProblem:
    def __init__(self, dist, service_time, work_time):
        self.dist = dist
        self.service_time = service_time
        self.work_time = 100 * work_time
        self.cnt_terminals = dist['from_int'].max() + 1

    def print_solution(self, data, manager, routing, solution):
        """Prints solution on console."""
        print(f'Objective: {solution.ObjectiveValue()}')
        max_route_distance = 0
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
            route_distance = 0
            while not routing.IsEnd(index):
                plan_output += ' {} -> '.format(manager.IndexToNode(index))
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id)
            plan_output += '{}\n'.format(manager.IndexToNode(index))
            plan_output += 'Distance of the route: {}m\n'.format(route_distance)
            print(plan_output)
            max_route_distance = max(route_distance, max_route_distance)
        print('Maximum of the route distances: {}m'.format(max_route_distance))

    def get_distance_matrix(self):
        cnt_terminals = self.cnt_terminals
        distance_matrix = np.ones((cnt_terminals + 2, cnt_terminals + 2)) * INF
        for i, j, w in zip(self.dist['from_int'], self.dist['to_int'], self.dist['Total_Time']):
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
        return distance_matrix

    def get_routing(self, vrp_data):
        manager = pywrapcp.RoutingIndexManager(len(vrp_data['distance_matrix']),
                                               vrp_data['num_vehicles'],
                                               # vrp_data['depot']
                                               vrp_data['starts'],
                                               vrp_data['ends'])

        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)

        # Create and register a transit callback.
        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return vrp_data['distance_matrix'][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)

        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add Distance constraint.
        dimension_name = 'Distance'
        routing.AddDimension(
            transit_callback_index,
            0,  # no slack
            self.work_time,  # vehicle maximum travel distance
            True,  # start cumul to zero
            dimension_name)
        distance_dimension = routing.GetDimensionOrDie(dimension_name)
        distance_dimension.SetGlobalSpanCostCoefficient(100)
        return routing, manager

    def get_search_parameters(self, solution_limit=100, time_limit=30):
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC)
        search_parameters.solution_limit = solution_limit
        search_parameters.time_limit.seconds = time_limit
        return search_parameters

    def find_vrp(self, num_vehicles, verbose=False):
        distance_matrix = self.get_distance_matrix()
        vrp_data = {'distance_matrix': distance_matrix,
                    'num_vehicles': num_vehicles,
                    # 'depot': 0,
                    'num_locations': self.cnt_terminals + 2}

        vrp_data['starts'] = [0] * vrp_data['num_vehicles']
        vrp_data['ends'] = [int(self.cnt_terminals + 1)] * vrp_data['num_vehicles']

        routing, manager = self.get_routing(vrp_data)
        search_parameters = self.get_search_parameters()

        solution = routing.SolveWithParameters(search_parameters)

        if solution:
            if verbose:
                self.print_solution(vrp_data, manager, routing, solution)
            return True
        else:
            return False


if __name__ == '__main__':
    dist = pd.read_csv('data/times v4.csv')
    le = preprocessing.LabelEncoder()
    le.fit(dist['Origin_tid'])
    dist['from_int'] = le.transform(dist['Origin_tid'])
    dist['to_int'] = le.transform(dist['Destination_tid'])
    myvrp = VehicleRoutingProblem(dist, 10, 10 * 60)
    result = myvrp.find_vrp(43, True)
    if not result:
        print("Didn't find")


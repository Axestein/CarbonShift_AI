import numpy as np
import pandas as pd
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

class GreenRouteOptimizer:
    def __init__(self):
        self.locations = {}
        self.distance_matrix = []
        self.emission_matrix = []
    
    def create_data_model(self, locations, distances, emissions=None):
        """Create the data for the problem"""
        if emissions is None:
            # Default emissions based on distance
            emissions = distances * 0.21  # kg CO2 per km
        
        data = {}
        data['distance_matrix'] = distances
        data['emission_matrix'] = emissions
        data['num_vehicles'] = 1
        data['depot'] = 0  # Starting point
        return data
    
    def optimize_route(self, locations, distance_matrix, emission_matrix=None):
        """Solve the routing problem with emissions consideration"""
        data = self.create_data_model(locations, distance_matrix, emission_matrix)
        
        # Create routing index manager
        manager = pywrapcp.RoutingIndexManager(
            len(data['distance_matrix']), data['num_vehicles'], data['depot']
        )
        
        # Create routing model
        routing = pywrapcp.RoutingModel(manager)
        
        # Define cost functions for both distance and emissions
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['distance_matrix'][from_node][to_node]
        
        def emission_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(data['emission_matrix'][from_node][to_node] * 100)  # Scale for integer
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        emission_callback_index = routing.RegisterTransitCallback(emission_callback)
        
        # Set arc cost evaluators
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Add emission dimension
        routing.AddDimension(
            emission_callback_index,
            0,  # no slack
            300000,  # maximum emissions (scaled)
            True,  # start cumul to zero
            'Emission'
        )
        
        emission_dimension = routing.GetDimensionOrDie('Emission')
        emission_dimension.SetGlobalSpanCostCoefficient(100)
        
        # Set search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        
        # Solve the problem
        solution = routing.SolveWithParameters(search_parameters)
        
        if solution:
            return self.get_solution(manager, routing, solution, data)
        return None
    
    def get_solution(self, manager, routing, solution, data):
        """Extract solution from routing model"""
        index = routing.Start(0)
        route_distance = 0
        route_emissions = 0
        route_nodes = []
        
        while not routing.IsEnd(index):
            route_nodes.append(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
            route_emissions += data['emission_matrix'][manager.IndexToNode(previous_index)][manager.IndexToNode(index)]
        
        route_nodes.append(manager.IndexToNode(index))
        
        return {
            'route': route_nodes,
            'total_distance': route_distance,
            'total_emissions': route_emissions
        }
    
    def calculate_route_metrics(self, optimized_route, baseline_route):
        """Calculate savings from optimized route"""
        distance_savings = baseline_route['total_distance'] - optimized_route['total_distance']
        emission_savings = baseline_route['total_emissions'] - optimized_route['total_emissions']
        savings_percentage = (distance_savings / baseline_route['total_distance']) * 100
        
        return {
            'distance_savings_km': distance_savings,
            'emission_savings_kg': emission_savings,
            'savings_percentage': savings_percentage,
            'optimized_distance': optimized_route['total_distance'],
            'baseline_distance': baseline_route['total_distance']
        }
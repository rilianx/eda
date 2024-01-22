import numpy as np
from .TSP_state import TSP_State


from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

def create_data_model(distance_matrix, scale_factor=10000, start_node=None, end_node=None):
    """Almacena los datos del problema."""
    data = {}
    # Matriz de distancias
    data['distance_matrix'] = distance_matrix
    data['scaled_distance_matrix'] = [[int(dist * scale_factor) for dist in row] for row in data['distance_matrix']]
    data['num_vehicles'] = 1
    data['starts'] =  data['ends'] = [0]  # Punto de inicio
    if start_node is not None:
      data['starts'] = [start_node]      # Punto de inicio
    if end_node is not None:
      data['ends'] = [end_node]      # Punto de fin


    return data

"""Soluciona el TSP."""
def solve(city_points, start_node=None, end_node=None, time_limit=0):
  distance_matrix = np.sqrt(np.sum((city_points[:, np.newaxis, :] - city_points[np.newaxis, :, :]) ** 2, axis=-1))

  data = create_data_model(distance_matrix, start_node=start_node, end_node=end_node)

  # Crea el modelo de enrutamiento
  manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['starts'], data['ends'])
  routing = pywrapcp.RoutingModel(manager)

  def distance_callback(from_index, to_index):
      """Devuelve la distancia entre los dos nodos."""
      from_node = manager.IndexToNode(from_index)
      to_node = manager.IndexToNode(to_index)
      return data['scaled_distance_matrix'][from_node][to_node]

  transit_callback_index = routing.RegisterTransitCallback(distance_callback)
  routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

  # Configura parámetros de búsqueda
  search_parameters = pywrapcp.DefaultRoutingSearchParameters()
  search_parameters.first_solution_strategy = (
      routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

  if time_limit !=0:
    search_parameters.time_limit.seconds = time_limit  # 300 segundos = 5 minutos


  # Resuelve el problema
  solution = routing.SolveWithParameters(search_parameters)

  visited = []
  if solution:
      index = routing.Start(0)
      visited.append(manager.IndexToNode(index))
      while not routing.IsEnd(index):
          index = solution.Value(routing.NextVar(index))
          visited.append(manager.IndexToNode(index))
      #visited.append(manager.IndexToNode(index))
  else:
      print('No se encontró una solución.')

  TSP_State.initClass(city_points)

  sol_state = TSP_State(visited=visited)
  return sol_state


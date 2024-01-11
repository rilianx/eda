import numpy as np
from .TSP_state import TSP_State

def generate_random_points_and_distance_matrix(num_points, dim=2):
    """
    Genera puntos aleatorios 3D en el rango [0, 1] y calcula la matriz de distancias.

    :param num_points: Número de puntos a generar.
    :return: Tuple de (puntos, matriz de distancias)
    """
    # Generar puntos aleatorios
    points = np.random.rand(num_points, dim)

    # Calcular la matriz de distancias
    distance_matrix = np.sqrt(np.sum((points[:, np.newaxis, :] - points[np.newaxis, :, :]) ** 2, axis=-1))

    return points, distance_matrix

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
def solve(city_points, start_node=None, end_node=None):
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

import matplotlib.pyplot as plt

def plot_tour(points, visited, start_node=False):
    # Asegurarse de que 'visited' contiene índices válidos para 'points'
    if not all(0 <= i < len(points) for i in visited):
        raise ValueError("Los índices en 'visited' deben ser válidos para 'points'")

    # Separar las coordenadas x e y de los puntos
    x = [points[i][0] for i in visited]
    y = [points[i][1] for i in visited]

    # Agregar el primer punto al final para cerrar el tour
    if start_node==False:
      x.append(x[0])
      y.append(y[0])

    # Graficar los puntos
    plt.scatter(x, y)

    # Graficar las líneas del tour
    plt.plot(x, y)

    # Agregar títulos y etiquetas si es necesario
    plt.title("Tour de puntos 2D")
    plt.xlabel("Coordenada X")
    plt.ylabel("Coordenada Y")

    # Mostrar el gráfico
    plt.show()

def calculate_tour_cost(visited, distance_matrix):
    if not visited or len(distance_matrix) == 0:
        return 0

    distance = 0
    n_old = visited[0]

    for n in visited[1:]:
        distance += distance_matrix[n_old, n] * 10000
        n_old = n

    # Si quieres que el tour regrese al punto de inicio
    distance += distance_matrix[n_old, visited[0]] * 10000

    return distance

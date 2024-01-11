import random
import numpy as np
from copy import deepcopy

class TSP_State():
    distance_matrix = None  # Variable de clase común a todas las instancias
    city_points = None  # Variable de clase común a todas las instancias

    # static class initializer
    @classmethod
    def initClass(cls, city_points):
        cls.city_points = city_points
        cls.distance_matrix = np.sqrt(np.sum((city_points[:, np.newaxis, :] - city_points[np.newaxis, :, :]) ** 2, axis=-1))

    def __init__(self,visited):
        self.visited = [0]
        self.moves = []
        self.cost = 0
        self.not_visited = set()

        for visit in visited[1:]:
            self.transition(("constructive-move", visit), evaluate=False)
        
        self.cost = self.get_cost()

    def get_cost(self):
        """
        Calcula el coste total del tour actual basado en la secuencia de ciudades y la matriz de distancias.
        """
        cost = 0
        if self.visited:
            for i in range(len(self.visited) - 1):
              cost += TSP_State.distance_matrix[self.visited[i]][self.visited[i + 1]]

            if self.isCompleteSolution():
              # Agregar el coste de volver al nodo inicial para completar el circuito
              cost += TSP_State.distance_matrix[self.visited[-1]][self.visited[0]]
            self.cost = cost
        return cost

    def transition(self, move, evaluate=True):
        if move[0]=="constructive-move":
          self.visited.append(move[1])
          self.not_visited.remove(move[1])
          self.moves.append(move)
        elif move[0]=="2-change":
          if evaluate:
            #evaluación incremental O(1)
            self.cost = self.get_cost_after_move(move)

          i, j = move[1:]
          self.visited[i+1:j+1] = reversed(self.visited[i+1:j+1])
        else:
          raise NotImplementedError(f"Tipo de movimiento '{move[0]}' no está implementado")


    def get_cost_after_move(self, move):
        if move[0] == "2-change":
            n = len(self.visited)
            # evaluación incremental O(1)
            i, j = move[1:]

            distancia_actual_i = TSP_State.distance_matrix[self.visited[i]][self.visited[(i+1)%n]]
            distancia_actual_j = TSP_State.distance_matrix[self.visited[j]][self.visited[(j+1)%n]]
            nueva_distancia_i = TSP_State.distance_matrix[self.visited[i]][self.visited[j]]
            nueva_distancia_j = TSP_State.distance_matrix[self.visited[(i+1)%n]][self.visited[(j+1)%n]]

            # Calcular el cambio en el costo
            cambio_costo = (nueva_distancia_i + nueva_distancia_j) - (distancia_actual_i + distancia_actual_j)
            new_cost = self.cost + cambio_costo
            return new_cost
        else:
            raise NotImplementedError(f"eval_solution_after_move para movimiento '{move[0]}' no está implementado")

    def isCompleteSolution(self):
        """
        Determina si el estado actual representa una solución completa, es decir,
        un tour completo que visita todas las ciudades una vez.
        """
        return len(self.visited) == len(TSP_State.distance_matrix)

    #se necesita implementar para poder hacer un heap de estados
    def __lt__(self, other):
        return True


    def __str__(self):
        """
        Representa el estado actual del TSP como una cadena de texto.
        """
        return f"Tour actual: {self.visited}, Coste total: {self.get_cost()}"

#Greedy methods
def getConstructiveMoves(tsp_state):
    """
    Genera una lista de acciones posibles.
    """
    moves = [("constructive-move", ciudad) for ciudad in tsp_state.not_visited]
    return moves

def evalConstructiveMoves(tsp_state):
    """
    Evalúa los movimientos constructivos válidos.
    :return: -Costos adicionales de agregar cada ciudad.
    """
    moves = getConstructiveMoves(tsp_state)
    evals = []
    for move in moves:
        ultima_ciudad = tsp_state.visited[-1] if tsp_state.visited else 0
        eval = -TSP_State.distance_matrix[ultima_ciudad][move[1:]]
        evals.append((move,eval))
    return evals


# Local Search methods
def get_2change_moves(tsp_state, k=None):
    n = len(tsp_state.visited)
    moves = [("2-change", i, j) for i in range(n - 1) for j in range(i + 2, n)]

    # Si k no es None y es menor que la longitud de moves, selecciona k movimientos
    if k is not None and k < len(moves):
        return moves[:k]
    else:
        return moves
    

# A* admissible heuristic
def AdmissibleHeuristic(tsp_state):
    # Heurística: Distancia estimada para completar el tour.
    # Por simplicidad, podrías usar la distancia más corta desde la última ciudad hasta el punto de inicio,
    # aunque hay mejores heurísticas para TSP.
    if tsp_state.isCompleteSolution(): return 0

    ultima_ciudad = tsp_state.visited[-1]
    inicio = tsp_state.visited[0]
    return TSP_State.distance_matrix[ultima_ciudad][inicio]

# Una heurística admisible válida en el problema es retornar 0, ya que si las
# distancias entre ciudades son positivas, el costo para llegar desde el nodo actual
# al nodo destino no puede ser inferior a 0.
def naiveAdmissibleHeuristic(tsp_state):
    return 0


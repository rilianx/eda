import random
from copy import deepcopy
import numpy as np

class TSP_Instance:
    def __init__(self, city_locations):
        self.city_locations = city_locations
        self.num_cities = len(city_locations)
        self.distance_matrix = np.sqrt(np.sum((city_locations[:, np.newaxis, :] -  city_locations[np.newaxis, :, :]) ** 2, axis=-1))


class TSP_State:
    def __init__(self, inst_info, visited=None):
        self.visited = visited if visited is not None else []
        self.not_visited = set(range(len(inst_info.distance_matrix))) - set(self.visited)
        self.is_complete = len(self.not_visited) == 0
        self.inst_info = inst_info
        self.cost = self.update_cost()  # Aquí se realiza el cálculo inicial del coste.

    def update_cost(self):
        cost = 0
        if len(self.visited) > 1:
          for i in range(len(self.visited) - 1):
              cost += self.inst_info.distance_matrix[self.visited[i]][self.visited[i + 1]]
        self.cost = cost
        return cost

    def __deepcopy__(self, memo):
        # Crear una nueva instancia de la clase
        new_instance = type(self)(
            visited=deepcopy(self.visited),
            inst_info=self.inst_info # referencia
        )
        return new_instance

    def __str__(self):
        return f"Tour actual: {self.visited}, \nCoste total: {self.cost}"


import numpy as np

class TSP_Environment():
    @staticmethod
    def gen_constructive_actions(state):
        actions = [("constructive", city) for city in state.not_visited]
        for action in actions:
	        yield action

    @staticmethod
    def gen_2opt_random_actions(state):
      n = len(state.visited)
      actions = [("2-opt", (i, j)) for i in range(n - 1) for j in range(i + 2, n-1)]
      random.shuffle(actions)
      for action in actions:
        yield action

    @staticmethod
    def state_transition(state, action):
        # constructive-move: agrega una ciudad al tour
        if action[0]=="constructive" and state.is_complete==False:
          state.visited.append(action[1])
          state.not_visited.remove(action[1])

          if len(state.not_visited) == 0: # se completó el tour
             state.visited.append(state.visited[0])
             state.update_cost() #solo se actualiza en soluciones completas
             state.is_complete = True

        # 2-opt: intercambia dos aristas del tour
        elif action[0]=="2-opt" and state.is_complete==True:
           state.cost = calculate_cost_after_action(state, action)
           i, j = action[1]
           state.visited[i+1:j+1] = reversed(state.visited[i+1:j+1])
        else:
           raise NotImplementedError(f"Movimiento '{action}' no válido para estado {state}")

        return state

def calculate_cost_after_action(state, action):
    visited = state.visited
    dist_matrix = state.inst_info.distance_matrix

    n = len(visited)
    i, j = action[1]
    dist_actual_i = dist_matrix[visited[i]][visited[(i+1)%n]]
    dist_actual_j = dist_matrix[visited[j]][visited[(j+1)%n]]
    nueva_dist_i = dist_matrix[visited[i]][visited[j]]
    nueva_dist_j = dist_matrix[visited[(i+1)%n]][visited[(j+1)%n]]

    # Calcular el cambio en el costo
    cambio_costo = (nueva_dist_i + nueva_dist_j) - (dist_actual_i + dist_actual_j)
    new_cost = state.cost + cambio_costo
    return new_cost

def evalConstructiveActions(tsp_state, gen_constructive_actions):
    evals = []
    for action in gen_constructive_actions(tsp_state):
      ultima_ciudad = tsp_state.visited[-1] if tsp_state.visited else 0
      eval = tsp_state.inst_info.distance_matrix[ultima_ciudad][action[1]]
      evals.append((action,1.0-eval))

    return evals

class GreedyAgent():
    def __init__(self, eval_actions, gen_actions):
        self.eval_actions= eval_actions
        self.gen_actions= gen_actions

    def action_policy(self, state):
      evals = self.eval_actions(state, self.gen_actions)
      if len(evals)==0: return None

      # Seleccionar la acción que maximiza su evaluación
      action = max(evals, key=lambda x: x[1])[0]
      return action

import math

def sigmoid(x, k=1):
    return 1 / (1 + math.exp(-k * x))

class StochasticGreedyAgent():
    def __init__(self, eval_actions, gen_actions, steepness=1):
        self.eval_actions = eval_actions
        self.gen_actions = gen_actions
        self.steepness=steepness

    def __call__(self, state):
        evals = self.eval_actions(state, self.gen_actions)
        if len(evals) == 0: return None

        # Normalizar las evaluaciones usando la función sigmoid
        max_eval = max(evals, key=lambda x: x[1])[1]
        probabilities = [sigmoid(e[1] - max_eval, self.steepness) for e in evals]

        # Asegurar que la suma de probabilidades sea 1
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]

        # Seleccionar una acción basada en la distribución de probabilidad
        action = random.choices([e[0] for e in evals], weights=probabilities, k=1)[0]
        return action
    

class FirstImprovementAgent():
    def __init__(self, gen_random_actions):
        self.gen_random_actions = gen_random_actions

    def action_policy(self, state):
        current_cost=state.cost
        for action in self.gen_random_actions(state):
            new_cost = calculate_cost_after_action(state, action)
            if current_cost-new_cost > 0:
                return action
        return None
    
class Solver():
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def solve(self, state):
        while not state.is_complete:
            action = self.agent.action_policy(state)
            if action is None:
                break
            state = self.env.state_transition(state, action)
        return state
    

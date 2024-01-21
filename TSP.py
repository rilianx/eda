import random
from copy import deepcopy
import numpy as np
import math
from utils import handle_multiple_states

def sigmoid(x, k=1):
    return 1 / (1 + math.exp(-k * x))

def distance(punto1, punto2):
    return math.sqrt((punto1[0] - punto2[0])**2 + (punto1[1] - punto2[1])**2)


### TSP environment

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
    
    def calculate_cost(self, visited):
        cost = 0
        if len(visited) > 1:
          for i in range(len(visited) - 1):
              cost += self.inst_info.distance_matrix[visited[i]][visited[i + 1]]
        return cost

    def update_cost(self):
        self.cost = self.calculate_cost(self.visited)
        return self.cost
    
    def __deepcopy__(self, memo):
        # Crear una nueva instancia de la clase
        new_instance = type(self)(
            visited=deepcopy(self.visited),
            inst_info=self.inst_info # referencia
        )
        return new_instance


    def state2vecSeq(self):
        # creamos dos diccionarios para mantenre un mapeo de los
        # movimientos con los índices de la secuencia del modelo de aprendizaje

        city_locations = self.inst_info.city_locations

        idx2move = dict()
        move2idx = dict()
        origin = city_locations[self.visited[0]]
        destination = city_locations[self.visited[-1]]

        origin_dist = 0.0
        dest_dist = distance(origin, destination)

        seq = [list(origin) + [1,0] + [origin_dist, dest_dist]] # Última ciudad visitada (origen)


        idx2move[0] = ("constructive", self.visited[-1])
        move2idx[tsp_s.visited[-1]] = 0

        idx = 1
        for i in self.not_visited:
            point = list(self.city_points[i])
            origin_dist = distance( point, origin)
            dest_dist = distance( point, destination)
            if i == self.visited[0]:
                city_vector = point + [0, 1] + [origin_dist, 0.0]  # Ciudad final
            else:
                city_vector = point + [0, 0] + [origin_dist, dest_dist] # Otras ciudades

            seq.append(city_vector)
            idx2move[idx] = ("constructive", i)
            move2idx[i] = idx
            idx += 1

        return seq, idx2move, move2idx

    def __str__(self):
        return f"Tour actual: {self.visited}, \nCoste total: {self.cost}"

class TSP_Environment():
    @staticmethod
    def gen_actions(state, type, shuffle = False):
        if type == "constructive":
            actions = [("constructive", city) for city in state.not_visited]
        elif type == "2-opt":
            n = len(state.visited)
            actions = [(type, (i, j)) for i in range(n - 1) for j in range(i + 2, n-1)]
        elif type == "3-opt":
            n = len(state.visited)
            actions = [(type, (i, j, k)) for i in range(n - 3) for j in range(i + 2, n - 2) for k in range(j + 2, n-1)]
        elif type == "swap":
            n = len(state.visited)
            actions = [(type, (i, j)) for i in range(1, n - 1) for j in range(i + 1, n-1)]

        else:
            raise NotImplementedError(f"Tipo de acción '{type}' no implementado")
    
        if shuffle:
            random.shuffle(actions)

        for action in actions:
            yield action

    @staticmethod
    def apply_3_opt(state, indices):
        # Asume que 'indices' es una tupla de tres índices (i, j, k)
        i, j, k = indices  # Asegúrate de que los índices estén en orden

        if not (0 <= i < j < k < len(state.visited) - 1):
            raise ValueError("Índices inválidos para 3-opt")

        # Define los segmentos
        segment1 = state.visited[:i + 1]
        segment2 = state.visited[i + 1:j + 1]
        segment3 = state.visited[j + 1:k + 1]
        segment4 = state.visited[k + 1:]

        # Genera todas las combinaciones de reconexión
        combinations = [
            segment1 + segment2 + segment3[::-1] + segment4,  # 1ra reconexión
            segment1 + segment2[::-1] + segment3 + segment4,  # 2da reconexión
            segment1 + segment3 + segment2 + segment4,        # 3ra reconexión
            segment1 + segment3 + segment2[::-1] + segment4,  # 4ta reconexión
            segment1 + segment3[::-1] + segment2[::-1] + segment4,  # 5ta reconexión
            # ... añade más reconexiones según sea necesario
        ]

        # Encuentra la mejor combinación
        best_combination = min(combinations, key=state.calculate_cost)
        state.visited = best_combination
        state.cost = state.calculate_cost(best_combination)

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
           state.cost = TSP_Environment.calculate_cost_after_action(state, action)
           i, j = action[1]
           state.visited[i+1:j+1] = reversed(state.visited[i+1:j+1])
        
        elif action[0]=="3-opt" and state.is_complete==True:
            TSP_Environment.apply_3_opt(state, action[1])
            state.update_cost()

        # swap2: intercambia dos ciudades no adyacentes
        elif action[0] == "swap" and state.is_complete == True:
            i, j = action[1]
            state.visited[i], state.visited[j] = state.visited[j], state.visited[i]
            state.update_cost()  

        else:
           raise NotImplementedError(f"Movimiento '{action}' no válido para estado {state}")

        return state
    
    def calculate_cost_after_action(state, action):
        if action[0] == "2-opt": #optimización para 2-opt
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

        else:
            visited = deepcopy(state.visited)
            cost = state.cost
            TSP_Environment.state_transition(state, action)
            new_cost = state.cost
            state.visited = visited
            state.cost = cost

        return new_cost
    
#########################

@handle_multiple_states
def evalConstructiveActions(tsp_state, env):
    evals = []
    for action in env.gen_actions(tsp_state, "constructive"):
      ultima_ciudad = tsp_state.visited[-1] if tsp_state.visited else 0
      eval = tsp_state.inst_info.distance_matrix[ultima_ciudad][action[1]]
      evals.append((action,1.0-eval))

    return evals

class GreedyAgent():
    def __init__(self, eval_actions):
        self.eval_actions= eval_actions

    def reset(self):
        pass

    def select_action(self, evals):
        return max(evals, key=lambda x: x[1])[0]

    def action_policy(self, state, env):
      evals = self.eval_actions(state, env)
      if len(evals)==0: return None

      # Seleccionar la acción que maximiza su evaluación
      return self.select_action(self, evals)

    def __deepcopy__(self, memo):
        # Crear una nueva instancia de la clase
        new_instance = type(self)(
            eval_actions=self.eval_actions # pasamos por referencia
        )
        return new_instance


class StochasticGreedyAgent(GreedyAgent):
    def __init__(self, eval_actions, steepness=1):
        super().__init__(eval_actions)
        self.steepness=steepness

    def select_action(self, evals):
        # Normalizar las evaluaciones usando la función sigmoid
        max_eval = max(evals, key=lambda x: x[1])[1]
        probabilities = [sigmoid(e[1] - max_eval, self.steepness) for e in evals]

        # Asegurar que la suma de probabilidades sea 1
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]

        # Seleccionar una acción basada en la distribución de probabilidad
        return random.choices([e[0] for e in evals], weights=probabilities, k=1)[0]

    def __deepcopy__(self, memo):
        # Crear una nueva instancia de la clase
        new_instance = type(self)(
            eval_actions = self.eval_actions, # pasamos por referencia
            steepness = self.steepness
        )
        return new_instance

class FirstImprovementAgent():
    def __init__(self, action_type=None):
        self.action_type = action_type

    def reset(self):
        pass

    def action_policy(self, state, env):
        current_cost=state.cost
        for action in env.gen_actions(state, self.action_type, shuffle=True):
            new_cost = env.calculate_cost_after_action(state, action)
            if current_cost-new_cost > 0:
                return action
        return None
    
class SingleAgentSolver():
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def solve(self, state, track_best_state=False, save_history=False, max_actions=0):
        history = None
        if save_history: history = [(None, state.cost)]
        if max_actions==0: max_actions = 99999999

        best_state = None
    
        if track_best_state: best_state = deepcopy(state)

        self.agent.reset()
        n_actions=0
        while n_actions < max_actions:
            action = self.agent.action_policy(state, self.env)
            if action is None:
                break
            state = self.env.state_transition(state, action)
            n_actions+=1

            if track_best_state and state.cost < best_state.cost:
                best_state = deepcopy(state)

            if save_history:
                history.append((action, state.cost))

        if track_best_state:
            return best_state, history
        else:
            return state, history, n_actions
        
    def multi_solve(self, states, track_best_state=False, save_history=False, max_actions=0):
        agents = [deepcopy(self.agent) for _ in range(len(states))]
        history = [None]*len(states)
        best_state = [None]*len(states)
        n_actions = [None]*len(states)

        if max_actions==0: max_actions = 99999999

        for i in range(len(states)):
            agents[i].reset()
            n_actions[i] = 0
            history[i] = []
            if track_best_state: best_state[i] = deepcopy(states[i])

        live_states_idx = list(range(len(states)))

        for _ in range(max_actions):
            evals = agents[0].eval_actions([states[i] for i in live_states_idx], self.env)
            
            new_idx = []
            for i in live_states_idx:
                eval = evals[live_states_idx.index(i)]
                if eval == []: continue

                action = agents[i].select_action(eval)

                states[i] = self.env.state_transition(states[i], action)
                n_actions[i]+=1

                new_idx.append(i)

                if track_best_state and states[i].cost < best_state.cost:
                    best_state[i] = deepcopy(states[i])

                if save_history:
                    history[i].append((action, states[i].cost))
            
            live_states_idx = new_idx
            if new_idx == []: break

        if track_best_state:
            return best_state, history
        else:
            return states, history, n_actions
        





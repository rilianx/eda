import random
from copy import deepcopy
import heapq

## GREEDY (constructive algorithm)
class Greedy:
  def __init__(self, evalConstructiveMoves, random=False, elite_factor=1.0):
    self.evalConstructiveMoves = evalConstructiveMoves
    self.random=random
    self.elite_factor=elite_factor

  def __call__(self, initial_state):
    current_state = initial_state

    while not current_state.isCompleteSolution():
         # Evaluate moves and store their scores
        scored_elements = self.evalConstructiveMoves(current_state)

        if self.random==True:
          total_score = sum(score for _, score in scored_elements)
          normalized_scores = [(element, (score / total_score)**self.elite_factor) for element, score in scored_elements]

          best_move = random.choices([element for element, _ in normalized_scores],
                              weights=[score for _, score in normalized_scores],
                              k=1)[0]
        else:
          # Find the element with the highest score
          best_move, _ = max(scored_elements, key=lambda x: x[1])


        # Realiza la transición al estado siguiente agregando la ciudad más cercana al tour
        current_state.transition(best_move)

    return current_state

class Parallel_Greedy: #agregando aleatoriedad en la elección
  def __init__(self, evalConstructiveMoves, random=False, elite_factor=1.0):
    self.evalConstructiveMoves = evalConstructiveMoves
    self.random=random
    self.elite_factor=elite_factor
    if hasattr(evalConstructiveMoves, 'parallel_support'):
      self.parallel_eval = True
    else:
      self.parallel_eval = False


  def __call__(self, current_states):
    while not [current_state.isCompleteSolution() for current_state in current_states] == [True]*len(current_states):
        # Evaluate moves of non-complete solutions
        uncomplete_states = [current_state for current_state in current_states if not current_state.isCompleteSolution()]

        if self.parallel_eval == True:
          scored_elements = self.evalConstructiveMoves(uncomplete_states)
        

        for k in range(len(uncomplete_states)):
          if self.parallel_eval == True:
            scores = scored_elements[k]
          else:
            scores = self.evalConstructiveMoves(uncomplete_states[k])

          if self.random==True:
            total_score = sum(score for _, score in scores)
            normalized_scores = [(element, (score / total_score)**self.elite_factor) for element, score in scores]

            best_move = random.choices([element for element, _ in normalized_scores],
                                weights=[score for _, score in normalized_scores],
                                k=1)[0]
          else:
            # Find the element with the highest score
            best_move, _ = max(scores, key=lambda x: x[1])


          # Realiza la transición al estado siguiente agregando la ciudad más cercana al tour
          uncomplete_states[k].transition(best_move)

    return current_states


## SLS (Stochastic Local Search)
class SLS:
    def __init__(self, get_moves, first_improvement=True):
      self.first_improvement=first_improvement
      self.get_moves=get_moves

    def __call__(self, current_solution):
        while True:
            # Obtener el mejor vecino
            min_cost = current_solution.get_cost()
            best_move = None

            moves = self.get_moves(current_solution)
            random.shuffle(moves)
            for move in moves:
                # Comprobar si el movimiento mejora
                solution_cost = current_solution.get_cost_after_move(move)
                if solution_cost < min_cost:
                    min_cost = solution_cost
                    best_move = move
                    if self.first_improvement: break

            # Si no se encontraron mejoras, terminar la búsqueda
            if best_move is None:
                break
            else:
                # Si se encontró mejora, se modifica la solución actual
                current_solution.transition(best_move)
                #print(current_solution)

        # Retornar el estado con la mejor solución encontrada
        return current_solution

## ILS (Iterated Local Search)
class Perturbation:
    def __init__(self, get_moves, pert_size=3):
        self.pert_size = pert_size
        self.get_moves = get_moves

    def __call__(self, state):
        for _ in range(self.pert_size):
          moves = self.get_moves(state)
          move = random.choice(moves)
          state.transition(move)
        return state

class DefaultAcceptanceCriterion:
    def __call__(self, min_cost, new_cost):
      return new_cost <= min_cost

class ILS:
  def __init__(self, local_search, perturbation, 
             acceptance_criterion=DefaultAcceptanceCriterion(), max_iterations=50):
    self.local_search = local_search
    self.perturbation = perturbation
    self.acceptance_criterion = acceptance_criterion
    self.max_iterations = max_iterations

  def __call__(self,initial_solution):
    current_solution = initial_solution
    current_solution = self.local_search(current_solution)

    best_solution = deepcopy(current_solution)
    best_solution_cost = best_solution.get_cost()

    for _ in range(self.max_iterations):
        # Perturb the current solution to escape local optima
        perturbed_solution = self.perturbation(current_solution)

        # Apply local search on the perturbed solution
        local_optimum = self.local_search(perturbed_solution)
        cost = local_optimum.get_cost()

        # Decide whether to accept the new solution
        if self.acceptance_criterion(best_solution_cost, cost):
            current_solution = local_optimum
            if cost < best_solution_cost:
                best_solution = deepcopy(current_solution)
                best_solution_cost = cost

    return best_solution

import heapq

## A_star
class AStar:
    def __init__(self, admissibleHeuristic, getConstructiveMoves):
        self.admissibleHeuristic = admissibleHeuristic
        self.getConstructiveMoves = getConstructiveMoves

    def __call__(self, initial_state):
        # Lista de prioridad para los nodos a explorar
        open_set = []
        heapq.heappush(open_set, (0, initial_state))
        iters = 0

        while open_set:
            # Seleccionar el estado con el menor costo estimado para explorar
            _, current_state = heapq.heappop(open_set)

            # Si la solución está completa, retorna el estado
            if current_state.isCompleteSolution():
                return current_state, iters

            iters += 1
            # Generar movimientos constructivos
            for move in self.getConstructiveMoves(current_state):
                new_state = deepcopy(current_state)
                new_state.transition(move)

                # Calcular el costo total estimado (costo actual + heurística)
                estimated_total_cost = new_state.get_cost() + self.admissibleHeuristic(new_state)
                heapq.heappush(open_set, (estimated_total_cost, new_state))

        return None, iters  # No se encontró solución

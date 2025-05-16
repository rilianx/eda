from copy import deepcopy
import math 
import random
from .utils import handle_multiple_states


def sigmoid(x, k=1):
    return 1 / (1 + math.exp(-k * x))

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
      return self.select_action(evals)

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

class LocalSearchAgent(GreedyAgent):
    def __init__(self, action_type=None, first_improvement=True):
        self.action_type = action_type
        self.first_improvement = first_improvement

    @handle_multiple_states
    def eval_actions(self, state, env):
        current_cost=state.cost
        evals = []
        for action in env.gen_actions(state, self.action_type, shuffle=True):
            new_cost = env.calculate_cost_after_action(state, action)
            if current_cost-new_cost > 0:
                evals.append((action, new_cost))
                if self.first_improvement: return evals         
        return evals

    def __deepcopy__(self, memo):
        # Crear una nueva instancia de la clase
        new_instance = type(self)(
            action_type=self.action_type,
            first_improvement=self.first_improvement
        )
        return new_instance
    
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
        
    def multistate_solve(self, states, track_best_state=False, save_history=False, max_actions=0):
        self.agents = [deepcopy(self.agent) for _ in range(len(states))]
        history = [None]*len(states)
        best_state = [None]*len(states)
        n_actions = [None]*len(states)

        if max_actions==0: max_actions = 99999999

        for i in range(len(states)):
            self.agents[i].reset()
            n_actions[i] = 0
            history[i] = []
            if track_best_state: best_state[i] = deepcopy(states[i])

        live_states_idx = list(range(len(states)))

        for _ in range(max_actions):
            evals = self.agents[0].eval_actions([states[i] for i in live_states_idx], self.env)
            
            new_idx = []
            for i in live_states_idx:
                eval = evals[live_states_idx.index(i)]
                if eval == []: continue

                action = self.agents[i].select_action(eval)

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
        


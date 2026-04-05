import numpy as np

from problem import *
from planner import *

# Test Cases

# First problem
P1_B0 = np.array([["U", "U", "U", "U", "U"], ["U", "U", "U", "U", "U"],
                  ["U", "U", "U", "U", "U"], ["U", "U", "U", "U", "U"],
                  ["U", "U", "U", "U", "U"], ["U", "U", "U", "U", "U"]])

P1_B1 = np.array([["C", "S", "C", "C", "C"], ["S", "U", "U", "U", "U"],
                  ["S", "U", "U", "U", "U"], ["S", "U", "U", "U", "U"],
                  ["C", "U", "U", "U", "U"], ["C", "C", "C", "C", "C"]])

P1_G0 = np.array([["C", "S", "C", "C", "C"], ["S", "F", "S", "C", "C"],
                  ["S", "F", "S", "S", "S"], ["S", "F", "F", "F", "F"],
                  ["C", "S", "S", "S", "S"], ["C", "C", "C", "C", "C"]])

# Second problem
P2_B1 = np.array([["C", "S", "C", "C", "C"], ["S", "U", "U", "C", "U"],
                  ["S", "U", "U", "C", "U"], ["S", "U", "U", "U", "U"],
                  ["C", "U", "U", "C", "U"], ["C", "C", "C", "C", "C"]])

P2_G0 = np.array([["C", "S", "C", "C", "C"], ["S", "F", "S", "C", "C"],
                  ["S", "F", "S", "C", "S"], ["S", "F", "F", "S", "F"],
                  ["C", "S", "S", "C", "S"], ["C", "C", "C", "C", "C"]])


def test_policy(belief_map, true_map, problem, policy):
    """Test a policy on a SearchAndRescue problem.

    Args:
        belief_map: A numpy array specifying the belief map
        true_map:   A numpy array specifying the state map
        problem:    A SearchAndRescueProblem instance
        policy:     A policy returned by a policy making fn.
                    e.g. make_planner_policy(problem, planner)
    """
    height, width = true_map.shape
    bottom, right = height - 1, width - 1
    robot = (0, right)
    hospital = (bottom, right)
    people = {'pp': (bottom, right - 1)}  # Peter Parker
    carrying = None
    # Environment state
    env_state = State(robot=robot,
                      hospital=hospital,
                      people=people,
                      carrying=carrying,
                      state_map=true_map)
    # Initial belief: omniscient
    b0 = BeliefState(robot=robot,
                     hospital=hospital,
                     people=people,
                     carrying=carrying,
                     state_map=belief_map)
    # Do it
    return agent_loop(problem, env_state, policy, b0)


def agent_loop(problem: SearchAndRescueProblem, initial_state: State, policy, initial_belief: BeliefState, max_steps=200):
    """See MP01 introduction."""
    state = initial_state
    state.render(msg="initial state")
    belief = initial_belief
    belief.render(msg="initial belief")
    # An initial observation
    observation = problem.get_observation(state)
    print("Initial observation", observation)
    # Update the belief, first with transition, then with observation
    belief = belief.update(problem, observation)
    belief.render(msg="new belief")
    num_steps = 0
    for step in range(max_steps):
        action = policy(belief)
        if action in ("*Success*", "*Failure*"):
            print("Terminate with", action)
            return action, state, belief, num_steps
        # Resulting state
        state, valid = problem.get_next_state(state, action)
        assert valid, "Attempted to execute invalid action"
        num_steps += not ("look" in action)

        # Get observation of grid squares around the robot
        observation = problem.get_observation(state)
        # Update the belief, first with transition, then with observation
        belief = belief.update(problem, observation, action)
        print("agent_loop: step", step, "action", action, "observation", observation)
        state.render(msg="new state")
        belief.render(msg="new belief")
    return "*Failure*", state, belief, num_steps


### Test for 9

# %%
problem = SearchAndRescueProblem()

def make_safestupid():
    return make_safe_but_not_so_smart_policy(problem)

def make_safesmart(search_algo='gbf', heuristic='hff'):
    my_planner = SearchAndRescuePlanner(search_algo=search_algo, heuristic=heuristic)
    return make_planner_policy(problem, my_planner)

def make_reckless(search_algo='gbf', heuristic='hff'):
    my_planner = SearchAndRescuePlanner(search_algo=search_algo, heuristic=heuristic, unsafe=True)
    return make_reckless_planner_policy(problem, my_planner)

def make_safesmartreckless(search_algo='gbf', heuristic='hff'):
    my_planner = SearchAndRescuePlanner(search_algo=search_algo, heuristic=heuristic, unsafe=True)
    return make_combination_planner_policy(problem, my_planner)

def make_lookleap(search_algo='gbf', heuristic='hff'):
    my_planner = LookLeapSARPlanner(search_algo=search_algo, heuristic=heuristic)
    return make_planner_policy(problem, my_planner)

#

problems = [(P1_B0, P1_G0), (P1_B1, P2_G0), (P2_B1, P2_G0)]
policies = {
    # "safe_stupid": make_safestupid(),
    # "safe_smart": make_safesmart('astar', 'lmcut'),
    # "reckless": make_reckless('astar', 'lmcut'),
    "safe_smart_reckless": make_safesmartreckless('astar', 'lmcut'),
    # "look_leap": make_lookleap('astar', 'lmcut'),
}

evals = {0: {}, 1: {}, 2: {}}

for name, policy in policies.items():
    for i, (belief_map, true_map) in enumerate(problems):
        end_state, _, _, num_steps = test_policy(belief_map, true_map, problem, policy)
        evals[i][name] = {"Number of steps": num_steps, 
                          "Finished plan with": end_state}


for name in policies:
    print(f"*** Evaluating policy {name}")
    sum_steps = 0
    for i in range(len(problems)):
        print(f"** For problem {i}:")
        for eval, value in evals[i][name].items():
            print(f"    ** {eval} was: {value} ")
            if eval == "Number of steps":
                sum_steps += value
    
    print(f"** Total number of steps was {sum_steps}")


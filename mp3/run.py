from planner import *
from tests import agent_loop
from problem import SearchAndRescueProblem
from problem import State, BeliefState

# %%
def print_problem_pddl(my_planner, state):
    obj_str = my_planner.get_obj_strs(state)
    init_str = my_planner.get_init_strs(state)
    goal_str = my_planner.get_goal_strs(state)
    problem_pddl = f"""(define (problem searchandrescue) (:domain letsgoo)
    (:objects
    {obj_str}
    )
    (:init
    {init_str}
    )
    (:goal (and {goal_str}))
    )"""

    print(my_planner.generate_domain_pddl("letsgoo"))
    print(problem_pddl)

def exercise1():
    # %%
    problem = SearchAndRescueProblem()
    state = State()
    state.people = {
                "p1": (4, 0),
                "p2": (6, 0)}

    # my_planner = SearchAndRescuePlanner(search_algo='gbf', heuristic='hff')
    my_planner = SearchAndRescuePlanner()
    plan, success = my_planner.get_plan(state)
    print(plan)

    state = execute_plan(problem, plan, state)

# %%
def exercise2():
    problem = SearchAndRescueProblem()
    my_planner = SearchAndRescuePlanner(search_algo='gbf', heuristic='hff')
    belief = BeliefState()
    state = State()
    policy = make_planner_policy(problem, my_planner)

    agent_loop(problem, state, policy, belief)

# %%
def exercise3():
    problem = SearchAndRescueProblem()
    my_planner = SearchAndRescuePlanner(unsafe=True)
    belief = BeliefState()
    state = State()
    policy = make_planner_policy(problem, my_planner)

    agent_loop(problem, state, policy, belief)


# %%
def exercise4():
    problem = SearchAndRescueProblem()
    my_planner = LookLeapSARPlanner(search_algo='gbf', heuristic='hff')
    belief = BeliefState()
    state = State()
    policy = make_planner_policy(problem, my_planner)

    agent_loop(problem, state, policy, belief)



exercise4()

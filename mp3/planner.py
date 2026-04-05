import os
import tempfile
import time
from pyperplan.pddl.parser import Parser
from pyperplan import grounding, planner
from problem import State, BeliefState, SearchAndRescueProblem

class SearchAndRescuePlanner:
    """A planner for a search and rescue problem.

    The core function in this class is 'get_plan'
    This function does the following:
        1. Create PDDL domain and problem strings for search and rescue. The operators should work for any grid size, obstacles, people locations, and hospital location.
        2. Invoke `run_planning` using the given `search_algo` search algorithm with the `heuristic` heuristic.
        3. Convert the output of run_planning (pyperplan Operators) into actions
           that can be given to the SearchAndRescueProblem.

    Example Usage:
        problem = SearchAndRescueProblem()
        state = State()

        planner = SearchAndRescuePlanner(search_algo='astar', heuristic='lmcut')
        plan, plan_time = planner.get_plan(state)
        state = execute_plan(problem, plan, state)

    'get_plan' Returns:
        plan: A list of actions; each action is a str, see SearchAndRescueProblem.
        plan_time: Total planning time(sec) used for plan searching.

    For reference, 'get_plan' takes ~1-2 seconds to run with our implementation if using 'gbf' search and 'lmcut' heuristic.
    """

    def __init__(self, search_algo='astar', heuristic='lmcut', unsafe=False):
        self.search_algo = search_algo
        self.heuristic = heuristic
        self.unsafe = unsafe

    def generate_domain_pddl(self,
                             domain_name,
                             added_operators='',
                             added_predicates=''):
        predicates_str = """(conn ?v0 - location ?v1 - location ?v2 - direction)
        (is-safe ?v0 - location)
        (on ?v0 - location)
        (person-on ?v0 - person ?v1 - location)
        (carrying ?v0 - person)
        (hands-free ?v0 - robot)"""

        operators_str = """(:action move-robot
    :parameters (?from - location ?to - location ?dir - direction)
    :precondition (and
      (conn ?from ?to ?dir)
      (on ?from)
      (is-safe ?to)
    )
    :effect (and
      (on ?to)
      (not (on ?from))
    )
  )
  (:action pickup-person
    :parameters (?person - person ?loc - location)
    :precondition (and
      (hands-free agent)
      (person-on ?person ?loc)
      (on ?loc)
    )
    :effect (and
      (not (hands-free agent))
      (not (person-on ?person ?loc))
      (carrying ?person)
    )
  )
  (:action dropoff-person
    :parameters (?person - person ?loc - location)
    :precondition (and
      (carrying ?person)
      (on ?loc)
    )
    :effect (and
      (person-on ?person ?loc)
      (not (carrying ?person))
      (hands-free agent)
    )
  )"""

        domain_pddl = f"""(define (domain {domain_name})
    (:requirements :typing)
    (:types person location direction robot)
    (:constants
      down - direction
      left - direction
      right - direction
      up - direction
      agent - robot
    )
    (:predicates
      {predicates_str}
      {added_predicates}
    )
    {operators_str}
    {added_operators}
)"""
        return domain_pddl

    def get_plan(self, state):
        search_algo, heuristic = self.search_algo, self.heuristic
        domain_name, added_predicate, added_operator = self.update_pddl_domain()
        domain_pddl = self.generate_domain_pddl(
            domain_name,
            added_operators=added_operator,
            added_predicates=added_predicate)
        # Create objects str
        obj_str = self.get_obj_strs(state)

        # Create init str
        init_str = self.get_init_strs(state)

        # Create goal str
        goal_str = self.get_goal_strs(state)

        problem_pddl = f"""(define (problem searchandrescue) (:domain {domain_name})
      (:objects
      {obj_str}
      )
      (:init
      {init_str}
      )
      (:goal (and {goal_str}))
    )"""

        start_time = time.time()
        plan = run_planning(domain_pddl, problem_pddl, search_algo, heuristic)
        time_elapsed = time.time() - start_time
        if plan is None:
            print("Failed to find a plan.")
            return None, time_elapsed

        # Convert operators to actions
        actions = self.parse_plan(plan)
        return actions, time_elapsed

    def get_obj_strs(self, state):
        height, width = state.state_map.shape

        objects_strs = [f"{person} - person" for person in state.people]
        if state.carrying is not None:
            objects_strs.append(f"{state.carrying} - person")
        
        objects_strs.extend(
            [f"l{r}-{c} - location" for r in range(height) for c in range(width)]
        )

        objects_str = " ".join(objects_strs)
        return objects_str

    def get_init_strs(self, state: State):
        height, width = state.state_map.shape
        robot_r, robot_c = state.robot
        init_strs = []

        deltas = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1),
        }
        for r in range(height):
            for c in range(width):
                # Here we're going to add one (conn ...) atom for every pair
                # of adjacent locations.
                for direction, (dr, dc) in deltas.items():
                    if not (0 <= r + dr < height and 0 <= c + dc < width):
                        continue
                    # For example, if r == 0, c == 0, dr == 0, dc == 1, then
                    # this line adds the atom (conn l0-0 l0-1 right).
                    init_strs.append(
                        f"(conn l{r}-{c} l{r + dr}-{c + dc} {direction})")
                
                # Mark the safe locations - where the robot can move
                if state.state_map[r][c] in ['C', 'S'] or (self.unsafe and state.state_map[r][c] == 'U'):
                    init_strs.append(
                        f"(is-safe l{r}-{c})"
                    )
                
        for person, (r, c) in state.people.items():
            init_strs.append(
                f"(person-on {person} l{r}-{c})"
            )

        if state.carrying is not None:
            init_strs.append(
                f"(carrying {state.carrying})"
            )
        else:
            init_strs.append(
                f"(hands-free agent)"
            )

        init_strs.append(
            f"(on l{robot_r}-{robot_c})"
        )

        init_str = " ".join(init_strs)
        return init_str

    def get_goal_strs(self, state: State):
        goal_strs = []
        hospital_r, hospital_c = state.hospital
        
        for person in state.people:
            goal_strs.append(
                f"(person-on {person} l{hospital_r}-{hospital_c})"
            )

        if state.carrying is not None:
            goal_strs.append(
                f"(person-on {state.carrying} l{hospital_r}-{hospital_c})"
            )
        
        goal_str = " ".join(goal_strs)
        return goal_str

    def update_pddl_domain(self):
        domain_name = 'searchandrescue'
        added_predicate = ''
        added_operator = ''
        return domain_name, added_predicate, added_operator

    def parse_plan(self, plan):
        actions = []
        for op in plan:
            if "move-robot" in op.name:
                _, direction = op.name[:-1].rsplit(" ", 1)
                action = direction
            elif "pickup-person" in op.name:
                _, person, _ = op.name.split(" ")
                action = f"pickup-{person}"
            else:
                assert "dropoff-person" in op.name
                action = "dropoff"
            actions.append(action)
        return actions


class LookLeapSARPlanner(SearchAndRescuePlanner):
    
    def get_init_strs(self, state: State):
        height, width = state.state_map.shape
        init_strs = [super().get_init_strs(state)]

        for r in range(height):
            for c in range(width):
                # add the predicates related to unknown cells
                if state.state_map[r][c] == 'U':
                    init_strs.append(
                        f"(is-unknown l{r}-{c})"
                    )


        init_str = " ".join(init_strs)
        return init_str

    def update_pddl_domain(self):
        domain_name = 'searchandrescueleaplook'
        added_predicate = '(is-unknown ?v0 - location)'
        added_operator = """(:action look
    :parameters (?from - location ?to - location ?dir - direction)
    :precondition (and
      (conn ?from ?to ?dir)
      (on ?from)
      (is-unknown ?to)
    )
    :effect (and
      (not (is-unknown ?to))
      (is-safe ?to)
    )
  )"""
        return domain_name, added_predicate, added_operator
    
    
    def parse_plan(self, plan):
        actions = []
        for op in plan:
            if "move-robot" in op.name:
                _, direction = op.name[:-1].rsplit(" ", 1)
                action = direction
            elif "pickup-person" in op.name:
                _, person, _ = op.name.split(" ")
                action = f"pickup-{person}"
            elif "look" in op.name:
                _, _, _, direction = op.name[:-1].split(" ")
                action = f"look-{direction}"
            else:
                assert "dropoff-person" in op.name
                action = "dropoff"
            actions.append(action)
        return actions



def run_planning(domain_pddl_str,
                 problem_pddl_str,
                 search_alg_name,
                 heuristic=None):
    """Plan a sequence of actions to solve the given PDDL problem.

    This function is a lightweight wrapper around pyperplan.

    Args:
      domain_pddl_str: A str, the contents of a domain.pddl file.
      problem_pddl_str: A str, the contents of a problem.pddl file.
      search_alg_name: A str, the name of a search algorithm in
        pyperplan. Options: astar, wastar, gbf, bfs, ehs, ids, sat.
      heuristic: A str or a pyperplan `Heuristic` class.
        A str, the name of a heuristic in pyperplan.
          Options: blind, hadd, hmax, hsa, hff, lmcut, landmark.
        A pyperplan `Heuristic` class.
          See: https://github.com/aibasel/pyperplan/blob/main/doc/documentation.md#implementing-new-heuristics

    Returns:
      plan: A list of actions; each action is a pyperplan Operator.
    """
    # Parsing the PDDL
    domain_file = tempfile.NamedTemporaryFile(delete=False)
    problem_file = tempfile.NamedTemporaryFile(delete=False)
    with open(domain_file.name, 'w') as f:
        f.write(domain_pddl_str)
    with open(problem_file.name, 'w') as f:
        f.write(problem_pddl_str)
    parser = Parser(domain_file.name, problem_file.name)
    domain = parser.parse_domain()
    problem = parser.parse_problem(domain)
    os.remove(domain_file.name)
    os.remove(problem_file.name)

    # Ground the PDDL
    task = grounding.ground(problem)

    # Get the search alg
    search_alg = planner.SEARCHES[search_alg_name]

    if heuristic is None:
        return search_alg(task)

    if isinstance(heuristic, str):
        # Get the heuristic from pyperplan
        heuristic_initialized = planner.HEURISTICS[heuristic](task)
    else:
        # Use customized heuristic
        heuristic_initialized = heuristic(task)

    # Run planning
    return search_alg(task, heuristic_initialized)


def execute_plan(problem, plan, state):
    """See MP01 introduction."""
    for action in plan:
        state.render(msg=f"execute_plan: {action}")
        # Resulting state
        state, valid = problem.get_next_state(state, action)
        assert valid, "Attempted to execute invalid action"
    state.render(msg=f"execute_plan: Final state")
    return state



def get_num_delivered(state):
    """Returns the number of people located in the hospital."""
    num_delivered = 0
    for loc in state.people.values():
        if loc == state.hospital:
            num_delivered += 1
    return num_delivered


def execute_count_num_delivered(problem, state, plan):
    """Execute a plan for search and rescue and count the number of people
    delivered.

    Args:
      problem: A SearchAndRescueProblem
      plan: A list of action strs, see SearchAndRescueProblem.

    Returns:
      num_delivered: int
    """
    state = execute_plan(problem=problem, plan=plan, state=state)
    return get_num_delivered(state)


def manhattan_distance(r1, c1, r2, c2):
    return abs(r1 - r2) + abs(c1 - c2)

def get_valid_neighbors(r, c, height, width):
    deltas = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }
    for dir, (d_r, d_c) in deltas.items():
        new_r, new_c = r + d_r, c + d_c
        if 0 <= new_r < height and 0 <= new_c < width:
            yield dir, (new_r, new_c)

def make_safe_but_not_so_smart_policy(problem):
    def policy(belief: BeliefState):
        """Returns an action or '*Failure*"""
        if belief.robot == belief.hospital:
            return "*Success*"
        
        hospital_r, hospital_c = belief.hospital
        robot_r, robot_c = belief.robot
        dist = manhattan_distance(robot_r, robot_c, hospital_r, hospital_c)
        height, width = belief.state_map.shape

        for dir, (neigh_r, neigh_c) in get_valid_neighbors(robot_r, robot_c, height, width):
            if (belief.state_map[neigh_r][neigh_c] in ['S', 'C'] and 
                manhattan_distance(neigh_r, neigh_c, hospital_r, hospital_c) < dist):
                return dir
        
        return "*Failure*"

    # return the policy function
    return policy


def make_planner_policy(problem: SearchAndRescueProblem, planner: SearchAndRescuePlanner):
    # Keep memory of plan and which step we're on
    status = {"plan": None, "step": None}

    def policy(belief: BeliefState):
        """Returns an action string or '*Failure*' or '*Success*'."""
        # if we've brought all the people to the hospital
        if get_num_delivered(belief) == (len(belief.people) + (belief.carrying is not None)):
            return "*Success*"

        # if we already have a plan
        if status["plan"] is not None and status["step"] + 1 < len(status["plan"]):
            next_action = status["plan"][status["step"] + 1] 
            next_state, valid = problem.get_next_state(belief, next_action)

            # is the square where we are heading safe?
            if valid:
                status["step"] += 1
                return next_action

        # if the action assigned by the old plan does not lead to safety or we have no plan,
        # make new plan        
        plan, _ = planner.get_plan(belief.get_careful_state())
        print(f"IMPORTANT: the new plan is {plan}")

        # if no plan can be made, return false
        if not plan:
            return "*Failure*"
        
        status['plan'] = plan
        status["step"] = 0
        return status["plan"][0]

    # return the policy function
    return policy


def make_reckless_planner_policy(problem: SearchAndRescueProblem, planner: SearchAndRescuePlanner):
    # Keep memory of plan and which step we're on
    status = {"plan": None, "step": None}

    def policy(belief: BeliefState):
        """Returns an action string or '*Failure*' or '*Success*'."""
        # if we've brought all the people to the hospital
        if get_num_delivered(belief) == (len(belief.people) + (belief.carrying is not None)):
            return "*Success*"

        # if we already have a plan
        if status["plan"] is not None and status["step"] + 1 < len(status["plan"]):
            next_action = status["plan"][status["step"] + 1] 
            next_state, valid = problem.get_next_state(belief, next_action)

            # is the square where we are heading safe?
            if valid:
                status["step"] += 1
                return next_action

        # if the action assigned by the old plan does not lead to safety or we have no plan,
        # make new plan        
        plan, _ = planner.get_plan(belief.get_optimistic_state())
        print(f"IMPORTANT: the new plan is {plan}")

        # if no plan can be made, return false
        if not plan:
            return "*Failure*"
        
        status['plan'] = plan
        status["step"] = 0
        return status["plan"][0]

    # return the policy function
    return policy

def make_combination_planner_policy(problem: SearchAndRescueProblem, planner: SearchAndRescuePlanner):
    # Keep memory of plan and which step we're on
    status = {"plan": None, "step": None, "plan_type": None}

    def policy(belief: BeliefState):
        """Returns an action string or '*Failure*' or '*Success*'."""
        # if we've brought all the people to the hospital
        if get_num_delivered(belief) == (len(belief.people) + (belief.carrying is not None)):
            return "*Success*"

        # if we already have a plan
        if status["plan"] is not None and status["step"] + 1 < len(status["plan"]):
            next_action = status["plan"][status["step"] + 1] 
            next_state, valid = problem.get_next_state(belief, next_action)

            # is the square where we are heading safe?
            if valid:
                status["step"] += 1
                return next_action

        # if the action assigned by the old plan does not lead to safety or we have no plan,
        # make new plan        
        plan, _ = planner.get_plan(belief.get_careful_state())
        print(f"IMPORTANT: the new plan is {plan}")

        # if no plan can be made, return false
        if not plan:
            plan, _ = planner.get_plan(belief.get_optimistic_state())

        if not plan:
            return "*Failure*"

        status['plan'] = plan
        status["step"] = 0
        return status["plan"][0]

    # return the policy function
    return policy


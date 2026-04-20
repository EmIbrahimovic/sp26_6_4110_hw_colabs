# Setup matplotlib animation
import matplotlib
matplotlib.rc('animation', html='jshtml')


from typing import (Callable, Iterable, List, Sequence, Tuple, Dict, Optional,
                    Any, Union, Set, ClassVar, Type, TypeVar)

import collections
import functools
import itertools
import random
import dataclasses
import numpy as np
import heapq as hq

from problem import *

class SearchFailed(ValueError):
    """Raise this exception whenever a search must fail."""
    pass


# A useful data structure for best-first search
BFSNode = collections.namedtuple("BFSNode",
                                 ["state", "parent", "action", "cost", "g"])


def run_best_first_search(
    problem: PathCostProblem,
    get_priority: Callable[[BFSNode], float],
    step_budget: int = 10000
) -> Tuple[List[State], List[Action], List[float], int]:
    """A generic heuristic search implementation.

    Depending on `get_priority`, can implement A*, GBFS, or UCS.

    The `get_priority` function here should determine the order
    in which nodes are expanded. For example, if you want to
    use path cost as part of this determination, then the
    path cost (node.g) should appear inside of get_priority,
    rather than in this implementation of `run_best_first_search`.

    Important: for determinism (and to make sure our tests pass),
    please break ties using the state itself. For example,
    if you would've otherwise sorted by `get_priority(node)`, you
    should now sort by `(get_priority(node), node.state)`.

    Args:
      problem: a path cost problem.
      get_priority: a callable taking in a search Node and returns the priority
      step_budget: maximum number of `problem.step` before giving up.

    Returns:
      state_sequence: A list of states.
      action_sequence: A list of actions.
      cost_sequence: A list of costs.
      num_steps: number of taken `problem.step`s. Must be less than or equal to `step_budget`.

    Raises:
      error: SearchFailed, if no plan is found.
    """
    num_steps = 0
    frontier = []
    reached = {}

    root_node = BFSNode(state=problem.initial,
                        parent=None,
                        action=None,
                        cost=None,
                        g=0)
    hq.heappush(frontier, (get_priority(root_node), problem.initial, root_node))
    reached[problem.initial] = root_node
    num_expansions = 0

    while frontier:
        pri, s, node = hq.heappop(frontier)
        # If reached the goal, return
        if problem.goal_test(node.state):
            return (*finish_plan(node), num_steps)

        num_expansions += 1
        # Generate successors
        for action in problem.actions(node.state):
            if num_steps >= step_budget:
                raise SearchFailed(
                    f"Failed to find a plan in {step_budget} steps")
            child_state = problem.step(node.state, action)
            num_steps += 1
            cost = problem.step_cost(node.state, action, child_state)
            path_cost = node.g + cost
            # If the state is already in explored or reached, don't bother
            if not child_state in reached or path_cost < reached[child_state].g:
                # Add new node
                child_node = BFSNode(state=child_state,
                                     parent=node,
                                     action=action,
                                     cost=cost,
                                     g=path_cost)
                priority = get_priority(child_node)
                hq.heappush(frontier, (priority, child_state, child_node))
                reached[child_state] = child_node
    raise SearchFailed(f"Frontier exhausted after {num_steps} steps")


def finish_plan(node: BFSNode):
    """Helper for run_best_first_search."""
    state_sequence = []
    action_sequence = []
    cost_sequence = []

    while node.parent is not None:
        action_sequence.insert(0, node.action)
        state_sequence.insert(0, node.state)
        cost_sequence.insert(0, node.cost)
        node = node.parent
    state_sequence.insert(0, node.state)

    return state_sequence, action_sequence, cost_sequence


def run_astar_search(problem: PathCostProblem, step_budget: int = 10000):
    """A* search.

    Use your implementation of `run_best_first_search`.
    """
    get_priority = lambda node: node.g + problem.h(node.state)
    return run_best_first_search(problem, get_priority, step_budget=step_budget)


@dataclasses.dataclass(frozen=False, eq=False)
class MCTStateNode:
    """Node in the Monte Carlo search tree, keeps track of the children states."""
    # For MDP, this is a state; for POMDP, this is an observation
    obs: Union[State, Observation]
    N: int
    horizon: int
    parent: Optional['MCTChanceNode']
    children: Dict['MCTChanceNode',
                   Action] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=False, eq=False)
class MCTChanceNode:
    U: float
    N: int
    parent: MCTStateNode
    children: Dict[State,
                   MCTStateNode] = dataclasses.field(default_factory=dict)


def ucb(n: MCTStateNode, C: float = 1.4) -> float:
    """UCB for a node, note the C argument"""
    return (np.inf if n.N == 0 else
            (n.U / n.N + C * np.sqrt(np.log(n.parent.N) / n.N)))


RolloutPolicy = Callable[[State], Action]


def random_rollout_policy(problem: MDP, state: State) -> Action:
    return random.choice(list(problem.actions(state)))


def run_mcts_search(problem: POMDP,
                    state: Optional[State] = None,
                    state_sampler: Iterable[State] = None,
                    horizon: Optional[int] = None,
                    C: float = np.sqrt(2),
                    iteration_budget: int = 100,
                    n_simulations: int = 10,
                    max_backup: bool = True,
                    rollout_policy: RolloutPolicy = None,
                    verbose: bool = False) -> Action:
    """A generic MCTS search implementation for MDP and POMDPs.

    For MDP, this is a standard MCTS implementation based on description in AIMA
    (with some additional features).
    For POMDP, this is an implementation of the POMCP algorithm.

    Args:
        problem: an MDP or POMDP.
        state: the initial state. If None, then `state_sampler` must be provided.
        state_sampler: an iterable of states. If None, then `state` must be provided. This is used to sample the initial state for POMCP.
        horizon: the horizon of the search. If None, then the horizon is set to the problem's horizon.
        C: the C parameter for UCB.
        iteration_budget: the maximum number of iterations.
        n_simulations: the number of simulations to run at each MCTS iteration. In AIMA,
            this is set 1, but in general, we run multiple simulations to reduce variance.
        max_backup: whether to use max backup or sum backup.
        rollout_policy: the rollout policy. If None, then the random rollout policy is used.
        verbose: whether to print debug information.

    Returns:
        action: the best action to take at the initial state according to the search.
    """
    if state is None:
        state = problem.initial

    if state_sampler is None:
        state_sampler = itertools.repeat(state)

    if horizon is None:
        horizon = problem.horizon

    if rollout_policy is None:
        rollout_policy = functools.partial(random_rollout_policy, problem)

    ucb_fixed_C = functools.partial(ucb, C=C)

    rewards = []

    def select(n: MCTStateNode, state: State) -> Tuple[MCTStateNode, State]:
        """select a leaf node in the tree."""
        if n.children:
            # Select the best child, break ties randomly
            children = list(n.children.keys())
            random.shuffle(children)
            ucb_pick: MCTChanceNode = max(children, key=ucb_fixed_C)
            act = n.children[ucb_pick]
            next_state = problem.step(state, act)
            rewards.append(problem.reward(state, act, next_state))
            next_obs = problem.get_observation(
                next_state)  # For MDP this is same as next_state
            if next_obs not in ucb_pick.children:
                new_leaf = MCTStateNode(next_obs,
                                        horizon=n.horizon - 1,
                                        parent=ucb_pick,
                                        N=0)
                ucb_pick.children[next_obs] = new_leaf
                return new_leaf, next_state
            return select(ucb_pick.children[next_obs], next_state)
        return n, state

    def expand(n: MCTStateNode, state: State) -> Tuple[MCTStateNode, State]:
        """expand the leaf node by adding all its children actions."""
        assert not n.children
        if n.horizon == 0 or problem.terminal(state):
            return n, state
        for action in problem.actions(state):
            new_chance_node = MCTChanceNode(parent=n, U=0, N=0)
            n.children[new_chance_node] = action
        chance_node, action = random.choice(list(n.children.items()))
        next_state = problem.step(state, action)
        rewards.append(problem.reward(state, action, next_state))
        next_obs = problem.get_observation(
            next_state)  # For MDP this is same as next_state
        new_node = MCTStateNode(next_obs,
                                N=0,
                                horizon=n.horizon - 1,
                                parent=chance_node)
        chance_node.children[next_obs] = new_node
        return new_node, next_state

    def simulate(node: MCTStateNode, state: State) -> float:
        """simulate the utility of current state by taking a rollout policy."""
        total_reward = 0
        disc = 1
        h = node.horizon
        while h > 0 and not problem.terminal(state):
            action = rollout_policy(state)
            next_state = problem.step(state, action)
            reward = problem.reward(state, action, next_state)
            total_reward += disc * reward
            state = next_state
            disc = disc * problem.discount
            h -= 1
        return total_reward

    def backup(state_node: MCTStateNode, value: float) -> None:
        """passing the utility back to all parent nodes."""
        state_node.N += 1
        if state_node.parent:
            # Need to include the reward on the action *into* n
            parent_chance_node = state_node.parent
            parent_state_node = parent_chance_node.parent
            r = rewards.pop()
            future_val = r + problem.discount * value
            parent_chance_node.U += future_val
            parent_chance_node.N += 1
            if max_backup:
                bk_val = max(0 if n.N == 0 else n.U / n.N
                             for n in parent_state_node.children)
            else:
                bk_val = future_val
            backup(parent_state_node, bk_val)

    state = next(state_sampler)
    root = MCTStateNode(obs=problem.get_observation(state),
                        horizon=horizon,
                        parent=None,
                        N=0)

    i = 0
    while i < iteration_budget:
        state = next(state_sampler)
        assert len(rewards) == 0
        leaf, state = select(root, state)
        child, state = expand(leaf, state)
        value = np.mean([simulate(child, state) for _ in range(n_simulations)])
        backup(child, value)
        i += 1

    children = list(root.children.keys())
    random.shuffle(children)
    act = root.children[max(children, key=lambda p: p.U / p.N)]
    if verbose:
        print(
            {
                act: (c.U / c.N if c.N > 0 else 0, c.N)
                for c, act in root.children.items()
            }, act)
    return act


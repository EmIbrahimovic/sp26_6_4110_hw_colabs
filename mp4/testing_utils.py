# Setup matplotlib animation
import matplotlib
matplotlib.rc('animation', html='jshtml')


from typing import (Callable, Iterable, List, Sequence, Tuple, Dict, Optional,
                    Any, Union, Set, ClassVar, Type, TypeVar)

import textwrap
import random
import numpy as np

from problem import *
from fire_problem import *
from agents import *

def get_problem(name: str) -> MDP:
    """Return a problem instance by name."""

    params = {
        "maze":
            dict(env_s="""\
                |R < . X . > H|
                |            X|
                |. X .   .   .|
                |        X   X|
                |. X .   . X .|
                |    X   ^    |
                |. X .   . X .|
                |    X   ^   ^|
                |P   . > . > .|
                """,
                 fire_process_kargs=dict(fire_weights=np.array([
                     [0, 1, 0],
                     [1, 10, 1],
                     [0, 1, 0],
                 ])),
                 _horizon=20),
        "just_wait":
            dict(env_s="""\
                |.   R   .   H|
                |X   v   X   ^|
                |. X . X .   .|
                |    v       ^|
                |. X . X .   .|
                |    v       ^|
                |. X . X .   .|
                |    v       ^|
                |F X . X F   .|
                |    v       ^|
                |. X . X .   F|
                |    v       ^|
                |. X . X .   .|
                |X   v   X   ^|
                |P   .   .   .|
                """,
                 fire_process_kargs=dict(fire_weights=np.array([
                     [0, 1, 0],
                     [1, 20, 1],
                     [0, 1, 0],
                 ]))),
        "the_circle":
            dict(env_s="""\
                |R   .   .   H|
                |    X   X   v|
                |. X .   . X .|
                |            v|
                |. X F   . X .|
                |            v|
                |. X F   . X .|
                |    X   X   v|
                |P   . < . < .|
                """,
                 fire_process_kargs=dict(fire_weights=np.array([
                     [0, 1, 0],
                     [1, 4, 1],
                     [0, 1, 0],
                 ]))),
        "the_choice":
            dict(
                env_s="""\
                |.   .   F   F   F   F|
                |.   X   X   X   X   X|
                |. X .   .   .   .   .|
                |X                    |
                |R > .   .   F   .   .|
                |v                    |
                |. X .   .   .   .   .|
                |v   X   X   X   X   v|
                |. X .   F   F   . X .|
                |v                   v|
                |. X F   .   .   . X .|
                |v                   v|
                |. X F   .   .   . X .|
                |v                   v|
                |. X F   F   .   . X .|
                |v   X   X   X   X   v|
                |. > . > H   P < . < .|
                """,
                fire_process_kargs=dict(fire_weights=np.array([
                    [0, 1, 0],
                    [1, 4, 1],
                    [0, 1, 0],
                ]),),
            )
    }

    if name not in params:
        raise ValueError(f"Unknown problem name: {name}")

    params[name]["env_s"] = textwrap.dedent(params[name]["env_s"])
    return FireMDP.from_str(**params[name])


def run_agent_on_problem(problem: Union[MDP, POMDP],
                         agent: Agent,
                         verbose: bool = True
                        ) -> Tuple[Sequence[State], Sequence[Action], float]:
    """Runs the agent on the problem and returns the trajectory."""
    random.seed(6)
    np.random.seed(4110)

    agent.reset()
    state = problem.initial
    obs = problem.get_observation(state)
    state_sequence = [state]
    action_sequence = []
    total_reward = 0
    while not problem.terminal(state) and len(state_sequence) < problem.horizon:
        action = agent.act(obs)
        next_state = problem.step(state, action)
        reward = problem.reward(state, action, next_state)
        total_reward += reward * problem.discount**len(state_sequence)
        if verbose:
            print(
                f"Action={action} reward={reward} total_reward={total_reward}")
        obs = problem.get_observation(next_state)
        state = next_state
        action_sequence.append(action)
        state_sequence.append(state)
    return state_sequence, action_sequence, total_reward


def animate_trajectory(problem: Union[MDP, POMDP],
                       trajectory: Tuple[Sequence[State], Sequence[Action]]):
    """Visualizes a trajectory.

    Args:
        problem: The problem.
        trajectory: A tuple of state and action sequences.

    Returns:
        A matplotlib animation.
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation

    state_sequence, action_sequence, *_ = trajectory

    fig, ax = plt.subplots()

    total_reward = 0.

    def animate(i):
        ax.clear()
        ax.set_aspect('equal')
        nonlocal total_reward
        if i == 0:
            total_reward = 0
            ax.set_title(f"Step {i}: begin, total_reward={total_reward:.2f}")
        elif i < len(state_sequence):
            action = action_sequence[i - 1]
            reward = problem.reward(state_sequence[i - 1], action,
                                    state_sequence[i])
            total_reward += reward * problem.discount**i
            ax.set_title(
                f"Step {i}: action={action}, "
                f"reward={reward:.2f}, total_reward={total_reward:.2f}")

        problem.render(state_sequence[i], ax=ax)

    anim = matplotlib.animation.FuncAnimation(fig,
                                              animate,
                                              frames=len(state_sequence),
                                              interval=500)
    return anim

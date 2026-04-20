
# Setup matplotlib animation
import matplotlib
matplotlib.rc('animation', html='jshtml')


from typing import (Callable, Iterable, List, Sequence, Tuple, Dict, Optional,
                    Any, Union, Set, ClassVar, Type, TypeVar)

from abc import abstractmethod, ABC
import numpy as np


State = Any
Observation = Any
Action = Any


class Problem(ABC):
    """The abstract base class for either a path cost problem or a reward problem."""

    @property
    @abstractmethod
    def initial(self) -> State:
        ...

    @abstractmethod
    def actions(self, state: State) -> Iterable[Action]:
        """Returns the allowed actions in a given state.

        The result would typically be a list. But if there are many
        actions, consider yielding them one at a time in an iterator,
        rather than building them all at once.
        """
        ...

    @abstractmethod
    def step(self, state: State, action: Action) -> State:
        """Returns the next state when executing a given action in a given
        state.

        The action must be one of self.actions(state).
        """
        ...


class PathCostProblem(Problem):
    """An abstract class for a path cost problem, based on AIMA.

    To formalize a path cost problem, you should subclass from this and
    implement the abstract methods. Then you will create instances of
    your subclass and solve them with the various search functions.
    """

    @abstractmethod
    def goal_test(self, state: State) -> bool:
        """Checks if the state is a goal."""
        ...

    @abstractmethod
    def step_cost(self, state1: State, action: Action, state2: State) -> float:
        """Returns the cost incurred at state2 from state1 via action."""
        ...

    def h(self, state: State) -> float:
        """Returns the heuristic value, a lower bound on the distance to goal."""
        return 0


class POMDP(Problem):
    """A generative-model-based POMDP."""

    @property
    def discount(self) -> float:
        """The discount factor."""
        return 1.

    @property
    def horizon(self) -> int:
        """The planning horizon."""
        return np.inf

    @abstractmethod
    def terminal(self, state: State) -> bool:
        """If this state is terminating (absorbing state)."""
        return False

    @abstractmethod
    def reward(self, state1: State, action: Action, state2: State) -> float:
        """Returns the reward given at state2 from state1 via action."""
        ...

    @abstractmethod
    def get_observation(self, state: State) -> State:
        """Sample an observation from current state."""
        ...


class MDP(POMDP):
    """An generative-model-based MDP."""

    def get_observation(self, state: State) -> State:
        """An MDP is fully observable."""
        return state


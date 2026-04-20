# Setup matplotlib animation
import matplotlib
matplotlib.rc('animation', html='jshtml')


from typing import (Callable, Iterable, List, Sequence, Tuple, Dict, Optional,
                    Any, Union, Set, ClassVar, Type, TypeVar)

import collections
import functools
import dataclasses
import numpy as np

from functools import cached_property

import scipy.signal

from problem import *
from search import *
from utils import *

@dataclasses.dataclass(frozen=True, eq=True, order=True)
class PickupProblemState:
    robot_loc: Tuple[int, int]
    carried_patient: bool


OneWayBlock = collections.namedtuple('OneWayBlock', ['loc', 'dest'])


@dataclasses.dataclass(frozen=True)
class PickupProblem(PathCostProblem):
    """A simple deterministic pickup problem in a grid.

    The robot starts at some location and can move up, down, left, or right.
    There is a patient at some location.
    The goal is to pick up the patient and drop them off at the hospital.
    The robot can only move to a location if there is a path from the robot's
    current location to the destination that does not pass through any roadblocks.
    """
    grid_shape: Tuple[int, int]

    initial_robot_loc: Tuple[int, int] = (0, 0)

    patient_loc: Tuple[int, int] = (4, 0)  # initial location of the patient
    hospital_loc: Tuple[int, int] = (0, 3)

    one_ways: List[OneWayBlock] = dataclasses.field(default_factory=list)

    grid_act_to_delta: ClassVar = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1)
    }
    all_grid_actions: ClassVar = tuple(grid_act_to_delta.keys())

    @property
    def initial(self) -> State:
        return PickupProblemState(self.initial_robot_loc,
                                  self.initial_robot_loc == self.patient_loc)

    @cached_property
    def _one_way_set(self) -> Set[OneWayBlock]:
        """Set of one-way blocks. Helps with fast lookup."""
        return set(self.one_ways)

    def actions(self, state: PickupProblemState) -> Iterable[Action]:
        """Actions from the current state: move up, down, left, or right unless blocked."""
        (r, c) = state.robot_loc
        actions = []
        for act in self.all_grid_actions:
            dr, dc = self.grid_act_to_delta[act]
            new_r, new_c = r + dr, c + dc
            if (new_r in range(self.grid_shape[0]) and
                    new_c in range(self.grid_shape[1]) and
                    OneWayBlock(state.robot_loc,
                                (new_r, new_c)) not in self._one_way_set):
                actions.append(act)
        return actions

    def step(self, state: PickupProblemState,
             action: Action) -> PickupProblemState:
        """We automatically pick up patient if we're on that square."""
        (r, c) = state.robot_loc
        dr, dc = self.grid_act_to_delta[action]
        return PickupProblemState(
            (r + dr, c + dc),
            state.carried_patient or self.patient_loc == (r + dr, c + dc),
        )

    def step_cost(self, state1, action, state2) -> float:
        """Cost of taking an action in a state.

        Actually not used in this project, but we keep it here for completeness.
        """
        return 1.

    def goal_test(self, state: PickupProblemState) -> bool:
        """True if at hospital and holding patient."""
        return state.robot_loc == self.hospital_loc and state.carried_patient

    def render(self, state: PickupProblemState, ax=None):
        """Render the state as a grid."""
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.gca()

        heatmap(np.zeros(self.grid_shape),
                ax=ax,
                cmap="YlOrRd",
                vmin=0,
                vmax=1,
                origin="upper")

        # Render the robot
        robot = plt.Circle(state.robot_loc[::-1], 0.5, color='blue')
        ax.add_patch(robot)

        # Render the patient
        patient_loc = (state.robot_loc
                       if state.carried_patient else self.patient_loc)
        patient = plt.Circle(patient_loc[::-1], 0.3, color='orange')
        ax.add_patch(patient)

        # Render the hospital
        plt.plot([self.hospital_loc[1]], [self.hospital_loc[0]],
                 marker='P',
                 color='r',
                 markersize=15)

        # Render the walls and one-way doors
        one_ways = set(self._one_way_set)
        while one_ways:
            one_way = one_ways.pop()
            rev = OneWayBlock(one_way.dest, one_way.loc)
            if rev in one_ways:
                one_ways.remove(rev)
                src, dst = one_way.loc, one_way.dest
                if src < dst:
                    dst, src = src, dst
                src, dst = np.array(src), np.array(dst)
                mid_pt = (src + dst) / 2
                delta = dst - src
                wall = plt.Rectangle(mid_pt[::-1] - delta / 2 +
                                     delta[::-1] * 0.05,
                                     delta[0] if delta[0] != 0 else 0.1,
                                     delta[1] if delta[1] != 0 else 0.1,
                                     color='black')
                ax.add_patch(wall)
            else:
                src, dst = np.array(one_way.loc), np.array(one_way.dest)
                mid_pt = (src + dst) / 2
                delta = dst - src
                base = mid_pt + delta * 0.07
                length = -delta * 0.2
                arrow = plt.arrow(base[1],
                                  base[0],
                                  length[1],
                                  length[0],
                                  width=0.2,
                                  length_includes_head=True,
                                  head_length=0.2,
                                  color='black')
                ax.add_patch(arrow)


conv2D = functools.partial(scipy.signal.convolve2d,
                           mode='same',
                           boundary='fill',
                           fillvalue=0)

FireGridT = np.ndarray  # boolean array


@dataclasses.dataclass(frozen=True)
class FireProcess:
    """A probabilistic model for the evolution of fire in a grid.

    At time step $t$, the probability of a cell being on fire is weighted probability
    of the neighboring cell being on fire at time step $t-1$.
    """

    initial_fire_grid: np.ndarray

    fire_weights: np.ndarray = dataclasses.field(
        default_factory=lambda: np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]]))
    attenuation: float = 1.0

    rng: np.random.Generator = dataclasses.field(
        default_factory=np.random.default_rng)

    @cached_property
    def normalized_fire_weights(self) -> np.ndarray:
        return self.attenuation * self.fire_weights / np.sum(self.fire_weights)

    def dist(self, fire_grid: FireGridT) -> np.ndarray:
        """Given the fire grid at time t, return a new grid with marginal distributions of fire for t + 1."""
        next_fire_dist = conv2D(fire_grid, self.normalized_fire_weights)
        return np.clip(next_fire_dist, 0, 1)  # clip for numerical stability

    def sample(self, fire_grid: FireGridT) -> FireGridT:
        """Given the fire grid at time t, return a new grid that's a sample from the distribution."""
        return self.rng.binomial(1, self.dist(fire_grid)).astype(bool)

    def render(self, fire_grid: FireGridT, ax=None):
        heatmap(fire_grid, ax=ax, cmap="YlOrRd", vmin=0, vmax=1, origin="upper")


@dataclasses.dataclass(frozen=True, eq=True)
class FireMDPState(PickupProblemState):
    """A state in the fire MDP extends the pickup problem state with a fire grid."""

    fire_grid: np.ndarray

    # Below we implement __eq__ and __hash__ to make FireMDPState hashable.
    def __eq__(self, other):
        if not isinstance(other, FireMDPState):
            return False
        return (self.robot_loc == other.robot_loc and
                self.carried_patient == other.carried_patient and
                np.all(self.fire_grid == other.fire_grid))

    def __hash__(self):
        return hash((super().__hash__(), self.fire_grid.tobytes()))


T = TypeVar('T', bound='FireProblemCommon')


@dataclasses.dataclass(frozen=True)
class FireProblemCommon:
    """Common code for the fire MDP and POMDP problems."""

    pickup_problem: PickupProblem
    fire_process: FireProcess

    _horizon: int = np.inf
    _discount: float = 0.999

    step_reward: float = 0.

    burn_reward: float = -1
    goal_reward: float = 1.

    @property
    def horizon(self):
        return self._horizon

    @property
    def discount(self):
        return self._discount

    @property
    def initial_robot_loc(self):
        return self.pickup_problem.initial_robot_loc

    def robot_burned(self, state) -> bool:
        return bool(state.fire_grid[state.robot_loc])

    def succeeded(self, state) -> bool:
        return self.pickup_problem.goal_test(state)

    def reward(self, state1, action, state2) -> float:
        if self.robot_burned(state2):
            return self.burn_reward
        if self.succeeded(state2):
            return self.goal_reward
        return self.step_reward

    def terminal(self, state) -> bool:
        return (self.robot_burned(state) or self.succeeded(state))

    @property
    def grid_shape(self):
        return self.pickup_problem.grid_shape

    @classmethod
    def from_str(cls: Type[T],
                 env_s: str,
                 fire_process_kargs: Optional[Dict[str, Any]] = None,
                 **kwargs) -> T:
        """Create a problem from a grid string.

        Legend:
            . = empty
            F = fire
            < = one way block (can go left)
            > = one way block (can go right)
            ^ = one way block (can go up)
            v = one way block (can go down)
            X = wall (two way block)
            R = robot
            P = patient
            H = hospital
        Each line must be the same length, and starts and ends with a `|`.
        Each character is separated by a space.

        Warning:
            The user needs to make ssure that no cell location is a dead end (due to roadblocks)
            since our `terminal` condition does not check for empty available actions.
        """
        if fire_process_kargs is None:
            fire_process_kargs = {}

        lines = env_s.splitlines()
        assert all(len(line) == len(lines[0]) for line in lines)
        assert all(line[0] == '|' and line[-1] == '|' for line in lines)
        # remove the first and last character of each line
        lines = [line[1:-1] for line in lines]
        # split each line into a list of characters
        lines = [line[::2] for line in lines]
        w = len(lines[0]) // 2 + 1
        h = len(lines) // 2 + 1
        robot_loc = None
        patient_loc = None
        hospital_loc = None
        fire_grid = np.zeros((h, w), dtype=bool)
        one_ways = []
        for i, line in enumerate(lines):
            i2 = i // 2
            if i % 2 == 0:
                for j, c in enumerate(line):
                    j2 = j // 2
                    if j % 2 == 0:
                        if c == 'F':
                            fire_grid[i2, j2] = True
                        elif c == 'R':
                            robot_loc = (i2, j2)
                        elif c == 'P':
                            patient_loc = (i2, j2)
                        elif c == 'H':
                            hospital_loc = (i2, j2)
                        else:
                            assert c == '.'
                    else:
                        if c in '<X':
                            one_ways.append(OneWayBlock((i2, j2), (i2, j2 + 1)))
                        if c in '>X':
                            one_ways.append(OneWayBlock((i2, j2 + 1), (i2, j2)))
            if i % 2 == 1:
                for j, c in enumerate(line[::2]):
                    if c in 'vX':
                        one_ways.append(
                            OneWayBlock(loc=(i2 + 1, j), dest=(i2, j)))
                    if c in '^X':
                        one_ways.append(
                            OneWayBlock(loc=(i2, j), dest=(i2 + 1, j)))

        if robot_loc is None:
            raise ValueError("No robot location specified")
        if patient_loc is None:
            raise ValueError("No patient location specified")
        if hospital_loc is None:
            raise ValueError("No hospital location specified")

        pickup_problem = PickupProblem(fire_grid.shape, robot_loc, patient_loc,
                                       hospital_loc, one_ways)
        fire_process = FireProcess(fire_grid, **fire_process_kargs)
        return cls(pickup_problem, fire_process, **kwargs)


@dataclasses.dataclass(frozen=True)
class FireMDP(FireProblemCommon, MDP):
    """The completely observable fire problem."""

    @property
    def initial(self) -> FireMDPState:
        return FireMDPState(*dataclasses.astuple(self.pickup_problem.initial),
                            self.fire_process.initial_fire_grid)

    def actions(self, state: State) -> Iterable[Action]:
        return self.pickup_problem.actions(state)

    def step(self, state: FireMDPState, action: Action) -> FireMDPState:
        return FireMDPState(
            *dataclasses.astuple(self.pickup_problem.step(state, action)),
            self.fire_process.sample(state.fire_grid))

    def render(self, state: FireMDPState, ax=None):
        self.pickup_problem.render(state, ax=ax)
        self.fire_process.render(state.fire_grid, ax=ax)


@dataclasses.dataclass(frozen=True, eq=True, order=True)
class DeterminizedFireMDPState(PickupProblemState):
    """A state for the DeterminizedFireMDP.

    The state is a pair of the PickupProblemState and a time step $t$.
    """
    time: int = 0

@dataclasses.dataclass(frozen=True)
class DeterminizedFireMDP(PathCostProblem):
    """Determinized version of the fire MDP --- tries to find the solution path
    that is most likely to succeed.
    """
    pickup_problem: PickupProblem
    fire_process: FireProcess

    # Additional cost for each step.
    # Can be 0 but we might have 0-cost arcs if the success probability is 1.
    action_cost = 1e-6

    # Use this to cache precomputed fire distributions, so we don't have to recompute them.
    fire_dists_cache: Dict[int, np.ndarray] = dataclasses.field(
        init=False,
        default_factory=dict,
    )

    def __post_init__(self):
        assert (self.pickup_problem.grid_shape ==
                self.fire_process.initial_fire_grid.shape)

    @property
    def initial(self) -> DeterminizedFireMDPState:
        return DeterminizedFireMDPState(
            *dataclasses.astuple(self.pickup_problem.initial),
            time=0,
        )

    def actions(self, state: DeterminizedFireMDPState) -> Iterable[Action]:
        return self.pickup_problem.actions(state)

    def step(self, state: DeterminizedFireMDPState,
             action: Action) -> State:
        """We automatically pick up patient if we're on that square."""
        next_state = self.pickup_problem.step(state, action)
        return DeterminizedFireMDPState(next_state.robot_loc, next_state.carried_patient, state.time + 1)

    def goal_test(self, state: DeterminizedFireMDPState) -> bool:
        """True if at hospital and holding patient."""
        return self.pickup_problem.goal_test(state)

    def step_cost(self, state1: DeterminizedFireMDPState, action: Action,
                  state2: DeterminizedFireMDPState) -> float:
        """Induce a small action_cost and the negative log-likelihood of no fire."""
        return self.action_cost - np.log(1 - self.fire_dist_at_time(state2.time)[state2.robot_loc])

    def fire_dist_at_time(self, t: int) -> np.ndarray:
        """Return the marginal distribution of fire grid at time $t$."""
        if t not in self.fire_dists_cache:
            if t == 0:
                self.fire_dists_cache[t] = self.fire_process.initial_fire_grid.astype(float)
            else:
                # after working out the algebra on the simplest possible example, 
                # this seems to be true
                self.fire_dists_cache[t] = self.fire_process.dist(self.fire_dist_at_time(t - 1))

        return self.fire_dists_cache[t]

    @staticmethod
    def _manhattan_distance(loc1: Tuple[int, int], loc2: Tuple[int, int]) -> int:
        return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])

    def h(self, state: DeterminizedFireMDPState) -> float:
        """heuristic based on the manhattan distance to the patient and hospital."""
        r_loc = state.robot_loc
        h_loc = self.pickup_problem.hospital_loc
        p_loc = self.pickup_problem.patient_loc

        r_p_loc = self._manhattan_distance(r_loc, p_loc) if not state.carried_patient else 0
        return (r_p_loc + self._manhattan_distance(p_loc, h_loc)) * self.action_cost
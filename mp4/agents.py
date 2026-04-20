import dataclasses
from utils import *
from fire_problem import *
from search import *


class Agent:
    """An agent that can act in an MDP or POMDP.

    A derived agent must keep track of its own internal state.
    """

    def reset(self):
        """Reset the agent's internal state."""
        pass

    @abstractmethod
    def act(self, obs: Union[Observation, State]) -> Action:
        """Return the agent's action given an observation.
        For MDP agents, `obs` will be the complete state"""
        ...


@dataclasses.dataclass
class OpenLoopAgent(Agent):
    """Agent that just follows a fixed sequence of actions."""

    actions: Sequence[Action]

    t: int = dataclasses.field(default=0, init=False)

    def reset(self):
        self.t = 0

    def act(self, obs) -> Action:
        del obs  # observation is not used
        assert self.t < len(self.actions)
        a = self.actions[self.t]
        self.t += 1
        return a# self.actions


@dataclasses.dataclass(frozen=True)
class FireMDPDeterminizedAStarAgent(Agent):
    """Agent that uses A* to plan a path to the goal in a determinized
    version of the problem. Does not need any internal state since we
    re-determinize the problem at each step.
    """

    problem: FireMDP
    step_budget: int = 10000

    def determinized_problem(self,
                             state: FireMDPState) -> DeterminizedFireMDP:
        """Returns a determinized approximation of the fire MDP."""
        # state contains: fire grid, robot loc, carried patient
        # we need to modify the current problem with this as the initial conditions
        pp = self.problem.pickup_problem
        new_pickupp_problem = PickupProblem(pp.grid_shape, 
                                            state.robot_loc, 
                                            state.robot_loc if state.carried_patient else pp.patient_loc,
                                            pp.hospital_loc,
                                            pp.one_ways)
        
        fp = self.problem.fire_process
        new_fire_process = FireProcess(state.fire_grid, 
                                       fp.fire_weights,
                                       fp.attenuation,
                                       fp.rng)
        
        return DeterminizedFireMDP(new_pickupp_problem, new_fire_process)

    def act(self, state: FireMDPState) -> Action:
        problem = self.determinized_problem(state)
        try:
            plan = run_astar_search(problem, self.step_budget)
        except SearchFailed:
            print("Search failed, performing a random action")
            return random.choice(list(self.problem.actions(state)))
        return plan[1][0]



@dataclasses.dataclass
class MCTSAgent(Agent):
  '''Agent that uses Monte Carlo Tree Search to plan a path to the goal.

  The agent simply wraps `run_mcts_search`, and it should work for any MDP.
  '''

  problem: MDP

  # An optional receding horizon to use for the planning
  # If not provided, the problem must have a finite horizon
  receding_horizon: Optional[int] = 40

  C: float = np.sqrt(2)
  iteration_budget: int = 1000

  t: int = dataclasses.field(default=0, init=False)

  def __post_init__(self):
    if self.receding_horizon is None:
      assert self.problem.horizon != np.inf

  def reset(self):
    self.t = 0

  @property
  def planning_horizon(self) -> int:
    '''Returns the planning horizon for the current time step.'''
    if self.receding_horizon is None:
      return self.problem.horizon - self.t
    return self.receding_horizon

  def act(self, state: State) -> Action:
    '''Return the action to take at state.'''
    self.t += 1
    return run_mcts_search(self.problem, state, horizon=self.planning_horizon, C=self.C, iteration_budget=self.iteration_budget)

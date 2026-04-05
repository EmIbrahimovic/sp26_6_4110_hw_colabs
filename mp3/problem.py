
import numpy as np


import copy

from infer import infer_unknown_values


class State:
    """States have the following attributes:

    "robot": A (row, col) representing the robot's loc.
    "hospital": A (row, col) representing the hospital's loc.
    "carrying": The str name of a person being carried,
      or None, if no person is being carried.
    "people": A dict mapping str people names to (row, col)
      locs. If a person is being carried, they do not
      appear in this dict.
    "state_map": A numpy array of str 'C', 'F', 'S', and 'W',
      where 'C' represents free space, 'F' represents fire,
      'S' represents smoke, and 'W' represents an obstacle(wall).
      The robot may safely enter any cell that is clear (‘C’)
      or contains smoke (‘S’).
    """

    robot: tuple[int, int]
    hospital: tuple[int, int]
    carrying: tuple[int, int] | None
    people: dict[str, tuple[int, int]]
    state_map: np.ndarray

    def __init__(self,
                 robot=None,
                 hospital=None,
                 carrying=None,
                 people=None,
                 state_map=None):
        default_state_map = np.array([['C', 'C', 'C', 'C', 'C', 'C', 'C'],
                                      ['C', 'W', 'W', 'C', 'C', 'W', 'W'],
                                      ['C', 'C', 'C', 'C', 'C', 'C', 'C'],
                                      ['C', 'C', 'W', 'C', 'C', 'C', 'C'],
                                      ['C', 'C', 'W', 'C', 'W', 'C', 'C'],
                                      ['C', 'C', 'C', 'C', 'C', 'W', 'C'],
                                      ['C', 'W', 'C', 'C', 'W', 'C', 'C']],
                                     dtype=np.str_)
        default_robot = (0, 0)  # top left corner
        default_hospital = (6, 6)  # bottom right corner
        default_carrying = None
        default_people = {
            "p1": (4, 0),
            "p2": (6, 0),
            "p3": (0, 6),
            "p4": (3, 3)
        }
        self.state_map = state_map if state_map is not None else default_state_map
        self.robot = robot if robot is not None else default_robot
        self.hospital = hospital if hospital is not None else default_hospital
        self.carrying = carrying if carrying is not None else default_carrying
        self.people = people if people is not None else default_people

    def get_safe_grid(self):
        """
        "safe_grid": A grid map of boolean values where `True`
        indicate the locations where the robot are allowed to move into.

        Clear and Smoke grid cells are safe to enter
        """
        safe_grid = np.logical_or(self.state_map == "C", self.state_map == "S")
        return safe_grid

    def render(self, msg=None):
        height, width = self.state_map.shape
        state_arr = np.full((height, width), "  ", dtype=object)
        state_arr[self.state_map == 'W'] = "##"
        state_arr[self.state_map == 'F'] = "XX"
        state_arr[self.state_map == 'S'] = "||"
        state_arr[self.state_map == 'U'] = "??"
        state_arr[self.hospital] = "Ho"
        state_arr[self.robot] = "Ro"
        # Draw the people not at the hospital
        for person, loc in self.people.items():
            if loc == self.hospital:
                continue
            elif loc == self.robot:
                person = "R" + person[-1]
            state_arr[loc] = person
        # Add padding
        padded_state_arr = np.full((height + 2, width + 2), "##", dtype=object)
        padded_state_arr[1:-1, 1:-1] = state_arr
        state_arr = padded_state_arr
        carrying_str = f"Carrying: {self.carrying}"
        # Print
        if msg:
            print(msg)
        for row in state_arr:
            print(''.join(row))
        print(carrying_str)
        print()

    def copy(self):
        state_copy = copy.copy(self)
        state_copy.state_map = self.state_map.copy()  # copy the numpy array
        state_copy.people = self.people.copy()
        return state_copy



class SearchAndRescueProblem:
    """Defines a search and rescue (SAR) problem.

    In search and rescue, a robot must navigate to, pick up, and
    drop off people that are in need of help.

    Actions are strs. The following actions are defined:
      "up" / "down" / "left" / "right" : Moves the robot. The
        robot cannot move into obstacles or off the map.
      "pickup-{person}": If the robot is at the person, and if
        the robot is not already carrying someone, picks them up.
      "dropoff": If the robot is carrying a person, they are
        dropped off at the robot's current location.
      "look...": later we'll allow these actions, but they
        have no effect on the state.

    This structure serves as a container for a transition model
    "get_next_state(state, action)", an observaton model "get_observation(state)"
    and an action model "get_legal_actions(state)"

    Example usage:
      problem = SearchAndRescueProblem()
      state = State()
      state.render()
      action = "down"
      next_state = problem.get_next_state(state, action)[0]
      next_state.render()
    """

    def __init__(self):
        self.action_deltas = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1),
        }

    @staticmethod
    def is_valid_location(loc_r, loc_c, state, verbose=False):
        if not (0 <= loc_r < state.state_map.shape[0] and
                0 <= loc_c < state.state_map.shape[1]):
            if verbose:
                print(
                    "WARNING: attempted to move out of bounds, action has no effect."
                )
            return False
        if not state.get_safe_grid()[loc_r, loc_c]:
            if verbose:
                print(
                    "WARNING: attempted to move into an obstacle/unsafe region, action has no effect."
                )
            return False
        return True

    @staticmethod
    def get_legal_actions(state):
        legal_actions = ["up", "down", "left", "right", "dropoff"]
        for person in state.people:
            legal_actions.append(f"pickup-{person}")
        return legal_actions

    def get_next_state(self, state, action, verbose=False) -> tuple[State, bool]:
        legal_actions = self.get_legal_actions(state)
        if action not in legal_actions and not action.startswith('look'):
            raise ValueError(
                f"Unrecognized action {action}. Actions must be one of: {legal_actions}"
            )

        if action in ["up", "down", "left", "right"]:
            dr, dc = self.action_deltas[action]
            r, c = state.robot
            if not self.is_valid_location(
                    r + dr, c + dc, state, verbose=verbose):
                if verbose:
                    print(f"Action {action} is invalid in {state}.")
                return state, False
            new_state = state.copy()
            new_state.robot = (r + dr, c + dc)
            return new_state, True

        elif action.startswith("pickup"):
            person = action.split("-")[1]
            if state.carrying is not None:
                if verbose:
                    print(
                        "WARNING: attempted to pick up a person while already carrying someone, action has no effect."
                    )
                return state, False
            if person not in state.people or (state.people[person] !=
                                              state.robot):
                if verbose:
                    print(
                        "WARNING: attempted to pick up a person not at the robot location, action has no effect."
                    )
                return state, False
            new_state = state.copy()
            del new_state.people[person]
            new_state.carrying = person
            return new_state, True

        elif action == "dropoff":
            if state.carrying is None:
                if verbose:
                    print(
                        "WARNING: attempted to dropoff while not carrying anyone, action has no effect."
                    )
                return state, False
            person = state.carrying
            new_state = state.copy()
            new_state.carrying = None
            new_state.people[person] = state.robot
            return new_state, True

        elif action.startswith('look'):
            return state, True

        else:
            raise KeyError

    def get_observation(self, state):
        """Return the states of the adjacent (non-wall) grid squares."""
        height, width = state.state_map.shape
        deltas = self.action_deltas
        r, c = state.robot
        observation = {(r, c): state.state_map[r, c]}
        for direction, (dr, dc) in deltas.items():
            nr = r + dr
            nc = c + dc
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            if state.state_map[nr, nc] == "W":
                continue
            observation[(nr, nc)] = state.state_map[nr, nc]
        return observation
    


class BeliefState(State):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "state_map" not in kwargs:
            self.state_map = np.array(
                [
                    ["U", "U", "U", "U", "U", "U", "U"],
                    ["U", "W", "W", "U", "U", "W", "W"],
                    ["U", "U", "U", "U", "U", "U", "U"],
                    ["U", "U", "W", "U", "U", "U", "U"],
                    ["U", "U", "W", "U", "W", "U", "U"],
                    ["U", "U", "U", "U", "U", "W", "U"],
                    ["U", "W", "U", "U", "W", "U", "U"],
                ],
                dtype=np.str_,
            )

    def update(self, problem: SearchAndRescueProblem, obs, action=None):
        """
        problem: SearchAndRescueProblem instance
        obs: {loc: entry, loc: entry,...}
        act: string or None

        # <<< TODO: >>>
            1. Do transition from action (if any)
            2. Update from observation
            3. Do inference (infer_unknown_values(grid))
        """
        if action is not None:
            next_state = problem.get_next_state(self, action)[0]
        else:
            next_state = self.copy()

        # update state map with observations
        for (r, c), value in obs.items():
            next_state.state_map[r][c] = value
        
        # perform inference with the new information
        next_state.state_map = np.array(infer_unknown_values(next_state.state_map))
        return next_state

    def get_optimistic_state(self):
        """Returns a copy of the belief with a completed map in which Unknowns
        are assumed to be Clear."""
        new_state = self.copy()
        new_state.state_map[self.state_map == "U"] = "C"
        return new_state

    def get_careful_state(self):
        """Returns a copy of the belief.

        Unknown states will not be treated as safe, see get_safe_grid.
        """
        return self.copy()


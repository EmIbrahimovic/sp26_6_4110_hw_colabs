from constraint import *
import typing
import copy
import pprint

def get_varname(i: int, j: int) -> str:
    return '_'.join(['a', str(i), str(j)])

def get_indices_from_varname(varname: str) -> tuple[int, int]:
    var_split = varname.split('_')
    return int(var_split[1]), int(var_split[2])

def get_neighbors(i: int, j: int, n: int, m: int) -> list[str]:
    d1 = [-1, 0,  0, 1]
    d2 = [ 0, 1, -1, 0]
    neighs = []
    for dd1, dd2 in zip(d1, d2):
        if 0 <= i + dd1 < n and 0 <= j + dd2 < m:
            neighs.append(get_varname(i + dd1, j + dd2))
    
    return neighs

def generate_observation(cell_value: str):
    if cell_value == 'U':
        raise ValueError("Unknown is not an observation")
    
    def _condition(var: str):
        return var == cell_value
    
    return _condition

def generate_fire_condition():
    def _condition(var: str, neighbor: str):
        return (var == 'F' and neighbor in ['F', 'S']) or var != 'F'
    
    return _condition


def generate_smoke_condition():
    
    def _condition(var: str, *neighbors: str):
        return (var == 'S' and 'F' in neighbors) or var != 'S'
    
    return _condition

def solve_available(board: list[list[str]]) -> list[dict]:
    if not isinstance(board, list):
        raise ValueError("Board must be a list of lists of strings")
    if not isinstance(board[0], list):
        raise ValueError("Board must be a list of lists of strings")
    if len(board[0]) > 0 and not isinstance(board[0][0], str):
        raise ValueError("Board must be a list of lists of strings")

    print(board)
    n = len(board)
    m = len(board[0])
    vars = [get_varname(i, j) for i in range(n) for j in range(m)]
    # print(vars)

    problem = Problem()
    problem.addVariables(
        vars,
        domain=['F', 'S', 'C']
    )

    board = copy.deepcopy(board)
    for i in range(n):
        for j in range(m):
            curr = get_varname(i, j)
            cell = copy.copy(board[i][j])

            if cell != 'U':
                problem.addConstraint(generate_observation(cell), [curr])
            
            neighbors = get_neighbors(i, j, n, m)

            # Fire condition
            if cell in ['U', 'F']:
                for neigh in neighbors:
                    problem.addConstraint(generate_fire_condition(), [curr, neigh])
            
            # # Smoke condition
            if cell in ['U', 'S'] and len(neighbors) > 0:
                problem.addConstraint(generate_smoke_condition(), [curr, *neighbors])

    allSols = problem.getSolutions()

    return allSols

def get_board_from_sols(sols: list[dict], n: int, m: int):
    board = []

    for i in range(n):
        row = []
        for j in range(m):
            varname = get_varname(i, j)
            final_value = 'U'

            for d in sols:
                possible_value = d[varname]
                if final_value != 'U' and possible_value != final_value:
                    final_value = 'U'
                    break
                else:
                    final_value = possible_value
            
            row.append(final_value)
        board.append(row)
    
    return board

def sol_dict_to_board(sol: dict, n: int, m: int):
    board = []

    for i in range(n):
        row = []
        for j in range(m):
            varname = get_varname(i, j)
            row.append(sol[varname])
        board.append(row)
    
    return board


def print_board(board: list[list[str]]):
    
    print("The board is: ")
    for i in range(len(board)):
        print(board[i])
    
    print()


def print_sol(sol: dict, n: int, m: int):
    board = sol_dict_to_board(sol, n, m)

    print_board(board)


if __name__ == "__main__":
    grid = [
        ['C', 'C', 'C', 'C', 'C', 'C', 'C'],
        ['C', 'C', 'S', 'U', 'S', 'C', 'C'],
        ['C', 'U', 'F', 'U', 'F', 'U', 'C'],
        ['C', 'U', 'U', 'U', 'S', 'C', 'C'],
        ['C', 'C', 'U', 'U', 'C', 'C', 'C'],
        ['C', 'U', 'C', 'U', 'U', 'C', 'C'],
        ['C', 'C', 'C', 'C', 'C', 'C', 'C']
    ]

    tests = [
        # [['F']],
        # [['C']],
        # [['S']],
        # [['F', 'U']],
        # [['U', 'C']],
        # [['U', 'S']],
        # [['F', 'S', 'U']],
        # [['F', 'U', 'S']],
        # [['U']],
        # [['F', 'U', 'U', 'C']],
        grid
    ]

    for t, test in enumerate(tests):
        print(f"\t ======== Test {t+1}  ==========")
        print_board(test)
        solutions = solve_available(test)
        sol_board = get_board_from_sols(solutions, len(test), len(test[0]))

        print("== Final solution")
        print_board(sol_board)

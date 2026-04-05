
from ortools.sat.python import cp_model

def infer_unknown_values(grid):
    R, C = len(grid), len(grid[0])

    def get_neighbors(r, c):
        return [
            (r + dr, c + dc)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            if 0 <= r + dr < R and 0 <= c + dc < C and grid[r + dr][c + dc] != "W"
        ]

    def build_base_model():
        model = cp_model.CpModel()
        is_F, is_S, is_C = {}, {}, {}

        for r in range(R):
            for c in range(C):
                if grid[r][c] == "W":
                    continue

                is_F[(r, c)] = model.new_bool_var(f"F_{r}_{c}")
                is_S[(r, c)] = model.new_bool_var(f"S_{r}_{c}")
                is_C[(r, c)] = model.new_bool_var(f"C_{r}_{c}")

                # Cell has to be exactly one of F, S, or C
                model.add_exactly_one([is_F[(r, c)], is_S[(r, c)], is_C[(r, c)]])

                # Add known values from given grid
                val = grid[r][c]
                if val == "F":
                    model.add(is_F[(r, c)] == 1)
                elif val == "S":
                    model.add(is_S[(r, c)] == 1)
                elif val == "C":
                    model.add(is_C[(r, c)] == 1)

        for r in range(R):
            for c in range(C):
                if grid[r][c] == "W":
                    continue
                nbrs = get_neighbors(r, c)

                # if fire, no neighbor can be clear
                for nr, nc in nbrs:
                    model.add(is_C[(nr, nc)] == 0).only_enforce_if(is_F[(r, c)])

                # if clear, no neighbor can be fire
                for nr, nc in nbrs:
                    model.add(is_F[(nr, nc)] == 0).only_enforce_if(is_C[(r, c)])

                # if smoke, at least one neighbor is fire
                if nbrs:
                    model.add_bool_or(
                        [is_F[(nr, nc)] for nr, nc in nbrs]
                    ).only_enforce_if(is_S[(r, c)])
                else:
                    model.add(is_S[(r, c)] == 0)

        return model, is_F, is_S, is_C

    def is_feasible(model):
        solver = cp_model.CpSolver()
        solver.parameters.log_search_progress = False
        status = solver.solve(model)
        return status in (cp_model.FEASIBLE, cp_model.OPTIMAL)

    inferred = [row[:] for row in grid]

    for r in range(R):
        for c in range(C):
            if grid[r][c] != "U":
                continue

            # Check each label by adding one extra pinning constraint
            feasible_labels = []
            for label, get_var in [
                ("F", lambda f, s, c: f),
                ("S", lambda f, s, c: s),
                ("C", lambda f, s, c: c),
            ]:
                model, is_F, is_S, is_C = build_base_model()
                var = get_var(is_F, is_S, is_C)[(r, c)]
                model.add(var == 1)  # Add constraint
                if is_feasible(model):
                    feasible_labels.append(label)

            # The cell is determined iff exactly one label is feasible
            if len(feasible_labels) == 1:
                inferred[r][c] = feasible_labels[0]

    return inferred


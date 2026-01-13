import numpy as np
import matplotlib.pyplot as plt

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize



# Demand (thermal demand only; excess elec = 0)
HEAT_DEMAND = 50.0

# Conversion factors (input -> output) 
CF_CHPH = 0.52   # NG input -> heat output
CF_LHP  = 3.0    # Elec input -> heat output (COP)

# Costs (source->tech input costs) 
COST_NG_TO_CHP = 0.15
COST_EL_TO_LHP = 0.45

# Emission factors on inputs
FCO2_NG = 240.0
FCO2_EL = 400.0

# Bounds 
XL = np.array([0.0, 0.0])        # [LHP_cap, CHP_cap]
XU = np.array([300.0, 500.0])


def cost_optimal_dispatch(LHP_cap: float, CHP_cap: float):
    

    # Use maximum LHP heat possible 
    z = min(LHP_cap, HEAT_DEMAND)          # LHP heat out
    y = HEAT_DEMAND - z                    # CHPH heat out

    # feasibility check: CHPH must be able to supply the remainder
    if y > CHP_cap + 1e-12:
        return None  # infeasible

    # compute inputs
    ng_in = y / CF_CHPH if y > 0 else 0.0
    el_in = z / CF_LHP  if z > 0 else 0.0

    cost = COST_NG_TO_CHP * ng_in + COST_EL_TO_LHP * el_in
    emissions = FCO2_NG * ng_in + FCO2_EL * el_in

    return cost, emissions, y, z, ng_in, el_in


class NativeHub2Obj(ElementwiseProblem):
    def __init__(self):
        super().__init__(
            n_var=2,
            n_obj=2,
            n_constr=0,          
            xl=XL,
            xu=XU,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        LHP_cap = float(x[0])
        CHP_cap = float(x[1])

        res = cost_optimal_dispatch(LHP_cap, CHP_cap)

        if res is None:
            # infeasible -> penalty 
            out["F"] = [1e12, 1e12]
            return

        cost, emissions, y, z, ng_in, el_in = res
        out["F"] = [cost, emissions]


if __name__ == "__main__":
    problem = NativeHub2Obj()

    algorithm = NSGA2(pop_size=40)
    termination = get_termination("n_gen", 60)

    res = minimize(problem, algorithm, termination, seed=42, verbose=True)

    if res.F is None:
        print("No feasible solutions found.")
    else:
        # Pareto plot
        plt.figure(figsize=(25, 20))
        sns.set(style='ticks', font_scale=2)
        plt.scatter(res.F[:, 0], res.F[:, 1])
        plt.xlabel("Total Cost")
        plt.ylabel("Emissions")
        plt.title("Pareto front (native pymoo)")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        # highlight best-by-cost point
        i_best = np.argmin(res.F[:, 0])
        plt.scatter([res.F[i_best, 0]], [res.F[i_best, 1]], s=120, marker="x")
        plt.show()

        print("Best-by-cost X [LHP_cap, CHP_cap]:", res.X[i_best])
        print("Best-by-cost F [cost, emissions]:", res.F[i_best])

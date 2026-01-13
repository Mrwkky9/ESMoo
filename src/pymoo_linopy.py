import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
import linopy

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize


FCO2_NGAS = 240.0  # g/kWh
FCO2_ELEC = 400.0  # g/kWh


def solve_linopy(LHP_cap, CHP_cap):
    m = linopy.Model()

    g = {"Sources": ["ng grid", "elec grid"]}
    i = {"Technologies": ["CHPE", "CHPH", "LHP"]}
    j = {"Sinks": ["excess elec", "thermal demand"]}

    Lg = xr.DataArray([3500, 6000], coords=g)
    
    Li = xr.DataArray([CHP_cap, CHP_cap, LHP_cap], coords=i)

    Dj = xr.DataArray([0, 50], coords=j)
    CF = xr.DataArray([0.37, 0.52, 3.0], coords=i)

    cgi = xr.DataArray([[0.15, 0.15, 0.0],
                        [0.0,  0.0,  0.45]],
                       coords=g | i)

    cij = xr.DataArray(np.zeros((3, 2)), coords=i | j)

    mask_xgi = np.ones([2, 3], dtype=bool)
    mask_xgi[0, 2] = False
    mask_xgi[1, 0] = False
    mask_xgi[1, 1] = False

    mask_xij = np.ones([3, 2], dtype=bool)
    mask_xij[0, 1] = False
    mask_xij[1, 0] = False
    mask_xij[2, 0] = False

    xgi = m.add_variables(lower=0.0, coords=cgi.coords, name="xgi", mask=mask_xgi)
    xij = m.add_variables(lower=0.0, coords=cij.coords, name="xij", mask=mask_xij)

    m.add_constraints(xgi.sum(dim="Sources") * CF == xij.sum(dim="Sinks"))
    m.add_constraints(xij.sum(dim="Sinks") <= Li)
    m.add_constraints(xgi.sum(dim="Technologies") <= Lg)
    m.add_constraints(xij.sum(dim="Technologies") == Dj)

    m.add_objective((cgi * xgi).sum())
    m.solve()

    cost = float(m.objective.value)

    # emissions from xgi by source
    xgi_sol = xgi.solution.fillna(0.0)
    ng_use = float(xgi_sol.loc[{"Sources": "ng grid"}].sum().values)
    el_use = float(xgi_sol.loc[{"Sources": "elec grid"}].sum().values)
    emissions = FCO2_NGAS * ng_use + FCO2_ELEC * el_use

    return cost, emissions


class Problem(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=2, n_obj=2, n_constr=0,
                         xl=np.array([0.0, 0.0]),
                         xu=np.array([300.0, 500.0]))

    def _evaluate(self, x, out, *args, **kwargs):
        LHP_cap, CHP_cap = float(x[0]), float(x[1])
        try:
            cost, emissions = solve_linopy(LHP_cap, CHP_cap)
            out["F"] = [cost, emissions]
        except Exception:
            out["F"] = [1e12, 1e12]


if __name__ == "__main__":
    problem = Problem()
    algorithm = NSGA2(pop_size=30)
    termination = get_termination("n_gen", 25)

    res = minimize(problem, algorithm, termination, seed=42, verbose=True)

    plt.figure(figsize=(25, 20))
    sns.set(style='ticks', font_scale=2)
    plt.scatter(res.F[:, 0], res.F[:, 1])
    plt.xlabel("Total Cost")
    plt.ylabel("Emissions")
    plt.title("Pareto front (linopy solve + reassess emissions)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
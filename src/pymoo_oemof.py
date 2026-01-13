import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from oemof import solph
from oemof.tools import economics
from pyomo.environ import SolverFactory

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize


FCO2_NGAS = 240.0  # g/kWh (example)


def build_model(LHP_max, CHP_max, periods=2):
    cost_ngas = 150 / 1000
    cost_el = 450 / 1000
    chp_eta_el = 0.37
    chp_eta_th = 0.52

    epc_hp = economics.annuity(capex=1500, n=15, wacc=0.05)
    epc_chp = economics.annuity(capex=1550, n=15, wacc=0.05)

    idx = pd.date_range("2025-01-01", periods=periods, freq="h")
    es = solph.EnergySystem(timeindex=idx, infer_last_interval=True)

    bgas = solph.Bus(label="bus_gas")
    bel = solph.Bus(label="bus_elec")
    bel2 = solph.Bus(label="bus_elec2")
    bth = solph.Bus(label="bus_heat")

    ng_grid = solph.components.Source("ng_grid", outputs={bgas: solph.Flow(variable_costs=cost_ngas)})
    e_grid  = solph.components.Source("e_grid",  outputs={bel:  solph.Flow(variable_costs=cost_el)})

    th_demand = solph.components.Sink(
        "th_demand",
        inputs={bth: solph.Flow(fix=500, nominal_value=1)}   # 500 heat per hour
    )
    el_excess = solph.components.Sink("el_excess", inputs={bel2: solph.Flow()})

    
    LHP = solph.components.Converter(
        label="LHP",
        inputs={bel: solph.Flow()},
        outputs={
            bth: solph.Flow(
                investment=solph.Investment(ep_costs=epc_hp, maximum=LHP_max),
                max=1.0,
            )
        },
        conversion_factors={bel: 1/3},
    )

    CHP = solph.components.Converter(
        label="CHP",
        inputs={bgas: solph.Flow()},
        outputs={
            bel2: solph.Flow(
                investment=solph.Investment(ep_costs=epc_chp, maximum=CHP_max),
                max=1.0,
            ),
            bth: solph.Flow(),
        },
        conversion_factors={bel2: chp_eta_el, bth: chp_eta_th},
    )

    es.add(bgas, bel, bel2, bth, ng_grid, e_grid, th_demand, el_excess, LHP, CHP)
    om = solph.Model(es)
    return om


def solve_and_reassess(om):
    solver = SolverFactory("cbc")

    # KEY: don't load infeasible results into model
    res = solver.solve(om, tee=False, load_solutions=False)
    term = str(res.solver.termination_condition).lower()

    if term not in ("optimal", "feasible"):
        # infeasible sample -> signal to caller
        raise RuntimeError(f"Solver termination: {term}")

    # only now load solution
    om.solutions.load_from(res)

    cost = float(om.objective())

    # reassess emissions from gas input (bus_gas -> CHP)
    results = solph.processing.results(om)
    gas_seq = solph.views.node(results, "bus_gas")["sequences"]

    gas_to_chp = 0.0
    for col in gas_seq.columns:
        if "CHP" in str(col):
            gas_to_chp += float(gas_seq[col].sum())

    emissions = FCO2_NGAS * gas_to_chp
    return cost, emissions


class Problem(ElementwiseProblem):
    def __init__(self):
        super().__init__(
            n_var=2, n_obj=2, n_constr=0,
            xl=np.array([0.0, 0.0]),
            xu=np.array([300.0, 500.0]),
        )

    def _evaluate(self, x, out, *args, **kwargs):
        LHP_max, CHP_max = float(x[0]), float(x[1])

        try:
            om = build_model(LHP_max, CHP_max, periods=2)
            cost, emissions = solve_and_reassess(om)
            out["F"] = [cost, emissions]
        except Exception:
            # Penalty for infeasible / solver failure
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
    plt.title("Pareto front (oemof solve + reassess emissions)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
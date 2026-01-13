import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import xarray as xr
import linopy

from oemof import solph


# -------------------------
# LINOPY 
# -------------------------
def solve_linopy():
    m = linopy.Model()

    g = {"Sources": ["ng grid", "elec grid"]}
    i = {"Technologies": ["CHPE", "CHPH", "LHP"]}
    j = {"Sinks": ["excess elec", "thermal demand"]}

    Lg = xr.DataArray([3500, 6000], coords=g, name="Nominal flow from source g")
    Li = xr.DataArray([50, 50, 30], coords=i, name="Nominal load of technology i")
    Dj = xr.DataArray([0, 50], coords=j, name="Demand at sink j")
    CF = xr.DataArray([0.37, 0.52, 3], coords=i, name="Conversion factor of technology i")

    cgi = xr.DataArray(
        [[0.15, 0.15, 0.0], [0.0, 0.0, 0.45]],
        coords=g | i,
        name="carrier cost in euro",
    )

    cij = xr.DataArray(
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        coords=i | j,
        name="loads cost in euro",
    )

    mask = np.ones([2, 3], dtype=bool)
    mask[0, 2] = False
    mask[1, 0] = False
    mask[1, 1] = False

    masky = np.ones([3, 2], dtype=bool)
    masky[0, 1] = False
    masky[1, 0] = False
    masky[2, 0] = False

    xgi = m.add_variables(lower=0.0, coords=cgi.coords, name="Loads from source", mask=mask)
    xij = m.add_variables(lower=0.0, coords=cij.coords, name="Energy generated", mask=masky)

    m.add_constraints(xgi.sum(dim="Sources") * CF == xij.sum(dim="Sinks"), name="converter balance")
    m.add_constraints(xij.sum(dim="Sinks") <= Li, name="Converter maximal load")
    m.add_constraints(xgi.sum(dim="Technologies") <= Lg, name="Source nominal capacity")
    m.add_constraints(xij.sum(dim="Technologies") == Dj, name="Demand fulfilment")

    m.add_objective((cgi * xgi).sum())
    m.solve()

    return float(m.objective.value), xgi.solution, xij.solution


# -------------------------
# OEMOF (match linopy structure 1:1)
# - Split CHP into CHPE (gas->elec2) and CHPH (gas->heat)
# - LHP uses elec->heat with COP=3
# -------------------------
def solve_oemof():
    cost_ngas = 150 / 1000
    cost_el = 450 / 1000

    periods = 1
    idx = pd.date_range("2025-01-01", periods=periods, freq="h")
    es = solph.EnergySystem(timeindex=idx, infer_last_interval=True)

    bgas = solph.Bus(label="bus_gas")
    bel = solph.Bus(label="bus_elec")
    bel2 = solph.Bus(label="bus_elec2")      # sink for "excess elec"
    bth = solph.Bus(label="bus_heat")        # sink for "thermal demand"

    ng_grid = solph.components.Source(
        label="ng grid", outputs={bgas: solph.Flow(variable_costs=cost_ngas)}
    )
    e_grid = solph.components.Source(
        label="elec grid", outputs={bel: solph.Flow(variable_costs=cost_el)}
    )

    thermal_demand = solph.components.Sink(
        label="thermal demand",
        inputs={bth: solph.Flow(fix=50, nominal_value=1)}
    )
    excess_elec = solph.components.Sink(
        label="excess elec",
        inputs={bel2: solph.Flow()}
    )

    # LHP: elec -> heat, COP=3 (heat = elec*3)
    LHP = solph.components.Converter(
        label="LHP",
        inputs={bel: solph.Flow()},
        outputs={bth: solph.Flow(nominal_value=30)},
        conversion_factors={bth: 3.0},
    )

    # CHPE: gas -> elec2 (eta_el=0.37)
    CHPE = solph.components.Converter(
        label="CHPE",
        inputs={bgas: solph.Flow()},
        outputs={bel2: solph.Flow(nominal_value=50)},
        conversion_factors={bel2: 0.37},
    )

    # CHPH: gas -> heat (eta_th=0.52)
    CHPH = solph.components.Converter(
        label="CHPH",
        inputs={bgas: solph.Flow()},
        outputs={bth: solph.Flow(nominal_value=50)},
        conversion_factors={bth: 0.52},
    )

    es.add(bgas, bel, bel2, bth, ng_grid, e_grid, thermal_demand, excess_elec, LHP, CHPE, CHPH)

    om = solph.Model(es)
    om.solve(solver="cbc", solve_kwargs={"tee": False})
    results = solph.processing.results(om)

    # extract key flows for comparison
    def sum_flow(bus_label, contains_text):
        seq = solph.views.node(results, bus_label)["sequences"]
        s = 0.0
        for col in seq.columns:
            if contains_text in str(col):
                s += float(seq[col].sum())
        return s

    flows = {
        "ng->CHPE": sum_flow("bus_gas", "CHPE"),
        "ng->CHPH": sum_flow("bus_gas", "CHPH"),
        "elec->LHP": sum_flow("bus_elec", "LHP"),
        "CHPE->excess elec": sum_flow("bus_elec2", "excess elec"),
        "CHPH->thermal": sum_flow("bus_heat", "thermal demand"),
        "LHP->thermal": sum_flow("bus_heat", "thermal demand"),  # both contribute; we split below
    }

    # split thermal contributions by looking at inflows on heat bus (columns contain converter labels)
    heat_seq = solph.views.node(results, "bus_heat")["sequences"]
    th_from_chph = 0.0
    th_from_lhp = 0.0
    for col in heat_seq.columns:
        if "CHPH" in str(col):
            th_from_chph += float(heat_seq[col].sum())
        if "LHP" in str(col):
            th_from_lhp += float(heat_seq[col].sum())
    flows["CHPH->thermal"] = th_from_chph
    flows["LHP->thermal"] = th_from_lhp

    return float(om.objective()), flows


# -------------------------
# MAIN + plots
# -------------------------
if __name__ == "__main__":
    lin_obj, xgi, xij = solve_linopy()
    oem_obj, oem_flows = solve_oemof()

    # Build comparable linopy flows
    lin_flows = {
        "ng->CHPE": float(xgi.loc[{"Sources": "ng grid", "Technologies": "CHPE"}].values),
        "ng->CHPH": float(xgi.loc[{"Sources": "ng grid", "Technologies": "CHPH"}].values),
        "elec->LHP": float(xgi.loc[{"Sources": "elec grid", "Technologies": "LHP"}].values),
        "CHPE->excess elec": float(xij.loc[{"Technologies": "CHPE", "Sinks": "excess elec"}].values),
        "CHPH->thermal": float(xij.loc[{"Technologies": "CHPH", "Sinks": "thermal demand"}].values),
        "LHP->thermal": float(xij.loc[{"Technologies": "LHP", "Sinks": "thermal demand"}].values),
    }

    print("\nLINOPY objective:", lin_obj)
    print("OEMOF objective :", oem_obj)
    print("\nLINOPY flows:", lin_flows)
    print("OEMOF flows :", oem_flows)

    # --- Plot objectives ---
    plt.figure(figsize=(25, 20))
    sns.set(style='ticks', font_scale=2)
    plt.bar(["linopy", "oemof"], [lin_obj, oem_obj])
    plt.title("Objective comparison (total cost)")
    plt.ylabel("Cost")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    # --- Plot flows ---
    labels = list(lin_flows.keys())
    lin_vals = [lin_flows[k] for k in labels]
    oem_vals = [oem_flows[k] for k in labels]

    x = np.arange(len(labels))
    w = 0.38

    plt.figure(figsize=(25, 20))
    sns.set(style='ticks', font_scale=2)
    plt.bar(x - w/2, lin_vals, width=w, label="linopy")
    plt.bar(x + w/2, oem_vals, width=w, label="oemof")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.title("Flow comparison (mapped 1:1)")
    plt.ylabel("Flow")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.show()
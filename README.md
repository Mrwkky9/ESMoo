# Multiobjective Energy System Optimization  
## A Comparison of Modelling Approaches and Tools

## Overview

Energy system optimisation problems are traditionally formulated with cost
minimisation as the primary objective. However, modern energy system planning
requires the consideration of multiple, often conflicting objectives, such as
emissions reduction and system performance.

This project investigates different modelling approaches for developing
multiobjective energy system optimisation models using open-source tools.
The focus is on **model formulation and integration with multiobjective
optimisation**, rather than on solver benchmarking.

Three approaches are examined:

1. Component-based modelling using **oemof.solph**
2. Algebraic linear programming using **linopy**
3. Direct mathematical formulation within **pymoo**

All approaches are evaluated using a common reference energy system.

---

## Reference Energy System

A simplified **energy hub** is used to ensure transparency and comparability.

**Energy sources**
- Natural gas grid
- Electricity grid

**Conversion technologies**
- Combined Heat and Power (CHP)
- Heat pump (HP)

**Energy sinks**
- Thermal demand (fixed)
- Electricity excess (slack sink)

The system is evaluated for a **single time step** representing steady-state
operation.

---

## Optimisation Objectives

The optimisation problem considers two conflicting objectives:

1. Minimisation of total system cost  
2. Minimisation of total CO₂ emissions  

The trade-offs between economic and environmental performance are analysed using
Pareto-optimal solutions.

---

## Key Findings

- All modelling approaches are capable of generating feasible solutions.
- Clear cost–emissions trade-offs are observed in all cases.
- The modelling effort and integration complexity differ significantly between
  approaches.

A visual comparison of the approaches is provided in the full methodology
document.

---

## Conclusion (Summary)

- **oemof.solph** offers intuitive, component-based energy system modelling but
  requires additional effort for multiobjective integration.
- **linopy** provides a transparent and flexible algebraic formulation suitable
  for algorithmic experimentation.
- **Direct pymoo formulation** offers maximum control and computational
  efficiency but requires careful mathematical formulation.

The choice of modelling approach depends on the balance between modelling
flexibility, transparency, and computational efficiency.

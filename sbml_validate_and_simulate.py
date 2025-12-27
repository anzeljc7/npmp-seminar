from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Union
import os
import numpy as np

# Optional dependencies
try:
    import libsbml
except Exception:
    libsbml = None

try:
    import tellurium as te
except Exception:
    te = None


@dataclass
class SBMLValidationResult:
    ok: bool
    errors: List[str]
    warnings: List[str]


@dataclass
class SimulationResult:
    ok: bool
    message: str
    time: Optional[np.ndarray] = None
    data: Optional[np.ndarray] = None
    columns: Optional[List[str]] = None
    final_values: Optional[Dict[str, float]] = None


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ---------------- VALIDATION ----------------

def validate_sbml(sbml: str) -> SBMLValidationResult:
    if libsbml is None:
        return SBMLValidationResult(
            ok=False,
            errors=["python-libsbml not installed. Install with: pip install python-libsbml"],
            warnings=[]
        )

    errors: List[str] = []
    warnings: List[str] = []

    reader = libsbml.SBMLReader()
    doc = reader.readSBMLFromString(sbml)

    # basic parsing
    for i in range(doc.getNumErrors()):
        e = doc.getError(i)
        msg = f"[{e.getSeverityAsString()}] line {e.getLine()}: {e.getMessage()}"
        if e.isError() or e.isFatal():
            errors.append(msg)
        else:
            warnings.append(msg)

    if doc.getModel() is None:
        errors.append("SBML document has no model element.")
        return SBMLValidationResult(ok=False, errors=errors, warnings=warnings)

    # deeper consistency check
    doc.checkConsistency()
    for i in range(doc.getNumErrors()):
        e = doc.getError(i)
        msg = f"[{e.getSeverityAsString()}] line {e.getLine()}: {e.getMessage()}"
        if e.isError() or e.isFatal():
            if msg not in errors:
                errors.append(msg)
        else:
            if msg not in warnings:
                warnings.append(msg)

    ok = (len(errors) == 0)
    return SBMLValidationResult(ok=ok, errors=errors, warnings=warnings)


# ---------------- SIMULATION ----------------

def simulate_sbml(
    sbml: str,
    start: float = 0.0,
    end: float = 100.0,
    points: int = 1001,
) -> SimulationResult:

    if te is None:
        return SimulationResult(
            ok=False,
            message="tellurium not installed. Install with: pip install tellurium"
        )

    try:
        rr = te.loadSBMLModel(sbml)
    except Exception as e:
        return SimulationResult(ok=False, message=f"Failed to load SBML: {e}")

    try:
        sim = rr.simulate(start, end, points)

        colnames = list(sim.colnames)
        arr = np.array(sim)

        if not np.all(np.isfinite(arr)):
            return SimulationResult(ok=False, message="Simulation contains NaN or Inf values")

        t = arr[:, colnames.index("time")] if "time" in colnames else arr[:, 0]

        final_vals: Dict[str, float] = {}
        for j, name in enumerate(colnames):
            if name == "time":
                continue
            final_vals[name] = float(arr[-1, j])

        return SimulationResult(
            ok=True,
            message="OK",
            time=t,
            data=arr,
            columns=colnames,
            final_values=final_vals
        )

    except Exception as e:
        return SimulationResult(ok=False, message=f"Simulation failed: {e}")


# ---------------- PLOTTING ----------------

def save_plot(sim: SimulationResult, filename: str = "simulation.png") -> Optional[str]:
    if not sim.ok or sim.data is None or sim.columns is None or sim.time is None:
        return None

    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 5))

    for i, name in enumerate(sim.columns):
        if name == "time":
            continue
        plt.plot(sim.time, sim.data[:, i], label=name)

    plt.xlabel("time")
    plt.ylabel("value / concentration")
    plt.title("SBML simulation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

    return filename


# ---------------- CSV EXPORT ----------------

def save_csv(sim: SimulationResult, filename: str = "simulation.csv") -> Optional[str]:
    if not sim.ok or sim.data is None or sim.columns is None:
        return None

    import csv
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(sim.columns)  # header
        for row in sim.data:
            writer.writerow(row.tolist())

    return filename


# ---------------- MAIN WRAPPER ----------------

def run_full_evaluation(
    sbml_source: Union[str, os.PathLike],
    is_path: bool = True,
    sim_end: float = 200.0
) -> Dict[str, Any]:

    sbml = read_text(str(sbml_source)) if is_path else str(sbml_source)

    out: Dict[str, Any] = {}

    # 1) validation
    v = validate_sbml(sbml)
    out["sbml_valid"] = v.ok
    out["validation_errors"] = v.errors
    out["validation_warnings"] = v.warnings

    # 2) simulation
    sim = simulate_sbml(sbml, 0, sim_end, int(sim_end * 10 + 1))
    out["simulation_ok"] = sim.ok
    out["simulation_message"] = sim.message

    if not sim.ok:
        return out

    out["columns"] = sim.columns
    out["final_values"] = sim.final_values

    # 3) save outputs for comparison
    out["plot_file"] = save_plot(sim, "simulation.png")
    out["csv_file"] = save_csv(sim, "simulation.csv")

    return out


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python sbml_validate_and_simulate.py model.xml")
        raise SystemExit(1)

    model_path = sys.argv[1]

    report = run_full_evaluation(
        model_path,
        is_path=True,
        sim_end=200.0
    )

    print(json.dumps(report, indent=2))

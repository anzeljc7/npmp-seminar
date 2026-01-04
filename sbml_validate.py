from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any
import os
import json
import sys

try:
    import libsbml
except Exception:
    libsbml = None


@dataclass
class SBMLValidationResult:
    ok: bool
    errors: List[str]
    warnings: List[str]


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def validate_sbml_string(sbml: str) -> SBMLValidationResult:
    """
    Validate SBML from a string using python-libsbml.
    Returns errors (fatal+error severities) and warnings (everything else).
    """
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

    # Parsing errors/warnings
    for i in range(doc.getNumErrors()):
        e = doc.getError(i)
        msg = f"[{e.getSeverityAsString()}] line {e.getLine()}: {e.getMessage()}"
        if e.isError() or e.isFatal():
            errors.append(msg)
        else:
            warnings.append(msg)

    if doc.getModel() is None:
        errors.append("SBML document has no <model> element.")
        return SBMLValidationResult(ok=False, errors=errors, warnings=warnings)

    # Consistency checks (adds more issues)
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


def validate_file(path: str) -> Dict[str, Any]:
    sbml = read_text(path)
    res = validate_sbml_string(sbml)
    return {
        "file": path,
        "sbml_valid": res.ok,
        "validation_errors": res.errors,
        "validation_warnings": res.warnings,
    }


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print("Usage: python sbml_validate_only.py <model1.xml> [model2.xml ...]")
        return 1

    reports: List[Dict[str, Any]] = []
    exit_code = 0

    for path in argv[1:]:
        if not os.path.exists(path):
            reports.append({
                "file": path,
                "sbml_valid": False,
                "validation_errors": [f"File not found: {path}"],
                "validation_warnings": [],
            })
            exit_code = max(exit_code, 2)
            continue

        rep = validate_file(path)
        reports.append(rep)
        if not rep["sbml_valid"]:
            exit_code = max(exit_code, 2)

    print(json.dumps(reports[0] if len(reports) == 1 else reports, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

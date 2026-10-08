"""Check the pinned Lean project and all 30 theorem dependencies.

Run from any directory: python <relative-repository-path>/proofs/verify.py
The first run needs elan/lake, Git and access to the pinned mathlib sources.
Use --setup once to fetch them and their compiled cache. Subsequent checks
require no dependency update. Results are written below outputs/proofs/.

The proof scope is exact scalar dynamics, held-input retiming, eligibility
and adjoint identities, and error-budget composition from per-look bounds.
It excludes the underlying PREDICT rank theorem, pseudorandom independence,
FPGA correctness, empirical accuracy, and exactness of approximate feedback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

LEAN_VERSION = "4.19.0"
MATHLIB_REVISION = "c44e0c8ee63ca166450922a373c7409c5d26b00b"
ALLOWED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}


def strip_comments(text):
    """Remove nested comments before an additional source-construct check."""
    out, index, depth = [], 0, 0
    while index < len(text):
        if text.startswith("/-", index):
            depth += 1
            index += 2
        elif depth and text.startswith("-/", index):
            depth -= 1
            index += 2
        elif depth:
            if text[index] == "\n":
                out.append("\n")
            index += 1
        elif text.startswith("--", index):
            end = text.find("\n", index)
            index = len(text) if end == -1 else end
        else:
            out.append(text[index])
            index += 1
    if depth:
        raise RuntimeError("Unterminated source comment")
    return "".join(out)


def axiom_audit(log, expected):
    pattern = re.compile(r"'([^']+)'\s+(?:depends on axioms:\s*\[([^\]]*)\]|does not depend on any axioms)", re.S)
    found = {}
    for match in pattern.finditer(log):
        found[match.group(1)] = [x.strip() for x in (match.group(2) or "").split(",") if x.strip()]
    missing = expected - found.keys()
    if missing:
        raise RuntimeError(f"Missing theorem dependency reports: {sorted(missing)}")
    disallowed = {name: sorted(set(found[name]) - ALLOWED_AXIOMS)
                  for name in expected if set(found[name]) - ALLOWED_AXIOMS}
    if disallowed:
        raise RuntimeError(f"Disallowed proof dependencies: {disallowed}")
    return {name: found[name] for name in sorted(expected)}


def run(command, directory, *, check=True):
    completed = subprocess.run(command, cwd=directory, text=True, encoding="utf-8",
                               errors="replace", stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, timeout=1800)
    if check and completed.returncode:
        raise RuntimeError(f"Command failed: {' '.join(command)}\n{completed.stdout[-12000:]}")
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", action="store_true", help="Fetch pinned dependencies and their compiled cache")
    args = parser.parse_args()
    project = Path(__file__).resolve().parent
    root = project.parent
    output = root / "outputs" / "proofs"
    output.mkdir(parents=True, exist_ok=True)
    result_file = output / "verification.json"
    result_file.unlink(missing_ok=True)
    if not shutil.which("lake"):
        raise RuntimeError("lake is unavailable. Install elan, then run this script with --setup.")
    print("[1/4] Checking Lean and mathlib versions", flush=True)
    version = run(["lean", "--version"], project).stdout
    if not re.search(rf"Lean \(version {re.escape(LEAN_VERSION)}(?:[ ,)])", version):
        raise RuntimeError(f"Unexpected Lean version: {version.strip()}")
    if args.setup:
        print("Fetching pinned mathlib sources and cache; this may take several minutes.", flush=True)
        for command in (["lake", "update"], ["lake", "exe", "cache", "get"]):
            completed = subprocess.run(command, cwd=project, timeout=3600)
            if completed.returncode:
                raise RuntimeError(f"Dependency setup failed: {' '.join(command)}")
    mathlib = project / ".lake" / "packages" / "mathlib"
    if not mathlib.is_dir():
        raise RuntimeError("Pinned mathlib has not been fetched. Run with --setup once.")
    revision = run(["git", "rev-parse", "HEAD"], mathlib).stdout.strip()
    if revision != MATHLIB_REVISION:
        raise RuntimeError("The mathlib checkout does not match the pinned revision")
    dirty = run(["git", "status", "--porcelain", "--untracked-files=no"], mathlib).stdout.strip()
    if dirty:
        raise RuntimeError("The pinned mathlib source has local modifications")
    dependency_revisions = {}
    pinned_dependencies = json.loads((mathlib / "lake-manifest.json").read_text(encoding="utf-8"))["packages"]
    for dependency in pinned_dependencies:
        checkout = project / ".lake" / "packages" / dependency["name"]
        actual = run(["git", "rev-parse", "HEAD"], checkout).stdout.strip()
        if actual != dependency["rev"]:
            raise RuntimeError(f"Dependency revision mismatch: {dependency['name']}")
        if run(["git", "status", "--porcelain", "--untracked-files=no"], checkout).stdout.strip():
            raise RuntimeError(f"Modified dependency source: {dependency['name']}")
        dependency_revisions[dependency["name"]] = actual
    source = project / "TemporalIdentities.lean"
    stripped = strip_comments(source.read_text(encoding="utf-8"))
    forbidden = re.search(r"\b(?:sorry|admit|native_decide)\b|^\s*(?:axiom|unsafe)\b|proofAsSorry", stripped, re.M)
    if forbidden:
        raise RuntimeError(f"Disallowed proof construct: {forbidden.group(0)}")
    names = re.findall(r"^theorem\s+([A-Za-z0-9_]+)", stripped, re.M)
    expected = {"CrispClear." + name for name in names}
    if len(names) != 30 or len(expected) != 30:
        raise RuntimeError("The expected 30-theorem inventory changed")
    print("[2/4] Kernel-checking all theorems", flush=True)
    checked = run(["lake", "env", "lean", "TemporalIdentities.lean"], project, check=False)
    (output / "lean.log").write_text(checked.stdout, encoding="utf-8")
    if checked.returncode:
        raise RuntimeError(f"Lean rejected the proof source:\n{checked.stdout[-12000:]}")
    inventory = axiom_audit(checked.stdout, expected)
    print("[3/4] Testing rejection of invalid and incomplete proofs", flush=True)
    cases = {
        "false_statement": "import Lean\ntheorem falseClaim : (0 : Nat) = 1 := by rfl\n",
        "unfinished_proof": "import Lean\ntheorem unfinished : False := by sorry\n#print axioms unfinished\n",
        "unproved_axiom": "import Lean\naxiom fabricated : False\ntheorem fake : False := fabricated\n#print axioms fake\n",
    }
    controls = {}
    with tempfile.TemporaryDirectory(prefix="controls_", dir=output) as scratch:
        for name, content in cases.items():
            path = Path(scratch) / f"{name}.lean"
            path.write_text(content, encoding="utf-8")
            relative = Path("..") / path.relative_to(root)
            checked = run(["lake", "env", "lean", str(relative)], project, check=False)
            (output / f"{name}.log").write_text(checked.stdout, encoding="utf-8")
            if name == "false_statement":
                detected = checked.returncode != 0 and "error:" in checked.stdout
            else:
                target = "unfinished" if name == "unfinished_proof" else "fake"
                forbidden_dependency = "sorryAx" if name == "unfinished_proof" else "fabricated"
                try:
                    axiom_audit(checked.stdout, {target})
                    detected = False
                except RuntimeError as error:
                    detected = (checked.returncode == 0 and "Disallowed proof dependencies" in str(error)
                                and forbidden_dependency in str(error))
            controls[name] = {"detected": detected, "lean_exit_code": checked.returncode}
            if not detected:
                raise RuntimeError(f"Failure detector did not catch {name}")
    print("[4/4] Writing proof-check results", flush=True)
    result = {
        "status": "PASS", "lean_version": LEAN_VERSION,
        "mathlib_revision": revision, "theorem_count": len(expected),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "axiom_allowlist": sorted(ALLOWED_AXIOMS), "theorem_axioms": inventory,
        "negative_controls": controls, "dependency_revisions": dependency_revisions,
    }
    result_file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("Saved outputs/proofs/verification.json", flush=True)


if __name__ == "__main__":
    main()

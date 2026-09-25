"""Linux powercap topology and optional DRAM policy; no sampling or scheduling."""
from __future__ import annotations

import os
from pathlib import Path
import re


def discover_rapl_topology(powercap_root="/sys/class/powercap") -> dict:
    root = Path(powercap_root)
    records = {}
    visited = set()
    errors = []

    def walk(path):
        real = os.path.realpath(path)
        if real in records:
            aliases = records[real]["aliases"]
            if str(path) not in aliases:
                aliases.append(str(path))
        if real in visited:
            return
        visited.add(real)
        is_domain = re.fullmatch(r"[\w-]*rapl(?:-mmio)?(?::\d+)+", path.name) is not None
        if is_domain:
            entry = {"id": path.name, "path": real, "aliases": [str(path)],
                     "parent_id": path.name.rsplit(":", 1)[0] if path.name.count(":") > 1 else None,
                     "name": None, "kind": "other", "energy_path": str(Path(real) / "energy_uj"),
                     "max_energy_range_uj": None, "enabled": None,
                     "status": "unavailable", "detail": "", "selected": False}
            records[real] = entry
            try:
                entry["name"] = (path / "name").read_text().strip()
                entry["kind"] = ("package" if entry["name"].startswith("package-") else
                                 "dram" if entry["name"].lower() == "dram" else "other")
                if (path / "enabled").exists():
                    entry["enabled"] = bool(int((path / "enabled").read_text()))
                if (path / "max_energy_range_uj").exists():
                    entry["max_energy_range_uj"] = int((path / "max_energy_range_uj").read_text())
                value = int((path / "energy_uj").read_text())
                if value < 0 or (entry["max_energy_range_uj"] is not None and entry["max_energy_range_uj"] <= 0):
                    raise ValueError("invalid RAPL energy/range")
                # enabled controls power capping, not the energy counter.
                entry["status"] = "available"
            except PermissionError as exc:
                entry.update(status="permission_denied", detail=str(exc))
            except (OSError, ValueError) as exc:
                entry["detail"] = str(exc)
        try:
            children = sorted(path.iterdir(), key=lambda item: item.name)
        except OSError as exc:
            errors.append({"path": str(path), "error": str(exc)})
            return
        for child in children:
            # Do not traverse unrelated sysfs device/subsystem links or cycles.
            if "rapl" in child.name and child.is_dir():
                walk(child)

    if root.is_dir():
        walk(root)
    domains = sorted(records.values(), key=lambda entry: entry["id"])
    selected_packages = {}
    for entry in domains:
        if entry["kind"] == "package" and entry["status"] == "available":
            # Alternate interfaces to the same package (e.g. MSR/MMIO) are not additive.
            previous = selected_packages.get(entry["name"])
            if previous is None or ("mmio" in previous["id"] and "mmio" not in entry["id"]):
                if previous is not None:
                    previous["selected"] = False
                selected_packages[entry["name"]] = entry
                entry["selected"] = True
    selected_ids = {entry["id"] for entry in selected_packages.values()}
    for entry in domains:
        if entry["kind"] == "dram" and entry["parent_id"] in selected_ids and entry["status"] == "available":
            entry["selected"] = True
        entry["aliases"].sort()
        if not entry["selected"] and not entry["detail"]:
            entry["detail"] = "metadata-only subdomain or alternate package interface"
    missing = sorted(package for package in selected_ids if not any(
        entry["kind"] == "dram" and entry["parent_id"] == package and entry["selected"] for entry in domains))
    missing.extend(entry["id"] for entry in domains if entry["kind"] == "package"
                   and entry["name"] not in selected_packages)
    dram_status = "available" if selected_ids and not missing else "unavailable"
    if missing and any(e["kind"] == "dram" and e["parent_id"] in missing
                       and e["status"] == "permission_denied" for e in domains):
        dram_status = "permission_denied"
    return {"schema_version": 1, "source": str(root), "domains": domains, "errors": errors,
            "dram_status": dram_status, "dram_missing_packages": missing,
            "counter_assumption": "at most one wrap per domain between adjacent samples"}


def dram_policy(mode: str, selection: str) -> bool:
    if selection not in {"auto", "off", "required"}:
        raise ValueError("DRAM energy must be auto/off/required")
    if mode not in {"basic", "full"}:
        raise ValueError("profiling mode must be basic/full")
    if mode == "basic" and selection == "required":
        raise ValueError("--dram-energy required needs --profiling-mode full")
    return mode == "full" and selection != "off"

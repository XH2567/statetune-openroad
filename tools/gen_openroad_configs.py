#!/usr/bin/env python3
"""
Generate OpenROAD (ORFS) StateTune configs for the 2 PDK x 3 design matrix.

Template: configs/nangate_aes/openroad_tuner.yaml (hand-written, commented).
Every other case is that template plus the per-case overrides below, written to
configs/<case>/openroad_tuner.yaml. Edit the template or CASES here and re-run:

    python tools/gen_openroad_configs.py            # all cases except the template
    python tools/gen_openroad_configs.py asap7_aes  # selected cases

Per-case values follow ORFS's own reference configs (flow/designs/<platform>/<design>/config.mk);
floorplan/placement ranges are centred on those references.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "configs" / "nangate_aes" / "openroad_tuner.yaml"

IBEX_FILES = [
    "./benchmark/ibex/ibex_pkg.sv",
    "./benchmark/ibex/ibex_alu.sv",
    "./benchmark/ibex/ibex_compressed_decoder.sv",
    "./benchmark/ibex/ibex_controller.sv",
    "./benchmark/ibex/ibex_core.sv",
    "./benchmark/ibex/ibex_counter.sv",
    "./benchmark/ibex/ibex_cs_registers.sv",
    "./benchmark/ibex/ibex_csr.sv",
    "./benchmark/ibex/ibex_decoder.sv",
    "./benchmark/ibex/ibex_ex_block.sv",
    "./benchmark/ibex/ibex_fetch_fifo.sv",
    "./benchmark/ibex/ibex_id_stage.sv",
    "./benchmark/ibex/ibex_if_stage.sv",
    "./benchmark/ibex/ibex_load_store_unit.sv",
    "./benchmark/ibex/ibex_multdiv_fast.sv",
    "./benchmark/ibex/ibex_multdiv_slow.sv",
    "./benchmark/ibex/ibex_pmp.sv",
    "./benchmark/ibex/ibex_prefetch_buffer.sv",
    "./benchmark/ibex/ibex_register_file_ff.sv",
    "./benchmark/ibex/ibex_wb_stage.sv",
    "./benchmark/ibex/prim_clock_gating.v",
    # prim_assert.sv is an include file (`include "prim_assert.sv"), not a compilation unit
]

DESIGNS = {
    "aes": {
        "name": "aes_cipher_top",
        "verilog": ["./benchmark/aes/*.v"],
        "verilog_include_dirs": ["./benchmark/aes"],
        "extra_vars": {"SKIP_INCREMENTAL_REPAIR": 1},
    },
    "jpeg": {
        "name": "jpeg_encoder",
        "verilog": ["./benchmark/jpeg/*.v"],
        "verilog_include_dirs": ["./benchmark/jpeg/include"],
        "extra_vars": {},
    },
    "ibex": {
        "name": "ibex_core",
        "verilog": IBEX_FILES,
        "verilog_include_dirs": ["./benchmark/ibex"],
        "extra_vars": {"SYNTH_HDL_FRONTEND": "slang"},
    },
}

# (platform, design) -> overrides. util = CORE_UTILIZATION range (percent), cts_diam in um.
CASES = {
    "nangate_aes":  {"platform": "nangate45", "design": "aes",  "sdc": "./benchmark/aes/aes_constraint_nangate45.sdc",
                     "util": [30, 55], "util0": 45, "cts_diam": [50, 150], "cts_diam0": 100},
    "nangate_jpeg": {"platform": "nangate45", "design": "jpeg", "sdc": "./benchmark/jpeg/jepg_constraint_45nm.sdc",
                     "util": [45, 70], "util0": 60, "cts_diam": [50, 150], "cts_diam0": 100},
    "nangate_ibex": {"platform": "nangate45", "design": "ibex", "sdc": "./benchmark/ibex/ibex_constraint_nangate45.sdc",
                     "util": [35, 60], "util0": 50, "cts_diam": [50, 150], "cts_diam0": 100},
    "asap7_aes":    {"platform": "asap7", "design": "aes",  "sdc": "./benchmark/aes/aes_constraint_asap7.sdc",
                     "util": [50, 70], "util0": 60, "cts_diam": [20, 80], "cts_diam0": 50},
    "asap7_jpeg":   {"platform": "asap7", "design": "jpeg", "sdc": "./benchmark/jpeg/jpeg_encoder15_7nm.sdc",
                     "util": [50, 70], "util0": 60, "cts_diam": [20, 80], "cts_diam0": 50},
    "asap7_ibex":   {"platform": "asap7", "design": "ibex", "sdc": "./benchmark/ibex/ibex_constraint_asap7.sdc",
                     "util": [30, 55], "util0": 40, "cts_diam": [20, 80], "cts_diam0": 50,
                     "extra_vars": {"ENABLE_DPO": 0}},
}

TIMEOUTS = {  # (quick_timeout_sec, full_timeout_sec); larger designs get more headroom
    "aes": (1800, 3600),
    "jpeg": (3600, 7200),
    "ibex": (3600, 7200),
}


def build(case: str, tmpl: dict) -> dict:
    c = CASES[case]
    d = DESIGNS[c["design"]]
    cfg = copy.deepcopy(tmpl)
    tag = f"openroad_{c['design']}_{'ng45' if c['platform'] == 'nangate45' else 'asap7'}"

    cfg["project_name"] = tag
    cfg["work_dir"] = f"./runs/{tag}"
    cfg["rag"]["index_path"] = f"./runs/{tag}/rag_index"

    cfg["orfs"]["platform"] = c["platform"]
    cfg["orfs"]["extra_vars"] = {"LEC_CHECK": 0, **d["extra_vars"], **c.get("extra_vars", {})}

    cfg["design"] = {
        "name": d["name"],
        "verilog": list(d["verilog"]),
        "verilog_include_dirs": list(d["verilog_include_dirs"]),
        "sdc": c["sdc"],
    }

    cfg["default_params"]["CORE_UTILIZATION"] = c["util0"]
    cfg["default_params"]["CTS_CLUSTER_DIAMETER"] = c["cts_diam0"]
    for p in cfg["tuning"]["param_space"]:
        if p["name"] == "CORE_UTILIZATION":
            p["range"] = list(c["util"])
        elif p["name"] == "CTS_CLUSTER_DIAMETER":
            p["range"] = list(c["cts_diam"])

    q, f = TIMEOUTS[c["design"]]
    cfg["fidelity"]["quick_timeout_sec"] = q
    cfg["fidelity"]["full_timeout_sec"] = f
    return cfg


def main(argv: list[str]) -> int:
    tmpl = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    cases = argv or [k for k in CASES if k != "nangate_aes"]
    for case in cases:
        cfg = build(case, tmpl)
        out = ROOT / "configs" / case / "openroad_tuner.yaml"
        hdr = (
            f"# StateTune on OpenROAD-flow-scripts — {case}\n"
            f"# GENERATED by tools/gen_openroad_configs.py from configs/nangate_aes/openroad_tuner.yaml\n"
            f"# (see that file for comments). Edit the generator/template and re-run instead of editing here.\n"
        )
        out.write_text(hdr + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

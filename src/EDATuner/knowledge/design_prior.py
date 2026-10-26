# src/EDATuner/knowledge/design_prior.py
"""
Design Prior Knowledge Generator
=================================
On the first run, this module scans design files automatically, ranks them by importance,
feeds them to the LLM in batches, and persists the distilled prior knowledge.
On later runs, it loads the refined text directly (~500-1500 words) instead of
sending the raw source code again.

Usage:
    prior = DesignPriorBuilder(cfg, llm_client)
    prior_text = prior.get_or_build()   # Load from cache when available, otherwise build it.
"""
from __future__ import annotations

import glob
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# File importance scoring
# ---------------------------------------------------------------------------

@dataclass
class DesignFile:
    """A design file to be evaluated."""
    path: str
    category: str          # "sdc" | "rtl_top" | "rtl_module" | "tcl" | "lib" | "other"
    importance: float      # 0-100, higher means more important
    size_bytes: int = 0
    content_preview: str = ""   # Truncated content actually sent to the LLM

    def __repr__(self) -> str:
        return f"DesignFile({self.category}, imp={self.importance:.0f}, {Path(self.path).name})"


# File category -> base importance
_CATEGORY_BASE_SCORE: Dict[str, float] = {
    "sdc":         90,   # SDC constraints are highest priority (clock, IO delay, false path)
    "rtl_top":     80,   # Top-level module (ports and instantiation structure)
    "rtl_module":  40,   # Submodule RTL (many files, lower information density per file)
    "tcl":         60,   # Innovus/Genus Tcl scripts (affect flow behavior)
    "lef":         30,   # LEF (process-related, usually not needed by the LLM)
    "lib":         20,   # Liberty (usually too large; only inspect the header)
    "config":      50,   # Config files (parameter definitions and flow configuration)
    "other":       10,
}

# Filename keywords -> bonus score
_FILENAME_BONUS: List[Tuple[re.Pattern, float]] = [
    (re.compile(r"top|wrapper|chip", re.I),          +15),
    (re.compile(r"clk|clock|pll", re.I),             +12),
    (re.compile(r"ctrl|control|fsm|state", re.I),    +10),
    (re.compile(r"datapath|alu|dct|fft", re.I),       +8),
    (re.compile(r"mem|ram|fifo|buffer", re.I),         +5),
    (re.compile(r"pad|io_?cell", re.I),                +3),
    (re.compile(r"test|scan|jtag|bist", re.I),         -5),  # Test logic is usually less helpful for tuning
]


def _classify_file(path: str, design_name: str = "") -> Tuple[str, float]:
    """Classify a file by path/suffix and return ``(category, importance)``."""
    p = Path(path)
    name = p.stem.lower()
    suffix = p.suffix.lower()

    # SDC
    if suffix == ".sdc":
        return "sdc", _CATEGORY_BASE_SCORE["sdc"]

    # RTL
    if suffix in (".v", ".sv", ".svh", ".vh"):
        # Top-level module: filename contains the design name or "top"
        is_top = False
        if design_name and design_name.lower() in name:
            is_top = True
        if "top" in name or "wrapper" in name or "chip" in name:
            is_top = True
        cat = "rtl_top" if is_top else "rtl_module"
        score = _CATEGORY_BASE_SCORE[cat]

        # Apply filename keyword bonus/penalty
        for pat, bonus in _FILENAME_BONUS:
            if pat.search(name):
                score += bonus

        return cat, score

    # TCL
    if suffix in (".tcl",):
        return "tcl", _CATEGORY_BASE_SCORE["tcl"]

    # LEF
    if suffix in (".lef", ".tlef"):
        return "lef", _CATEGORY_BASE_SCORE["lef"]

    # Liberty
    if suffix in (".lib", ".lib.gz"):
        return "lib", _CATEGORY_BASE_SCORE["lib"]

    # Config
    if suffix in (".yaml", ".yml", ".json", ".cfg"):
        return "config", _CATEGORY_BASE_SCORE["config"]

    return "other", _CATEGORY_BASE_SCORE["other"]


def _read_preview(path: str, max_chars: int) -> str:
    """Read file content for LLM analysis. For large files, keep only the head and tail."""
    try:
        p = Path(path)
        if not p.is_file():
            return ""
        size = p.stat().st_size

        # Skip binary files
        if p.suffix.lower() in (".gz", ".db", ".oa", ".enc", ".dat"):
            return f"[binary file, {size} bytes]"

        txt = p.read_text(encoding="utf-8", errors="replace")

        if len(txt) <= max_chars:
            return txt

        # Large file: keep the first 60% and the last 30% with a gap marker
        head_n = int(max_chars * 0.6)
        tail_n = int(max_chars * 0.3)
        return (
            txt[:head_n]
            + f"\n\n... [{len(txt) - head_n - tail_n} chars truncated] ...\n\n"
            + txt[-tail_n:]
        )
    except Exception:
        return ""


def _extract_rtl_skeleton(verilog_text: str, max_chars: int = 1500) -> str:
    """
    Extract a structural skeleton from Verilog/SV (module declaration, ports, and
    instantiations) while removing detailed always/assign logic to keep the file compact.
    """
    lines = verilog_text.splitlines()
    keep: List[str] = []
    in_module_header = False
    paren_depth = 0

    for line in lines:
        stripped = line.strip()

        # Module declaration header
        if re.match(r"\bmodule\b", stripped, re.I):
            in_module_header = True
            paren_depth = 0

        if in_module_header:
            keep.append(line)
            paren_depth += stripped.count("(") - stripped.count(")")
            if ";" in stripped and paren_depth <= 0:
                in_module_header = False
            continue

        # Keep input/output/inout/wire/reg declarations
        if re.match(r"\s*(input|output|inout|wire|reg|logic|parameter|localparam)\b", stripped, re.I):
            keep.append(line)
            continue

        # Keep instantiation lines (module name + instance name)
        if re.match(r"\s*\w+\s+\w+\s*\(", stripped) and not re.match(r"\s*(if|for|while|case|always|assign|function|task)\b", stripped, re.I):
            keep.append(line)
            continue

        # endmodule
        if re.match(r"\s*endmodule\b", stripped, re.I):
            keep.append(line)
            continue

    result = "\n".join(keep)
    if len(result) > max_chars:
        result = result[:max_chars] + "\n... [skeleton truncated]"
    return result


# ---------------------------------------------------------------------------
# File scanner
# ---------------------------------------------------------------------------

def scan_design_files(
    cfg: Dict[str, Any],
    *,
    max_files: int = 50,
) -> List[DesignFile]:
    """
    Infer design file locations from the config, scan them, and sort them by importance.

    Scan sources:
    1. ``config.design.verilog`` (RTL path or glob)
    2. ``config.design.sdc``
    3. Only these benchmark locations:
       - the sibling ``benchmark`` directory above ``work_dir``
       - the grandparent ``benchmark`` directory above ``work_dir``
    4. ``knowledge/`` or ``docs/`` under ``work_dir``
    """
    design_name = str((cfg.get("design", {}) or {}).get("name", "")).strip()
    work_dir = Path(cfg.get("work_dir", ".")).resolve()

    candidates: Dict[str, DesignFile] = {}  # path -> DesignFile (dedup)

    def _add(path_str: str) -> None:
        try:
            p = Path(path_str).resolve()
            key = str(p)
            if key in candidates:
                return
            if not p.is_file():
                return
            cat, imp = _classify_file(str(p), design_name)
            candidates[key] = DesignFile(
                path=str(p),
                category=cat,
                importance=imp,
                size_bytes=p.stat().st_size,
            )
        except Exception:
            pass

    # 1) Explicit design files from config
    design_cfg = cfg.get("design", {}) or {}

    # Verilog files (may contain globs)
    verilog_list = design_cfg.get("verilog", [])
    if isinstance(verilog_list, str):
        verilog_list = [verilog_list]
    for v in (verilog_list or []):
        for expanded in glob.glob(str(v)):
            _add(expanded)

    # SDC
    sdc = design_cfg.get("sdc", "")
    if sdc:
        _add(str(sdc))

    # Verilog include dirs
    for inc_dir in (design_cfg.get("verilog_include_dirs", []) or []):
        inc_path = Path(str(inc_dir))
        if inc_path.is_dir():
            for f in inc_path.rglob("*"):
                if f.suffix.lower() in (".v", ".sv", ".svh", ".vh", ".h"):
                    _add(str(f))

    # 2) Scan benchmark directories:
    #    Only scan the subdirectory that matches the current design,
    #    so ibex configs don't pollute with jpeg/aes files and vice versa.
    benchmark_dirs = [
        work_dir.parent / "benchmark",
        work_dir.parent.parent / "benchmark",
    ]

    # Derive design keyword from design_name for filtering benchmark subdirs.
    # e.g. "ibex_core" -> "ibex", "aes_cipher_top" -> "aes", "jpeg_encoder" -> "jpeg"
    _design_keyword = ""
    if design_name:
        # Take the first segment before '_' as the keyword
        _kw = design_name.split("_")[0].lower()
        if len(_kw) >= 2:
            _design_keyword = _kw

    for scan_dir in benchmark_dirs:
        if not scan_dir.is_dir():
            continue

        if _design_keyword:
            # Try to find a matching subdirectory (e.g. "ibex", "ibex_sv", "aes", "jpeg")
            matched_subdir = None
            for child in sorted(scan_dir.iterdir()):
                if child.is_dir() and _design_keyword in child.name.lower():
                    matched_subdir = child
                    break  # take the first match

            if matched_subdir:
                for f in matched_subdir.rglob("*"):
                    if f.suffix.lower() in (".v", ".sv", ".svh", ".vh", ".sdc", ".tcl"):
                        _add(str(f))
                continue  # skip full benchmark scan

        # Fallback: scan entire benchmark dir (original behavior)
        for f in scan_dir.rglob("*"):
            if f.suffix.lower() in (".v", ".sv", ".svh", ".vh", ".sdc", ".tcl"):
                _add(str(f))


    # 3) Knowledge dir (user might have put notes there)
    for kdir in ("knowledge", "docs"):
        kp = work_dir / kdir
        if kp.is_dir():
            for f in kp.rglob("*"):
                if f.suffix.lower() in (".txt", ".md", ".json", ".yaml"):
                    _add(str(f))

    # Sort by importance descending, then by size ascending
    sorted_files = sorted(
        candidates.values(),
        key=lambda df: (-df.importance, df.size_bytes),
    )

    # 4) If nothing is found, derive a minimal summary from the config without exposing raw paths
    if not candidates:
        remote_cfg = (cfg.get("cadence", {}) or {}).get("remote", {}) or {}
        remote_rtl = remote_cfg.get("rtl_dir", "")
        remote_sdc = remote_cfg.get("sdc_file", "")
        design_name = str((cfg.get("design", {}) or {}).get("name", ""))

        info_text = f"Design: {design_name}\n"
        if remote_rtl:
            info_text += "Remote RTL location is configured.\n"
        if remote_sdc:
            info_text += "Remote SDC file is configured.\n"
        defaults = cfg.get("default_params", {})
        if defaults:
            info_text += f"Default params: {json.dumps(defaults, ensure_ascii=False)}\n"
        param_space = (cfg.get("tuning", {}) or {}).get("param_space", [])
        if param_space:
            info_text += f"Param space: {json.dumps(param_space, ensure_ascii=False)}\n"

        if info_text.strip():
            candidates["__config_info__"] = DesignFile(
                path="<from_config>",
                category="config",
                importance=50,
                size_bytes=len(info_text),
                content_preview=info_text,
            )

        sorted_files = sorted(
            candidates.values(),
            key=lambda df: (-df.importance, df.size_bytes),
        )

    return sorted_files[:max_files]


# ---------------------------------------------------------------------------
# Prior builder core
# ---------------------------------------------------------------------------

# Per-request character budget sent to the LLM
_BATCH_BUDGET_CHARS = 12000
# Maximum skeleton length extracted from a single RTL module
_RTL_SKELETON_MAX = 1500
# Preview size for high-priority files such as SDC/Tcl
_HIGH_PRIO_PREVIEW_MAX = 3000
# Preview size for lower-priority files
_LOW_PRIO_PREVIEW_MAX = 800

# Filename used to persist prior knowledge
PRIOR_FILENAME = "design_prior.json"


@dataclass
class DesignPrior:
    """Persisted prior knowledge."""
    design_name: str = ""
    generated_ts: float = 0.0
    # Core payload: distilled prior text injected into the system prompt
    prior_text: str = ""
    # Auxiliary metadata: scanned files and scores for debugging/rebuilds
    scanned_files: List[Dict[str, Any]] = field(default_factory=list)
    # Intermediate summaries produced for each LLM batch
    batch_summaries: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "design_name": self.design_name,
            "generated_ts": self.generated_ts,
            "prior_text": self.prior_text,
            "scanned_files": self.scanned_files,
            "batch_summaries": self.batch_summaries,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DesignPrior":
        return cls(
            design_name=str(d.get("design_name", "")),
            generated_ts=float(d.get("generated_ts", 0)),
            prior_text=str(d.get("prior_text", "")),
            scanned_files=d.get("scanned_files", []),
            batch_summaries=d.get("batch_summaries", []),
        )


class DesignPriorBuilder:
    """
    Two-stage prior knowledge manager.

    Stage 1 (first run):
        scan -> prioritize -> batch -> LLM summarize -> merge -> save

    Stage 2 (later runs):
        load -> return ``prior_text``
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        llm_client: Any = None,  # OpenAIResponsesClient
        *,
        model: str = "",
        max_output_tokens: int = 2000,
    ):
        self.cfg = cfg
        self.llm_client = llm_client
        self.model = model or str((cfg.get("llm", {}) or {}).get("model", "deepseek-v3.2"))
        self.max_output_tokens = max_output_tokens

        self.design_name = str((cfg.get("design", {}) or {}).get("name", "unknown"))

        # Persistence path
        work_dir = Path(cfg.get("work_dir", ".")).resolve()
        kb_dir = work_dir / "kb"
        kb_dir.mkdir(parents=True, exist_ok=True)
        self.prior_path = kb_dir / PRIOR_FILENAME

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_or_build(self, *, force_rebuild: bool = False) -> str:
        """
        Return the distilled prior text.
        - If a cached prior exists and ``force_rebuild=False``, return it directly.
        - Otherwise scan files, build with the LLM, and save the result.
        - If the LLM is unavailable, fall back to a statistics-only summary.
        """
        if not force_rebuild:
            cached = self._load_cached()
            if cached and cached.prior_text.strip():
                print(f"[DesignPrior] loaded cached prior ({len(cached.prior_text)} chars)")
                return cached.prior_text

        print("[DesignPrior] building prior knowledge for design '%s' ..." % self.design_name)

        # Step 1: scan files
        files = scan_design_files(self.cfg)
        print(f"[DesignPrior] scanned {len(files)} files:")
        for f in files[:15]:
            print(f"  {f.importance:5.0f}  {f.category:12s}  {Path(f.path).name}")
        if len(files) > 15:
            print(f"  ... and {len(files) - 15} more")

        # Step 2: prepare a preview for each file
        self._prepare_previews(files)

        # Step 3: send file batches to the LLM
        if self.llm_client and getattr(self.llm_client, "is_ready", lambda: False)():
            prior = self._build_with_llm(files)
        else:
            print("[DesignPrior] LLM not available, generating stats-only prior")
            prior = self._build_without_llm(files)

        # Step 4: save the result
        self._save(prior)
        print(f"[DesignPrior] saved prior to {self.prior_path} ({len(prior.prior_text)} chars)")

        return prior.prior_text

    # ------------------------------------------------------------------
    # Internal: prepare file previews
    # ------------------------------------------------------------------

    def _prepare_previews(self, files: List[DesignFile]) -> None:
        """Prepare a content preview based on file category and importance."""
        for f in files:
            if f.category == "sdc":
                f.content_preview = _read_preview(f.path, _HIGH_PRIO_PREVIEW_MAX)

            elif f.category == "rtl_top":
                raw = _read_preview(f.path, 8000)
                # Top-level module: extract a structural skeleton (module, ports, instantiations)
                skeleton = _extract_rtl_skeleton(raw, max_chars=_RTL_SKELETON_MAX * 2)
                f.content_preview = skeleton if skeleton.strip() else raw[:_HIGH_PRIO_PREVIEW_MAX]

            elif f.category == "rtl_module":
                raw = _read_preview(f.path, 5000)
                skeleton = _extract_rtl_skeleton(raw, max_chars=_RTL_SKELETON_MAX)
                f.content_preview = skeleton if skeleton.strip() else raw[:_LOW_PRIO_PREVIEW_MAX]

            elif f.category == "tcl":
                f.content_preview = _read_preview(f.path, _HIGH_PRIO_PREVIEW_MAX)

            elif f.category in ("lib", "lef"):
                # Liberty/LEF: only inspect the first few lines for process/site metadata
                f.content_preview = _read_preview(f.path, 500)

            else:
                f.content_preview = _read_preview(f.path, _LOW_PRIO_PREVIEW_MAX)

    # ------------------------------------------------------------------
    # Internal: batched LLM generation
    # ------------------------------------------------------------------

    def _build_with_llm(self, files: List[DesignFile]) -> DesignPrior:
        """Feed file batches to the LLM, summarize each batch, then merge them into the final prior."""

        # Batch files by importance, keeping each batch under the character budget
        batches = self._create_batches(files)
        print(f"[DesignPrior] created {len(batches)} batches for LLM analysis")

        batch_summaries: List[str] = []

        for i, batch in enumerate(batches):
            print(f"[DesignPrior] processing batch {i+1}/{len(batches)} "
                  f"({len(batch)} files, {sum(len(f.content_preview) for f in batch)} chars)")

            summary = self._analyze_batch(batch, batch_index=i, total_batches=len(batches))
            for _retry in range(2):
                if summary:
                    break
                # Transient empty/failed responses happen; the prior is built once per run, so retry
                print(f"[DesignPrior]   retrying batch {i+1} ({_retry + 1}/2)")
                time.sleep(5)
                summary = self._analyze_batch(batch, batch_index=i, total_batches=len(batches))
            if summary:
                batch_summaries.append(summary)
                print(f"[DesignPrior]   → summary: {summary[:120]}...")

        # Final merge
        if len(batch_summaries) > 1:
            final_text = self._merge_summaries(batch_summaries)
        elif batch_summaries:
            final_text = batch_summaries[0]
        else:
            # If all LLM calls fail, fall back to the statistics-only summary
            return self._build_without_llm(files)

        prior = DesignPrior(
            design_name=self.design_name,
            generated_ts=time.time(),
            prior_text=final_text,
            scanned_files=[{"path": Path(f.path).name, "category": f.category, "importance": f.importance} for f in files],
            batch_summaries=batch_summaries,
        )
        return prior

    def _create_batches(self, files: List[DesignFile]) -> List[List[DesignFile]]:
        """Group files into batches from high to low importance, capped by total preview size."""
        batches: List[List[DesignFile]] = []
        current_batch: List[DesignFile] = []
        current_chars = 0

        for f in files:
            preview_len = len(f.content_preview)
            if preview_len == 0:
                continue

            # If a single file exceeds the budget, place it in its own batch
            if preview_len > _BATCH_BUDGET_CHARS:
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_chars = 0
                batches.append([f])
                continue

            if current_chars + preview_len > _BATCH_BUDGET_CHARS:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [f]
                current_chars = preview_len
            else:
                current_batch.append(f)
                current_chars += preview_len

        if current_batch:
            batches.append(current_batch)

        return batches

    def _analyze_batch(self, batch: List[DesignFile], *, batch_index: int, total_batches: int) -> str:
        """Call the LLM on a file batch and produce an intermediate summary."""
        file_sections = []
        for f in batch:
            file_sections.append(
                f"--- FILE: {Path(f.path).name} (category={f.category}, importance={f.importance:.0f}) ---\n"
                f"{f.content_preview}\n"
            )

        content = "\n".join(file_sections)

        system_msg = (
            "You are an expert EDA/IC-design analyst. "
            "You will receive source files from a chip design project. "
            "Your job: extract ONLY the information relevant to PD (physical design) parameter tuning. "
            "Focus on:\n"
            "1. Clock constraints (period, uncertainty, false/multi-cycle paths)\n"
            "2. Design complexity (module count, register count, datapath vs control logic)\n"
            "3. Critical path characteristics (combinational depth, cross-hierarchy paths)\n"
            "4. IO constraints and special timing requirements\n"
            "5. Technology/library hints (node, cell types, voltage corners)\n"
            "6. Any clue about which PD parameters (utilization, placement density, CTS, routing) "
            "would be most impactful for this design\n"
            "\n"
            "Output a CONCISE summary (300-500 words max). "
            "Do NOT repeat the raw code. Extract insights only. "
            "Use technical EDA terminology. "
            "If a file is not useful for PD tuning, skip it entirely."
        )

        user_msg = (
            f"Design: {self.design_name}\n"
            f"Batch {batch_index + 1}/{total_batches}\n"
            f"Files in this batch:\n\n"
            f"{content}"
        )

        try:
            text, _raw = self.llm_client.create_text(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.1,
                max_output_tokens=self.max_output_tokens,
            )
            return (text or "").strip()
        except Exception as e:
            print(f"[DesignPrior] LLM batch analysis failed: {e}")
            return ""

    def _merge_summaries(self, summaries: List[str]) -> str:
        """Merge multiple intermediate summaries into the final prior text."""
        joined = "\n\n---\n\n".join(summaries)

        system_msg = (
            "You are an expert EDA engineer. "
            "You will receive multiple analysis summaries of the same chip design, "
            "each covering different source files. "
            "Merge them into ONE coherent prior knowledge document for PD parameter tuning.\n"
            "\n"
            "Requirements:\n"
            "- 400-800 words max\n"
            "- Structure: Design Overview → Timing Profile → Complexity Analysis → "
            "PD Tuning Recommendations\n"
            "- Be specific: include numbers (clock period, module count, etc.)\n"
            "- Deduplicate: if multiple summaries mention the same fact, state it once\n"
            "- End with 3-5 concrete tuning hypotheses (e.g. 'UTIL > 0.7 likely causes "
            "congestion due to high register count')"
        )

        try:
            text, _raw = self.llm_client.create_text(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": f"Design: {self.design_name}\n\nBatch summaries:\n\n{joined}"},
                ],
                temperature=0.1,
                max_output_tokens=self.max_output_tokens,
            )
            return (text or "").strip()
        except Exception as e:
            print(f"[DesignPrior] merge failed: {e}")
            # Fallback: concatenate directly and truncate
            return joined[:3000]

    # ------------------------------------------------------------------
    # No-LLM fallback: statistics-only summary
    # ------------------------------------------------------------------

    def _build_without_llm(self, files: List[DesignFile]) -> DesignPrior:
        """Build prior knowledge from simple file statistics without calling the LLM."""
        lines: List[str] = []
        lines.append(f"Design: {self.design_name}")

        # SDC information
        for f in files:
            if f.category == "sdc" and f.content_preview:
                clk_period = self._extract_clock_period(f.content_preview)
                if clk_period:
                    lines.append(f"Clock period: {clk_period}")
                lines.append(f"SDC file: {Path(f.path).name}")
                break

        # RTL statistics
        rtl_files = [f for f in files if f.category in ("rtl_top", "rtl_module")]
        lines.append(f"RTL files: {len(rtl_files)}")

        top_files = [f for f in files if f.category == "rtl_top"]
        if top_files:
            lines.append(f"Top modules: {', '.join(Path(f.path).stem for f in top_files)}")

        # Total code volume
        total_lines = 0
        for f in rtl_files:
            try:
                total_lines += Path(f.path).read_text(encoding="utf-8", errors="replace").count("\n")
            except Exception:
                pass
        if total_lines > 0:
            lines.append(f"Total RTL lines: ~{total_lines}")

        # Configuration information
        cfg_info = []
        param_space = (self.cfg.get("tuning", {}) or {}).get("param_space", [])
        if param_space:
            cfg_info.append(f"Tunable params: {', '.join(p.get('name','?') for p in param_space)}")
        default_params = self.cfg.get("default_params", {})
        if default_params:
            cfg_info.append(f"Defaults: {json.dumps(default_params, ensure_ascii=False)}")
        lines.extend(cfg_info)

        text = "\n".join(lines)

        return DesignPrior(
            design_name=self.design_name,
            generated_ts=time.time(),
            prior_text=text,
            scanned_files=[{"path": Path(f.path).name, "category": f.category, "importance": f.importance} for f in files],
            batch_summaries=[],
        )

    @staticmethod
    def _extract_clock_period(sdc_text: str) -> Optional[str]:
        """Extract the clock period from SDC text."""
        # create_clock -period 680
        m = re.search(r"create_clock\s+.*?-period\s+([0-9.]+)", sdc_text, re.I)
        if m:
            return f"{m.group(1)} (from SDC create_clock)"
        return None

    # ------------------------------------------------------------------
    # Cache read/write
    # ------------------------------------------------------------------

    def _load_cached(self) -> Optional[DesignPrior]:
        try:
            if not self.prior_path.is_file():
                return None
            data = json.loads(self.prior_path.read_text(encoding="utf-8"))
            prior = DesignPrior.from_dict(data)
            # Verify that the cached record belongs to the same design
            if prior.design_name != self.design_name:
                print(f"[DesignPrior] cached prior is for '{prior.design_name}', "
                      f"current design is '{self.design_name}'; rebuilding")
                return None
            return prior
        except Exception:
            return None

    def _save(self, prior: DesignPrior) -> None:
        try:
            self.prior_path.parent.mkdir(parents=True, exist_ok=True)
            self.prior_path.write_text(
                json.dumps(prior.to_dict(), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception as e:
            print(f"[DesignPrior] failed to save: {e}")

    def refine_with_history(
        self,
        current_prior: str,
        history_recent: List[Dict[str, Any]],
        pareto: List[Dict[str, Any]],
        failure_summary: str = "",
        sensitivity_summary: str = "",
    ) -> str:
        """Update prior knowledge with the latest experiment data and return the refreshed text."""
        if not self.llm_client or not getattr(self.llm_client, "is_ready", lambda: False)():
            return current_prior

        succ = [r for r in history_recent if r.get("success", False)]
        fail = [r for r in history_recent if not r.get("success", False)]
        history_stats = f"Recent {len(history_recent)} runs: {len(succ)} success, {len(fail)} fail.\n"

        pareto_info = ""
        if pareto:
            pareto_info = "Current Pareto front:\n"
            for p in pareto[:5]:
                m = p.get("metrics", {}) or {}
                params = p.get("params", {}) or {}
                pareto_info += (
                    f"  WNS={m.get('wns','?')} Area={m.get('area','?')} "
                    f"Power={m.get('power','?')} params={params}\n"
                )

        system_msg = (
            "You are an expert EDA engineer. Update the design prior knowledge document "
            "based on new experimental evidence.\n"
            "Rules:\n"
            "- Keep 400-800 words\n"
            "- Preserve correct info from original; correct anything contradicted by data\n"
            "- Add new insights from failure analysis and sensitivity data\n"
            "- End with updated tuning recommendations\n"
            "- Be specific with numbers and parameter values"
        )

        user_msg = f"ORIGINAL PRIOR:\n{current_prior}\n\nNEW EVIDENCE:\n{history_stats}\n{pareto_info}\n"
        if failure_summary:
            user_msg += f"FAILURE ANALYSIS:\n{failure_summary}\n\n"
        if sensitivity_summary:
            user_msg += f"SENSITIVITY ANALYSIS:\n{sensitivity_summary}\n\n"
        user_msg += "Please produce the UPDATED prior knowledge document."

        try:
            text, _raw = self.llm_client.create_text(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.1,
                max_output_tokens=self.max_output_tokens,
            )
            if text and text.strip():
                updated = text.strip()
                prior = DesignPrior(
                    design_name=self.design_name,
                    generated_ts=time.time(),
                    prior_text=updated,
                    scanned_files=[],
                    batch_summaries=[f"[refined] {updated[:200]}..."],
                )
                self._save(prior)
                print(f"[DesignPrior] refined prior saved ({len(updated)} chars)")
                return updated
        except Exception as e:
            print(f"[DesignPrior] refine failed: {e}")

        return current_prior

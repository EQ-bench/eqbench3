# File: ai/eqbench3/eqbench3.py

"""
Main entry point for the EQBench3 benchmark based on multi-turn scenarios.
Includes scenario simulation, debriefing, optional rubric scoring, and optional ELO analysis.
"""
import argparse
import sys
import signal
import logging
from datetime import datetime, timezone
import os # For path joining
from typing import Dict, Any
# Added import for leaderboard printing
import unicodedata

# Load environment variables early
from dotenv import load_dotenv
load_dotenv()

from utils.logging_setup import setup_logging, get_verbosity
from utils.file_io import load_json_file, update_run_data, save_json_file # save_json_file needed for reset
from core.benchmark import run_eq_bench3 # Import the main benchmark runner
import utils.constants as C # Import constants for default file paths


def signal_handler(signum, frame):
    """Handles graceful shutdown on signals like Ctrl+C."""
    print(f"\n[INFO] Signal {signum} received. Shutting down gracefully...")
    logging.info(f"Shutdown signal {signum} received.")
    # Perform any necessary cleanup here if needed
    sys.exit(1)

def print_summary_box(run_key: str, local_runs_file: str, run_elo: bool, run_rubric: bool):
    """
    Prints a formatted summary box of the benchmark run.
    Reads data ONLY from the local runs file.
    """
    try:
        # Load only local runs data for the summary box
        runs = load_json_file(local_runs_file)
        run_data = runs.get(run_key)
        if not run_data:
            print(f"\nError: Could not find run data for key {run_key} in local runs file {local_runs_file}")
            logging.warning(f"Summary box generation failed: Run key {run_key} not found in {local_runs_file}")
            return

        # Use model_name if available, fallback to test_model (legacy), then api_model_id
        model_name = run_data.get("model_name", run_data.get("test_model", "N/A"))
        api_model_id = run_data.get("api_model_id", "N/A")
        judge_model = run_data.get("judge_model", "N/A") # Judge model used for ELO and Rubric (legacy)
        rubric_judge_model = run_data.get("rubric_judge_model")
        elo_judge_model = run_data.get("elo_judge_model")
        start_time_str = run_data.get("start_time")
        end_time_str = run_data.get("end_time")
        run_status = run_data.get("status", "Unknown")

        duration_str = "N/A"
        if start_time_str and end_time_str:
            try:
                start_time = datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
                end_time = datetime.fromisoformat(end_time_str.replace("Z", "+00:00"))
                if start_time.tzinfo is None: start_time = start_time.replace(tzinfo=timezone.utc)
                if end_time.tzinfo is None: end_time = end_time.replace(tzinfo=timezone.utc)
                duration = end_time - start_time
                total_seconds = duration.total_seconds()
                hours, remainder = divmod(total_seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                duration_str = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
            except ValueError as e:
                duration_str = "Error parsing time"
                logging.error(f"Error parsing run duration times: {start_time_str}, {end_time_str} - {e}")

        results = run_data.get("results", {})

        # --- Rubric Score (Calculate 0-100) ---
        rubric_score_0_20 = results.get("average_rubric_score", "N/A") # This is the 0-20 score
        rubric_error = results.get("rubric_error")
        rubric_score_100_str = "N/A" # Initialize the display string

        if not run_rubric:
            rubric_score_100_str = "Skipped"
        elif rubric_error:
            rubric_score_100_str = f"Error ({rubric_error[:20]}...)"
        elif isinstance(rubric_score_0_20, (int, float)):
            # Scale the 0-20 score to 0-100
            rubric_score_0_100 = rubric_score_0_20 * 5.0
            rubric_score_100_str = f"{rubric_score_0_100:.2f}" # Format the 0-100 score
        else:
            # Handle cases where score might be "N/A" or other non-numeric string
             rubric_score_100_str = str(rubric_score_0_20)


        # --- ELO Score ---
        elo_raw = results.get("elo_raw", "N/A")
        elo_norm = results.get("elo_normalized", "N/A")
        elo_error = results.get("elo_error")
        elo_raw_str = "N/A"
        elo_norm_str = "N/A"

        if not run_elo:
            elo_raw_str = "Skipped"
            elo_norm_str = "Skipped"
        elif elo_error:
             elo_raw_str = f"Error ({elo_error[:20]}...)"
             elo_norm_str = f"Error ({elo_error[:20]}...)"
        else:
            elo_raw_str = f"{elo_raw:.2f}" if isinstance(elo_raw, (int, float)) else str(elo_raw)
            elo_norm_str = f"{elo_norm:.2f}" if isinstance(elo_norm, (int, float)) else str(elo_norm)

        import unicodedata

        # ──────────  helper: printable width  ──────────
        def cell_width(s: str) -> int:
            """Return the on‑screen width of s in monospaced fonts."""
            w = 0
            for ch in s:
                # ‘F’=Full, ‘W’=Wide → 2 columns; everything else → 1
                w += 2 if unicodedata.east_asian_width(ch) in ("F", "W") else 1
            return w

        def crop_to_width(s: str, max_w: int, ellipsis="…") -> str:
            """Trim string so its visual width ≤ max_w (add ellipsis if trimmed)."""
            if cell_width(s) <= max_w:
                return s + " " * (max_w - cell_width(s))
            keep_w = max_w - cell_width(ellipsis)
            out = ""
            for ch in s:
                ch_w = 2 if unicodedata.east_asian_width(ch) in ("F", "W") else 1
                if cell_width(out) + ch_w > keep_w:
                    break
                out += ch
            return out + ellipsis

        # ──────────  box parameters  ──────────
        BOX_W      = 80          # total width including borders
        PAD        = 1           # spaces inside each cell
        LINE       = "─"
        TOP        = "┌" + LINE * (BOX_W - 2) + "┐"
        BOTTOM     = "└" + LINE * (BOX_W - 2) + "┘"
        ROW_SEP    = "├" + LINE * (BOX_W - 2) + "┤"
        TITLE_SEP  = "╞" + LINE * (BOX_W - 2) + "╡"

        # ──────────  collect rows first  ──────────
        rows = [
            ("Run Key:",               run_key),
            ("Model Name:",            model_name), # Changed label
            ("API Model ID:",          api_model_id), # Added API ID
        ]
        if run_rubric:
            rows.append(("Judge (Rubric):", rubric_judge_model or judge_model))
        if run_elo:
            rows.append(("Judge (ELO):", elo_judge_model or judge_model))

        rows.extend([
            ("Status:",                run_status),
            ("Duration:",              duration_str),
            ("Rubric Score (0‑100):",  rubric_score_100_str),
            ("ELO Raw:",               elo_raw_str),
            ("ELO (Normalised):",      elo_norm_str),
        ])

        # ──────────  calculate column widths  ──────────
        max_label = max(cell_width(lbl) for lbl, _ in rows)
        # cap label col so value column is at least 15‑char wide
        # constant characters per row:
        #   • 3 border glyphs  │ │ │
        #   • 4 padding spaces (PAD on each side of each cell)
        CONST_CHARS = 3 + 4*PAD          # = 7 when PAD == 1

        label_col = min(max(cell_width(lbl) for lbl, _ in rows),
                        BOX_W - CONST_CHARS - 15)     # leave ≥15 for value col
        value_col = BOX_W - CONST_CHARS - label_col   # <-- corrected here

        def make_row(lbl: str, val: str) -> str:
            lbl_fmt = crop_to_width(lbl,  label_col)
            val_fmt = crop_to_width(val,  value_col)
            return (f"│{' '*PAD}{lbl_fmt}{' '*PAD}│"
                    f"{' '*PAD}{val_fmt}{' '*PAD}│")

        # ──────────  render  ──────────
        print("\n" + TOP)
        print(make_row("", "EQBench3 Results Summary".center(value_col)))
        print(TITLE_SEP)
        for (lbl, val) in rows:
            if lbl == "Duration:":
                print(ROW_SEP)
            if lbl == "Rubric Score (0‑100):":
                print(ROW_SEP)
                print(make_row("", "ELO Analysis Results".center(value_col)))
                print(ROW_SEP)
            print(make_row(lbl, str(val)))
        print(BOTTOM)


    except Exception as e:
        print(f"\nError generating summary box: {e}")
        logging.error(f"Error generating summary box for run {run_key}", exc_info=True)


def print_leaderboard_summary(elo_data: Dict[str, Any], highlight_model: str):
    """Prints a formatted leaderboard summary, sorted by ELO Norm."""
    if not elo_data:
        print("\n[INFO] No ELO data available to display leaderboard.")
        return

    # --- Prepare data ---
    leaderboard_entries = []
    for model, data in elo_data.items():
        if model == "__metadata__" or not isinstance(data, dict):
            continue
        # Use .get with defaults for robustness
        elo_norm = data.get("elo_norm", 0.0) # Default to 0 for sorting if missing
        elo_raw = data.get("elo", "N/A")
        sigma = data.get("sigma", "N/A")
        ci_low_norm = data.get("ci_low_norm", "N/A")
        ci_high_norm = data.get("ci_high_norm", "N/A")

        # Format CI string
        ci_str = "N/A"
        if ci_low_norm != "N/A" and ci_high_norm != "N/A":
            try:
                # Use norm CI for display
                ci_str = f"{ci_low_norm:.0f} - {ci_high_norm:.0f}"
            except (TypeError, ValueError):
                ci_str = "[Err]" # Handle non-numeric CI values

        leaderboard_entries.append({
            "name": model,
            "elo_norm": elo_norm if isinstance(elo_norm, (int, float)) else 0.0, # Ensure numeric for sort
            "elo_norm_disp": f"{elo_norm:.0f}" if isinstance(elo_norm, (int, float)) else str(elo_norm),
            "elo_raw_disp": f"{elo_raw:.0f}" if isinstance(elo_raw, (int, float)) else str(elo_raw),
            "ci_norm_disp": ci_str,
            "sigma_disp": f"{sigma:.1f}" if isinstance(sigma, (int, float)) else str(sigma),
        })

    # Sort by ELO Norm descending
    leaderboard_entries.sort(key=lambda x: x["elo_norm"], reverse=True)

    # --- Formatting Helpers (from print_summary_box) ---
    def cell_width(s: str) -> int:
        w = 0
        for ch in s: w += 2 if unicodedata.east_asian_width(ch) in ("F", "W") else 1
        return w

    def crop_to_width(s: str, max_w: int, ellipsis="…") -> str:
        s = str(s) # Ensure string conversion
        if cell_width(s) <= max_w: return s + " " * (max_w - cell_width(s))
        keep_w = max_w - cell_width(ellipsis)
        out = ""
        current_w = 0
        for ch in s:
            ch_w = 2 if unicodedata.east_asian_width(ch) in ("F", "W") else 1
            if current_w + ch_w > keep_w: break
            out += ch
            current_w += ch_w
        return out + ellipsis + " " * (max_w - cell_width(out + ellipsis))


    # --- Table Parameters ---
    COL_PAD = 1
    RANK_W = 4
    MODEL_W = 35 # Adjust as needed
    ELO_NORM_W = 8
    ELO_RAW_W = 8
    CI_W = 12 # Width for "[XXXX - YYYY]"
    SIGMA_W = 7

    cols = [RANK_W, MODEL_W, ELO_NORM_W, ELO_RAW_W, CI_W, SIGMA_W]
    LINE = "─"

    # --- Corrected Separator and Box Width Calculation ---
    # Calculate the width of each line segment including padding
    segments = [LINE * (w + 2 * COL_PAD) for w in cols]
    # Join segments with the correct column separator (┼)
    ROW_SEP_INNER = "┼".join(segments)
    # Create the full row separator line
    ROW_SEP = f"├{ROW_SEP_INNER}┤"
    # Calculate the total box width based on the correctly calculated separator
    BOX_W = len(ROW_SEP)

    # --- Recalculate other lines based on correct BOX_W ---
    TOP = "┌" + LINE * (BOX_W - 2) + "┐"
    BOTTOM = "└" + LINE * (BOX_W - 2) + "┘"
    TITLE_SEP = "╞" + LINE * (BOX_W - 2) + "╡"

    # --- Header Row ---
    headers = ["Rank", "Model Name", "ELO Norm", "ELO Raw", "95% CI Norm", "Sigma"]
    header_cells = [
        crop_to_width(h, w).center(w) # Center header text within its column width
        for h, w in zip(headers, cols)
    ]
    # Construct the header row string with padding and column separators
    header_parts = []
    for cell_content in header_cells:
        header_parts.append(f"{' ' * COL_PAD}{cell_content}{' ' * COL_PAD}")
    header_row = f"│{'│'.join(header_parts)}│" # Join padded cells with │

    # --- Print Table ---
    print("\n" + TOP)
    print(f"│{'EQBench3 Elo Results'.center(BOX_W - 2)}│") # Center title within the box
    print(TITLE_SEP)
    print(header_row)
    print(ROW_SEP) # Print the correctly calculated row separator directly

    for rank, entry in enumerate(leaderboard_entries, 1):
        is_highlighted = entry["name"] == highlight_model
        prefix = ">" if is_highlighted else " "
        rank_str = f"{prefix}{rank}"

        cells = [
            crop_to_width(rank_str, RANK_W),
            crop_to_width(entry["name"], MODEL_W),
            crop_to_width(entry["elo_norm_disp"], ELO_NORM_W).rjust(ELO_NORM_W),
            crop_to_width(entry["elo_raw_disp"], ELO_RAW_W).rjust(ELO_RAW_W),
            crop_to_width(entry["ci_norm_disp"], CI_W).center(CI_W),
            crop_to_width(entry["sigma_disp"], SIGMA_W).rjust(SIGMA_W),
        ]
        row_str = f"│{' ' * COL_PAD}" + f"{' ' * COL_PAD}│{' ' * COL_PAD}".join(cells) + f"{' ' * COL_PAD}│"
        print(row_str)

    print(BOTTOM)

def print_rubric_summary(runs_data: Dict[str, Any], highlight_model: str):
    """Prints a formatted leaderboard summary based on Rubric Scores."""
    if not runs_data:
        print("\n[INFO] No run data available to display Rubric leaderboard.")
        return

    # --- Prepare data ---
    rubric_entries = []
    for run_key, run_data in runs_data.items():
        if not isinstance(run_data, dict):
            continue

        model_name = run_data.get("model_name", run_data.get("test_model", "Unknown"))
        results = run_data.get("results", {})
        rubric_score_0_20 = results.get("average_rubric_score") # 0-20 scale
        rubric_error = results.get("rubric_error")

        score_100 = 0.0 # Default for sorting if missing/error
        score_disp = "N/A"

        if rubric_error:
            score_disp = "Error"
        elif rubric_score_0_20 == "Skipped":
             score_disp = "Skipped"
        elif isinstance(rubric_score_0_20, (int, float)):
            score_100 = rubric_score_0_20 * 5.0
            score_disp = f"{score_100:.1f}" # Display with 1 decimal place
        elif rubric_score_0_20 is not None: # Handle other non-numeric strings like "N/A"
             score_disp = str(rubric_score_0_20)

        # Only include entries that have a valid score attempt (not skipped and has results)
        if "results" in run_data and rubric_score_0_20 != "Skipped":
            rubric_entries.append({
                "name": model_name,
                "score_100": score_100,
                "score_disp": score_disp,
            })

    # Sort by Rubric Score (0-100) descending
    rubric_entries.sort(key=lambda x: x["score_100"], reverse=True)

    # --- Formatting Helpers (copied from print_leaderboard_summary) ---
    def cell_width(s: str) -> int:
        w = 0
        for ch in s: w += 2 if unicodedata.east_asian_width(ch) in ("F", "W") else 1
        return w

    def crop_to_width(s: str, max_w: int, ellipsis="…") -> str:
        s = str(s) # Ensure string conversion
        if cell_width(s) <= max_w: return s + " " * (max_w - cell_width(s))
        keep_w = max_w - cell_width(ellipsis)
        out = ""
        current_w = 0
        for ch in s:
            ch_w = 2 if unicodedata.east_asian_width(ch) in ("F", "W") else 1
            if current_w + ch_w > keep_w: break
            out += ch
            current_w += ch_w
        return out + ellipsis + " " * (max_w - cell_width(out + ellipsis))

    # --- Table Parameters ---
    COL_PAD = 1
    RANK_W = 4
    MODEL_W = 50 # Wider for model names
    SCORE_W = 15 # Width for "XXX.X / 100"

    cols = [RANK_W, MODEL_W, SCORE_W]
    LINE = "─"

    # --- Separator and Box Width Calculation ---
    segments = [LINE * (w + 2 * COL_PAD) for w in cols]
    ROW_SEP_INNER = "┼".join(segments)
    ROW_SEP = f"├{ROW_SEP_INNER}┤"
    BOX_W = len(ROW_SEP)

    # --- Recalculate other lines based on correct BOX_W ---
    TOP = "┌" + LINE * (BOX_W - 2) + "┐"
    BOTTOM = "└" + LINE * (BOX_W - 2) + "┘"
    TITLE_SEP = "╞" + LINE * (BOX_W - 2) + "╡"

    # --- Header Row ---
    headers = ["Rank", "Model Name", "Rubric (0-100)"]
    header_cells = [
        crop_to_width(h, w).center(w)
        for h, w in zip(headers, cols)
    ]
    header_parts = [f"{' ' * COL_PAD}{cell_content}{' ' * COL_PAD}" for cell_content in header_cells]
    header_row = f"│{'│'.join(header_parts)}│"

    # --- Print Table ---
    print("\n" + TOP)
    print(f"│{'EQBench3 Rubric Score Summary'.center(BOX_W - 2)}│")
    print(TITLE_SEP)
    print(header_row)
    print(ROW_SEP)

    for rank, entry in enumerate(rubric_entries, 1):
        is_highlighted = entry["name"] == highlight_model
        prefix = ">" if is_highlighted else " "
        rank_str = f"{prefix}{rank}"

        cells = [
            crop_to_width(rank_str, RANK_W),
            crop_to_width(entry["name"], MODEL_W),
            crop_to_width(entry["score_disp"], SCORE_W).rjust(SCORE_W),
        ]
        row_parts = [f"{' ' * COL_PAD}{cell_content}{' ' * COL_PAD}" for cell_content in cells]
        row_str = f"│{'│'.join(row_parts)}│"
        print(row_str)

    print(BOTTOM)

def _reset_model_data(model_name: str, local_runs_file: str, local_elo_file: str):
    """
    Remove all entries for the logical `model_name` from the LOCAL files:
      • local_runs_file (main run data)
      • local_elo_file (ELO results and pairwise comparisons)
    Does NOT touch leaderboard files.
    """
    # --- purge local runs file ---
    runs = load_json_file(local_runs_file) or {}
    to_delete = [k for k,v in runs.items()
                 if isinstance(v, dict) and v.get("model_name", v.get("test_model")) == model_name] # Check model_name first, fallback to test_model
    deleted_count = 0
    for k in to_delete:
        if k in runs:
            del runs[k]
            deleted_count += 1
    if deleted_count > 0:
        save_json_file(runs, local_runs_file)
        logging.info(f"[RESET] Removed {deleted_count} run(s) for '{model_name}' from local runs file: {local_runs_file}")

    # --- purge local ELO file ---
    if not os.path.exists(local_elo_file):
        logging.info(f"[RESET] Local ELO file not found, skipping purge: {local_elo_file}")
        return

    elo = load_json_file(local_elo_file) or {}
    elo_changed = False
    # Check if the logical model name exists as a top-level key
    if model_name in elo:
        del elo[model_name]
        elo_changed = True
        logging.info(f"[RESET] Removed top-level ELO entry for '{model_name}' from {local_elo_file}")

    meta = elo.setdefault("__metadata__", {})
    comps = meta.get("global_pairwise_comparisons", [])
    initial_comp_count = len(comps)
    # Filter comparisons based on logical model names stored in the pair dict
    comps_filtered = [
        c for c in comps
        if model_name not in (
            c.get("pair",{}).get("test_model"), # These should hold logical names
            c.get("pair",{}).get("neighbor_model")
        )
    ]
    if len(comps_filtered) < initial_comp_count:
        meta["global_pairwise_comparisons"] = comps_filtered
        elo_changed = True
        logging.info(f"[RESET] Pruned {initial_comp_count - len(comps_filtered)} comparisons involving '{model_name}' from {local_elo_file}")

    if elo_changed:
        save_json_file(elo, local_elo_file)
        logging.info(f"[RESET] Saved updated local ELO file: {local_elo_file}")
    else:
        logging.info(f"[RESET] No data found for '{model_name}' in local ELO file: {local_elo_file}")


def _redo_elo_for_model(model_name: str,
                        local_runs_file: str,
                        local_elo_file: str,
                        leaderboard_runs_file: str,
                        leaderboard_elo_file: str):
    """
    Purge all ELO comparisons for *model_name* so that ELO judging is
    re-done from scratch on the next (resumed) run.

    Algorithm:
        1. Unmerge ALL models from canonical -> local
        2. Purge every comparison involving *model_name* from local
        3. Merge back only the models that were originally in canonical
           (minus *model_name*), so pre-existing local models are untouched
        4. Recalculate canonical ELO after the merge-back
        5. Atomically save all four files
    """
    import copy
    from utils.merge_utils import (
        unmerge_data, merge_data, purge_model_comparisons,
        _recalculate_elo_ratings, _atomic_multi_save,
    )

    logging.info(f"[REDO-ELO] Starting ELO reset for model '{model_name}'")

    # --- Load all four files ---
    canonical_runs = load_json_file(leaderboard_runs_file) or {}
    canonical_elo  = load_json_file(leaderboard_elo_file)  or {}
    local_runs     = load_json_file(local_runs_file)       or {}
    local_elo      = load_json_file(local_elo_file)        or {}
    for d in [canonical_runs, canonical_elo, local_runs, local_elo]:
        d.setdefault("__metadata__", {})

    # Work on deep copies so we can abort without touching originals
    canonical_runs_c = copy.deepcopy(canonical_runs)
    canonical_elo_c  = copy.deepcopy(canonical_elo)
    local_runs_c     = copy.deepcopy(local_runs)
    local_elo_c      = copy.deepcopy(local_elo)

    # --- Step 1: record which models are currently in canonical ---
    originally_canonical = set(k for k in canonical_elo_c if k != "__metadata__")
    logging.info(f"[REDO-ELO] {len(originally_canonical)} models currently in canonical")

    # --- Step 2: unmerge ALL canonical -> local ---
    if originally_canonical:
        all_candidates = [{"model_name": m, "run_keys": []} for m in originally_canonical]
        if not unmerge_data(all_candidates, local_runs_c, local_elo_c,
                            canonical_runs_c, canonical_elo_c):
            logging.error("[REDO-ELO] Unmerge failed. Aborting.")
            sys.exit(1)

    # --- Step 3: purge target model's comparisons + ELO entry from local ---
    purge_model_comparisons(local_elo_c, model_name)

    # --- Step 4: merge back only the originally-canonical models (minus target) ---
    models_to_merge_back = originally_canonical - {model_name}
    if models_to_merge_back:
        merge_candidates = [{"run_key": "redo-elo", "model_name": m}
                            for m in models_to_merge_back]
        if not merge_data(merge_candidates, local_runs_c, local_elo_c,
                          canonical_runs_c, canonical_elo_c):
            logging.error("[REDO-ELO] Merge-back failed. Aborting.")
            sys.exit(1)

        # Recalculate canonical ELO after merge-back
        logging.info("[REDO-ELO] Recalculating canonical ELO after merge-back...")
        if not _recalculate_elo_ratings(canonical_elo_c):
            logging.error("[REDO-ELO] Canonical ELO recalculation failed. Aborting.")
            sys.exit(1)

    # --- Step 5: atomically save all four files ---
    files_to_save = {
        local_runs_file:       local_runs_c,
        local_elo_file:        local_elo_c,
    }
    # Only include canonical files if they were actually loaded (not --ignore-canonical)
    if leaderboard_runs_file:
        files_to_save[leaderboard_runs_file] = canonical_runs_c
    if leaderboard_elo_file:
        files_to_save[leaderboard_elo_file] = canonical_elo_c
    if not _atomic_multi_save(files_to_save):
        logging.error("[REDO-ELO] Atomic save failed. No files were modified.")
        sys.exit(1)

    logging.info(f"[REDO-ELO] Successfully purged ELO data for '{model_name}'. "
                 f"ELO judging will be re-done on the resumed run.")


def main():
    parser = argparse.ArgumentParser(description="Run EQBench3 Scenario Benchmark.")
    # --- Model Identifiers ---
    parser.add_argument("--test-model", required=True, help="Identifier for the model sent to the API (e.g., 'openai/gpt-4o'). This is the API Model ID.")
    parser.add_argument("--model-name", help="Logical identifier for the model (e.g., 'gpt-4o-june-2024') used for tracking and leaderboards. Defaults to the value of --test-model if not provided.")
    parser.add_argument("--judge-model", help="Identifier for the judge model used for ELO pairwise comparisons and/or Rubric scoring.")
    # --- File Paths ---
    parser.add_argument("--runs-file", default=C.DEFAULT_LOCAL_RUNS_FILE, help=f"File to store local run data (default: {C.DEFAULT_LOCAL_RUNS_FILE}).")
    parser.add_argument("--elo-results-file", default=C.DEFAULT_LOCAL_ELO_FILE, help=f"File to store local ELO results and comparisons (default: {C.DEFAULT_LOCAL_ELO_FILE}).")
    parser.add_argument("--leaderboard-runs-file", default=C.CANONICAL_LEADERBOARD_RUNS_FILE, help=f"Path to the canonical leaderboard runs file (read-only, default: {C.CANONICAL_LEADERBOARD_RUNS_FILE}).")
    parser.add_argument("--leaderboard-elo-file", default=C.CANONICAL_LEADERBOARD_ELO_FILE, help=f"Path to the canonical leaderboard ELO file (read-only, default: {C.CANONICAL_LEADERBOARD_ELO_FILE}).")
    # --- Run Control ---
    parser.add_argument("--run-id", help="Optional: Resume or specify a run ID prefix for the local run.")
    parser.add_argument("--threads", type=int, default=4, help="Number of parallel threads for API calls.")
    parser.add_argument("--verbosity", choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'], default="INFO", help="Logging verbosity level.")
    parser.add_argument("--save-interval", type=int, default=2, help="How often (in tasks) to save partial progress to local files.")
    parser.add_argument("--iterations", type=int, default=1, help="Number of times to run each scenario (for assessing variance).")
    # --- Feature Flags ---
    parser.add_argument(
        "--ignore-canonical",
        action="store_true",
        default=False,
        help="If set, do not load or use default canonical leaderboard files. Runs will be based on local files only."
    )
    parser.add_argument("--no-elo", action="store_true", default=False, help="Disable the ELO analysis step.")
    parser.add_argument("--no-rubric", action="store_true", default=False, help="Disable the Rubric scoring step.")
    parser.add_argument("--redo-rubric-judging", action="store_true", default=False,
                        help="If set, tasks in the local runs file that have completed rubric scoring will be reset so the rubric step is re-run.")
    parser.add_argument(
        "--repair-rubric",
        action="store_true",
        default=False,
        help="Repair only interrupted, failed, or score-missing rubric tasks in the resumed run.",
    )
    parser.add_argument(
        "--repair-scenario",
        action="store_true",
        default=False,
        help="Repair only interrupted or failed scenario and debrief tasks in the resumed run.",
    )
    parser.add_argument(
        "--reset-model",
        action="store_true",
        default=False,
        help="Delete all existing run data and ELO comparisons for the logical model name from the LOCAL files before running.",
    )
    parser.add_argument(
        "--redo-elo",
        action="store_true",
        default=False,
        help="Purge all ELO comparisons for the model (from both canonical and local), then resume the run so ELO judging is re-done from scratch. Requires --run-id.",
    )
    # --- Removed file path arguments (now handled via constants) ---
    # parser.add_argument("--scenario-prompts-file", ...)
    # parser.add_argument("--debrief-prompt-file", ...)
    # parser.add_argument("--pairwise-prompt-file", ...)
    # parser.add_argument("--rubric-criteria-file", ...)
    # parser.add_argument("--rubric-prompt-file", ...)

    args = parser.parse_args()

    # Determine the logical model name and API model ID
    api_model_id = args.test_model
    logical_model_name = args.model_name if args.model_name else api_model_id

    # Setup logging first
    setup_logging(get_verbosity(args.verbosity))

    actual_leaderboard_runs_file = args.leaderboard_runs_file
    actual_leaderboard_elo_file = args.leaderboard_elo_file

    if args.ignore_canonical:
        logging.info("--ignore-canonical flag is set. Canonical leaderboard files will not be loaded or used.")
        actual_leaderboard_runs_file = None # load_json_file will return {} for this
        actual_leaderboard_elo_file = None  # load_json_file will return {} for this
    else:
        # Optional: Check existence and warn if not found, though load_json_file handles it
        if not os.path.exists(args.leaderboard_runs_file):
            logging.warning(f"Canonical leaderboard runs file not found: {args.leaderboard_runs_file}. Will proceed as if empty. Use --ignore-canonical to run purely locally and suppress this warning.")
        if not os.path.exists(args.leaderboard_elo_file):
            logging.warning(f"Canonical leaderboard ELO file not found: {args.leaderboard_elo_file}. Will proceed as if empty. Use --ignore-canonical to run purely locally and suppress this warning.")

    logging.info("--- EQBench3 Run Start ---")
    logging.info(f"Logical Model Name: {logical_model_name}")
    logging.info(f"API Model ID: {api_model_id}")
    logging.info(f"Local Runs File: {args.runs_file}")
    logging.info(f"Local ELO File: {args.elo_results_file}")
    logging.info(f"Leaderboard Runs File (effective): {actual_leaderboard_runs_file if actual_leaderboard_runs_file else 'Ignored'}")
    logging.info(f"Leaderboard ELO File (effective): {actual_leaderboard_elo_file if actual_leaderboard_elo_file else 'Ignored'}")
    logging.debug(f"Full Arguments: {args}")

    if args.reset_model:
        # Reset using the logical model name and LOCAL file paths
        _reset_model_data(logical_model_name, args.runs_file, args.elo_results_file)

    if args.redo_elo:
        if not args.run_id:
            parser.error("--redo-elo requires --run-id to resume the existing run after purging ELO data.")
        _redo_elo_for_model(
            model_name=logical_model_name,
            local_runs_file=args.runs_file,
            local_elo_file=args.elo_results_file,
            leaderboard_runs_file=actual_leaderboard_runs_file,
            leaderboard_elo_file=actual_leaderboard_elo_file,
        )

    # Validate arguments
    run_elo_flag = not args.no_elo
    run_rubric_flag = not args.no_rubric

    if (args.repair_rubric or args.repair_scenario) and not args.run_id:
        parser.error("--repair-rubric and --repair-scenario require --run-id to target an existing run.")
    if args.repair_rubric and args.no_rubric:
        parser.error("--repair-rubric cannot be used with --no-rubric.")
    if args.repair_rubric and args.redo_rubric_judging:
        parser.error("--repair-rubric and --redo-rubric-judging are mutually exclusive.")

    if (run_elo_flag or run_rubric_flag) and not args.judge_model:
        parser.error("--judge-model is required unless both --no-elo and --no-rubric are specified.")
        sys.exit(1)

    # Hook signals for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    run_key = None
    try:
        # Call run_eq_bench3 with logical name, API ID, and all file paths
        run_key = run_eq_bench3(
            model_name=logical_model_name,
            api_model_id=api_model_id,
            judge_model=args.judge_model if (run_elo_flag or run_rubric_flag) else None,
            # File Paths
            local_runs_file=args.runs_file,
            local_elo_file=args.elo_results_file,
            leaderboard_runs_file=actual_leaderboard_runs_file,
            leaderboard_elo_file=actual_leaderboard_elo_file,
            # Run Control
            num_threads=args.threads,
            run_id=args.run_id,
            save_interval=args.save_interval,
            iterations=args.iterations,
            # Feature Flags
            run_elo=run_elo_flag,
            run_rubric=run_rubric_flag,
            redo_judging=args.redo_rubric_judging,
            repair_scenarios=args.repair_scenario,
            repair_rubric=args.repair_rubric,
            truncate_for_rubric=False, # Hardcoded for now, could be arg
        )

        logging.info(f"EQBench3 run completed. Run key: {run_key}")
        print(f"\nEQBench3 benchmark completed. Run key: {run_key}")

    except Exception as e:
        logging.critical(f"An unhandled error occurred during the benchmark run: {e}", exc_info=True)
        print(f"\nFATAL ERROR during benchmark run: {e}")
        if run_key and args.runs_file:
            try:
                # Update status in the LOCAL runs file
                update_run_data(args.runs_file, run_key, {
                    "status": "error",
                    "error": f"Unhandled exception: {str(e)}",
                    "end_time": datetime.now(timezone.utc).isoformat()
                })
                logging.info(f"Marked run {run_key} as errored in {args.runs_file}")
            except Exception as update_e:
                logging.error(f"Could not update run status to error for {run_key}: {update_e}")
        sys.exit(1)

    # Print Summary Box if run completed (even with errors), using LOCAL runs file
    if run_key:        
        # Print Rubric Leaderboard Summary if Rubric scoring was run
        if run_rubric_flag and run_key:
            try:
                logging.info("Loading merged run data for Rubric leaderboard display...")
                # Load final data from both sources, respecting --ignore-canonical
                final_leaderboard_runs = {}
                if not args.ignore_canonical:
                    final_leaderboard_runs = load_json_file(actual_leaderboard_runs_file)
                else:
                    logging.info("Rubric summary: Not loading canonical runs due to --ignore-canonical.")
                final_local_runs = load_json_file(args.runs_file)
                final_merged_runs = {**final_leaderboard_runs, **final_local_runs}

                if final_merged_runs:
                    print_rubric_summary(final_merged_runs, logical_model_name)
                else:
                    logging.warning("Could not load merged run data for Rubric leaderboard display.")
            except Exception as e:
                logging.error(f"Failed to load or print Rubric leaderboard summary: {e}", exc_info=True)
        elif not run_rubric_flag:
            logging.info("Skipping Rubric leaderboard display because Rubric scoring was disabled.")
            
        # Print Leaderboard Summary if ELO was run successfully
        if run_elo_flag:
            # Check the status in the run file to see if ELO step had an error stored
            final_run_data = load_json_file(args.runs_file).get(run_key, {})
            elo_step_error = final_run_data.get("results", {}).get("elo_error")

            if not elo_step_error:
                try:
                    logging.info(f"Loading final ELO data from {args.elo_results_file} for leaderboard display.")
                    # Load the local ELO file which now contains the full snapshot
                    final_local_elo_data = load_json_file(args.elo_results_file)
                    if final_local_elo_data:
                        print_leaderboard_summary(final_local_elo_data, logical_model_name)
                    else:
                        logging.warning(f"Could not load local ELO data from {args.elo_results_file} for leaderboard display.")
                except Exception as e:
                    logging.error(f"Failed to load or print leaderboard summary: {e}", exc_info=True)
            else:
                logging.warning(f"Skipping leaderboard display because ELO analysis reported an error: {elo_step_error}")
        else:
            logging.info("Skipping leaderboard display because ELO analysis was disabled.")
            
        print_summary_box(run_key, args.runs_file, run_elo_flag, run_rubric_flag)


if __name__ == "__main__":
    main()
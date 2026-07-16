
# File: ai/eqbench3/core/benchmark.py

# core/benchmark.py

import os
import re
import uuid
import time
import logging
import json # For constructing rubric output format
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import queue
import threading
import statistics # For averaging rubric scores
from pathlib import Path

from utils.file_io import load_json_file, update_run_data, save_json_file
from utils.api import APIClient
from core.conversation import ScenarioTask
from core.elo import run_elo_analysis_eqbench3 # Keep existing import
# Import constants including file paths and scenario type IDs
import utils.constants as C
from collections import defaultdict

ALLOW_INCOMPLETE_RESPONSES = True

# --- Helper Function for the Save Worker Thread ---
def _save_worker(save_queue: queue.Queue, local_runs_file: str, batch_size: int = 10):
    """
    Save‑worker thread. Writes ONLY to the local runs file.
    Accumulates `batch_size` queue items before writing to disk, to reduce I/O.
    If a sentinel (None) is received, any remaining queued items are flushed
    immediately before the thread exits.

    Args:
        save_queue (queue.Queue):   Queue populated by producer threads.
        local_runs_file (str):      Path to the LOCAL JSON file holding run data.
        batch_size (int, optional): Number of tasks to buffer before saving. Defaults to 10.
    """
    logging.info(f"[SaveWorker] Save worker thread started. Target file: {local_runs_file}")

    # -----------------------------  helper: flush_batch  ----------------------------- #
    def flush_batch(item_batch: list):
        """
        Write all queued items in `item_batch` to the local runs file, grouped by run_key.
        Clears `item_batch` on completion.
        """
        if not item_batch:
            return

        # Group pending updates by run_key to minimise file operations
        grouped_updates: dict[str, dict] = defaultdict(lambda: {"scenario_tasks": {}})

        for run_key, iteration_index, scenario_id, task_data in item_batch:
            iter_dict = grouped_updates[run_key]["scenario_tasks"].setdefault(str(iteration_index), {})
            iter_dict[str(scenario_id)] = task_data

        for run_key, update_dict in grouped_updates.items():
            # Ensure writing ONLY to the local file
            ok = update_run_data(local_runs_file, run_key, update_dict, max_retries=5, retry_delay=0.75)
            if ok:
                logging.debug(
                    f"[SaveWorker] Flushed {len(item_batch)} tasks for run {run_key} to {local_runs_file}."
                )
            else:
                logging.error(f"[SaveWorker] Failed to flush batch for run {run_key} to {local_runs_file}.")

        item_batch.clear()
    # ------------------------------------------------------------------------------- #

    pending_items: list[tuple[str, int, int, dict]] = []

    while True:
        try:
            item = save_queue.get()  # block until an item arrives

            # Sentinel => flush anything buffered, then exit
            if item is None:
                logging.info("[SaveWorker] Sentinel received. Flushing remaining tasks.")
                flush_batch(pending_items)
                save_queue.task_done()
                break

            pending_items.append(item)

            # If buffer full, write to disk
            if len(pending_items) >= batch_size:
                flush_batch(pending_items)

            save_queue.task_done()

        except Exception as e:
            logging.error("[SaveWorker] Error handling queue item: %s", e, exc_info=True)
            try:
                save_queue.task_done()
            except ValueError:
                pass  # task_done() called too many times

    logging.info("[SaveWorker] Save worker thread finished.")



# --- (Keep parse_scenario_prompts as is) ---
def parse_scenario_prompts(file_path: str) -> Dict[str, List[str]]:
    """Parses the scenario prompts file into a dictionary (Revised Logic)."""
    scenarios: Dict[str, List[str]] = {}
    current_scenario_id: Optional[str] = None
    current_prompts_for_scenario: List[str] = []
    current_prompt_lines: List[str] = []
    in_prompt_content = False # Flag to indicate if we are currently reading lines for a prompt

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, raw_line in enumerate(f, 1):
                line = raw_line.strip()

                # Check for delimiters first
                scenario_match = re.match(r'^########\s*(\S+)', line)
                prompt_match = re.match(r'^#######\s*Prompt(\d+)', line)

                # --- Handle Scenario Start ---
                if scenario_match:
                    # 1. Finalize the last prompt of the previous scenario (if any)
                    if current_prompt_lines:
                        prompt_text = "\n".join(current_prompt_lines).strip()
                        if prompt_text:
                            current_prompts_for_scenario.append(prompt_text)
                        current_prompt_lines = [] # Reset for the new scenario

                    # 2. Store the completed previous scenario (if any)
                    if current_scenario_id and current_prompts_for_scenario:
                        scenarios[current_scenario_id] = current_prompts_for_scenario
                        logging.debug(f"Stored scenario {current_scenario_id} with {len(current_prompts_for_scenario)} prompts.")

                    # 3. Start the new scenario
                    current_scenario_id = scenario_match.group(1)
                    current_prompts_for_scenario = []
                    in_prompt_content = False # Reset flag, wait for a Prompt delimiter
                    logging.debug(f"Starting parse for scenario {current_scenario_id} (Line {line_num})")
                    continue # Move to next line

                # --- Handle Prompt Start ---
                elif prompt_match:
                    if current_scenario_id is None:
                        logging.warning(f"Line {line_num}: Found prompt delimiter but no active scenario ID: {line}")
                        continue

                    # 1. Finalize the previous prompt within the current scenario (if any)
                    if current_prompt_lines:
                        prompt_text = "\n".join(current_prompt_lines).strip()
                        if prompt_text:
                            current_prompts_for_scenario.append(prompt_text)

                    # 2. Start collecting lines for the new prompt
                    current_prompt_lines = []
                    in_prompt_content = True # Start collecting content lines
                    logging.debug(f"Starting Prompt {prompt_match.group(1)} for scenario {current_scenario_id} (Line {line_num})")
                    continue # Move to next line

                # --- Handle Content Lines ---
                elif current_scenario_id and in_prompt_content:
                    # Append the raw line (preserving leading/trailing whitespace within the prompt)
                    # but skip truly empty lines between prompts if desired (using strip check above)
                    current_prompt_lines.append(raw_line.rstrip('\n\r')) # Keep indentation, remove trailing newline

                # --- Handle other lines (e.g., comments, blank lines between scenarios/prompts) ---
                elif line: # Log unexpected non-empty lines if not collecting content
                     if not current_scenario_id:
                          logging.debug(f"Line {line_num}: Skipping non-empty line before first scenario: {line[:50]}...")
                     elif not in_prompt_content:
                          logging.debug(f"Line {line_num}: Skipping non-empty line before first prompt in scenario {current_scenario_id}: {line[:50]}...")
                     # else: line is content, handled above

            # --- After loop: Finalize the last prompt and scenario ---
            if current_prompt_lines:
                prompt_text = "\n".join(current_prompt_lines).strip()
                if prompt_text:
                    current_prompts_for_scenario.append(prompt_text)

            if current_scenario_id and current_prompts_for_scenario:
                scenarios[current_scenario_id] = current_prompts_for_scenario
                logging.debug(f"Stored final scenario {current_scenario_id} with {len(current_prompts_for_scenario)} prompts.")

    except FileNotFoundError:
        logging.error(f"Scenario prompts file not found: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error parsing scenario prompts file {file_path}: {e}", exc_info=True)
        raise

    if not scenarios:
         logging.warning(f"Parsing finished, but no scenarios were loaded from {file_path}.")
    else:
         logging.info(f"Successfully parsed {len(scenarios)} scenarios from {file_path}")

    return scenarios


# --- Function to calculate final rubric score ---
def calculate_final_rubric_score(run_data: Dict[str, Any]) -> Tuple[Optional[float], Optional[str]]:
    """
    Calculates the average rubric score across all completed tasks in a run.
    Considers different criteria sets for standard/drafting vs analysis tasks.
    Averages scores for allowed criteria within a task first, then averages these task scores.
    Returns (average_score, error_message).
    """
    # Define the specific criteria to include for standard/drafting tasks
    STANDARD_ALLOWED_RUBRIC_CRITERIA = {
        "demonstrated_empathy",
        "pragmatic_ei",
        "depth_of_insight",
        "social_dexterity",
        "emotional_reasoning",
        "message_tailoring",
        "theory_of_mind",
        "subtext_identification",
        "intellectual_grounding",
        "correctness"
    }
    # Define the specific criteria to include for analysis tasks
    # Assumes analysis criteria are loaded from ANALYSIS_RUBRIC_CRITERIA_FILE
    # We need to load them here or pass them in. Let's load them.
    analysis_criteria = []
    try:
        with open(C.ANALYSIS_RUBRIC_CRITERIA_FILE, 'r', encoding='utf-8') as f:
            analysis_criteria = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
        ANALYSIS_ALLOWED_RUBRIC_CRITERIA = set(analysis_criteria)
        if not ANALYSIS_ALLOWED_RUBRIC_CRITERIA:
             logging.warning(f"Analysis rubric criteria file loaded but no criteria found: {C.ANALYSIS_RUBRIC_CRITERIA_FILE}")
    except Exception as e:
        logging.error(f"Failed to load analysis rubric criteria for score calculation: {e}. Analysis tasks will not contribute to score.")
        ANALYSIS_ALLOWED_RUBRIC_CRITERIA = set()


    logging.info(f"Calculating final rubric score using criteria:")
    logging.info(f"  Standard/Drafting: {', '.join(sorted(STANDARD_ALLOWED_RUBRIC_CRITERIA))}")
    logging.info(f"  Analysis: {', '.join(sorted(ANALYSIS_ALLOWED_RUBRIC_CRITERIA))}")

    scenario_tasks_data = run_data.get("scenario_tasks", {})
    task_average_scores = []
    tasks_considered = 0
    tasks_with_scores = 0
    total_valid_scores_considered = 0 # Track scores used in the average
    run_key_for_log = run_data.get("run_key", "UnknownRun") # Get run key for logging

    for iter_str, scenario_dict in scenario_tasks_data.items():
        if not isinstance(scenario_dict, dict): continue
        for sid, task_info in scenario_dict.items():
            if isinstance(task_info, dict) and task_info.get("status") == "rubric_scored":
                tasks_considered += 1
                rubric_scores = task_info.get("rubric_scores")
                is_analysis = sid in C.ANALYSIS_SCENARIO_IDS
                allowed_criteria = ANALYSIS_ALLOWED_RUBRIC_CRITERIA if is_analysis else STANDARD_ALLOWED_RUBRIC_CRITERIA

                if isinstance(rubric_scores, dict) and rubric_scores:
                    # Calculate average score for this task using ONLY allowed criteria for its type
                    valid_task_scores = []
                    for metric, score in rubric_scores.items():
                        if isinstance(score, (int, float)) and metric in allowed_criteria:
                            weight = 3 if metric == "overall_eq" else 1
                            valid_task_scores.extend([score] * weight)


                    if valid_task_scores:
                        task_avg = statistics.mean(valid_task_scores)
                        task_average_scores.append(task_avg)
                        tasks_with_scores += 1
                        total_valid_scores_considered += len(valid_task_scores) # Count only used scores
                    else:
                        logging.warning(f"Task {run_key_for_log}/{iter_str}/{sid} (Analysis: {is_analysis}) has status 'rubric_scored' but no valid numeric scores found for the *allowed* criteria ({', '.join(sorted(allowed_criteria))}).")
                else:
                    logging.warning(f"Task {run_key_for_log}/{iter_str}/{sid} (Analysis: {is_analysis}) has status 'rubric_scored' but 'rubric_scores' is missing or empty.")

    if not task_average_scores:
        if tasks_considered > 0:
            return None, f"No valid rubric scores found for allowed criteria in {tasks_considered} potentially scored tasks."
        else:
            return None, "No tasks reached the 'rubric_scored' state."

    final_average_score = statistics.mean(task_average_scores)
    logging.info(f"Calculated final rubric score: {final_average_score:.2f} (averaged from {tasks_with_scores} tasks with scores, using {total_valid_scores_considered} total score points from allowed criteria).")
    return round(final_average_score, 2), None


# --- Helper function for executing rubric scoring in a thread ---
def _execute_rubric_scoring_task(
    task: ScenarioTask,
    api_clients: Dict[str, APIClient],
    judge_model_id: str,
    # Pass the specific template and format string for this task type
    rubric_prompt_template: str,
    rubric_output_format_str: str,
    save_queue: queue.Queue,
    run_key: str,
    truncate_for_rubric: bool # Keep flag
):
    """Target function for rubric scoring threads."""
    judge_api = api_clients.get("judge")
    if not judge_api:
        logging.error(f"Judge API client not found for task {task.scenario_id} (Iter {task.iteration_index}).")
        task.status = "error"
        task.error = "Rubric Scoring Error: Judge API client missing."
        task.rubric_run_error = "Judge API client missing."
        task._save_progress(save_queue, run_key)
        return

    # Set status and save progress before starting
    task.status = "running_rubric_scoring"
    task.rubric_run_error = None # Clear previous error if retrying
    task.error = None
    task._save_progress(save_queue, run_key)

    try:
        # 1. Prepare the prompt using the task's helper method, passing the truncation flag
        # The helper method now handles different task types (incl. analysis)
        prompt_text = task.prepare_rubric_prompt_text(
            rubric_prompt_template, # Pass the specific template
            rubric_output_format_str, # Pass the specific format string
            truncate_for_rubric # Pass the flag
        )
        prompt_text = prompt_text.replace('*', '').replace('#', '')
        # print(f"--- Rubric Prompt (Task: {task.scenario_id}, Iter: {task.iteration_index}) ---")
        # print(prompt_text)
        # print("--- End Rubric Prompt ---")
        if prompt_text is None: # Handle error during prompt preparation
            # Error is logged and status set within prepare_rubric_prompt_text if it fails
            # Ensure state is saved
            task._save_progress(save_queue, run_key)
            return

        # 2. Call the Judge API
        logging.debug(f"Calling judge API ({judge_model_id}) for rubric scoring: Task {task.scenario_id} (Iter {task.iteration_index})")
        response_text = judge_api.generate(
            model=judge_model_id,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0.0, # Zero temp for deterministic scoring
            max_tokens=8000, # Enough for scores and reasoning
            min_p=None # No min_p for judge
        )

        # 3. Process the response using the task's helper method
        task.process_rubric_response(response_text.strip())
        # Status (rubric_scored or error), scores, errors are set within process_rubric_response

    except Exception as e:
        # Catch errors during API call or unexpected errors in helpers
        error_msg = f"Rubric Scoring Error: API call failed or processing error: {str(e)}"
        logging.error(f"Error during rubric scoring execution for task {task.scenario_id} (Iter {task.iteration_index}): {e}", exc_info=True)
        task.status = "error"
        task.error = error_msg
        task.rubric_run_error = str(e)
        # Clear potentially partial results from failed processing
        task.rubric_scores = None
        task.raw_rubric_judge_text = None

    finally:
        # 4. Save the final state of the task after this step
        task._save_progress(save_queue, run_key)

# ---------- completeness-check helpers ------------------------------------
_MIN_RAW_LEN = 100

MANDATORY_SECTIONS_ROLEPLAY = {"thinking_feeling",
                               "their_thinking_feeling",
                               "response"}
MANDATORY_SECTIONS_DRAFTING = {"perspective_taking",
                               "draft_brainstorming",
                               "draft"}


def _task_has_all_expected_responses(task: "ScenarioTask") -> bool:
    """
    Decide whether *task* is complete enough for rubric scoring.

    Baseline (even when `ALLOW_INCOMPLETE_RESPONSES` is True):
    ▸ There must be **at least one** assistant message in the main
      conversation (i.e., not the debrief) whose raw text length is
      ≥ `_MIN_RAW_LEN`.

    • NO_RP / Analysis : at least one assistant message ≥ `_MIN_RAW_LEN`
    • Role-play / Draft:
        – every assistant turn must have either
            • raw text ≥ `_MIN_RAW_LEN`  **or**
            • all mandatory parsed sections non-blank
        – plus a non-empty debrief.
    """
    sid = task.scenario_id
    is_analysis = sid in C.ANALYSIS_SCENARIO_IDS
    is_drafting = sid in C.MESSAGE_DRAFTING_SCENARIO_IDS
    is_no_rp    = sid in C.NO_RP_SCENARIO_IDS

    assistants = [m for m in (task.conversation_history or [])
                  if m.get("role") == "assistant"]

    # ------------------------------------------------------------------
    # If incomplete responses are allowed, enforce only the baseline.
    # ------------------------------------------------------------------
    if ALLOW_INCOMPLETE_RESPONSES:
        return any(len(m.get("content", "").strip()) >= _MIN_RAW_LEN
                   for m in assistants)

    # ── NO_RP / analysis ───────────────────────────────────────────────
    if is_analysis or is_no_rp:
        return any(len(m.get("content", "").strip()) >= _MIN_RAW_LEN
                   for m in assistants)

    # ── role-play / drafting ───────────────────────────────────────────
    parsed = task.parsed_responses or []
    if len(parsed) < len(assistants):          # parsed list too short
        return False

    required = (MANDATORY_SECTIONS_DRAFTING
                if is_drafting else
                MANDATORY_SECTIONS_ROLEPLAY)

    def turn_ok(pr: dict) -> bool:
        raw = (pr.get("raw") or "").strip()
        if len(raw) >= _MIN_RAW_LEN:
            return True                        # long raw content suffices
        return all(pr.get(k, "").strip() for k in required)

    if not all(turn_ok(parsed[i]) for i in range(len(assistants))):
        return False

    # Debrief is compulsory for role-play & drafting scenarios.
    return bool(task.debrief_response and task.debrief_response.strip())



def run_eq_bench3(
    model_name: str, # Logical name
    api_model_id: str, # API model ID
    # File Paths
    local_runs_file: str,
    local_elo_file: str,
    leaderboard_runs_file: str,
    leaderboard_elo_file: str,
    # Run Control
    num_threads: int = 4,
    run_id: Optional[str] = None,
    save_interval: int = 2,
    iterations: int = 1,
    # Feature Flags & Models
    run_elo: bool = True,
    run_rubric: bool = True,
    judge_model: Optional[str] = None, # Judge model ID string
    redo_judging: bool = False,
    repair_scenarios: bool = False,
    repair_rubric: bool = False,
    truncate_for_rubric: bool = False,
) -> str:
    """
    Main function to run the EQBench3 benchmark.
    Orchestrates scenario simulation, debriefing, optional rubric scoring,
    and optional ELO analysis across iterations. Uses asynchronous saving.
    Handles standard, message drafting, and analysis task types.
    Uses logical model_name for tracking and api_model_id for API calls.
    Loads leaderboard data for context but writes ONLY to local files.
    """
    # --- Argument Validation ---
    if run_elo and not judge_model:
         raise ValueError("A --judge-model must be specified when running ELO analysis (--no-elo not set).")
    if run_rubric and not judge_model:
         raise ValueError("A --judge-model must be specified when running Rubric scoring (--no-rubric not set).")

    # --- Load Leaderboard Data (Read-Only) ---
    logging.info(f"Loading leaderboard runs data from: {leaderboard_runs_file}")
    leaderboard_runs = load_json_file(leaderboard_runs_file)
    logging.info(f"Loading leaderboard ELO data from: {leaderboard_elo_file}")
    leaderboard_elo = load_json_file(leaderboard_elo_file) # Loaded here, passed to ELO function

    # --- Load Local Data (Read/Write) ---
    logging.info(f"Loading local runs data from: {local_runs_file}")
    local_runs = load_json_file(local_runs_file)

    # --- Duplicate Model Name Check (Across Leaderboard & Local, only if ELO enabled) ---
    if run_elo:
        logging.info(f"Checking for duplicate model name '{model_name}' in run files (ELO enabled)...")
        # Check Leaderboard Runs
        for existing_run_key, run_data in leaderboard_runs.items():
            if isinstance(run_data, dict):
                existing_model = run_data.get("model_name", run_data.get("test_model"))
                if existing_model == model_name:
                    raise ValueError(
                        f"\nERROR: Logical model name '{model_name}' already exists in the LEADERBOARD runs file ('{leaderboard_runs_file}') under run key '{existing_run_key}'.\n"
                        f"       Unique model names are required when ELO is enabled to ensure comparisons are correctly attributed.\n"
                        f"       Please choose a different --model-name."
                    )

        # Check Local Runs (excluding the run being resumed, if applicable)
        for existing_run_key, run_data in local_runs.items():
            if isinstance(run_data, dict):
                existing_model = run_data.get("model_name", run_data.get("test_model"))
                if existing_model == model_name:
                    # Check if this is the exact run we are trying to resume
                    is_resuming_this_run = False
                    if run_id: # run_id is the prefix passed via CLI
                        # Construct potential run key for comparison
                        sanitized_model_for_check = re.sub(r'[^a-zA-Z0-9_.-]+', '_', model_name)
                        potential_resume_key = f"{run_id}_{sanitized_model_for_check}"
                        if existing_run_key == potential_resume_key:
                            is_resuming_this_run = True

                    if not is_resuming_this_run:
                        raise ValueError(
                            f"\nERROR: Logical model name '{model_name}' already exists in the LOCAL runs file ('{local_runs_file}') under run key '{existing_run_key}'.\n"
                            f"       Unique model names (that don't collide with other runs) are required when ELO is enabled.\n"
                            f"       This prevents ELO analysis from incorrectly reusing comparisons from a different run/version of the model.\n"
                            f"       Please choose a different --model-name or resume the existing run using --run-id {existing_run_key.split('_')[0]}"
                        )
        logging.info(f"Model name '{model_name}' is unique across run files.")
    else:
        logging.info("Skipping duplicate model name check as ELO is disabled.")


    # --- Run Key Setup ---
    def sanitize_model_name(name: str) -> str:
        # Sanitize the logical model name for the run key
        return re.sub(r'[^a-zA-Z0-9_.-]+', '_', name)

    sanitized_model = sanitize_model_name(model_name) # Use logical name here
    base_id = run_id if run_id else str(uuid.uuid4().hex[:8])
    run_key = f"{base_id}_{sanitized_model}"

    # --- Init or resume run (in Local Runs) ---
    if run_key not in local_runs:
        init_dict = {
            "run_key": run_key,
            "model_name": model_name, # Store logical name
            "api_model_id": api_model_id, # Store API ID
            "test_model": model_name, # Store logical name in legacy field
            "judge_model": judge_model if (run_elo or run_rubric) else "N/A",
            "rubric_judge_model": judge_model if run_rubric else "N/A",
            "elo_judge_model": judge_model if run_elo else "N/A",
            "start_time": datetime.now(timezone.utc).isoformat(),
            "status": "initializing",
            # Store paths used for reference (using constants)
            "scenario_prompts_file": C.STANDARD_SCENARIO_PROMPTS_FILE,
            "scenario_master_prompt_file": C.STANDARD_MASTER_PROMPT_FILE,
            "message_drafting_master_prompt_file": C.MESSAGE_DRAFTING_MASTER_PROMPT_FILE,
            "analysis_master_prompt_file": C.ANALYSIS_MASTER_PROMPT_FILE,
            "debrief_prompt_file": C.STANDARD_DEBRIEF_PROMPT_FILE,
            "rubric_criteria_file_standard": C.STANDARD_RUBRIC_CRITERIA_FILE if run_rubric else "N/A",
            "rubric_prompt_file_standard": C.STANDARD_RUBRIC_PROMPT_FILE if run_rubric else "N/A",
            "rubric_criteria_file_analysis": C.ANALYSIS_RUBRIC_CRITERIA_FILE if run_rubric else "N/A",
            "rubric_prompt_file_analysis": C.ANALYSIS_RUBRIC_PROMPT_FILE if run_rubric else "N/A",
            "truncate_for_rubric": truncate_for_rubric,
            "iterations_requested": iterations,
            "scenario_tasks": {},
            "results": {}
        }
        # Update ONLY the local runs file
        update_run_data(local_runs_file, run_key, init_dict)
        logging.info(f"Created new run in local file: {run_key} for model '{model_name}' (API ID: '{api_model_id}')")
        local_runs = load_json_file(local_runs_file)  # Reload local runs after update
    else:
        logging.info(f"Resuming run: {run_key} for model '{model_name}' (API ID: '{api_model_id}') from local file {local_runs_file}")
        # Minimal update logic for resuming, mainly status and start time if needed
        update_payload = {}
        current_run_data = local_runs[run_key] # Read from local_runs
        # Check if the model identifiers match the resumed run
        existing_model_name = current_run_data.get("model_name", current_run_data.get("test_model"))
        existing_api_id = current_run_data.get("api_model_id", current_run_data.get("test_model")) # Fallback needed?
        if existing_model_name != model_name:
             logging.warning(f"Resuming run {run_key} but logical model name mismatch! Run has '{existing_model_name}', requested '{model_name}'. Continuing with run's name.")
             # Use the name already associated with the run_key
             model_name = existing_model_name
        if existing_api_id != api_model_id:
             logging.warning(f"Resuming run {run_key} but API model ID mismatch! Run has '{existing_api_id}', requested '{api_model_id}'. Updating API ID in run data.")
             # Update the API ID in the run data if it changed
             update_payload["api_model_id"] = api_model_id

        if "start_time" not in current_run_data:
            update_payload["start_time"] = datetime.now(timezone.utc).isoformat()
        if current_run_data.get("status") not in ["running", "initializing", "completed_with_errors", "error"]: # Allow resuming from intermediate/error states
            current_status = current_run_data.get("status")
            logging.info(f"Run {run_key} status is '{current_status}'. Resetting status to 'running'.")
            update_payload["status"] = "running"
        # Add missing config info if resuming an older run format
        if "model_name" not in current_run_data: update_payload["model_name"] = model_name
        if "api_model_id" not in current_run_data: update_payload["api_model_id"] = api_model_id
        if "test_model" not in current_run_data: update_payload["test_model"] = model_name # Backfill legacy field
        if "iterations_requested" not in current_run_data: update_payload["iterations_requested"] = iterations
        # Add missing file paths if needed (less critical now, but good for consistency)
        if "scenario_master_prompt_file" not in current_run_data: update_payload["scenario_master_prompt_file"] = C.STANDARD_MASTER_PROMPT_FILE
        if "message_drafting_master_prompt_file" not in current_run_data: update_payload["message_drafting_master_prompt_file"] = C.MESSAGE_DRAFTING_MASTER_PROMPT_FILE
        if "analysis_master_prompt_file" not in current_run_data: update_payload["analysis_master_prompt_file"] = C.ANALYSIS_MASTER_PROMPT_FILE
        if run_rubric:
            if "rubric_criteria_file_standard" not in current_run_data: update_payload["rubric_criteria_file_standard"] = C.STANDARD_RUBRIC_CRITERIA_FILE
            if "rubric_prompt_file_standard" not in current_run_data: update_payload["rubric_prompt_file_standard"] = C.STANDARD_RUBRIC_PROMPT_FILE
            if "rubric_criteria_file_analysis" not in current_run_data: update_payload["rubric_criteria_file_analysis"] = C.ANALYSIS_RUBRIC_CRITERIA_FILE
            if "rubric_prompt_file_analysis" not in current_run_data: update_payload["rubric_prompt_file_analysis"] = C.ANALYSIS_RUBRIC_PROMPT_FILE
            if "truncate_for_rubric" not in current_run_data: update_payload["truncate_for_rubric"] = truncate_for_rubric

        if update_payload:
            # Update ONLY the local runs file
            update_run_data(local_runs_file, run_key, update_payload)
            local_runs = load_json_file(local_runs_file) # Reload local runs after update

    # --- Merge Run Data for Processing ---
    # Local runs override leaderboard runs on key collision
    merged_runs = {**leaderboard_runs, **local_runs}
    logging.info(f"Merged run data: {len(leaderboard_runs)} leaderboard runs, {len(local_runs)} local runs -> {len(merged_runs)} total runs for context.")


    # --- Redo Judging Logic (Reset tasks in the LOCAL file) ---
    if redo_judging and run_rubric:
        logging.info(f"Processing --redo-judging flag: resetting tasks in LOCAL file {local_runs_file}...")
        # Load local data directly for modification
        current_local_runs_data = load_json_file(local_runs_file)
        my_run_data = current_local_runs_data.get(run_key, {})
        scenario_tasks_data = my_run_data.get("scenario_tasks", {})

        updated_scenario_tasks = {}
        tasks_reset_count = 0

        for iter_str, scenario_dict in scenario_tasks_data.items():
            if not isinstance(scenario_dict, dict):
                updated_scenario_tasks[iter_str] = scenario_dict
                continue

            updated_scen_dict = {}
            for sid, task_info in scenario_dict.items():
                if isinstance(task_info, dict) and task_info.get("status") == "rubric_scored":
                    new_task_info = task_info.copy()
                    is_analysis = sid in C.ANALYSIS_SCENARIO_IDS
                    reset_status = "scenario_completed" if is_analysis else "completed"
                    new_task_info["status"] = reset_status
                    # Clear old rubric data
                    new_task_info.pop("rubric_scores", None)
                    new_task_info.pop("raw_rubric_judge_text", None)
                    new_task_info.pop("rubric_run_error", None)
                    tasks_reset_count += 1
                    updated_scen_dict[sid] = new_task_info
                    logging.debug(f"Resetting task {sid} (Iter {iter_str}) to '{reset_status}' for rubric re-judging.")
                else:
                    updated_scen_dict[sid] = task_info # Keep others

            updated_scenario_tasks[iter_str] = updated_scen_dict

        if tasks_reset_count > 0:
            # Update ONLY the local runs file
            update_success = update_run_data(local_runs_file, run_key, {"scenario_tasks": updated_scenario_tasks})
            if update_success:
                logging.info(f"[redo-judging] Reset {tasks_reset_count} task(s) in {local_runs_file}. The rubric scoring step will be re-run.")
            else:
                 logging.error(f"[redo-judging] Failed to save the reset task data to the local runs file: {local_runs_file}.")
            local_runs = load_json_file(local_runs_file) # Reload local runs after potential modification
            merged_runs = {**leaderboard_runs, **local_runs} # Re-merge after modification
        else:
            logging.info("[redo-judging] No tasks in 'rubric_scored' status were found in the local run to reset.")
    elif redo_judging and not run_rubric:
         logging.warning("--redo-judging flag ignored because --no-rubric is set.")

    # --- Load Prompts and Templates (Remains the same) ---
    try:
        scenarios = parse_scenario_prompts(C.STANDARD_SCENARIO_PROMPTS_FILE)
        if not scenarios:
            logging.error(f"No scenarios parsed from {C.STANDARD_SCENARIO_PROMPTS_FILE}. Aborting.")
            update_run_data(local_runs_file, run_key, {"status": "error", "error": "No scenarios parsed"}) # Write error to local
            return run_key
    except Exception as e:
        logging.error(f"Failed to load or parse scenario prompts: {e}", exc_info=True)
        update_run_data(local_runs_file, run_key, {"status": "error", "error": f"Failed to load scenarios: {e}"}) # Write error to local
        return run_key

    # Load Master Prompt Templates (Remains the same)
    try:
        standard_master_template = Path(C.STANDARD_MASTER_PROMPT_FILE).read_text(encoding='utf-8')
        if not standard_master_template.strip(): raise ValueError("Standard master prompt template file is empty.")
        logging.info(f"Loaded standard master prompt template from {C.STANDARD_MASTER_PROMPT_FILE}")

        drafting_master_template = Path(C.MESSAGE_DRAFTING_MASTER_PROMPT_FILE).read_text(encoding='utf-8')
        if not drafting_master_template.strip(): raise ValueError("Drafting master prompt template file is empty.")
        logging.info(f"Loaded drafting master prompt template from {C.MESSAGE_DRAFTING_MASTER_PROMPT_FILE}")

        analysis_master_template = Path(C.ANALYSIS_MASTER_PROMPT_FILE).read_text(encoding='utf-8')
        if not analysis_master_template.strip(): raise ValueError("Analysis master prompt template file is empty.")
        logging.info(f"Loaded analysis master prompt template from {C.ANALYSIS_MASTER_PROMPT_FILE}")

    except Exception as e:
        logging.error(f"Failed to load one or more master prompt templates: {e}", exc_info=True)
        update_run_data(local_runs_file, run_key, {"status": "error", "error": f"Failed to load master prompt template: {e}"}) # Write error to local
        return run_key

    # Load Debrief Prompt (only for standard/drafting) (Remains the same)
    try:
        standard_debrief_prompt = Path(C.STANDARD_DEBRIEF_PROMPT_FILE).read_text(encoding='utf-8')
        if not standard_debrief_prompt.strip(): raise ValueError("Debrief prompt file is empty.")
        logging.info(f"Loaded standard debrief prompt from {C.STANDARD_DEBRIEF_PROMPT_FILE}")
    except Exception as e:
        logging.error(f"Failed to load debrief prompt from {C.STANDARD_DEBRIEF_PROMPT_FILE}: {e}", exc_info=True)
        update_run_data(local_runs_file, run_key, {"status": "error", "error": f"Failed to load debrief prompt: {e}"}) # Write error to local
        return run_key


    # --- Load Rubric Scoring Files (if enabled) (Remains the same) ---
    standard_rubric_criteria = []
    standard_rubric_prompt_template = None
    standard_rubric_output_format_str = "{}"
    analysis_rubric_criteria = []
    analysis_rubric_prompt_template = None
    analysis_rubric_output_format_str = "{}"

    if run_rubric:
        # Load Standard Rubric Files
        try:
            with open(C.STANDARD_RUBRIC_CRITERIA_FILE, 'r', encoding='utf-8') as f:
                standard_rubric_criteria = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
            if not standard_rubric_criteria: raise ValueError("Standard rubric criteria file is empty.")
            logging.info(f"Loaded {len(standard_rubric_criteria)} standard rubric criteria from {C.STANDARD_RUBRIC_CRITERIA_FILE}")

            output_format_dict_std = {"chain_of_thought_reasoning": "detailed chain of thought reasoning about the coming scoring decisions"}
            for criterion in standard_rubric_criteria: output_format_dict_std[criterion] = 0
            standard_rubric_output_format_str = json.dumps(output_format_dict_std, indent=2).replace(': 0', ': 0-20')

            standard_rubric_prompt_template = Path(C.STANDARD_RUBRIC_PROMPT_FILE).read_text(encoding='utf-8')
            if not standard_rubric_prompt_template or "{transcript}" not in standard_rubric_prompt_template or "{debrief}" not in standard_rubric_prompt_template or "{output_format}" not in standard_rubric_prompt_template:
                 raise ValueError("Standard rubric prompt template missing required placeholders ({transcript}, {debrief}, {output_format}).")
            logging.info(f"Loaded standard rubric prompt template from {C.STANDARD_RUBRIC_PROMPT_FILE}")

        except Exception as e:
            logging.error(f"Failed to load standard rubric files: {e}", exc_info=True)
            update_run_data(local_runs_file, run_key, {"status": "error", "error": f"Failed to load standard rubric files: {e}"}) # Write error to local
            return run_key

        # Load Analysis Rubric Files
        try:
            with open(C.ANALYSIS_RUBRIC_CRITERIA_FILE, 'r', encoding='utf-8') as f:
                analysis_rubric_criteria = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
            if not analysis_rubric_criteria: raise ValueError("Analysis rubric criteria file is empty.")
            logging.info(f"Loaded {len(analysis_rubric_criteria)} analysis rubric criteria from {C.ANALYSIS_RUBRIC_CRITERIA_FILE}")

            output_format_dict_anl = {"chain_of_thought_reasoning": "detailed chain of thought reasoning about the coming scoring decisions"}
            for criterion in analysis_rubric_criteria: output_format_dict_anl[criterion] = 0
            analysis_rubric_output_format_str = json.dumps(output_format_dict_anl, indent=2).replace(': 0', ': 0-20') # Assuming 0-20 scale

            analysis_rubric_prompt_template = Path(C.ANALYSIS_RUBRIC_PROMPT_FILE).read_text(encoding='utf-8')
            # Analysis prompt should NOT have {debrief} placeholder
            if not analysis_rubric_prompt_template or "{transcript}" not in analysis_rubric_prompt_template or "{output_format}" not in analysis_rubric_prompt_template:
                 raise ValueError("Analysis rubric prompt template missing required placeholders ({transcript}, {output_format}).")
            if "{debrief}" in analysis_rubric_prompt_template:
                 logging.warning(f"Analysis rubric prompt template ({C.ANALYSIS_RUBRIC_PROMPT_FILE}) contains a '{{debrief}}' placeholder, which is not used for analysis tasks.")
            logging.info(f"Loaded analysis rubric prompt template from {C.ANALYSIS_RUBRIC_PROMPT_FILE}")

        except Exception as e:
            logging.error(f"Failed to load analysis rubric files: {e}", exc_info=True)
            update_run_data(local_runs_file, run_key, {"status": "error", "error": f"Failed to load analysis rubric files: {e}"}) # Write error to local
            return run_key

        logging.info(f"Rubric scoring enabled. Truncation for rubric: {truncate_for_rubric}")
    else:
        logging.info("Rubric scoring is disabled.")


    # --- Build API clients (Remains the same) ---
    api_clients = {"test": APIClient(model_type="test")}
    if run_elo or run_rubric:
        api_clients["judge"] = APIClient(model_type="judge")
        judge_usage = []
        if run_rubric: judge_usage.append("Rubric")
        if run_elo: judge_usage.append("ELO")
        logging.info(f"Judge model ({'/'.join(judge_usage)}): {judge_model}")


    # --- Prepare Task Objects (Load or Create from Merged Data) ---
    # Use merged_runs to find existing task data, allowing resumption of leaderboard tasks locally
    run_data_for_tasks = merged_runs.get(run_key, {})
    existing_tasks_data = run_data_for_tasks.get("scenario_tasks", {})
    tasks_to_process: List[ScenarioTask] = []
    recovered_tasks: List[ScenarioTask] = []

    total_tasks_expected = len(scenarios) * iterations
    logging.info(f"Preparing {total_tasks_expected} total tasks ({len(scenarios)} scenarios x {iterations} iterations)...")

    for i in range(1, iterations + 1):
        i_str = str(i)
        for scenario_id, prompts_list in scenarios.items():
            # Determine task type and select appropriate templates/prompts (Remains the same)
            is_analysis = scenario_id in C.ANALYSIS_SCENARIO_IDS
            is_drafting = scenario_id in C.MESSAGE_DRAFTING_SCENARIO_IDS

            chosen_master_template = None
            chosen_debrief_prompt = None
            if is_analysis:
                chosen_master_template = analysis_master_template
                chosen_debrief_prompt = None # Analysis tasks have no debrief
            elif is_drafting:
                chosen_master_template = drafting_master_template
                chosen_debrief_prompt = standard_debrief_prompt
            else: # Standard role-play
                chosen_master_template = standard_master_template
                chosen_debrief_prompt = standard_debrief_prompt

            task_obj = None
            if (i_str in existing_tasks_data and
                    scenario_id in existing_tasks_data[i_str]):
                task_data = existing_tasks_data[i_str][scenario_id]
                # Check if loaded task data matches the current run's logical model name
                # 'test_model' key holds the logical name in task data
                task_model_name = task_data.get("test_model")
                if isinstance(task_data, dict) and task_model_name == model_name:
                    try:
                        task_obj = ScenarioTask.from_dict(task_data)
                        # Update templates/prompts in case they changed or were missing
                        task_obj.master_prompt_template = chosen_master_template
                        task_obj.debrief_prompt = chosen_debrief_prompt # Update debrief prompt (or set to None for analysis)

                        if task_obj.iteration_index != i:
                            logging.warning(f"Mismatch iteration index in loaded task data for {scenario_id} (expected {i}, got {task_obj.iteration_index}). Resetting.")
                            task_obj.iteration_index = i
                        if (repair_scenarios or repair_rubric) and task_obj.recover_for_resume(
                            repair_scenario=repair_scenarios,
                            repair_rubric=repair_rubric,
                            valid_rubric_criteria=set(
                                analysis_rubric_criteria if is_analysis else standard_rubric_criteria
                            ),
                        ):
                            recovered_tasks.append(task_obj)
                        logging.debug(f"Resuming task: Scenario {scenario_id}, Iteration {i}, Status: {task_obj.status}")
                    except Exception as e:
                        logging.error(f"Failed to load task from dict for scenario={scenario_id}, iter={i}: {e}. Creating new task.", exc_info=True)
                        task_obj = None
                else:
                    # Model name mismatch or invalid data, create new task
                    if isinstance(task_data, dict) and task_model_name != model_name:
                         logging.warning(f"Task data found for scenario={scenario_id}, iter={i} belongs to a different model ('{task_model_name}' vs '{model_name}'). Creating new task.")
                    else:
                         logging.warning(f"Invalid or mismatched task data found for scenario={scenario_id}, iter={i}. Creating new task.")
                    task_obj = None

            if task_obj is None:
                task_obj = ScenarioTask(
                    scenario_id=scenario_id,
                    prompts=prompts_list,
                    debrief_prompt=chosen_debrief_prompt, # Pass None for analysis
                    iteration_index=i,
                    test_model=model_name, # Pass logical name to task constructor
                    master_prompt_template=chosen_master_template,
                )
                logging.debug(f"Creating new task: Scenario {scenario_id}, Iteration {i} (Analysis: {is_analysis})")

            tasks_to_process.append(task_obj)

    logging.info(f"Prepared {len(tasks_to_process)} task objects across {iterations} iteration(s).")


    # --- Setup Asynchronous Saving (Targeting LOCAL file) ---
    save_queue = queue.Queue()
    save_thread = threading.Thread(
        target=_save_worker,
        args=(save_queue, local_runs_file), # Pass the LOCAL runs file path
        name="SaveWorkerThread",
        daemon=True
    )
    save_thread.start()
    logging.info("Save worker thread started.")
    for task in recovered_tasks:
        task._save_progress(save_queue, run_key)
    if recovered_tasks:
        logging.info(
            "Persisted recovery state for %s interrupted or failed task(s).",
            len(recovered_tasks),
        )
    recovered_task_keys = {
        (task.iteration_index, task.scenario_id)
        for task in recovered_tasks
    }

    # --- Execute Tasks (Remains largely the same, save worker handles target file) ---
    tasks_completed_this_run = 0 # Tracks tasks fully completed (incl. rubric if enabled)

    try:
        # 1. Run scenario steps
        tasks_needing_scenario = []
        if repair_scenarios or repair_rubric:
            tasks_needing_scenario = [
                t for t in tasks_to_process
                if (t.iteration_index, t.scenario_id) in recovered_task_keys
                and t.status == "initialized"
            ]
        else:
            tasks_needing_scenario = [t for t in tasks_to_process if t.status in ["initialized", "error"]]
        if tasks_needing_scenario:
            logging.info(f"Running scenario simulation for {len(tasks_needing_scenario)} tasks...")
            with ThreadPoolExecutor(max_workers=num_threads, thread_name_prefix="ScenarioRun") as executor:
                futures = {
                    # Pass api_model_id for API calls
                    executor.submit(t.run_scenario, api_clients, save_queue, run_key, api_model_id): t
                    for t in tasks_needing_scenario
                }
                future_list = list(futures.keys())
                for future in tqdm(as_completed(future_list), total=len(future_list), desc="Running Scenarios"):
                    task = futures[future]
                    try:
                        future.result() # Wait for completion, errors handled within method
                    except Exception as e:
                        logging.error(f"Unhandled executor error during scenario for task {task.scenario_id} (Iter {task.iteration_index}): {e}", exc_info=True)
                        # Check status before overwriting, run_scenario sets error status internally
                        if task.status not in ["error", "scenario_completed", "completed", "rubric_scored"]:
                             task.status = "error"; task.error = f"Unhandled Executor Error: {e}"; task._save_progress(save_queue, run_key)

        else:
            logging.info("No tasks require scenario simulation based on initial status.")


        # 2. Run debrief steps (Skip for Analysis tasks)
        tasks_needing_debrief = []
        if repair_scenarios or repair_rubric:
            tasks_needing_debrief = [
                t for t in tasks_to_process
                if (t.iteration_index, t.scenario_id) in recovered_task_keys
                and t.status == "scenario_completed"
                and t.scenario_id not in C.ANALYSIS_SCENARIO_IDS
            ]
        else:
            tasks_needing_debrief = [
                t for t in tasks_to_process
                if t.status == "scenario_completed" and t.scenario_id not in C.ANALYSIS_SCENARIO_IDS
            ]
        if tasks_needing_debrief:
            logging.info(f"Running debrief for {len(tasks_needing_debrief)} non-analysis tasks...")
            with ThreadPoolExecutor(max_workers=num_threads, thread_name_prefix="DebriefRun") as executor:
                futures = {
                    # Pass api_model_id for API calls
                    executor.submit(t.run_debrief, api_clients, save_queue, run_key, api_model_id): t
                    for t in tasks_needing_debrief
                }
                future_list = list(futures.keys())
                for future in tqdm(as_completed(future_list), total=len(future_list), desc="Running Debriefs"):
                    task = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        logging.error(f"Unhandled executor error during debrief for task {task.scenario_id} (Iter {task.iteration_index}): {e}", exc_info=True)
                        # Check status before overwriting
                        if task.status not in ["error", "completed", "rubric_scored"]:
                             task.status = "error"; task.error = f"Unhandled Executor Error (Debrief): {e}"; task._save_progress(save_queue, run_key)
        else:
            logging.info("No non-analysis tasks require debriefing based on current status.")


        # 3. Run Rubric Scoring steps (if enabled) - Handles different task types
        if run_rubric:
            # Standard/Drafting tasks need rubric if status is 'completed'
            # Analysis tasks need rubric if status is 'scenario_completed'
            rubric_ready_tasks = [
                t for t in tasks_to_process
                if (
                    not (repair_scenarios or repair_rubric)
                    or redo_judging
                    or (
                        repair_rubric
                        and (t.iteration_index, t.scenario_id) in recovered_task_keys
                    )
                ) and (
                    (
                        t.scenario_id in C.ANALYSIS_SCENARIO_IDS and t.status == "scenario_completed"
                    ) or (
                        t.scenario_id not in C.ANALYSIS_SCENARIO_IDS and t.status == "completed"
                    )
                )
            ]
            tasks_needing_rubric = [
                t for t in rubric_ready_tasks
                if _task_has_all_expected_responses(t)
            ]
            incomplete_rubric_tasks = [
                t for t in rubric_ready_tasks
                if not _task_has_all_expected_responses(t)
            ]
            for task in incomplete_rubric_tasks:
                task.status = "error"
                task.error = "Rubric Scoring Error: Task output incomplete; rubric scoring skipped."
                task.rubric_run_error = "Task output incomplete"
                task._save_progress(save_queue, run_key)
                logging.warning(
                    "Skipping rubric scoring for incomplete task %s (Iter %s).",
                    task.scenario_id,
                    task.iteration_index,
                )

            if tasks_needing_rubric:
                logging.info(f"Running rubric scoring for {len(tasks_needing_rubric)} tasks using judge model '{judge_model}' (Truncation: {truncate_for_rubric})...")
                with ThreadPoolExecutor(max_workers=num_threads, thread_name_prefix="RubricRun") as executor:
                    futures = {}
                    for t in tasks_needing_rubric:
                        is_analysis = t.scenario_id in C.ANALYSIS_SCENARIO_IDS
                        # Select appropriate rubric template and format string
                        rubric_template = analysis_rubric_prompt_template if is_analysis else standard_rubric_prompt_template
                        rubric_format = analysis_rubric_output_format_str if is_analysis else standard_rubric_output_format_str

                        future = executor.submit(
                            _execute_rubric_scoring_task,
                            task=t,
                            api_clients=api_clients,
                            judge_model_id=judge_model,
                            rubric_prompt_template=rubric_template,
                            rubric_output_format_str=rubric_format,
                            save_queue=save_queue,
                            run_key=run_key,
                            truncate_for_rubric=truncate_for_rubric
                        )
                        futures[future] = t

                    future_list = list(futures.keys())
                    for future in tqdm(as_completed(future_list), total=len(future_list), desc="Running Rubric Scoring"):
                        task = futures[future]
                        try:
                            future.result() # Wait for thread completion. Errors handled inside helper.
                            # Log progress based on tasks reaching the final state
                            if task.status == "rubric_scored":
                                tasks_completed_this_run += 1
                                if save_interval > 0 and (tasks_completed_this_run % save_interval == 0):
                                    logging.info(f"Completed {tasks_completed_this_run} tasks (incl. rubric) in this run.")
                        except Exception as e:
                            # This catches errors *outside* the helper's try/except
                            logging.error(
                                f"Unhandled executor error during rubric scoring future processing for task {task.scenario_id} (Iter {task.iteration_index}): {e}",
                                exc_info=True
                            )
                            # Ensure task status reflects error if it wasn't already set
                            if task.status not in ["error", "rubric_scored"]:
                                 task.status = "error"
                                 task.error = f"Unhandled Executor Error (Rubric Future): {e}"
                                 task._save_progress(save_queue, run_key) # Attempt to save error state
            else:
                logging.info("No tasks require rubric scoring based on current status.")
        else:
            # If rubric is disabled, count tasks reaching 'completed' (standard/drafting)
            # or 'scenario_completed' (analysis) as done.
            tasks_reaching_final_state = sum(
                1 for t in tasks_to_process
                if (t.scenario_id in C.ANALYSIS_SCENARIO_IDS and t.status == "scenario_completed") or \
                   (t.scenario_id not in C.ANALYSIS_SCENARIO_IDS and t.status == "completed")
            )
            tasks_completed_this_run = tasks_reaching_final_state
            logging.info("Rubric scoring disabled. Tasks reaching their respective pre-rubric completed state are considered finished for this run.")


    finally:
        # Signal the save worker to exit and wait for it
        logging.info("All task processing submitted. Waiting for save queue to empty...")
        save_queue.put(None)
        save_queue.join()
        logging.info("Save queue finished processing.")
        save_thread.join(timeout=10)
        if save_thread.is_alive():
            logging.warning("Save worker thread did not terminate after queue processing and join timeout.")

    # --- Calculate Final Rubric Score (if enabled, using LOCAL data) ---
    if run_rubric:
        logging.info(f"Calculating final average rubric score from local file: {local_runs_file}...")
        # Load the latest local data for calculation
        final_local_run_data_for_rubric = load_json_file(local_runs_file).get(run_key, {})
        avg_rubric_score, rubric_err = calculate_final_rubric_score(final_local_run_data_for_rubric)

        # Update results in the LOCAL file
        current_local_run_data = load_json_file(local_runs_file).get(run_key, {})
        current_results = current_local_run_data.get("results", {})
        current_results["average_rubric_score"] = avg_rubric_score if avg_rubric_score is not None else "N/A"
        current_results["rubric_calculation_time"] = datetime.now(timezone.utc).isoformat()
        current_results["rubric_error"] = rubric_err
        update_run_data(local_runs_file, run_key, {"results": current_results, "rubric_judge_model": judge_model})

        if rubric_err: logging.error(f"Rubric score calculation failed: {rubric_err}")
        elif avg_rubric_score is not None: logging.info(f"Final Average Rubric Score: {avg_rubric_score:.2f}")
        else: logging.warning("Rubric score calculation resulted in None, but no specific error message.")
    else:
         # Update results in the LOCAL file if skipped
         current_local_run_data = load_json_file(local_runs_file).get(run_key, {})
         current_results = current_local_run_data.get("results", {})
         if "average_rubric_score" not in current_results:
             current_results["average_rubric_score"] = "Skipped"; current_results["rubric_error"] = None
             update_run_data(local_runs_file, run_key, {"results": current_results})


    # --- Final ELO analysis (if enabled) ---
    final_elo_snapshot = {} # To store the solved ratings from the ELO run
    elo_error_msg = None # Initialize error message for ELO step

    # --- Reload local runs data AFTER saving is complete ---
    logging.info(f"Reloading local runs data from {local_runs_file} before ELO analysis...")
    local_runs = load_json_file(local_runs_file) # Reload the updated local runs
    merged_runs = {**leaderboard_runs, **local_runs} # Re-merge with the read-only leaderboard data
    logging.info(f"Refreshed merged run data: {len(leaderboard_runs)} leaderboard runs, {len(local_runs)} local runs -> {len(merged_runs)} total runs for ELO context.")

    if run_elo:
        logging.info("Starting ELO analysis using merged leaderboard/local data...")
        try:
            # Pass merged run data, leaderboard/local ELO paths
            # ELO function now loads prompts internally based on scenario type
            # Pass the logical model name as test_model
            # Capture the returned snapshot and error message
            final_elo_snapshot, elo_error_msg = run_elo_analysis_eqbench3(
                run_key=run_key,
                # ELO Files
                leaderboard_elo_file=leaderboard_elo_file, # Passed into run_eq_bench3
                local_elo_file=local_elo_file,             # Passed into run_eq_bench3
                # Run Data
                merged_runs_data=merged_runs, # Use the merged data prepared earlier
                # Models
                test_model=model_name, # Logical name passed into run_eq_bench3
                judge_model=judge_model,
                api_clients=api_clients,
                # Other params
                scenarios_data=scenarios,
                concurrency=num_threads,
                recompute_existing=True
            )

            # Extract scores for the current model from the *solved* snapshot returned
            elo_raw, elo_norm = "N/A", "N/A"
            current_model_elo_data = final_elo_snapshot.get(model_name) # Use model_name passed into run_eq_bench3

            if isinstance(current_model_elo_data, dict):
                 elo_raw = current_model_elo_data.get("elo", "N/A")
                 elo_norm = current_model_elo_data.get("elo_norm", "N/A")
            elif isinstance(current_model_elo_data, (int, float)): # Handle older format if necessary
                 elo_raw = current_model_elo_data
                 # Attempt to get norm from potentially updated local file as fallback
                 final_local_elo_data = load_json_file(local_elo_file)
                 if isinstance(final_local_elo_data.get(model_name), dict):
                     elo_norm = final_local_elo_data[model_name].get("elo_norm", "N/A")

            # Update results in the LOCAL run file
            current_local_run_data = load_json_file(local_runs_file).get(run_key, {})
            current_results = current_local_run_data.get("results", {})
            current_results.update({
                "elo_raw": elo_raw,
                "elo_normalized": elo_norm,
                "elo_calculation_time": datetime.now(timezone.utc).isoformat(),
                "elo_error": elo_error_msg # Store error message from ELO run
            })
            update_run_data(local_runs_file, run_key, {"results": current_results, "elo_judge_model": judge_model})

            if elo_error_msg is None:
                logging.info(f"ELO scores for {model_name} (from solved snapshot): Raw={elo_raw}, Normalized={elo_norm}")
                # NO leaderboard printing here
            else:
                logging.error(f"ELO calculation finished with message: {elo_error_msg}")

        except FileNotFoundError as e:
            logging.error(f"ELO analysis skipped: Required file not found: {e}")
            elo_error_msg = f"File not found: {e}"
            current_local_run_data = load_json_file(local_runs_file).get(run_key, {})
            current_results = current_local_run_data.get("results", {})
            current_results.update({"elo_error": elo_error_msg, "elo_raw": "Error", "elo_normalized": "Error"})
            update_run_data(local_runs_file, run_key, {"results": current_results})
        except Exception as e:
            logging.error(f"ELO analysis failed: {e}", exc_info=True)
            elo_error_msg = str(e)
            current_local_run_data = load_json_file(local_runs_file).get(run_key, {})
            current_results = current_local_run_data.get("results", {})
            current_results.update({"elo_error": elo_error_msg, "elo_raw": "Error", "elo_normalized": "Error"})
            update_run_data(local_runs_file, run_key, {"results": current_results})
    else:
        logging.info("Skipping ELO analysis as per --no-elo flag.")
        # Update results in the LOCAL file if skipped
        current_local_run_data = load_json_file(local_runs_file).get(run_key, {})
        current_results = current_local_run_data.get("results", {})
        if "elo_raw" not in current_results: current_results["elo_raw"] = "Skipped"
        if "elo_normalized" not in current_results: current_results["elo_normalized"] = "Skipped"
        current_results["elo_error"] = None
        update_run_data(local_runs_file, run_key, {"results": current_results})


    # --- Mark run as completed or completed_with_errors (in LOCAL file) ---
    final_status = "completed"
    # Load final data from LOCAL file
    final_local_run_data = load_json_file(local_runs_file).get(run_key, {})
    final_tasks_data = final_local_run_data.get("scenario_tasks", {})
    tasks_in_error_count = 0
    tasks_not_fully_completed_count = 0
    error_examples = []

    for iter_str, scenarios_in_iter in final_tasks_data.items():
        if isinstance(scenarios_in_iter, dict):
            for scenario_id, task_data in scenarios_in_iter.items():
                if isinstance(task_data, dict):
                    task_status = task_data.get("status")
                    is_analysis = scenario_id in C.ANALYSIS_SCENARIO_IDS
                    # Define expected final state based on task type and whether rubric is run
                    if run_rubric:
                        final_expected_status = "rubric_scored"
                    else:
                        final_expected_status = "scenario_completed" if is_analysis else "completed"

                    if task_status == "error":
                        tasks_in_error_count += 1
                        if len(error_examples) < 5: error_examples.append(f"Iter {iter_str}, Scenario {scenario_id}: {task_data.get('error', 'Unknown error')}")
                    elif task_status != final_expected_status:
                         tasks_not_fully_completed_count += 1
                         if len(error_examples) < 5: error_examples.append(f"Iter {iter_str}, Scenario {scenario_id}: Status '{task_status}' (expected '{final_expected_status}')")

    if tasks_in_error_count > 0 or tasks_not_fully_completed_count > 0:
        final_status = "completed_with_errors"
        warning_msg = f"Run {run_key} finished, but issues detected: "
        if tasks_in_error_count > 0: warning_msg += f"{tasks_in_error_count} task(s) ended in error. "
        if tasks_not_fully_completed_count > 0: warning_msg += f"{tasks_not_fully_completed_count} task(s) did not reach final expected status. "
        logging.warning(warning_msg)
        for err_ex in error_examples: logging.warning(f"  - Example Issue: {err_ex}")

    # Update status in LOCAL file
    update_run_data(local_runs_file, run_key, {
        "status": final_status,
        "end_time": datetime.now(timezone.utc).isoformat()
    })
    logging.info(f"Run {run_key} marked as {final_status} in {local_runs_file}.")

    return run_key
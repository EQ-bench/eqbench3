
# File: ai/eqbench3/core/pairwise_judging.py

# core/pairwise_judging.py

import logging
import math
import re
from typing import Dict, Any, List, Tuple, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from tqdm import tqdm

from utils.utils import robust_json_loads
from utils.constants import ANALYSIS_SCENARIO_IDS, DEBRIEF_CHAR_LIMIT, ANALYSIS_RESPONSE_CHAR_LIMIT
from .elo_config import (
    CONVERSATION_HISTORY_TRUNCATION_TOKENS,
    MAX_POSSIBLE_DIFF,
    MAX_ITEMS_PER_MODEL_MATCHUP,
    scenario_notes, # Import the global variable
    analysis_scenario_notes # Import the global variable
)
from .elo_helpers import (
    format_conversation_history,
    should_ignore_scenario,
    _data_has_all_expected_responses,
    downscale_analysis_pair
)
from .matchup_selection import (
    create_matchup_signature
)


##############################################
# Pairwise Comparison Logic (RESTORED & Adapted)
##############################################

def interpret_pairwise_result(result_dict):
    """
    Restores the original multi-criteria logic that:
      - Counts how many '+' signs appear in each dimension.
      - Punishes certain keys (like 'coherence', 'avoids_verbosity') by subtracting pluses from the other side.
      - Returns (outcome_for_A, plus_for_A, plus_for_B) in {0.0, 0.5, 1.0}.

    This matches the snippet from the old code:
      outcome_for_A = 1.0 if A's total > B's total, 0.0 if B's total > A's total, else 0.5
    plus_for_A and plus_for_B are the final tallies.
    """
    if not result_dict:
        return 0.5, 0, 0

    a_score = 0
    b_score = 0

    # Example keys that are "punished" if other model gets the plus
    punish_keys = {
        #"correctness": 4
        }

    for key, val in result_dict.items():
        if key == 'chain_of_thought_reasoning':
            continue

        # Ensure val is a string before calling .count()
        if isinstance(val, str):
            plus_count = val.count('+')
            weight = 1

            # Weight correctness strongly as it's high signal to differentiate stronger/weaker models
            # (this criteria is only relevant in analysis tasks)
            #if key in ["correctness"]: weight = 4
            #if key in ["depth_of_insight", "authentic_eu", "causal_attribution", "theory_of_mind", "incisiveness", "reading_between_lines", "correctness", "overall_eq"]: weight = 0
            #if key in ["authentic_eu", "causal_attribution", "theory_of_mind", "incisiveness", "reading_between_lines", "correctness"]: weight = 0.1
            #if key not in ["demonstrated_empathy"]: weight = 0

            weighted_plus = plus_count * weight

            if "A0493" in val:
                if plus_count > 0:
                    a_score += weighted_plus
                if key in punish_keys:
                    b_score -= plus_count * punish_keys[key]

            elif "A0488" in val:
                if plus_count > 0:
                    b_score += weighted_plus
                if key in punish_keys:
                    a_score -= plus_count * punish_keys[key]
        else:
            logging.debug(f"Skipping non-string value in judge result for key '{key}': {val}")


    if a_score > b_score:
        return 1.0, a_score, b_score  # A wins
    elif b_score > a_score:
        return 0.0, a_score, b_score  # B wins
    else:
        return 0.5, a_score, b_score  # Tie

# RESTORED ORIGINAL
def custom_blend(x: float, linear_gradient=5, sigmoid_power=0.75, transition_start=0.0, transition_end=0.11) -> float:
    """
    Transforms a value in [0,1] by blending a linear slope with a sigmoid curve
    around [transition_start..transition_end].
    """
    return x
    x = max(0.0, min(1.0, x))
    # Linear portion
    linear = linear_gradient * x
    # Sigmoid portion
    k = 3
    # Avoid potential math domain error for x=0 with sigmoid_power < 1
    if x == 0:
        sig = 0.0
    else:
        # Ensure base is non-negative before exponentiation if power < 1
        base = max(0.0, x)
        sig = (1.0 - math.exp(-k * (base**sigmoid_power))) / (1.0 - math.exp(-k))


    # Blend factor (smoothstep)
    if x <= transition_start:
        blend = 0.0
    elif x >= transition_end:
        blend = 1.0
    else:
        t = (x - transition_start)/(transition_end - transition_start)
        blend = t*t*(3-2*t) # Smoothstep interpolation

    return (1.0 - blend)*linear + blend*sig

# RESTORED ORIGINAL (but max diff needs context)
def compute_fraction_for_test(outcome_for_test: float, plus_for_test: int, plus_for_other: int) -> Tuple[float, int, float, float]:
    """
    RESTORED ORIGINAL LOGIC. Calculates margin-based fraction.
    1) plus_diff = abs(plus_for_test - plus_for_other)
    2) normalized = plus_diff / MAX_POSSIBLE_DIFF (e.g., 5 based on typical # criteria)
    3) diff_blended = custom_blend(normalized)
    4) margin = diff_blended/2 + 0.5  => in [0.5..1]
    5) if outcome_for_test=1 => fraction_for_test=margin
       if outcome_for_test=0 => fraction_for_test=1 - margin
       if outcome_for_test=0.5 => fraction_for_test=0.5
    """
    # Determine MAX_POSSIBLE_DIFF based on the number of criteria in the judge prompt
    # Example: If there are 5 criteria, max diff could be 5 (if one model gets all A0493, other gets none)
    # *** ADJUST THIS BASED ON YOUR PAIRWISE PROMPT ***
    # Common default if unsure, assuming ~5 comparison points in the prompt.


    diff = abs(plus_for_test - plus_for_other)
    diff_norm = diff / MAX_POSSIBLE_DIFF if MAX_POSSIBLE_DIFF > 0 else 0.0
    diff_norm = min(diff_norm, 1.0) # Cap normalization at 1.0

    diff_blend = custom_blend(diff_norm, 5, 0.75, 0.0, 0.11) # Using original blend params
    margin = diff_blend / 2.0 + 0.5  # [0.5..1.0]

    if outcome_for_test == 0.5:
        final_fraction = 0.5
    elif outcome_for_test == 1.0:
        final_fraction = margin
    else: # outcome_for_test == 0.0
        final_fraction = 1.0 - margin

    # Ensure fraction is within [0, 1] due to potential floating point issues
    final_fraction = max(0.0, min(1.0, final_fraction))

    return final_fraction, diff, diff_norm, diff_blend




def do_pairwise_judge(
    scenario_id: str,
    scenario_desc: str,
    history_A: List[Dict[str, str]],
    debrief_A: Optional[str],
    history_B: List[Dict[str, str]],
    debrief_B: Optional[str],
    pairwise_prompt_template: str,
    judge_model: str,
    api_clients: Dict[str, Any],
    parsed_responses_A: Optional[List[Dict[str, str]]] = None,
    parsed_responses_B: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Perform a pairwise comparison between two analysis‑oriented responses
    (or standard role‑play responses, depending on scenario_id).

    • For ordinary scenarios the function behaves exactly as before.
    • For analysis‑type scenarios it now substitutes the new placeholders
      {transcript_being_analysed}, {response_A}, {response_B}.
    """
    # Access global notes defined in elo_config
    global scenario_notes, analysis_scenario_notes

    judge_api   = api_clients["judge"]
    is_analysis = scenario_id in ANALYSIS_SCENARIO_IDS

    # --- 1. build standard / legacy strings --------------------------------
    formatted_history_A = format_conversation_history(
        history_A,
        CONVERSATION_HISTORY_TRUNCATION_TOKENS,
        parsed_responses_A,
        scenario_id,
    )
    formatted_history_B = format_conversation_history(
        history_B,
        CONVERSATION_HISTORY_TRUNCATION_TOKENS,
        parsed_responses_B,
        scenario_id,
    )

    debrief_A_str = debrief_A if debrief_A else "[No Debrief Provided]"
    debrief_B_str = debrief_B if debrief_B else "[No Debrief Provided]"
    
    # Strip thinking tags from debrief responses
    if '<think>' in debrief_A_str and '</think>' in debrief_A_str:
        post_think = debrief_A_str.find('</think>') + len('</think>')
        debrief_A_str = debrief_A_str[post_think:].strip()
    if '<think>' in debrief_B_str and '</think>' in debrief_B_str:
        post_think = debrief_B_str.find('</think>') + len('</think>')
        debrief_B_str = debrief_B_str[post_think:].strip()
    
    debrief_A_str = debrief_A_str.replace('*', '').replace('#', '')
    debrief_B_str = debrief_B_str.replace('*', '').replace('#', '')

    if debrief_A and len(debrief_A_str) > DEBRIEF_CHAR_LIMIT:
        debrief_A_str = debrief_A_str[:DEBRIEF_CHAR_LIMIT] + "... [truncated]"
    if debrief_B and len(debrief_B_str) > DEBRIEF_CHAR_LIMIT:
        debrief_B_str = debrief_B_str[:DEBRIEF_CHAR_LIMIT] + "... [truncated]"

    scenario_num  = scenario_id.replace("scenario", "").split("_")[0]
    notes_dict    = analysis_scenario_notes if is_analysis else scenario_notes
    scenario_note = notes_dict.get(scenario_num, "[No scenario notes available]")

    # --- 2. start filling the template -------------------------------------
    final_prompt = pairwise_prompt_template
    final_prompt = final_prompt.replace("{scenario_description}", scenario_desc)
    final_prompt = final_prompt.replace("{scenario_notes}",      scenario_note)
    final_prompt = final_prompt.replace("{conversation_history_A}", formatted_history_A)
    final_prompt = final_prompt.replace("{conversation_history_B}", formatted_history_B)

    if not is_analysis:
        final_prompt = final_prompt.replace("{debrief_A}", debrief_A_str)
        final_prompt = final_prompt.replace("{debrief_B}", debrief_B_str)

    # --- 3. NEW — analysis‑specific placeholders ---------------------------
    if is_analysis:
        # a) transcript_being_analysed  == the sole user prompt (prompt‑1)
        transcript_text = scenario_desc.strip()

        # b) helper to grab the first assistant reply in a history
        def _first_assistant_raw(hist: List[Dict[str, str]]) -> str:
            for msg in hist:
                if msg.get("role") == "assistant":
                    return msg.get("content", "")
            return ""


        raw_A = _first_assistant_raw(history_A)
        raw_B = _first_assistant_raw(history_B)
        raw_A = raw_A.replace('*', '').replace('#', '')
        raw_B = raw_B.replace('*', '').replace('#', '')

        # fallback to parsed_responses if history is unexpectedly empty
        if not raw_A and parsed_responses_A:
            raw_A = parsed_responses_A[0].get("raw", "")
        if not raw_B and parsed_responses_B:
            raw_B = parsed_responses_B[0].get("raw", "")

        # Strip thinking tags before sending to judge
        if '<think>' in raw_A and '</think>' in raw_A:
            post_think = raw_A.find('</think>') + len('</think>')
            raw_A = raw_A[post_think:].strip()
        if '<think>' in raw_B and '</think>' in raw_B:
            post_think = raw_B.find('</think>') + len('</think>')
            raw_B = raw_B[post_think:].strip()

        # truncate to analysis limit
        if len(raw_A) > ANALYSIS_RESPONSE_CHAR_LIMIT:
            raw_A = raw_A[:ANALYSIS_RESPONSE_CHAR_LIMIT] + "... [truncated]"
        if len(raw_B) > ANALYSIS_RESPONSE_CHAR_LIMIT:
            raw_B = raw_B[:ANALYSIS_RESPONSE_CHAR_LIMIT] + "... [truncated]"

        final_prompt = final_prompt.replace("{transcript_being_analysed}", transcript_text)
        final_prompt = final_prompt.replace("{response_A}", raw_A)
        final_prompt = final_prompt.replace("{response_B}", raw_B)

    # --- 4. debugging print (unchanged) ------------------------------------
    #print("\n\n\n\n\n")
    #print("----------------------------")
    #print(f"--- Pairwise Judge Prompt (Scenario: {scenario_id}, Analysis: {is_analysis}) ---")
    #print(final_prompt)
    #print("----------------------------")
    #print("\n\n\n\n\n")

    # --- 5. call judge model ----------------------------------------------
    judge_messages = [{"role": "user", "content": final_prompt}]
    response_text  = ""
    try:
        response_text = judge_api.generate(
            model      = judge_model,
            messages   = judge_messages,
            temperature= 0.0,
            max_tokens = 8000,
            min_p      = None,
        )

        # extract JSON using existing robust logic -------------------------
        #start = response_text.find("{")
        #end   = response_text.rfind("}")
        #if start != -1 and end != -1 and end > start:
        #    json_str = response_text[start : end + 1]
        result   = robust_json_loads(response_text)
        if isinstance(result, dict) and result:
            return result

        # fallback attempts ------------------------------------------------
        #json_match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
        #if json_match:
        #    result = robust_json_loads(json_match.group(1))
        #   if isinstance(result, dict):
        #        return result

        # last fallback – try whole response
        #result = robust_json_loads(response_text)
        #if isinstance(result, dict):
        #    return result

        return {"error": "No valid JSON block found", "raw_response": response_text}

    except Exception as e:
        logging.error(f"Error during pairwise judge API call for scenario {scenario_id}: {e}", exc_info=True)
        return {"error": f"API call failed: {str(e)}", "raw_response": response_text or "No response"}


def _judge_scenario_pairs_in_parallel(
    test_model_name: str, # Logical name
    neighbor_model_name: str, # Logical name
    test_model_results: Dict[str, Dict[str, Any]],
    neighbor_model_results: Dict[str, Dict[str, Any]],
    concurrency: int,
    standard_pairwise_prompt_template: str,
    analysis_pairwise_prompt_template: str,
    scenarios_data: Dict[str, List[str]],
    judge_model: str,
    api_clients: Dict[str, Any],
    max_pairs_per_model_matchup: int = MAX_ITEMS_PER_MODEL_MATCHUP,
    existing_matchups: Optional[set] = None
) -> List[Dict[str, Any]]:
    """
    Judges overlapping scenarios between two models (using logical names) with essential debug logging.
    Selects the correct pairwise prompt based on scenario type.
    Stores logical names in the comparison results.
    """
    comparisons = []
    tasks_to_submit = []
    processed_signatures = set()

    if existing_matchups is None:
        existing_matchups = set()

    common_scenario_ids = sorted(list(set(test_model_results.keys()) & set(neighbor_model_results.keys())))
    logging.info(f"Found {len(common_scenario_ids)} common scenarios between {test_model_name} and {neighbor_model_name}.")

    total_possible_comparisons = 0
    skipped_existing_comparisons = 0
    skipped_for_other_reasons = 0

    # Iterate through common scenarios
    for scenario_id in common_scenario_ids:
        if should_ignore_scenario(scenario_id):
            logging.debug(f"Skipping ignored scenario {scenario_id}")
            continue

        test_iters = test_model_results.get(scenario_id, {})
        neighbor_iters = neighbor_model_results.get(scenario_id, {})

        if not test_iters or not neighbor_iters:
            continue

        # Find common iteration indices for this scenario
        common_iter_indices = sorted(list(set(test_iters.keys()) & set(neighbor_iters.keys())))

        # Iterate through common iteration indices
        for iter_idx in common_iter_indices:
            # quick pass: both sides must be fully populated
            if not (
                _data_has_all_expected_responses(test_iters[iter_idx],     scenario_id)
                and _data_has_all_expected_responses(neighbor_iters[iter_idx], scenario_id)
            ):
                continue

            task_data_A = test_iters[iter_idx]
            task_data_B = neighbor_iters[iter_idx]

            is_analysis = scenario_id in ANALYSIS_SCENARIO_IDS
            # Analysis tasks are ready after 'scenario_completed'
            # Standard/Drafting tasks are ready after 'completed' (post-debrief) or 'rubric_scored'
            required_status_A = "rubric_scored" if is_analysis else ["completed", "rubric_scored"]
            required_status_B = "rubric_scored" if is_analysis else ["completed", "rubric_scored"]

            status_A_ok = task_data_A.get("status") == required_status_A if isinstance(required_status_A, str) else task_data_A.get("status") in required_status_A
            status_B_ok = task_data_B.get("status") == required_status_B if isinstance(required_status_B, str) else task_data_B.get("status") in required_status_B

            if not (
                _data_has_all_expected_responses(task_data_A, scenario_id)
                and _data_has_all_expected_responses(task_data_B, scenario_id)
            ):
                skipped_for_other_reasons += 1
                continue

            # Ensure both tasks for this iteration are in the correct state and have necessary data
            if not status_A_ok or not status_B_ok:
                skipped_for_other_reasons += 1
                continue

            if not task_data_A.get("conversation_history") or not task_data_B.get("conversation_history"):
                skipped_for_other_reasons += 1
                continue

            # Analysis tasks don't have debriefs
            if not is_analysis and (task_data_A.get("debrief_response") is None or task_data_B.get("debrief_response") is None):
                 skipped_for_other_reasons += 1
                 continue

            total_possible_comparisons += 1

            # Create a matchup signature using logical names
            matchup_sig = create_matchup_signature(
                test_model_name, neighbor_model_name, scenario_id, iter_idx
            )

            # Skip if this exact matchup already exists in previous results
            if matchup_sig in existing_matchups:
                skipped_existing_comparisons += 1
                continue

            # Only submit if this logical pair hasn't been processed in this function call
            if matchup_sig not in processed_signatures:
                scenario_desc = scenarios_data.get(scenario_id, ["Scenario description not found."])[0]
                # Choose the correct pairwise prompt template
                pairwise_template = analysis_pairwise_prompt_template if is_analysis else standard_pairwise_prompt_template

                # Submit forward comparison (test model as A, neighbor as B)
                tasks_to_submit.append({
                    "scenario_id": scenario_id,
                    "scenario_desc": scenario_desc,
                    "model_A": test_model_name, "iter_A": iter_idx, "data_A": task_data_A, # Logical name
                    "model_B": neighbor_model_name, "iter_B": iter_idx, "data_B": task_data_B, # Logical name
                    "direction": "forward", "signature": matchup_sig,
                    "pairwise_template": pairwise_template
                })
                # Submit reverse comparison (neighbor as A, test model as B)
                tasks_to_submit.append({
                    "scenario_id": scenario_id,
                    "scenario_desc": scenario_desc,
                    "model_A": neighbor_model_name, "iter_A": iter_idx, "data_A": task_data_B, # Logical name
                    "model_B": test_model_name, "iter_B": iter_idx, "data_B": task_data_A, # Logical name
                    "direction": "reversed", "signature": matchup_sig,
                    "pairwise_template": pairwise_template
                })
                processed_signatures.add(matchup_sig)

    logging.info(f"Prepared {len(tasks_to_submit)} tasks ({len(processed_signatures)} unique matchups) between {test_model_name} and {neighbor_model_name}")
    logging.debug(f"Skipped {skipped_existing_comparisons} existing matchups and {skipped_for_other_reasons} for other reasons")

    # ------------------------------------------------------------------
    # Deterministic, evenly‑spaced down‑sampling of logical pairs
    # (cap is reduced by the number of *existing* logical pairs so that
    #  total ≤ max_pairs_per_model_matchup for this opponent)
    # ------------------------------------------------------------------
    model_lo, model_hi = sorted([test_model_name, neighbor_model_name])
    existing_for_pair = sum(
        1
        for sig in existing_matchups
        if sig[0] == model_lo and sig[1] == model_hi
    )

    effective_cap = None
    if max_pairs_per_model_matchup is not None:
        effective_cap = max(0, max_pairs_per_model_matchup - existing_for_pair)

    # ---------- EARLY‑EXIT FIX ----------------------------------------
    if effective_cap == 0:
        logging.info(
            f"[Matchups] Cap satisfied for {neighbor_model_name}: "
            f"{existing_for_pair}/{max_pairs_per_model_matchup} logical pairs. "
            f"No new comparisons scheduled."
        )
        return []
    # ------------------------------------------------------------------

    num_logical_pairs = len(processed_signatures)
    if effective_cap and num_logical_pairs > effective_cap:
        # ---- 1. stable ordering --------------------------------------
        # prefer scenario-id "404" when stage-1’s cap == 1
        signatures_sorted = sorted(
            processed_signatures,
            key=lambda s: (0, int(s[3])) if s[2] == "404" else (1, s[2], int(s[3]))
        )

        # ---- 2. evenly‑spaced sampler --------------------------------
        def evenly_sample(seq, k):
            if k <= 0 or not seq:
                return []
            if k >= len(seq):
                return list(seq)
            if k == 1:
                return [seq[0]]
            step = (len(seq) - 1) / (k - 1)
            return [seq[int(round(i * step))] for i in range(k)]

        signatures_to_keep = set(
            evenly_sample(signatures_sorted, effective_cap)
        )

        # ---- 3. filter tasks -----------------------------------------
        tasks_to_submit = [
            task for task in tasks_to_submit
            if task["signature"] in signatures_to_keep
        ]

    # ---------- SUMMARY LOG ------------------------------------
    if tasks_to_submit:
        unique_sigs = {task["signature"] for task in tasks_to_submit}
        scenario_list = sorted({sig[2] for sig in unique_sigs})
        summary_block = (
            "\n"
            "================  Matchups scheduled  =================\n"
            f"{test_model_name}  vs  {neighbor_model_name}\n"
            f"Existing logical pairs : {existing_for_pair}\n"
            f"New logical pairs      : {len(unique_sigs)}\n"
            f"API tasks submitted    : {len(tasks_to_submit)}\n"
            "Scenarios:\n" +
            "\n".join(f"  • {sid}" for sid in scenario_list) +
            "\n=======================================================\n"
        )
        logging.info(summary_block)
    # ------------------------------------------------------------------

    if not tasks_to_submit:
        logging.info(f"No new comparisons to submit between {test_model_name} and {neighbor_model_name}.")
        return []

    # Track which signatures have completed forward and reverse directions
    received_directions = defaultdict(set)

    # (everything below is unchanged – building futures, collecting results, etc.)
    # ------------------------------------------------------------------
    # Run judgments in parallel
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for task in tasks_to_submit:
            future = executor.submit(
                do_pairwise_judge,
                task["scenario_id"], task["scenario_desc"],
                task["data_A"]["conversation_history"], task["data_A"].get("debrief_response"),
                task["data_B"]["conversation_history"], task["data_B"].get("debrief_response"),
                task["pairwise_template"],
                judge_model, api_clients,
                task["data_A"].get("parsed_responses", []),
                task["data_B"].get("parsed_responses", [])
            )
            futures.append((future, task))

        for future, task in tqdm(futures, desc=f"Judging {test_model_name} vs {neighbor_model_name}", leave=False):
            scenario_id = task["scenario_id"]
            direction = task["direction"]
            iter_idx = task["iter_A"] # Iteration index is the same for A and B here
            matchup_sig = task["signature"]

            try:
                judge_result = future.result()

                if isinstance(judge_result, dict) and "error" not in judge_result:
                    comparison_result = None

                    # Calculate average response length for both models
                    model_A_responses = [msg.get("content", "") for msg in task["data_A"]["conversation_history"]
                                    if msg.get("role") == "assistant"]
                    model_A_avg_length = sum(len(response) for response in model_A_responses) / max(1, len(model_A_responses))

                    model_B_responses = [msg.get("content", "") for msg in task["data_B"]["conversation_history"]
                                    if msg.get("role") == "assistant"]
                    model_B_avg_length = sum(len(response) for response in model_B_responses) / max(1, len(model_B_responses))

                    if direction == "forward":
                        # model_A is test_model_name, model_B is neighbor_model_name
                        outcome_for_test_model, test_model_plus_count, neighbor_plus_count = interpret_pairwise_result(judge_result)
                        #fracTest, diff, diff_norm, diff_blend = compute_fraction_for_test(outcome_for_test_model, test_model_plus_count, neighbor_plus_count)
                        (test_model_plus_count,
                        neighbor_plus_count,
                        diff,
                        diff_norm,
                        diff_blend,
                        fracTest) = downscale_analysis_pair(
                                        scenario_id, outcome_for_test_model,
                                        test_model_plus_count, neighbor_plus_count)

                        comparison_result = {
                            "scenario_id": scenario_id,
                            "pair": {
                                "test_model": test_model_name, # Logical name
                                "neighbor_model": neighbor_model_name, # Logical name
                                "iteration_index": iter_idx,
                            },
                            "order": "A0493:test / A0488:other",
                            "judge_response": judge_result,
                            "outcome_for_test_model": outcome_for_test_model,
                            "plus_for_test": test_model_plus_count,
                            "plus_for_other": neighbor_plus_count,
                            "plus_diff": diff,
                            "plus_diff_normalized": diff_norm,
                            "plus_diff_blended": diff_blend,
                            "fraction_for_test": fracTest,
                            "test_model_avg_response_length": model_A_avg_length,
                            "neighbor_model_avg_response_length": model_B_avg_length
                        }
                    else: # direction == "reversed"
                        # model_A is neighbor_model_name, model_B is test_model_name
                        outcome_for_neighbor_model, neighbor_plus_count, test_model_plus_count = interpret_pairwise_result(judge_result)
                        # Invert outcome to be relative to the *test* model
                        if outcome_for_neighbor_model == 1.0:
                            outcome_for_test_model = 0.0
                        elif outcome_for_neighbor_model == 0.0:
                            outcome_for_test_model = 1.0
                        else:
                            outcome_for_test_model = 0.5
                        #fracTest, diff, diff_norm, diff_blend = compute_fraction_for_test(outcome_for_test_model, test_model_plus_count, neighbor_plus_count)                        
                        (test_model_plus_count,
                        neighbor_plus_count,
                        diff,
                        diff_norm,
                        diff_blend,
                        fracTest) = downscale_analysis_pair(
                                        scenario_id, outcome_for_test_model,
                                        test_model_plus_count, neighbor_plus_count)
                        
                        comparison_result = {
                            "scenario_id": scenario_id,
                            "pair": {
                                # Store consistently with test_model_name as 'test_model'
                                "test_model": test_model_name, # Logical name
                                "neighbor_model": neighbor_model_name, # Logical name
                                "iteration_index": iter_idx,
                            },
                            "order": "A0493:other / A0488:test", # Reflects the A/B assignment in this specific judge call
                            "judge_response": judge_result,
                            "outcome_for_test_model": outcome_for_test_model,
                            "plus_for_test": test_model_plus_count,
                            "plus_for_other": neighbor_plus_count,
                            "plus_diff": diff,
                            "plus_diff_normalized": diff_norm,
                            "plus_diff_blended": diff_blend,
                            "fraction_for_test": fracTest,
                            # Lengths correspond to the logical models
                            "test_model_avg_response_length": model_B_avg_length, # B was test model in this call
                            "neighbor_model_avg_response_length": model_A_avg_length # A was neighbor in this call
                        }

                    comparisons.append(comparison_result)
                    received_directions[matchup_sig].add(direction)

                else:
                    logging.warning(f"Judge error for scenario {scenario_id} iter {iter_idx} ({direction}): {judge_result.get('error', 'Unknown error')}")
                    # Store error with logical names
                    comparisons.append({
                        "scenario_id": scenario_id,
                        "pair": {"test_model": test_model_name, "neighbor_model": neighbor_model_name,
                                "iteration_index": iter_idx},
                        "order": f"{direction}",
                        "error": f"Judge Error: {judge_result.get('error', 'Unknown error')}",
                        "raw_judge_response": judge_result.get('raw_response', '')
                    })

            except Exception as e:
                logging.error(f"Exception for scenario {scenario_id} iter {iter_idx} ({direction}): {str(e)}")
                # Store error with logical names
                comparisons.append({
                    "scenario_id": scenario_id,
                    "pair": {"test_model": test_model_name, "neighbor_model": neighbor_model_name,
                            "iteration_index": iter_idx},
                    "order": f"{direction}",
                    "error": f"Future processing error: {str(e)}"
                })

    # Check for incomplete pairs
    complete_pairs = 0
    incomplete_pairs = 0
    for sig, directions in received_directions.items():
        if len(directions) == 2:
            complete_pairs += 1
        else:
            incomplete_pairs += 1

    if incomplete_pairs > 0:
        logging.warning(f"Found {incomplete_pairs} incomplete pairs (missing one direction) out of {complete_pairs + incomplete_pairs} total")

    logging.info(f"Finished judging {test_model_name} vs {neighbor_model_name}. Generated {len(comparisons)} comparison results.")



    # ── DEBUG: what actually survived the helper’s filter? ──────────────
    ok = [c for c in comparisons if "error" not in c]
    bad = [c for c in comparisons if "error" in c]
    logging.info(
        f"[ELO-DBG] {test_model_name} vs {neighbor_model_name}: "
        f"{len(ok)} good / {len(bad)} error comparisons generated"
    )
    return comparisons


def print_matchup_results(comparisons):
    """
    Print lines like:
        model_A vs model_B  ->  model_A winner
    or “… -> tie” if the judge called it even.

    Parameters
    ----------
    comparisons : Iterable[dict]
        Each item must contain at least:
            • 'pair' : {'test_model': str, 'neighbor_model': str} # These are logical names
            • either
                - 'fraction_for_test'   (preferred)  OR
                - 'outcome_for_test_model'
    """
    for comp in comparisons:
        if "error" in comp:                # skip judge‑error entries
            continue

        pair = comp.get("pair", {})
        a, b = pair.get("test_model"), pair.get("neighbor_model") # Logical names
        if not a or not b:
            continue                       # malformed record

        # ----- decide winner --------------------------------------------
        if "fraction_for_test" in comp:    # 0‥1 ( >.5  ⇒  test model wins )
            frac = comp["fraction_for_test"]
            if   frac > 0.5: winner = a
            elif frac < 0.5: winner = b
            else:            winner = "tie"
        else:                              # fallback to legacy field
            out = comp.get("outcome_for_test_model")
            if   out == 1.0: winner = a
            elif out == 0.0: winner = b
            else:            winner = "tie"

        print(f"{a} vs {b}  ->  {winner}")


def _recompute_comparison_stats(comp: Dict[str, Any]) -> None:
    """
    Re-derives plus counts, outcome, margin blend and fraction_for_test
    from an *existing* comparison record – in-place.
    Assumes pair['test_model'] and pair['neighbor_model'] hold logical names.

    Skips entries that contain an 'error' key or lack 'judge_response'.
    """
    if "error" in comp or "judge_response" not in comp:
        return                                # nothing to do / unusable

    judge_dict = comp["judge_response"]
    outcome_A, plus_A, plus_B = interpret_pairwise_result(judge_dict)

    # Figure out whether the *logical* test model corresponds to A0493 or A0488
    # We encoded this once in the human-readable 'order' string.
    order_str = comp.get("order", "")
    a_is_test = order_str.startswith("A0493:test") # This reflects the A/B assignment during the judge call

    if a_is_test:
        plus_test, plus_other = plus_A, plus_B
        outcome_test = outcome_A
    else: # A was the neighbor model during the judge call
        plus_test, plus_other = plus_B, plus_A
        # invert outcome because roles were flipped relative to the logical test model
        if   outcome_A == 1.0: outcome_test = 0.0
        elif outcome_A == 0.0: outcome_test = 1.0
        else:                  outcome_test = 0.5

    frac, diff, diff_norm, diff_blend = compute_fraction_for_test(
        outcome_test, plus_test, plus_other
    )

    (plus_test,
     plus_other,
     diff,
     diff_norm,
     diff_blend,
     frac) = downscale_analysis_pair(
                 comp["scenario_id"], outcome_test,
                 plus_test, plus_other)

    # overwrite the stored fields
    comp.update({
        "plus_for_test":          plus_test,
        "plus_for_other":         plus_other,
        "plus_diff":              diff,
        "plus_diff_normalized":   diff_norm,
        "plus_diff_blended":      diff_blend,
        "outcome_for_test_model": outcome_test,
        "fraction_for_test":      frac,
    })

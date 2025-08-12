# core/elo_helpers.py

import logging
import statistics
from typing import Dict, Any, List, Tuple, Optional, Set

# Assuming constants are imported from a central location or defined elsewhere if needed
from utils.constants import (
    NO_RP_SCENARIO_IDS,
    MESSAGE_DRAFTING_SCENARIO_IDS,
    ANALYSIS_SCENARIO_IDS,
    SECTION_CHAR_LIMITS,
    SECTION_CHAR_LIMITS_MESSAGE_DRAFT,
    RAW_RESPONSE_CHAR_LIMIT,
    ANALYSIS_RESPONSE_CHAR_LIMIT,
    # DEBRIEF_CHAR_LIMIT # Not directly used here, but format_conversation_history uses it implicitly via do_pairwise_judge
)
from .elo_config import (
    IGNORE_SCENARIOS_FOR_ELO,
    DEFAULT_ELO,
    ALLOW_INCOMPLETE_RESPONSES,
    MANDATORY_SECTIONS_ROLEPLAY,
    MANDATORY_SECTIONS_DRAFTING,
    _MIN_RAW_LEN
)


##############################################
# Helper Functions (Adapted/Restored)
##############################################

def should_ignore_scenario(scenario_id: str) -> bool:
    """Check if scenario_id should be ignored for ELO."""
    # Handle potential iteration suffixes like "scenario1_iter2" if needed
    base_id = scenario_id.split('_iter')[0] if '_iter' in scenario_id else scenario_id
    return base_id in IGNORE_SCENARIOS_FOR_ELO


def truncate_text(text: str, limit: int) -> str:
    if not text or len(text) <= limit:
        return text
    return text[:limit] + "... [truncated]"


def truncate_parsed_sections(parsed: Dict[str, str], limits: Dict[str, int]) -> Dict[str, str]:
    """
    Truncate each section of parsed response according to character limits.

    Args:
        parsed: Dictionary containing parsed sections
        limits: Dictionary mapping section names to character limits

    Returns:
        Dictionary with truncated sections
    """
    truncated = {}

    for section_name, content in parsed.items():
        # Skip non-content keys or keys not in limits
        if section_name == "raw" or section_name not in limits:
            continue

        if content:
            char_limit = limits[section_name]
            if len(content) > char_limit:
                truncated[section_name] = content[:char_limit] + "... [truncated]"
            else:
                truncated[section_name] = content
        else:
            truncated[section_name] = content # Keep empty content as is

    # Ensure all expected keys from limits are present, even if empty
    for key in limits:
        if key not in truncated:
            truncated[key] = ""

    return truncated


def format_parsed_response(sections: Dict[str, str]) -> str:
    """
    Render parsed sections back into a readable block.
    Works for both role‑play and message‑drafting layouts.
    """
    formatted = []

    # Role-play format
    if "thinking_feeling" in sections:
        formatted.append("# I'm thinking & feeling")
        formatted.append(sections.get("thinking_feeling", "")) # Use .get for safety
        formatted.append("# They're thinking & feeling")
        formatted.append(sections.get("their_thinking_feeling", ""))
        formatted.append("# My response")
        formatted.append(sections.get("response", ""))

    # Message-drafting format (using updated keys)
    elif "perspective_taking" in sections:
        formatted.append("# Perspective-taking")
        formatted.append(sections.get("perspective_taking", ""))
        formatted.append("# Draft brainstorming")
        formatted.append(sections.get("draft_brainstorming", ""))
        formatted.append("# Draft") # Changed header slightly for consistency
        formatted.append(sections.get("draft", ""))

    # Add separator if content was added
    if formatted:
        formatted.append("--")

    return "\n\n".join(formatted)


# The corrected format_conversation_history function:
def format_conversation_history(
    history: List[Dict[str, str]],
    max_tokens: Optional[int] = None,
    parsed_responses: Optional[List[Dict[str, str]]] = None,
    scenario_id: Optional[str] = None
) -> str:
    """
    Formats conversation history into a readable string, optionally truncating.
    Handles standard, message drafting, NO_RP, and analysis scenarios.

    Args:
        history: List of message dictionaries with 'role' and 'content'
        max_tokens: Optional maximum token count for the entire formatted history
        parsed_responses: Optional list of parsed response dictionaries from the model
        scenario_id: Optional scenario ID to check if it's a special scenario

    Returns:
        Formatted conversation history as a string
    """
    formatted = []
    total_tokens = 0
    assistant_response_index = 0  # Track which assistant response we're processing

    # Check scenario type
    no_rp_scenario = scenario_id in NO_RP_SCENARIO_IDS if scenario_id else False
    is_drafting = scenario_id in MESSAGE_DRAFTING_SCENARIO_IDS if scenario_id else False
    is_analysis = scenario_id in ANALYSIS_SCENARIO_IDS if scenario_id else False

    # Process messages in order
    for i, msg in enumerate(history):
        role = msg.get("role", "unknown").capitalize()
        content = msg.get("content", "")

        # For assistant messages
        if role.lower() == "assistant":
            # Strip thinking tags before sending to judge
            if '<think>' in content and '</think>' in content:
                post_think = content.find('</think>') + len('</think>')
                content = content[post_think:].strip()
            
            # Standard Role-Play or Message Drafting: Format parsed sections
            if parsed_responses and assistant_response_index < len(parsed_responses) and not no_rp_scenario and not is_analysis:
                parsed = parsed_responses[assistant_response_index]
                limits = SECTION_CHAR_LIMITS_MESSAGE_DRAFT if is_drafting else SECTION_CHAR_LIMITS
                truncated_sections = truncate_parsed_sections(parsed, limits)
                content = format_parsed_response(truncated_sections)
                assistant_response_index += 1
            # NO_RP or Analysis: Use raw content, apply truncation if needed
            elif no_rp_scenario or is_analysis:
                # Apply truncation for special scenarios only during judging (if needed)
                # Currently using RAW_RESPONSE_CHAR_LIMIT for NO_RP, consider if analysis needs different limit
                if no_rp_scenario:
                    if len(content) > RAW_RESPONSE_CHAR_LIMIT:
                        content = content[:RAW_RESPONSE_CHAR_LIMIT] + "... [truncated]"
                else:
                    # Use ANALYSIS_RESPONSE_CHAR_LIMIT for analysis scenarios
                    if len(content) > ANALYSIS_RESPONSE_CHAR_LIMIT:
                        content = content[:ANALYSIS_RESPONSE_CHAR_LIMIT] + "... [truncated]"

                # Still increment the index even if using raw content
                if parsed_responses and assistant_response_index < len(parsed_responses):
                    assistant_response_index += 1
            # Fallback if parsed_responses are missing/mismatched
            else:
                 logging.warning(f"Mismatch or missing parsed_responses for assistant turn {assistant_response_index} in scenario {scenario_id}. Using raw content.")
                 # Apply truncation to raw content as a fallback
                 if len(content) > RAW_RESPONSE_CHAR_LIMIT:
                     content = content[:RAW_RESPONSE_CHAR_LIMIT] + "... [truncated]"
                 # Increment index if possible
                 if parsed_responses and assistant_response_index < len(parsed_responses):
                     assistant_response_index += 1


        content = content.replace('*', '').replace('#', '')
        entry = f"{role}:\n{content}\n---"
        # Simple token estimation (split by space) - consider a tokenizer for accuracy
        entry_tokens = len(content.split())

        if max_tokens is not None and total_tokens + entry_tokens > max_tokens:
            # Truncate if needed (similar to before)
            allowed_tokens = max_tokens - total_tokens
            if allowed_tokens > 50:  # Only include truncated message if reasonably long
                words = content.split()
                # Truncate content
                truncated_content = " ".join(words[:allowed_tokens]) + " ... [TRUNCATED]"
                entry = f"{role}:\n{truncated_content}\n---"
                formatted.append(entry)
                total_tokens += allowed_tokens
            else:
                if not any("[... HISTORY TRUNCATED ...]" in f for f in formatted):
                    formatted.append("[... HISTORY TRUNCATED ...]")
            break  # Stop adding more messages

        formatted.append(entry)
        total_tokens += entry_tokens

    return "\n".join(formatted)


def get_median_elo(existing_analyses: Dict[str, Any]) -> float:
    """Calculates the median ELO from existing model analyses."""
    elo_scores = [
        info.get("elo") for model, info in existing_analyses.items()
        if model != "__metadata__" and isinstance(info.get("elo"), (int, float))
    ]
    if not elo_scores:
        return DEFAULT_ELO
    return statistics.median(elo_scores)

# ---------- completeness-check helpers ------------------------------------
def _data_has_all_expected_responses(task: Dict[str, Any],
                                     scenario_id: str) -> bool:
    """
    Return True  ⟺  *every* assistant turn that we will judge contains
    **either** a sufficiently long `raw` field **or** fully-populated
    parsed sections appropriate to the scenario-type.

    Baseline (even when `ALLOW_INCOMPLETE_RESPONSES` is True):
    ▸ There must be **at least one** assistant message in the main
      conversation (i.e., not the debrief) whose raw text length is
      ≥ `_MIN_RAW_LEN`.

    – NO_RP / analysis : only need one assistant message ≥ `_MIN_RAW_LEN`
    – role-play / drafting : need all assistant turns + a non-empty debrief
    """
    # ---------------------------------------------------------------------
    # Gather assistant turns once so the baseline check works everywhere.
    # ---------------------------------------------------------------------
    conv = task.get("conversation_history") or []
    assistants = [m for m in conv if m.get("role") == "assistant"]

    # ---------------------------------------------------------------------
    # If incomplete responses are allowed, enforce only the baseline.
    # ---------------------------------------------------------------------
    if ALLOW_INCOMPLETE_RESPONSES:
        return any(len(m.get("content", "").strip()) >= _MIN_RAW_LEN
                   for m in assistants)

    # ---------------------------------------------------------------------
    # Normal (strict) validation follows.
    # ---------------------------------------------------------------------
    is_analysis = scenario_id in ANALYSIS_SCENARIO_IDS
    is_drafting = scenario_id in MESSAGE_DRAFTING_SCENARIO_IDS
    is_no_rp    = scenario_id in NO_RP_SCENARIO_IDS

    # ── NO_RP  &  analysis ───────────────────────────────────────────────
    if is_analysis or is_no_rp:
        return any(len(m.get("content", "").strip()) >= _MIN_RAW_LEN
                   for m in assistants)

    # ── role-play / drafting ─────────────────────────────────────────────
    parsed = task.get("parsed_responses")
    if not isinstance(parsed, list) or not parsed:
        return False                         # nothing parsed → incomplete

    required = (MANDATORY_SECTIONS_DRAFTING if is_drafting
                else MANDATORY_SECTIONS_ROLEPLAY)

    def turn_ok(pr: Dict[str, str]) -> bool:
        raw = (pr.get("raw") or "").strip()
        if len(raw) >= _MIN_RAW_LEN:
            return True                      # long raw text is enough
        return all(pr.get(k, "").strip()     # otherwise need all sections
                   for k in required)

    if not all(turn_ok(pr) for pr in parsed[:len(assistants)]):
        return False

    # Debrief remains compulsory in role-play & drafting scenarios.
    return bool(task.get("debrief_response", "").strip())



def load_scenario_notes(file_path: str) -> Dict[str, str]:
    """
    Load scenario notes from file.
    Format is # <scenario_id>\n<notes>\n# <scenario_id>\n<notes>...
    """
    notes = {}
    current_id = None
    current_notes = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('# '):
                    # Save previous scenario notes if any
                    if current_id is not None and current_notes:
                        notes[current_id] = '\n'.join(current_notes).strip()
                        current_notes = []

                    # Extract new scenario ID
                    current_id = line.strip('# \n')
                else:
                    if current_id is not None:
                        current_notes.append(line.rstrip('\n'))

            # Save the last scenario notes
            if current_id is not None and current_notes:
                notes[current_id] = '\n'.join(current_notes).strip()
    except FileNotFoundError:
        logging.warning(f"Scenario notes file not found: {file_path}. No notes loaded.")
    except Exception as e:
        logging.error(f"Error loading scenario notes from {file_path}: {e}")

    return notes


DOWNSCALE_ANALYSIS_CONTRIBUTION = True   # global flag (leave as-is)

def downscale_analysis_pair(scenario_id: str,
                            outcome_test: float,
                            plus_test: int,
                            plus_other: int
                           ) -> Tuple[int, int, float, float, float, float]:
    """
    If the scenario is an analysis item (id ≥ 400) and the global flag is
    on, clamp the plus-counts to 1-0 / 0-1 / 0-0 so that each analysis
    matchup carries the minimum possible margin.

    Returns the full set of derived values:
        plus_test, plus_other, diff, diff_norm, diff_blend, fraction_for_test
    """
    # local import → avoids circularity
    from .pairwise_judging import compute_fraction_for_test
    
    # quick check: is analysis?
    if not DOWNSCALE_ANALYSIS_CONTRIBUTION:
        pass                     # leave untouched
    else:
        try:
            if int(scenario_id) >= 400:
                if   outcome_test == 1.0: plus_test, plus_other = 1, 0
                elif outcome_test == 0.0: plus_test, plus_other = 0, 1
                else:                     plus_test = plus_other = 0
        except ValueError:
            # non-numeric id -> treat as non-analysis
            pass

    # re-compute diff / margin & fraction
    diff, diff_norm, diff_blend = None, None, None   # just to declare
    fraction, diff, diff_norm, diff_blend = compute_fraction_for_test(
        outcome_test, plus_test, plus_other)

    return plus_test, plus_other, diff, diff_norm, diff_blend, fraction
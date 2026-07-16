# EQ-Bench 3

EQ-Bench 3 is a multi-turn emotional intelligence benchmark. It assesses active EQ skills, interpersonal skills, psychological insight and analytical depth. It challenges language models with role-play or analysis tasks that require empathy, depth of insight, social dexterity, and more. An auxiliary judge model (by default, Claude Opus 4.6) scores or pairwise-compares the outputs.

For full details on the benchmark including methodology, criteria, bias analysis, repeatabilty experiments and more, click [here](http://localhost:8000/about.html#long).

**Features**  
- **Role-Play Scenarios**: The tested model is placed in conversation-based scenarios (e.g., parenting, relationship conflict, workplace tension). It must articulate what it (and others) feel/think before delivering its final response.  
- **Analysis Scenarios**: The model is asked to read or interpret a transcript and perform an in-depth analysis of human dynamics and subtext.  
- **Rubric Scoring**: A judge LLM assigns a multi-criteria rubric score (0–20, scaled up to 0–100) to each scenario outcome, focusing on empathy, emotional reasoning, social skill, etc.  
- **Pairwise ELO Analysis**: The judge compares transcripts from two different models for the same scenario, awarding a “win” margin for each of several criteria. A TrueSkill/ELO-like solver aggregates all pairwise comparisons to produce a final ranking.

## Leaderboard
![image](https://github.com/user-attachments/assets/bcd7b765-bf46-42c8-8081-415762d24263)

The full EQ-Bench leaderboard is viewable at: [https://eqbench.com/](https://eqbench.com/)

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Installation & Requirements](#installation--requirements)  
3. [Quickstart](#quickstart)  
4. [Running the Benchmark](#running-the-benchmark)  
   - [Command-Line Arguments](#command-line-arguments)  
   - [Example Commands](#example-commands)  
   - [Rubric vs ELO](#rubric-vs-elo)  
   - [Local vs Canonical Results Files](#local-vs-canonical-results-files)  
5. [Example Use Case: Compare a single model to a baseline with Elo](#5-example-use-case-compare-a-single-model-to-a-baseline-with-elo)
6. [Merging Local Results into the Canonical Leaderboard](#merging-local-results-into-the-canonical-leaderboard)  
7. [Folder and File Structure](#folder-and-file-structure)  
8. [Limitations and Notes](#limitations-and-notes)  
9. [Contact and License](#contact-and-license)  
10. [Citation](#citation)

---

## 1. Project Overview

EQ-Bench 3 aims to measure **active emotional intelligence** abilities in LLMs. Rather than knowledge-based or short-answer questions, tasks here are multi-turn dialogues or analysis questions that test empathy, social dexterity, and psychological insight. The evaluated model’s responses are then graded by a judge model:
- **Rubric pass**: The judge model issues a numerical score for each scenario.  
- **ELO pass**: The judge model performs pairwise comparisons of transcripts from different models, resulting in an overall ELO ranking (via TrueSkill).

### Key Points
- **Scenarios** vary from relationship drama to conflict mediation, pushing the tested model to reason about others’ emotions.  
- **Analysis tasks** require deeper reflection on a provided transcript or scenario.  
- **Judge model**: By default, a Claude model (Opus 4.6) is used, but any LLM accessible via an OpenAI-compatible endpoint can serve as judge.  
- **Truncation**: Pairwise judgments truncate outputs to level the playing field. Rubric judgments typically do not truncate (to preserve detail).  

---

## 2. Installation & Requirements

1. Clone this repository:

   ```bash
   git clone https://github.com/EQ-bench/eqbench3.git
   cd eqbench3
   ```

2. (Optional) Create a virtual environment and activate it:

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Configure your API keys in `.env`:

   ```bash
   cp .env.example .env
   # then edit .env with your API keys, e.g.:
   # TEST_API_KEY=sk-...
   # JUDGE_API_KEY=sk-...
   ```

   - `TEST_API_KEY` & `TEST_API_URL` are used when calling the tested model.  
   - `JUDGE_API_KEY` & `JUDGE_API_URL` are used by the judge model.  

---

## 3. Quickstart

1. **Run a single iteration** of EQ-Bench (Rubric Scoring only, no Elo):
   ```bash
   python eqbench3.py \
     --test-model openai/gpt-4.1-mini \
     --model-name gpt-4.1-mini-demo-run \
     --judge-model claude-opus-4-6 \
     --no-elo \
     --iterations 1
   ```
   - Runs 1 iteration of every scenario, scoring them with the rubric.  
   - Data is recorded in `eqbench3_runs.json`, and results displayed to console.

3. **Full benchmark run with Elo to place the model on the leaderboard**:
   ```bash
   python eqbench3.py \
     --test-model openai/gpt-4.1-mini \
     --model-name my-gpt4-run \
     --judge-model claude-opus-4-6
   ```
   - After the roleplay scenarios are completed, it does a multi-stage pairwise pass (comparing the evaluated model to known models in local+leaderboard data).  
   - Matchup results & ELO rating is stored in `elo_results_eqbench3.json`, and displayed to console.

---

## 4. Running the Benchmark

You interact with the main script [`eqbench3.py`](./eqbench3.py). It orchestrates:
- **Roleplay scenarios** (multi-turn, plus self-debrief).  
- **Rubric scoring** (judge LLM reads final transcripts and grades on several criteria).  
- **ELO analysis** (judge LLM does pairwise comparisons, then solves rating with TrueSkill).  

### Command-Line Arguments

| Argument                           | Description                                                                                                                             |
|------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| `--test-model` **(required)**      | API model identifier for the tested model (e.g. `openai/gpt-4.1-mini`).                                                                       |
| `--model-name`                     | Logical name used for storing ELO data (defaults to `--test-model` if not supplied). It must be unique to avoid collisions in ELO data. |
| `--judge-model`                    | Model to be used as the judge for ELO and/or rubric scoring. If `--no-elo` and `--no-rubric` are both set, you can skip this.           |
| `--runs-file`                      | Local runs data file (`.json` or `.json.gz`) storing scenario transcripts & statuses. Default: `eqbench3_runs.json`.                   |
| `--elo-results-file`               | Local ELO data file for storing pairwise comparisons & final ELO ratings. Default: `elo_results_eqbench3.json`.                        |
| `--leaderboard-runs-file`          | Path to the canonical leaderboard runs data (read-only). Default: `data/canonical_leaderboard_results.json.gz`.                         |
| `--leaderboard-elo-file`           | Path to the canonical leaderboard ELO data (read-only). Default: `data/canonical_leaderboard_elo_results.json.gz`.                     |
| `--run-id`                         | Custom prefix for the run key. If omitted, a random hex is used.                                                                        |
| `--threads`                        | Number of parallel threads (for scenario generation or scoring).                                                                        |
| `--verbosity`                      | Logging level: `DEBUG`, `INFO`, `WARNING`, etc. Default is `INFO`.                                                                      |
| `--save-interval`                  | Save progress every N tasks. Default is 2.                                                                                             |
| `--iterations`                     | Number of times to run each scenario. Default is 1.                                                                                     |
| `--no-elo`                         | If set, skip the ELO analysis step.                                                                                                      |
| `--no-rubric`                      | If set, skip the rubric scoring step.                                                                                                   |
| `--ignore-canonical`               | If set, do not load or use default canonical leaderboard files. Runs will be based on local files only.                                  |
| `--redo-rubric-judging`            | Reset tasks’ rubric status so that rubric scoring is done again.                                                                        |
| `--repair-rubric`                  | With `--run-id`, rerun only interrupted, failed, or score-missing rubric tasks. Cannot be combined with `--redo-rubric-judging`.        |
| `--repair-scenario`                | With `--run-id`, rerun only interrupted or failed scenario and debrief tasks. Combine with `--repair-rubric` to repair both phases.   |
| `--reset-model`                    | Wipes all local data (runs & ELO comparisons) pertaining to `--model-name` before starting.                                             |

Run `python eqbench3.py --help` for the full usage.

### Rubric vs ELO
- **Rubric**:  
  - Each scenario is scored by the judge on a set of criteria covering EQ abilities.  
  - Cheaper to run (about $1.50 in judging costs per iteration). Less variance in cost than Elo; no direct competition among models.
  - Good if you want an absolute measure for a single model, or when you only have one model to test.
  - Less discriminative compared to Elo, especially at the top ability range.

- **ELO** (TrueSkill)  
  - The judge compares transcripts from different models for each scenario, awarding a margin-based “win.”
  - Matchups are picked sparsely, efficiently narrowing in on the model's placement on the existing leaderboard.
  - Requires at least one baseline or previously completed model to compare against.  
  - Elo is more discriminative than the scoring rubric (by quite a lot), but also more expensive: around $10-20 in judging costs.
  - Can use either the canonical leaderboard (uses same results data as [eqbench.com](https://eqbench.com/)), or a user-specified local leaderboard.
  - Typically requires final results to be saved in `elo_results_eqbench3.json` (local) or the canonical `data/canonical_leaderboard_elo_results.json.gz`.

**You can run either or both**. If you do both, you end up with a final ELO rating as well as an average rubric score.

### Local vs Canonical Results Files
- **Local Files**:  
  - `eqbench3_runs.json`: Stores detailed transcripts, partial results, scenario statuses. This is the default local run results file, and can be set by user.
  - `elo_results_eqbench3.json`: Stores local pairwise comparisons plus your local ELO ratings for each model. This is the default local run results file, and can be set by user.
- **Canonical Files**:  
  - `data/canonical_leaderboard_results.json.gz`  
  - `data/canonical_leaderboard_elo_results.json.gz`  
  - These represent the official leaderboard data from models included on [eqbench.com](https://eqbench.com). By default, the canonical results are combined with your local results when running Elo. That means your evaluated model will be matched up against these leaderboard models, its placement on the leaderboard will be displayed at the end.

**Running Elo on local results only**
  - If you prefer not to run Elo against the leaderboard models, you can turn this off with `--ignore-canonical`, which means the evaluated model will only be matched against local results. Example flow:

**Specify your own canonical leaderboard**  
- You may wish to maintain your own local leaderboard, instead of using the canonical eqbench.com results.
**e.g. you may want to compare a fine-tuned model to a baseline.**
- you can point to a different `--leaderboard-runs-file` and `--leaderboard-elo-file`


## 5. Example Use Case: Compare a single model to a baseline with Elo

   ```bash
   python eqbench3.py \
     --test-model mymodel-baseline \
     --judge-model mistralai/mistral-medium-3 \
     --runs-file "exp01_results.json" \
     --elo-results-file "exp01_elo_results.json" \
     --threads 10 \
     --ignore-canonical
   ```

   ```bash
   python eqbench3.py \
     --test-model mymodel-finetune \
     --judge-model mistralai/mistral-medium-3 \
     --runs-file "exp01_results.json" \
     --elo-results-file "exp01_elo_results.json" \
     --threads 10 \
     --ignore-canonical
   ```

   **Example Output:**
   ```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        EQBench3 Rubric Score Summary                        │
╞─────────────────────────────────────────────────────────────────────────────╡
│ Rank │ Model Name                                         │ Rubric (0-100)  │
├──────┼────────────────────────────────────────────────────┼─────────────────┤
│  1   │ mistralai/mistral-medium-3                         │ 83.7            │
│ >2   │ mistralai/mistral-large-2411                       │ 76.8            │
└─────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────────────────┐
│                                EQBench3 Elo Results                                       │
╞───────────────────────────────────────────────────────────────────────────────────────────╡
│ Rank │ Model Name                          │ ELO Norm │ ELO Raw  │ 95% CI Norm  │ Sigma   │
├──────┼─────────────────────────────────────┼──────────┼──────────┼──────────────┼─────────┤
│  1   │ mistralai/mistral-medium-3          │ 1273     │ 1273     │ 1237 - 1308  │ 18.1    │
│ >2   │ mistralai/mistral-large-2411        │ 1127     │ 1127     │ 1092 - 1163  │ 18.1    │
└───────────────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                       │               EQBench3 Results Summary               │
╞──────────────────────────────────────────────────────────────────────────────╡
│ Run Key:              │ 1_mistralai_mistral-large-2411                       │
│ Model Name:           │ mistralai/mistral-large-2411                         │
│ API Model ID:         │ mistralai/mistral-large-2411                         │
│ Judge (Rubric/ELO):   │ mistralai/mistral-medium-3                           │
│ Status:               │ completed                                            │
├──────────────────────────────────────────────────────────────────────────────┤
│ Duration:             │ 00:02:59                                             │
├──────────────────────────────────────────────────────────────────────────────┤
│                       │                 ELO Analysis Results                 │
├──────────────────────────────────────────────────────────────────────────────┤
│ Rubric Score (0‑100): │ 76.75                                                │
│ ELO Raw:              │ 1127.25                                              │
│ ELO (Normalised):     │ 1127.25                                              │
└──────────────────────────────────────────────────────────────────────────────┘
   ```

  - With this usage, you can run pairwise matchups against just the models of interest, giving discriminative results even for small differences in performance.


---

## 6. Merging Local Results into the Canonical Leaderboard

Once you’re satisfied with your local run and want to merge your new model’s data into a canonical leaderboard:
1. **Review** your local results in `eqbench3_runs.json` and `elo_results_eqbench3.json`.
2. **Use** `merge_results_to_canonical.py` to push selected runs from local to canonical files:
   ```bash
   python merge_results_to_canonical.py \
     --local-runs eqbench3_runs.json \
     --local-elo elo_results_eqbench3.json \
     --canonical-runs data/canonical_leaderboard_results.json.gz \
     --canonical-elo data/canonical_leaderboard_elo_results.json.gz
   ```
   - This script interactively shows which runs can be merged and asks for confirmation.  
   - After merging, it recalculates the canonical ELO and updates canonical data transactionally.

3. **The newly integrated model** then appears in `data/canonical_leaderboard_elo_results.json.gz` with a final rating.

---

## 7. Folder and File Structure

- **`eqbench3.py`**: Main entry point for generating scenario outcomes, running rubric scoring, and orchestrating ELO.  
- **`merge_results_to_canonical.py`**: Script that merges local runs/ELO data with canonical results and re-runs an ELO solve.  
- **`data/`**: Contains canonical leaderboard data (`.json.gz`), scenario prompts, and supporting prompt templates.  
- **`utils/`**: Helper modules for file I/O, logging, API calls.  
- **`core/`**: Contains the main logic for running the roleplay scenarios, ELO, TrueSkill solver, pairwise judgement, and other components.  
- **`eqbench3_runs.json`** (local default): Your local scenario transcripts & statuses.  
- **`elo_results_eqbench3.json`** (local default): Your local ELO comparisons & ratings.  
- **`.env`**: Environment file specifying API credentials & certain timeouts.

---

## 8. Limitations and Notes

1. **Subjectivity**. Scores rely on a single LLM judge, which may exhibit biases or partialities.  
2. **Truncation**. Pairwise judgments deliberately truncate outputs to mitigate length bias.  
3. **Comparisons**. ELO is only as reliable as the coverage of pairwise matchups. With fewer references, the final rating is less stable.  
4. **Costs**. Each scenario run plus judge calls can be nontrivial in API usage. For large LLMs or many scenarios, expect higher costs.  
5. **Interpretation**. “EQ” is a broad concept. This benchmark does not aim for an absolute definition—only a measure of how well the model responds to intricate social/emotional scenarios under a consistent LLM judge.

---

## 9. Contact and License

- **License**: This project is open-sourced under an [MIT license](./LICENSE)
- **Contact**: For questions, refer to the [Issues](https://github.com/EQ-bench/eqbench3/issues) section of this repository.  

Contributions or suggestions are welcome—please open an issue or pull request.

## 10. Citation

If you use EQ-Bench 3 in academic work, please cite both the benchmark
repository and the original EQ-Bench paper.

**Citing EQ-Bench 3 (this repository):**
```bibtex
@misc{eqbench3_repo_2025,
  author       = {Samuel J. Paech},
  title        = {EQ-Bench 3: Emotional Intelligence Benchmark},
  year         = {2025},
  howpublished = {\url{https://github.com/EQ-bench/eqbench3}},
  note         = {Commit \<hash\> or release \<tag\>}
}
```

**Citing the legacy EQ-Bench paper:**
```bibtex
@misc{paech2023eqbench,
  title        = {EQ-Bench: An Emotional Intelligence Benchmark for Large Language Models},
  author       = {Samuel J. Paech},
  year         = {2023},
  eprint       = {2312.06281},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL}
}
```
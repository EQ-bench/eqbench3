import unittest

from core.conversation import ScenarioTask


def make_task(scenario_id: str = "001") -> ScenarioTask:
    return ScenarioTask(
        scenario_id=scenario_id,
        prompts=["Prompt one"],
        debrief_prompt=None if scenario_id == "401" else "Debrief prompt",
        iteration_index=1,
        test_model="test-model",
    )


class ScenarioTaskResumeRecoveryTests(unittest.TestCase):
    def test_rubric_error_retries_only_rubric_phase(self):
        task = make_task()
        task.status = "error"
        task.conversation_history = [{"role": "assistant", "content": "completed response"}]
        task.debrief_response = "completed debrief"
        task.rubric_run_error = "Judge API timed out"
        task.error = "Rubric Scoring Error: API call failed"

        self.assertTrue(task.recover_for_resume(repair_rubric=True))

        self.assertEqual(task.status, "completed")
        self.assertEqual(task.conversation_history[0]["content"], "completed response")
        self.assertEqual(task.debrief_response, "completed debrief")
        self.assertIsNone(task.rubric_run_error)
        self.assertIsNone(task.error)

    def test_debrief_error_reuses_completed_scenario(self):
        task = make_task()
        task.status = "error"
        task.conversation_history = [{"role": "assistant", "content": "completed response"}]
        task.debrief_run_error = "Test API timed out"

        self.assertTrue(task.recover_for_resume(repair_scenario=True))

        self.assertEqual(task.status, "scenario_completed")
        self.assertEqual(task.conversation_history[0]["content"], "completed response")
        self.assertIsNone(task.debrief_response)

    def test_scenario_error_preserves_completed_turns(self):
        task = make_task()
        task.status = "error"
        task.conversation_history = [
            {"role": "user", "content": "Prompt one"},
            {"role": "assistant", "content": "first completed turn"},
        ]
        task.parsed_responses = [{"raw": "first completed turn"}]
        task.scenario_run_error = "Test API timed out"

        self.assertTrue(task.recover_for_resume(repair_scenario=True))

        self.assertEqual(task.status, "initialized")
        self.assertEqual(task.conversation_history[1]["content"], "first completed turn")
        self.assertEqual(task.parsed_responses[0]["raw"], "first completed turn")

    def test_interrupted_rubric_returns_to_analysis_rubric_ready_state(self):
        task = make_task("401")
        task.status = "running_rubric_scoring"
        task.conversation_history = [{"role": "assistant", "content": "analysis response"}]

        self.assertTrue(task.recover_for_resume(repair_rubric=True))

        self.assertEqual(task.status, "scenario_completed")
        self.assertEqual(task.conversation_history[0]["content"], "analysis response")

    def test_interrupted_debrief_with_saved_response_is_completed(self):
        task = make_task()
        task.status = "running_debrief"
        task.conversation_history = [{"role": "assistant", "content": "completed response"}]
        task.debrief_response = "saved debrief"

        self.assertTrue(task.recover_for_resume(repair_scenario=True))

        self.assertEqual(task.status, "completed")
        self.assertEqual(task.debrief_response, "saved debrief")

    def test_repair_rubric_recovers_scored_task_without_numeric_scores(self):
        task = make_task()
        task.status = "rubric_scored"
        task.rubric_scores = {"judge_note": "No score returned"}
        task.raw_rubric_judge_text = '{"judge_note": "No score returned"}'

        self.assertTrue(task.recover_for_resume(repair_rubric=True))

        self.assertEqual(task.status, "completed")
        self.assertIsNone(task.rubric_scores)

    def test_repair_rubric_requires_an_allowed_numeric_score(self):
        task = make_task()
        task.status = "rubric_scored"
        task.rubric_scores = {"unexpected_numeric_field": 1.0}

        self.assertTrue(
            task.recover_for_resume(
                repair_rubric=True,
                valid_rubric_criteria={"overall_eq"},
            )
        )

        self.assertEqual(task.status, "completed")
        self.assertIsNone(task.rubric_scores)

    def test_repair_rubric_does_not_restart_scenario_errors(self):
        task = make_task()
        task.status = "error"
        task.scenario_run_error = "Test API timed out"

        self.assertFalse(task.recover_for_resume(repair_rubric=True))
        self.assertEqual(task.status, "error")

    def test_completed_rubric_is_not_changed(self):
        task = make_task()
        task.status = "rubric_scored"
        task.rubric_scores = {"overall_eq": 80.0}

        self.assertFalse(task.recover_for_resume(repair_rubric=True))

        self.assertEqual(task.status, "rubric_scored")
        self.assertEqual(task.rubric_scores, {"overall_eq": 80.0})


if __name__ == "__main__":
    unittest.main()

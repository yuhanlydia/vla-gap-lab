from vla_gap_lab.mu_vla_parity import evaluate_parity


def test_parity_requires_every_training_task_to_be_close_enough():
    references = {"A": 0.9, "B": 0.5}
    good = evaluate_parity({"A": 0.85, "B": 0.55}, references, tolerance_pp=15)
    bad = evaluate_parity({"A": 0.60, "B": 0.55}, references, tolerance_pp=15)
    assert good["passed"] is True
    assert bad["passed"] is False
    assert bad["tasks"]["A"]["absolute_error_pp"] == 30.0

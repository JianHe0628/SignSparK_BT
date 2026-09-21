"""Evaluation metrics, including the trimmed sacreBLEU."""
from signspark_bt import metrics


def test_identical_text_scores_perfectly():
    refs = ["the quick brown fox", "a second sentence here"]
    assert metrics.bleu(references=refs, hypotheses=refs)["bleu4"] > 99.9
    assert metrics.chrf(references=refs, hypotheses=refs) > 99.9
    assert metrics.rouge(references=refs, hypotheses=refs) > 99.9


def test_wer_counts_each_edit_type():
    assert metrics.wer_list(references=["a b c d"], hypotheses=["a b c d"])["wer"] == 0.0

    substitution = metrics.wer_list(references=["a b c d"], hypotheses=["a b x d"])
    assert substitution["sub_rate"] == 25.0 and substitution["wer"] == 25.0

    deletion = metrics.wer_list(references=["a b c d"], hypotheses=["a b c"])
    assert deletion["del_rate"] == 25.0

    insertion = metrics.wer_list(references=["a b c"], hypotheses=["a b c d"])
    assert insertion["ins_rate"] > 0.0


def test_bleu_reports_all_four_orders():
    scores = metrics.bleu(references=["a b c d e"], hypotheses=["a b c x e"])
    assert set(scores) == {"bleu1", "bleu2", "bleu3", "bleu4"}
    assert scores["bleu1"] > scores["bleu4"]


def test_unrelated_text_scores_near_zero():
    assert metrics.bleu(references=["the quick brown fox"],
                        hypotheses=["completely unrelated words"])["bleu4"] < 1.0

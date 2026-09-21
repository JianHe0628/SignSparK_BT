"""Config loading and environment variable expansion."""
import os

from signspark_bt.helpers import expand_env_vars, load_config


def test_expands_vars_anywhere_in_the_tree():
    os.environ["BT_TEST_ROOT"] = "/data/root"
    cfg = {
        "data": {"base_path": "${BT_TEST_ROOT}/lmdb"},
        "training": {"model_dir": "${BT_TEST_ROOT}/runs"},
        "list": ["${BT_TEST_ROOT}/a", "plain"],
    }
    out = expand_env_vars(cfg)
    assert out["data"]["base_path"] == "/data/root/lmdb"
    # the point of the change: expansion is not limited to data.base_path
    assert out["training"]["model_dir"] == "/data/root/runs"
    assert out["list"] == ["/data/root/a", "plain"]


def test_leaves_non_strings_alone():
    cfg = {"batch_size": 64, "lr": 1e-4, "flag": True, "none": None}
    assert expand_env_vars(cfg) == cfg


def test_undefined_variable_is_left_as_written():
    os.environ.pop("BT_DEFINITELY_UNSET", None)
    assert expand_env_vars("${BT_DEFINITELY_UNSET}/x") == "${BT_DEFINITELY_UNSET}/x"


def test_shipped_configs_load():
    for name in ("phoenix14t", "csl_daily", "how2sign"):
        cfg = load_config(f"configs/{name}.yaml")
        assert cfg["data"]["feature_size"] == 240
        assert cfg["model"]["encoder"]["type"] == "transformer"


def test_csl_config_disables_punctuation_normalization():
    """Chinese punctuation is full-width; jieba tokenizes it separately."""
    assert load_config("configs/csl_daily.yaml")["data"]["normalize_punctuation"] is False
    assert load_config("configs/csl_daily.yaml")["data"]["tokenizer"] == "jieba"
    # the other two keep the default
    for name in ("phoenix14t", "how2sign"):
        assert load_config(f"configs/{name}.yaml")["data"].get("normalize_punctuation", True)

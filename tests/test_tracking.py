import builtins

import pytest

from matchpredictor import tracking
from matchpredictor.evaluation import backtest as bt
from matchpredictor.evaluation import predictions_log as plog
from tests.synthetic import add_shots, synthetic_raw_two_divisions, write_as_csvs

mlflow = pytest.importorskip("mlflow")


def _run(tmp_path, run_id):
    mlflow.set_tracking_uri(f"file:{tmp_path / 'mlruns'}")
    return mlflow.get_run(run_id)


def test_log_run_stores_params_metrics_and_tags(tmp_path):
    uri = f"file:{tmp_path / 'mlruns'}"

    run_id = tracking.log_run(
        "prueba",
        params={"seasons": "a,b", "n": 3},
        metrics={"x.log_loss": 1.02, "nan": float("nan")},
        tags={"kind": "test"},
        tracking_uri=uri,
    )
    run = _run(tmp_path, run_id)

    assert run.data.params == {"seasons": "a,b", "n": "3"}
    assert run.data.metrics == {"x.log_loss": 1.02}  # el NaN se omite
    assert run.data.tags["kind"] == "test" and "git_commit" in run.data.tags
    assert run.info.run_name == "prueba"


def test_log_run_without_mlflow_explains_how_to_install(monkeypatch):
    real_import = builtins.__import__

    def fake(name, *args, **kwargs):
        if name == "mlflow":
            raise ImportError("no hay")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake)

    with pytest.raises(tracking.TrackingUnavailableError, match="requirements-tracking"):
        tracking.log_run("x", {}, {})


def test_backtest_track_logs_one_metric_per_model(tmp_path, monkeypatch):
    from matchpredictor.features import add_form_features, compute_elo

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{tmp_path / 'mlruns'}")
    games, _ = compute_elo(add_shots(synthetic_raw_two_divisions()))
    games = add_form_features(games)
    from matchpredictor.models import EloLogit, FrequencyBaseline

    seasons = bt.complete_seasons(games)[-2:]
    predictions = bt.run_backtest(
        games, {"frecuencias": FrequencyBaseline, "elo_logit": EloLogit}, seasons
    )

    run = _run(tmp_path, bt.track_backtest(predictions, seasons))

    assert {"frecuencias.log_loss", "elo_logit.log_loss", "market.log_loss"} <= set(
        run.data.metrics
    )
    assert run.data.params["seasons"] == ",".join(seasons)
    assert run.data.tags["kind"] == "backtest"
    expected = bt.score_table(predictions).set_index("model").loc["elo_logit", "log_loss"]
    assert run.data.metrics["elo_logit.log_loss"] == pytest.approx(expected)


def test_report_track_logs_the_gap_with_its_interval(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{tmp_path / 'mlruns'}")
    raw = add_shots(synthetic_raw_two_divisions())
    data = tmp_path / "raw"
    data.mkdir()
    write_as_csvs(raw, data)
    since = raw["date"].sort_values().iloc[len(raw) * 4 // 5].date().isoformat()
    base = ["--data", str(data), "--log", str(tmp_path / "log.csv")]
    plog.main(["replay", *base, "--since", since])
    capsys.readouterr()

    plog.main(["report", *base, "--track"])

    out = capsys.readouterr().out
    assert "Registrado en MLflow" in out
    client = mlflow.tracking.MlflowClient(f"file:{tmp_path / 'mlruns'}")
    experiment = client.get_experiment_by_name(tracking.DEFAULT_EXPERIMENT)
    run = client.search_runs([experiment.experiment_id])[0]
    metrics = run.data.metrics
    assert metrics["gap"] == pytest.approx(metrics["model.log_loss"] - metrics["market.log_loss"])
    assert metrics["gap_ci_low"] <= metrics["gap"] <= metrics["gap_ci_high"]
    assert run.data.tags["kind"] == "prediction-log"

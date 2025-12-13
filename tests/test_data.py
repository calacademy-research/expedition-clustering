import pandas as pd

from expedition_clustering.data import DatabaseConfig, fetch_table_by_ids


def test_database_config_connect_uses_pymysql(monkeypatch):
    captured_kwargs = {}

    def fake_connect(**kwargs):
        captured_kwargs.update(kwargs)
        return "fake-connection"

    monkeypatch.setattr("expedition_clustering.data.pymysql.connect", fake_connect)
    cfg = DatabaseConfig(host="h", user="u", password="p", database="d", port=1234, charset="utf8mb4")

    conn = cfg.connect()

    assert conn == "fake-connection"
    assert captured_kwargs["host"] == "h"
    assert captured_kwargs["user"] == "u"
    assert captured_kwargs["password"] == "p"
    assert captured_kwargs["database"] == "d"
    assert captured_kwargs["port"] == 1234
    assert captured_kwargs["charset"] == "utf8mb4"


def test_fetch_table_by_ids_returns_empty_when_no_valid_ids():
    cfg = DatabaseConfig()
    result = fetch_table_by_ids(cfg, "table", "id", ids=[None, float("nan")])
    assert isinstance(result, pd.DataFrame)
    assert result.empty

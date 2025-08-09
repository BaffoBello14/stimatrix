import types
from db.connect import DatabaseConnector, DatabaseConfig


def test_database_connector_engine_creation(monkeypatch):
    # Arrange a fake config
    cfg = DatabaseConfig(server="server", database="db", user="u", password="p")
    dc = DatabaseConnector(config=cfg)

    created = {}

    def fake_create_engine(self):
        created["called"] = True
        class FakeEngine:
            def connect(self):
                class Ctx:
                    def __enter__(self):
                        return self
                    def __exit__(self, exc_type, exc, tb):
                        return False
                    def execute(self, *_args, **_kwargs):
                        return None
                return Ctx()
        return FakeEngine()

    monkeypatch.setattr(DatabaseConnector, "_create_engine", fake_create_engine)

    # Act
    engine = dc.engine

    # Assert
    assert created.get("called") is True
    assert engine is not None
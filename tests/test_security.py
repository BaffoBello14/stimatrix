import pytest

from utils.security import InputValidator, SecurityError


class TestSanitizeColumnName:
    def test_allows_local_characters(self):
        assert InputValidator.sanitize_column_name("AI_Superficie_m²") == "AI_Superficie_m²"
        assert InputValidator.sanitize_column_name("Valore medio €") == "Valore medio €"

    def test_rejects_dangerous_sequences(self):
        with pytest.raises(SecurityError):
            InputValidator.sanitize_column_name("price; DROP TABLE users")
        with pytest.raises(SecurityError):
            InputValidator.sanitize_column_name("col--comment")

    def test_rejects_control_characters(self):
        with pytest.raises(SecurityError):
            InputValidator.sanitize_column_name("bad\nname")

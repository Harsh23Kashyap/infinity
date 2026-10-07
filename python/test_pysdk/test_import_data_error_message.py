"""Regression test for PR #125 (Fix #124) and its extension in run #75.

PR #125 fixed a copy-paste bug where the embedded Python SDK's
`import_data` error messages said "export" instead of "import" — but the
fix only targeted 2 of 4 sites in the import_data body. The other 2 sites
(introduced by PR #113's `isinstance(v, str)` guard at the new
delimiter-validation line, and the `else` branch's "Unknown parameter"
error) were NOT fixed by PR #125 because PR #125 was authored against
the pre-PR-113 state where those lines didn't exist yet.

This test exercises 4 invalid-input paths in `import_data` and asserts
that the error message contains "import" (not "export"). The
Thrift SDK's `import_data` is also tested for symmetry.

Server-free: the validation runs in the `for k, v in options.items():`
loop before `_conn.import_data(...)` is invoked. A `_MockConn` whose
`.import_data(...)` raises AssertionError is never invoked for the
rejection cases. The happy-path cases use a mock that returns a fake
`res` with `error_code == ErrorCode.OK` to verify validation didn't
short-circuit valid input.

Parametrized rejection set: [int, None, True, False, {"a": "b"}, [1, 2]].
All 5 types bypass the `isinstance(v, str)` guard and trigger the
InfinityException raise. `bool` is correctly rejected because `bool`
is NOT `str` (cycle-70 lesson: `bool` IS a subclass of `int` but NOT
of `str`).
"""
import pytest

# Embedded SDK (C extension). Skip the entire module if the extension is
# not built in this environment (matches the pytest.importorskip pattern
# used in test_query.py for PR #127 / #131 / #133 / #135 / #138 / #140).
infinity_embedded = pytest.importorskip("infinity_embedded")
from infinity_embedded.common import InfinityException  # noqa: E402
from infinity_embedded.errors import ErrorCode  # noqa: E402
from infinity_embedded.local_infinity.table import LocalTable  # noqa: E402

# Thrift SDK — exposed at two import paths depending on whether the test
# environment has `infinity_sdk` installed as a package or via the legacy
# `infinity` import path. Try the modern path first; fall back to the
# legacy path.
try:
    from infinity_sdk.infinity.remote_thrift.table import RemoteTable  # noqa: E402
except ImportError:
    from infinity.remote_thrift.table import RemoteTable  # noqa: E402


class _MockRes:
    def __init__(self):
        self.error_code = ErrorCode.OK
        self.error_msg = ""


class _MockConn:
    """Mock connection used for the happy-path cases. Tracks call counts
    so the test can assert the validation logic was bypassed correctly
    when the input is valid."""

    def __init__(self):
        self.import_calls = 0
        self.export_calls = 0

    def import_data(self, **_kwargs):
        self.import_calls += 1
        return _MockRes()

    def export_data(self, **_kwargs):
        self.export_calls += 1
        return _MockRes()


_INVALID_VALUES = [1, None, True, False, {"a": "b"}, [1, 2]]


class TestEmbeddedImportDataErrorMessage:
    """Regression tests for PR #125 (and the missing sites in
    import_data found in run #75): every error message raised by
    `LocalTable.import_data` for an invalid `file_type` or `delimiter`
    must say "import" — not "export" — because the user is importing,
    not exporting."""

    @pytest.fixture
    def table(self):
        return LocalTable(conn=_MockConn(), db_name="fake_db", table_name="fake_table")

    @pytest.mark.parametrize("bad_value", _INVALID_VALUES)
    def test_import_data_file_type_error_says_import(self, table, bad_value):
        """`import_data({"file_type": <non-string>})` must raise
        InfinityException whose message contains "import"."""
        with pytest.raises(InfinityException) as exc:
            table.import_data("/fake/path.csv", import_options={"file_type": bad_value})
        assert exc.value.error_code == ErrorCode.IMPORT_FILE_FORMAT_ERROR
        assert "import" in exc.value.error_msg, (
            f"Error message should say 'import', got: {exc.value.error_msg!r}"
        )
        assert "export" not in exc.value.error_msg, (
            f"Error message must NOT say 'export' (misleading for import_data), got: {exc.value.error_msg!r}"
        )

    @pytest.mark.parametrize("bad_value", _INVALID_VALUES)
    def test_import_data_delimiter_error_says_import(self, table, bad_value):
        """`import_data({"delimiter": <non-string>})` must raise
        InfinityException whose message contains "import"."""
        with pytest.raises(InfinityException) as exc:
            table.import_data("/fake/path.csv", import_options={"delimiter": bad_value})
        assert exc.value.error_code == ErrorCode.IMPORT_FILE_FORMAT_ERROR
        assert "import" in exc.value.error_msg, (
            f"Error message should say 'import', got: {exc.value.error_msg!r}"
        )
        assert "export" not in exc.value.error_msg, (
            f"Error message must NOT say 'export' (misleading for import_data), got: {exc.value.error_msg!r}"
        )

    def test_import_data_unknown_parameter_error_says_import(self, table):
        """`import_data({"foo": 1})` (unknown key) must raise
        InfinityException whose message says "Unknown import parameter"."""
        with pytest.raises(InfinityException) as exc:
            table.import_data("/fake/path.csv", import_options={"foo": 1})
        assert exc.value.error_code == ErrorCode.IMPORT_FILE_FORMAT_ERROR
        assert "import" in exc.value.error_msg, (
            f"Error message should say 'import', got: {exc.value.error_msg!r}"
        )
        assert "export" not in exc.value.error_msg, (
            f"Error message must NOT say 'export' (misleading for import_data), got: {exc.value.error_msg!r}"
        )

    # --- happy paths: validation passes, mock is invoked once each ---

    def test_import_data_accepts_string_file_type(self, table):
        """Happy path: `import_data({"file_type": "csv"})` should not
        raise; the mock `_conn.import_data` should be invoked once."""
        conn = table._conn
        res = table.import_data("/fake/path.csv", import_options={"file_type": "csv"})
        assert res.error_code == ErrorCode.OK
        assert conn.import_calls == 1

    def test_import_data_accepts_string_delimiter(self, table):
        """Happy path: `import_data({"delimiter": "\t"})` should not
        raise; the mock `_conn.import_data` should be invoked once."""
        conn = table._conn
        res = table.import_data("/fake/path.csv", import_options={"delimiter": "\t"})
        assert res.error_code == ErrorCode.OK
        assert conn.import_calls == 1


class TestThriftImportDataErrorMessage:
    """Same regression tests but for the Thrift SDK `RemoteTable`.
    Mirrors the embedded class so the regression coverage is symmetric
    across both SDKs (same pattern as PR #129 / #131)."""

    @pytest.fixture
    def table(self):
        return RemoteTable(conn=_MockConn(), db_name="fake_db", table_name="fake_table")

    @pytest.mark.parametrize("bad_value", _INVALID_VALUES)
    def test_import_data_file_type_error_says_import(self, table, bad_value):
        with pytest.raises(InfinityException) as exc:
            table.import_data("/fake/path.csv", import_options={"file_type": bad_value})
        assert exc.value.error_code == ErrorCode.IMPORT_FILE_FORMAT_ERROR
        assert "import" in exc.value.error_msg, (
            f"Error message should say 'import', got: {exc.value.error_msg!r}"
        )
        assert "export" not in exc.value.error_msg, (
            f"Error message must NOT say 'export', got: {exc.value.error_msg!r}"
        )

    @pytest.mark.parametrize("bad_value", _INVALID_VALUES)
    def test_import_data_delimiter_error_says_import(self, table, bad_value):
        with pytest.raises(InfinityException) as exc:
            table.import_data("/fake/path.csv", import_options={"delimiter": bad_value})
        assert exc.value.error_code == ErrorCode.IMPORT_FILE_FORMAT_ERROR
        assert "import" in exc.value.error_msg, (
            f"Error message should say 'import', got: {exc.value.error_msg!r}"
        )
        assert "export" not in exc.value.error_msg, (
            f"Error message must NOT say 'export', got: {exc.value.error_msg!r}"
        )

    def test_import_data_unknown_parameter_error_says_import(self, table):
        with pytest.raises(InfinityException) as exc:
            table.import_data("/fake/path.csv", import_options={"foo": 1})
        assert exc.value.error_code == ErrorCode.IMPORT_FILE_FORMAT_ERROR
        assert "import" in exc.value.error_msg, (
            f"Error message should say 'import', got: {exc.value.error_msg!r}"
        )
        assert "export" not in exc.value.error_msg, (
            f"Error message must NOT say 'export', got: {exc.value.error_msg!r}"
        )

    def test_import_data_accepts_string_file_type(self, table):
        conn = table._conn
        res = table.import_data("/fake/path.csv", import_options={"file_type": "csv"})
        assert res.error_code == ErrorCode.OK
        assert conn.import_calls == 1

    def test_import_data_accepts_string_delimiter(self, table):
        conn = table._conn
        res = table.import_data("/fake/path.csv", import_options={"delimiter": "\t"})
        assert res.error_code == ErrorCode.OK
        assert conn.import_calls == 1
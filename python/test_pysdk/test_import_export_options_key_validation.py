"""
Regression test for the import_data / export_data key-type validation.

Mirrors the cycle 78-80 pattern of server-free parametrized rejection tests:
construct Table / LocalTable with a Mock `_conn` that raises AssertionError
on any method call, then exercise `import_data` / `export_data` with
non-string dict keys and assert that the new isinstance guard fires BEFORE
the `_conn` call is reached.

This locks the fix at the cycle 81 site (https://github.com/Harsh23Kashyap/infinity/pull/151)
so that future regressions (e.g. a refactor that removes the guard) are caught
before they reach a real Infinity server.

The rejection set is the hashable subset from cycle 78 (parametrized rejection
across the isinstance(v, str) family):
    [None, True, False, 1, 1.5, (1, 2)]
The set is restricted to hashable types because dict keys must be hashable;
`{bad_key: "alpha"}` would raise TypeError at construction time, not at
the `.lower()` call site, masking the real bug.
"""

import os
import sys

import pytest

# Ensure the SDK package is importable when this test file is run in isolation
_SDK_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "python", "infinity_sdk")
)
if _SDK_ROOT not in sys.path:
    sys.path.insert(0, _SDK_ROOT)

_EMBEDDED_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "python", "infinity_embedded")
)
if _EMBEDDED_ROOT not in sys.path:
    sys.path.insert(0, _EMBEDDED_ROOT)


from infinity.common import InfinityException
from infinity.errors import ErrorCode

# Embedded SDK C extension is not buildable on this dev box (Apple Clang 17
# below the project's Clang 20+ requirement). Skip embedded tests rather than
# fail; the embedded path is covered in CI on Linux runners with the right
# toolchain.
EMBEDDED_AVAILABLE = False
try:
    import infinity_embedded.embedded_infinity_ext  # noqa: F401
    EMBEDDED_AVAILABLE = True
except ImportError:
    pass

skip_if_no_embedded = pytest.mark.skipif(
    not EMBEDDED_AVAILABLE,
    reason="infinity_embedded C extension not available (Clang 20+ required to build the engine); embedded tests skipped",
)


# Hashable non-string rejection set (cycle 78 lesson: dict keys must be
# hashable; unhashable types like dict / set / list raise TypeError at
# construction time, masking the real bug at the `.lower()` call site).
HASHABLE_NON_STRING_KEYS = [None, True, False, 1, 1.5, (1, 2)]


class _ConnRaisesIfCalled:
    """Mock `_conn` whose every method raises AssertionError.

    The validation in import_data / export_data must fire BEFORE
    `self._conn.import_data(...)` or `self._conn.export_data(...)` is
    called. If the guard is missing, the bad key would reach the
    `_conn` and trigger AssertionError instead of the expected
    InfinityException.
    """

    def __getattr__(self, name):
        def _raise(*args, **kwargs):
            raise AssertionError(
                f"_conn.{name} was called but should have been blocked by the key-type guard"
            )
        return _raise


def _make_thrift_table():
    """Construct a Thrift Table with a mock _conn, bypassing name validation."""
    from infinity.remote_thrift.table import RemoteTable
    return RemoteTable.__new__(RemoteTable)


def _make_embedded_table():
    """Construct an embedded LocalTable with a mock _conn."""
    from infinity_embedded.local_infinity.table import LocalTable
    return LocalTable.__new__(LocalTable)


# Thrift table tests
@pytest.mark.parametrize("bad_key", HASHABLE_NON_STRING_KEYS)
def test_thrift_import_data_rejects_non_string_key(bad_key):
    from infinity.remote_thrift.table import RemoteTable
    table = RemoteTable.__new__(RemoteTable)
    table._conn = _ConnRaisesIfCalled()
    table._db_name = "default"
    table._table_name = "t1"

    with pytest.raises(InfinityException) as exc_info:
        table.import_data("/tmp/whatever.csv", import_options={bad_key: "csv"})

    assert exc_info.value.error_code == ErrorCode.IMPORT_FILE_FORMAT_ERROR
    assert "import option key" in exc_info.value.error_msg
    assert repr(bad_key) in exc_info.value.error_msg


@pytest.mark.parametrize("bad_key", HASHABLE_NON_STRING_KEYS)
def test_thrift_export_data_rejects_non_string_key(bad_key):
    from infinity.remote_thrift.table import RemoteTable
    table = RemoteTable.__new__(RemoteTable)
    table._conn = _ConnRaisesIfCalled()
    table._db_name = "default"
    table._table_name = "t1"

    with pytest.raises(InfinityException) as exc_info:
        table.export_data("/tmp/whatever.csv", export_options={bad_key: "csv"})

    assert exc_info.value.error_code == ErrorCode.IMPORT_FILE_FORMAT_ERROR
    assert "export option key" in exc_info.value.error_msg
    assert repr(bad_key) in exc_info.value.error_msg


def test_thrift_import_data_accepts_string_key():
    """Happy path: string keys still work, the new guard is bypassed for valid input."""
    from infinity.remote_thrift.table import RemoteTable
    table = RemoteTable.__new__(RemoteTable)

    # Track whether _conn.import_data was called
    call_log = {"import_data": 0}

    class _SpyConn:
        def import_data(self, **kwargs):
            call_log["import_data"] += 1
            # Return a minimal sentinel — the validation guard runs BEFORE
            # the response is inspected, so any object with .error_code works.
            return type("R", (), {"error_code": ErrorCode.OK, "error_msg": ""})()

    table._conn = _SpyConn()
    table._db_name = "default"
    table._table_name = "t1"

    # Mixed valid string keys: file_type, delimiter, header
    table.import_data("/tmp/whatever.csv", import_options={
        "file_type": "csv",
        "delimiter": ",",
        "header": False,
    })
    assert call_log["import_data"] == 1


def test_thrift_export_data_accepts_string_key():
    """Happy path: string keys still work, the new guard is bypassed for valid input."""
    from infinity.remote_thrift.table import RemoteTable
    table = RemoteTable.__new__(RemoteTable)

    call_log = {"export_data": 0}

    class _SpyConn:
        def export_data(self, **kwargs):
            call_log["export_data"] += 1
            return type("R", (), {"error_code": ErrorCode.OK, "error_msg": ""})()

    table._conn = _SpyConn()
    table._db_name = "default"
    table._table_name = "t1"

    table.export_data("/tmp/whatever.csv", export_options={
        "file_type": "csv",
        "delimiter": ",",
        "header": False,
    })
    assert call_log["export_data"] == 1


# Embedded LocalTable tests (skipped on dev box without Clang 20+ C extension)
@skip_if_no_embedded
@pytest.mark.parametrize("bad_key", HASHABLE_NON_STRING_KEYS)
def test_embedded_import_data_rejects_non_string_key(bad_key):
    from infinity_embedded.local_infinity.table import LocalTable
    table = LocalTable.__new__(LocalTable)
    table._conn = _ConnRaisesIfCalled()
    table._db_name = "default"
    table._table_name = "t1"

    with pytest.raises(InfinityException) as exc_info:
        table.import_data("/tmp/whatever.csv", import_options={bad_key: "csv"})

    assert exc_info.value.error_code == ErrorCode.IMPORT_FILE_FORMAT_ERROR
    assert "import option key" in exc_info.value.error_msg
    assert repr(bad_key) in exc_info.value.error_msg


@skip_if_no_embedded
@pytest.mark.parametrize("bad_key", HASHABLE_NON_STRING_KEYS)
def test_embedded_export_data_rejects_non_string_key(bad_key):
    from infinity_embedded.local_infinity.table import LocalTable
    table = LocalTable.__new__(LocalTable)
    table._conn = _ConnRaisesIfCalled()
    table._db_name = "default"
    table._table_name = "t1"

    with pytest.raises(InfinityException) as exc_info:
        table.export_data("/tmp/whatever.csv", export_options={bad_key: "csv"})

    assert exc_info.value.error_code == ErrorCode.IMPORT_FILE_FORMAT_ERROR
    assert "export option key" in exc_info.value.error_msg
    assert repr(bad_key) in exc_info.value.error_msg


# Source-level guards: lock the 4 sites so a future refactor that removes
# the isinstance guard is caught by the test runner.
def test_source_guards_lock_4_sites():
    """Asserts that each of the 4 sites has an isinstance(k, str) guard
    immediately before the `key = k.lower()` call. Locks the fix in place.
    """
    import re

    def _count_guards(path):
        with open(path) as f:
            src = f.read()
        # Match the exact guard block (4 lines + the k.lower() line)
        pattern = re.compile(
            r"if not isinstance\(k, str\):\s*\n"
            r"\s*raise InfinityException\(ErrorCode\.IMPORT_FILE_FORMAT_ERROR,\s*\n"
            r'\s*f"Invalid (?:import|export) option key: \{k!r\} \(expected str, got \{type\(k\)\.__name__\}\)"\)\s*\n'
            r"\s*key = k\.lower\(\)",
            re.MULTILINE,
        )
        return len(pattern.findall(src))

    thrift_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "python", "infinity_sdk", "infinity", "remote_thrift", "table.py")
    )
    embedded_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "python", "infinity_embedded", "local_infinity", "table.py")
    )

    assert _count_guards(thrift_path) == 2, (
        f"Expected 2 key-type guards in {thrift_path} (import_data + export_data), "
        f"found {_count_guards(thrift_path)}"
    )
    assert _count_guards(embedded_path) == 2, (
        f"Expected 2 key-type guards in {embedded_path} (import_data + export_data), "
        f"found {_count_guards(embedded_path)}"
    )

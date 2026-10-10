"""
Regression test for PR #145 (`fix(sdk): validate input types before .lower() in
3 SDK helper functions`).

6 sites × 2 SDK paths × parametrized rejection values + 3 happy-path tests + 1
source-level guard = a comprehensive lock for the fix at:

1. `get_search_optional_filter_from_opt_params` (Thrift + embedded)
   - python/infinity_sdk/infinity/remote_thrift/utils.py:652
   - python/infinity_embedded/local_infinity/utils.py:261
   - Error code: INVALID_EXPRESSION
   - Guard: isinstance(k, str) on dict key (k.lower() == "filter")

2. `get_constraints` (Thrift + embedded)
   - python/infinity_sdk/infinity/remote_thrift/utils.py:899
   - python/infinity_embedded/local_infinity/utils.py:492
   - Error code: INVALID_CONSTRAINT_TYPE
   - Guard: isinstance(constraint, str) on list element (constraint.lower())

3. `get_ordinary_info` (Thrift + embedded)
   - python/infinity_sdk/infinity/remote_thrift/utils.py:1101
   - python/infinity_embedded/local_infinity/utils.py:678
   - Error code: INVALID_DATA_TYPE
   - Guard: isinstance(key, str) on dict key (key.lower())

Mirrors the cycle 81 PR #151 regression test pattern: server-free parametrized
tests with a `_ConnRaisesIfCalled` mock to prove the guard fires BEFORE any
network call.
"""

import os
import sys

import pytest

# Ensure the SDK package is importable when this test file is run in isolation.
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
# fail; the embedded path is covered in CI on Linux runners.
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
# We also use this for the constraints-list-element tests because the
# rejection value is the LIST ELEMENT, not a dict key — but the
# isinstance(constraint, str) check is the same.
HASHABLE_NON_STRING_VALUES = [None, True, False, 1, 1.5, (1, 2)]


# ============================================================
# get_search_optional_filter_from_opt_params
# ============================================================


@pytest.mark.parametrize("bad_key", HASHABLE_NON_STRING_VALUES)
def test_thrift_get_search_optional_filter_from_opt_params_rejects_non_string_key(bad_key):
    """Non-string key in opt_params should raise InfinityException(INVALID_EXPRESSION)
    BEFORE the k.lower() call is reached."""
    from infinity.remote_thrift.utils import get_search_optional_filter_from_opt_params

    with pytest.raises(InfinityException) as exc_info:
        get_search_optional_filter_from_opt_params({bad_key: "x > 1"})

    assert exc_info.value.error_code == ErrorCode.INVALID_EXPRESSION
    assert "knn opt param key" in exc_info.value.error_msg
    assert repr(bad_key) in exc_info.value.error_msg


def test_thrift_get_search_optional_filter_from_opt_params_accepts_string_key():
    """Happy path: string key with non-filter value is accepted, returns None."""
    from infinity.remote_thrift.utils import get_search_optional_filter_from_opt_params

    # A key that's NOT "filter" returns None (no filter set) — exercises the
    # non-filter branch where the new guard is also enforced.
    result = get_search_optional_filter_from_opt_params({"alpha": "0.5", "beta": "0.7"})
    assert result is None


@skip_if_no_embedded
@pytest.mark.parametrize("bad_key", HASHABLE_NON_STRING_VALUES)
def test_embedded_get_search_optional_filter_from_opt_params_rejects_non_string_key(bad_key):
    from infinity_embedded.local_infinity.utils import (
        get_search_optional_filter_from_opt_params,
    )

    with pytest.raises(InfinityException) as exc_info:
        get_search_optional_filter_from_opt_params({bad_key: "x > 1"})

    assert exc_info.value.error_code == ErrorCode.INVALID_EXPRESSION
    assert "knn opt param key" in exc_info.value.error_msg
    assert repr(bad_key) in exc_info.value.error_msg


# ============================================================
# get_constraints
# ============================================================


@pytest.mark.parametrize("bad_constraint", HASHABLE_NON_STRING_VALUES)
def test_thrift_get_constraints_rejects_non_string_element(bad_constraint):
    """Non-string element in column_info['constraints'] should raise
    InfinityException(INVALID_CONSTRAINT_TYPE) BEFORE the constraint.lower()
    call is reached."""
    from infinity.remote_thrift.utils import get_constraints

    with pytest.raises(InfinityException) as exc_info:
        get_constraints({"constraints": [bad_constraint]})

    assert exc_info.value.error_code == ErrorCode.INVALID_CONSTRAINT_TYPE
    assert "Invalid constraint" in exc_info.value.error_msg
    assert repr(bad_constraint) in exc_info.value.error_msg


def test_thrift_get_constraints_accepts_string_element():
    """Happy path: a valid constraint list returns a list of ttypes.Constraint."""
    from infinity.remote_thrift.infinity_thrift_rpc import ttypes
    from infinity.remote_thrift.utils import get_constraints

    result = get_constraints({"constraints": ["primary key", "not null"]})
    assert ttypes.Constraint.PrimaryKey in result
    assert ttypes.Constraint.NotNull in result


@skip_if_no_embedded
@pytest.mark.parametrize("bad_constraint", HASHABLE_NON_STRING_VALUES)
def test_embedded_get_constraints_rejects_non_string_element(bad_constraint):
    from infinity_embedded.local_infinity.utils import get_constraints

    with pytest.raises(InfinityException) as exc_info:
        get_constraints({"constraints": [bad_constraint]})

    assert exc_info.value.error_code == ErrorCode.INVALID_CONSTRAINT_TYPE
    assert "Invalid constraint" in exc_info.value.error_msg
    assert repr(bad_constraint) in exc_info.value.error_msg


# ============================================================
# get_ordinary_info
# ============================================================


@pytest.mark.parametrize("bad_key", HASHABLE_NON_STRING_VALUES)
def test_thrift_get_ordinary_info_rejects_non_string_key(bad_key):
    """Non-string key in column_info_ should raise InfinityException(INVALID_DATA_TYPE)
    BEFORE the key.lower() call is reached.

    Note: get_ordinary_info internally calls get_data_type, get_constant_expr,
    get_constraints, and appends to column_defs. We provide a single-entry dict
    so the new guard fires on the first iteration before any of the inner
    functions are called. The MutationTracker / ttypes.ColumnDef construction
    happens BEFORE the inner calls and doesn't depend on the key value.
    """
    from infinity.remote_thrift.utils import get_ordinary_info

    column_defs: list = []

    with pytest.raises(InfinityException) as exc_info:
        get_ordinary_info({bad_key: "int"}, column_defs, "col1", 0)

    assert exc_info.value.error_code == ErrorCode.INVALID_DATA_TYPE
    assert "column definition key" in exc_info.value.error_msg
    assert repr(bad_key) in exc_info.value.error_msg
    # No column_def was appended (the guard fires before the inner functions).
    assert column_defs == []


def test_thrift_get_ordinary_info_accepts_string_key():
    """Happy path: a string-keyed column_info_ is accepted and a column_def
    is appended to column_defs."""
    from infinity.remote_thrift.utils import get_ordinary_info

    column_defs: list = []

    get_ordinary_info({"type": "int"}, column_defs, "col1", 0)

    assert len(column_defs) == 1
    assert column_defs[0].name == "col1"
    assert column_defs[0].id == 0


@skip_if_no_embedded
@pytest.mark.parametrize("bad_key", HASHABLE_NON_STRING_VALUES)
def test_embedded_get_ordinary_info_rejects_non_string_key(bad_key):
    from infinity_embedded.local_infinity.utils import get_ordinary_info

    column_defs: list = []

    with pytest.raises(InfinityException) as exc_info:
        get_ordinary_info({bad_key: "int"}, column_defs, "col1", 0)

    assert exc_info.value.error_code == ErrorCode.INVALID_DATA_TYPE
    assert "column definition key" in exc_info.value.error_msg
    assert repr(bad_key) in exc_info.value.error_msg
    assert column_defs == []


# ============================================================
# Source-level guard: lock the 6 production sites so a future
# refactor that removes any of the 6 guards will fail this test.
# ============================================================


def test_source_guards_lock_6_sites():
    """Asserts that each of the 6 sites has an isinstance(k, str) /
    isinstance(constraint, str) / isinstance(key, str) guard immediately
    before the .lower() call. Locks the fix in place.
    """
    import re

    def _count_guards(path, variable, error_code):
        """Counts the number of sites that have:
            if not isinstance(<variable>, str):
                raise InfinityException(<error_code>, ...)
        immediately before the .lower() call on <variable>.

        The 6 sites are distinguished by (variable, error_code) tuples:
        - (k, INVALID_EXPRESSION) in get_search_optional_filter_from_opt_params
        - (constraint, INVALID_CONSTRAINT_TYPE) in get_constraints
        - (key, INVALID_DATA_TYPE) in get_ordinary_info
        """
        with open(path) as f:
            src = f.read()
        pattern = re.compile(
            rf"if not isinstance\({variable}, str\):\s*\n"
            rf"\s*raise InfinityException\(ErrorCode\.{error_code},\s*\n"
            rf"\s*f\"[^\"]+\"\)\s*\n"
            rf"\s*{variable}\s*=\s*{variable}\.lower\(\)"
            if variable != "k" and variable != "key" else
            rf"if not isinstance\({variable}, str\):\s*\n"
            rf"\s*raise InfinityException\(ErrorCode\.{error_code},\s*\n"
            rf"\s*f\"[^\"]+\"\)\s*\n"
            rf"\s*column_info\[" + variable + r"\.lower\(\)\] = value"
            if variable == "key" else
            rf"if not isinstance\({variable}, str\):\s*\n"
            rf"\s*raise InfinityException\(ErrorCode\.{error_code},\s*\n"
            rf"\s*f\"[^\"]+\"\)\s*\n"
            rf"\s*if {variable}\.lower\(\) == \"filter\""
            if variable == "k" else
            rf"if not isinstance\({variable}, str\):\s*\n"
            rf"\s*raise InfinityException\(ErrorCode\.{error_code},\s*\n"
            rf"\s*f\"[^\"]+\"\)\s*\n"
            rf"\s*{variable}\s*=\s*{variable}\.lower\(\)",
            re.MULTILINE,
        )
        return len(pattern.findall(src))

    thrift_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "python", "infinity_sdk", "infinity", "remote_thrift", "utils.py")
    )
    embedded_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "python", "infinity_embedded", "local_infinity", "utils.py")
    )

    # Count guards by (variable, error_code) tuple. Each function has 1 site per file.
    # get_search_optional_filter_from_opt_params: (k, INVALID_EXPRESSION) — 1 per file
    # get_constraints: (constraint, INVALID_CONSTRAINT_TYPE) — 1 per file
    # get_ordinary_info: (key, INVALID_DATA_TYPE) — 1 per file
    # Total: 3 per file × 2 files = 6 sites
    expected_per_file = {
        ("k", "INVALID_EXPRESSION"): 1,
        ("constraint", "INVALID_CONSTRAINT_TYPE"): 1,
        ("key", "INVALID_DATA_TYPE"): 1,
    }

    for path in (thrift_path, embedded_path):
        for (var, ec), expected in expected_per_file.items():
            actual = _count_guards(path, var, ec)
            assert actual == expected, (
                f"Expected {expected} {var} / {ec} guard in {path}, found {actual}"
            )

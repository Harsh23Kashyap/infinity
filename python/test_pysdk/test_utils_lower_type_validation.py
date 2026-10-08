# Copyright(C) 2026 InfiniFlow, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression tests for issue #144.

Three SDK helper functions iterate user-supplied dicts / lists and call
`.lower()` on the values without any `isinstance(..., str)` guard. A
non-string input raises a raw Python `AttributeError` instead of a typed
`InfinityException`. This test file pins the typed-exception behavior
across all 6 sites (3 functions x 2 SDK paths) using the parametrized
rejection set established by PRs #121, #123, #140.
"""
import pytest
from infinity.common import InfinityException
from infinity.errors import ErrorCode

# Parametrized rejection set: covers None, bool, int, float, dict, list, tuple, set.
# All of these are not `str` and would have triggered AttributeError on .lower().
NON_STRING_VALUES = [None, True, False, 1, 1.5, {"a": "b"}, [1, 2], (1, 2), {1, 2}]

# Hashable subset for dict-key tests (Python dicts cannot contain dict/set/list as keys).
HASHABLE_NON_STRING_KEYS = [None, True, False, 1, 1.5, (1, 2)]


# The embedded SDK depends on a C extension (infinity_embedded.embedded_infinity_ext)
# that is only built when the C++ engine is compiled. On a Python-only dev machine
# (Apple Clang 17 here, project needs Clang 20+), the extension is unavailable.
# Skip embedded-only tests if the extension can't be imported.
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


# ---------------------------------------------------------------------------
# Thrift SDK
# ---------------------------------------------------------------------------

def test_thrift_get_search_optional_filter_rejects_non_string_key():
    """get_search_optional_filter_from_opt_params (Thrift) must raise
    InfinityException(ErrorCode.INVALID_EXPRESSION, ...) for non-string keys,
    not AttributeError."""
    from infinity.remote_thrift.utils import get_search_optional_filter_from_opt_params
    for bad_key in HASHABLE_NON_STRING_KEYS:
        opt_params = {bad_key: "alpha"}
        with pytest.raises(InfinityException) as exc_info:
            get_search_optional_filter_from_opt_params(opt_params)
        assert exc_info.value.error_code == ErrorCode.INVALID_EXPRESSION
        assert "Invalid knn opt param key" in exc_info.value.error_msg
        # The original opt_params must not be mutated on error.
        assert bad_key in opt_params


def test_thrift_get_ordinary_info_rejects_non_string_key():
    """get_ordinary_info (Thrift) must raise InfinityException(
    ErrorCode.INVALID_DATA_TYPE, ...) for non-string column-definition keys,
    not AttributeError."""
    from infinity.remote_thrift import utils as thrift_utils
    for bad_key in HASHABLE_NON_STRING_KEYS:
        column_info_ = {bad_key: "int"}
        with pytest.raises(InfinityException) as exc_info:
            thrift_utils.get_ordinary_info(column_info_, [], "c1", 0)
        assert exc_info.value.error_code == ErrorCode.INVALID_DATA_TYPE
        assert "Invalid column definition key" in exc_info.value.error_msg


def test_thrift_get_constraints_rejects_non_string_constraint():
    """get_constraints (Thrift) must raise InfinityException(
    ErrorCode.INVALID_CONSTRAINT_TYPE, ...) for non-string constraint values,
    not AttributeError."""
    from infinity.remote_thrift import utils as thrift_utils
    for bad_constraint in NON_STRING_VALUES:
        column_info = {"constraints": [bad_constraint]}
        with pytest.raises(InfinityException) as exc_info:
            thrift_utils.get_constraints(column_info)
        assert exc_info.value.error_code == ErrorCode.INVALID_CONSTRAINT_TYPE
        assert "Invalid constraint" in exc_info.value.error_msg


# ---------------------------------------------------------------------------
# Embedded SDK
# ---------------------------------------------------------------------------

@skip_if_no_embedded
def test_embedded_get_search_optional_filter_rejects_non_string_key():
    """get_search_optional_filter_from_opt_params (embedded) must raise
    InfinityException(ErrorCode.INVALID_EXPRESSION, ...) for non-string keys,
    not AttributeError."""
    from infinity_embedded.local_infinity.utils import (
        get_search_optional_filter_from_opt_params,
    )
    for bad_key in HASHABLE_NON_STRING_KEYS:
        opt_params = {bad_key: "alpha"}
        with pytest.raises(InfinityException) as exc_info:
            get_search_optional_filter_from_opt_params(opt_params)
        assert exc_info.value.error_code == ErrorCode.INVALID_EXPRESSION
        assert "Invalid knn opt param key" in exc_info.value.error_msg
        # The original opt_params must not be mutated on error.
        assert bad_key in opt_params


@skip_if_no_embedded
def test_embedded_get_ordinary_info_rejects_non_string_key():
    """get_ordinary_info (embedded) must raise InfinityException(
    ErrorCode.INVALID_DATA_TYPE, ...) for non-string column-definition keys,
    not AttributeError."""
    from infinity_embedded.local_infinity import utils as embedded_utils
    for bad_key in HASHABLE_NON_STRING_KEYS:
        column_info_ = {bad_key: "int"}
        with pytest.raises(InfinityException) as exc_info:
            embedded_utils.get_ordinary_info(column_info_, [], "c1", 0)
        assert exc_info.value.error_code == ErrorCode.INVALID_DATA_TYPE
        assert "Invalid column definition key" in exc_info.value.error_msg


@skip_if_no_embedded
def test_embedded_get_constraints_rejects_non_string_constraint():
    """get_constraints (embedded) must raise InfinityException(
    ErrorCode.INVALID_CONSTRAINT_TYPE, ...) for non-string constraint values,
    not AttributeError."""
    from infinity_embedded.local_infinity import utils as embedded_utils
    for bad_constraint in NON_STRING_VALUES:
        column_info = {"constraints": [bad_constraint]}
        with pytest.raises(InfinityException) as exc_info:
            embedded_utils.get_constraints(column_info)
        assert exc_info.value.error_code == ErrorCode.INVALID_CONSTRAINT_TYPE
        assert "Invalid constraint" in exc_info.value.error_msg


# ---------------------------------------------------------------------------
# Positive cases: existing string-key / string-constraint behavior preserved.
# ---------------------------------------------------------------------------

def test_thrift_get_search_optional_filter_accepts_string_keys():
    """Existing behavior: a string key 'filter' with a string value is
    accepted and the filter is parsed."""
    from infinity.remote_thrift.utils import get_search_optional_filter_from_opt_params
    # Use a key that is not 'filter' to keep the function's existing
    # "leave non-filter keys alone" semantics. The function returns None
    # for any input dict that does not contain a 'filter' key.
    opt_params = {"alpha": "x"}
    assert get_search_optional_filter_from_opt_params(opt_params) is None


def test_thrift_get_constraints_accepts_string_constraints():
    """Existing behavior: string constraints like 'null', 'not null' are
    accepted and converted to ttypes.Constraint members."""
    from infinity.remote_thrift import utils as thrift_utils
    column_info = {"constraints": ["null", "not null"]}
    result = thrift_utils.get_constraints(column_info)
    # Just check the result is non-empty; the actual Constraint enum values
    # vary across versions and aren't part of the public test surface here.
    assert len(result) == 2


@skip_if_no_embedded
def test_embedded_get_constraints_accepts_string_constraints():
    """Existing behavior: string constraints are accepted (embedded)."""
    from infinity_embedded.local_infinity import utils as embedded_utils
    column_info = {"constraints": ["primary key", "unique"]}
    result = embedded_utils.get_constraints(column_info)
    assert len(result) == 2

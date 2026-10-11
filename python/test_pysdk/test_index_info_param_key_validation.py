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

"""Regression tests for issue #156.

The `IndexInfo` class in all three SDK paths (Thrift, embedded, HTTP) iterates
the user-supplied `params` dict to construct `InitParameter(param_name,
param_value)`. Each path validates the VALUE side with `isinstance(value, str)`
but the KEY side was unvalidated. A non-string key (int, None, tuple, dict)
silently coerced: Thrift would stringify at serialization, pybind11 would
convert, JSON would stringify — masking the user's intent and producing wrong
query results.

This test file pins the typed-exception behavior at all 3 sites using the
parametrized rejection set established by PRs #121, #123, #140, #145, #147,
#149, #151 (cycles 71-83).
"""
import os
import sys

import pytest

# Ensure the SDK packages are importable when this test is run in isolation.
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

# Hashable non-string key set (cycle 78 lesson: dict keys must be hashable;
# unhashable types like dict/set/list raise TypeError at dict-construction
# time, masking the real bug at the `.lower()`-equivalent call site).
HASHABLE_NON_STRING_KEYS = [None, True, False, 1, 1.5, (1, 2)]


# ---------------------------------------------------------------------------
# Thrift SDK: IndexInfo.to_ttype() — python/infinity_sdk/infinity/index.py:104
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_key", HASHABLE_NON_STRING_KEYS)
def test_thrift_index_info_to_ttype_rejects_non_string_key(bad_key):
    """IndexInfo.to_ttype() must raise InfinityException(
    ErrorCode.INVALID_INDEX_PARAM, ...) for non-string params dict keys,
    not silently coerce via ttypes.InitParameter (which stores `param_name`
    verbatim for later Thrift-side serialization that would stringify it
    invisibly).

    Use IndexType.IVF to avoid the SecondaryFunctional branch which calls
    parse_expr on target_name.
    """
    from infinity.index import IndexInfo, IndexType

    info = IndexInfo("col", IndexType.IVF, {bad_key: "alpha"})

    with pytest.raises(InfinityException) as exc_info:
        info.to_ttype()

    assert exc_info.value.error_code == ErrorCode.INVALID_INDEX_PARAM
    assert "Invalid index param key" in exc_info.value.error_msg
    assert repr(bad_key) in exc_info.value.error_msg
    assert f"got {type(bad_key).__name__}" in exc_info.value.error_msg


def test_thrift_index_info_to_ttype_accepts_string_key_and_value():
    """Happy path: a well-formed IndexInfo with string keys and values
    constructs ttypes.InitParameter entries and returns a ttypes.IndexInfo."""
    from infinity.index import IndexInfo, IndexType

    info = IndexInfo("col", IndexType.IVF, {"alpha": "0.5", "centroid_count": "100"})
    result = info.to_ttype()

    assert result.column_name == "col"
    assert len(result.index_param_list) == 2
    names = {p.param_name for p in result.index_param_list}
    values = {p.param_value for p in result.index_param_list}
    assert names == {"alpha", "centroid_count"}
    assert values == {"0.5", "100"}


def test_thrift_index_info_to_ttype_value_validation_still_active():
    """The pre-existing isinstance(value, str) check must still fire for a
    string key but non-string value. Locks in the dual guard pattern."""
    from infinity.index import IndexInfo, IndexType

    info = IndexInfo("col", IndexType.IVF, {"alpha": 1})  # int value, not int key

    with pytest.raises(InfinityException) as exc_info:
        info.to_ttype()

    assert exc_info.value.error_code == ErrorCode.INVALID_INDEX_PARAM
    assert "should be string type" in exc_info.value.error_msg


# ---------------------------------------------------------------------------
# Embedded SDK: IndexInfo.to_local_type()
# --- python/infinity_embedded/index.py:104
# ---------------------------------------------------------------------------


@skip_if_no_embedded
@pytest.mark.parametrize("bad_key", HASHABLE_NON_STRING_KEYS)
def test_embedded_index_info_to_local_type_rejects_non_string_key(bad_key):
    from infinity_embedded.index import IndexInfo, IndexType

    info = IndexInfo("col", IndexType.IVF, {bad_key: "alpha"})

    with pytest.raises(InfinityException) as exc_info:
        info.to_local_type()

    assert exc_info.value.error_code == ErrorCode.INVALID_INDEX_PARAM
    assert "Invalid index param key" in exc_info.value.error_msg
    assert repr(bad_key) in exc_info.value.error_msg
    assert f"got {type(bad_key).__name__}" in exc_info.value.error_msg


@skip_if_no_embedded
def test_embedded_index_info_to_local_type_accepts_string_key_and_value():
    from infinity_embedded.index import IndexInfo, IndexType

    info = IndexInfo("col", IndexType.IVF, {"alpha": "0.5", "centroid_count": "100"})
    result = info.to_local_type()

    assert result.column_name == "col"
    assert len(result.index_param_list) == 2
    names = {p.param_name for p in result.index_param_list}
    values = {p.param_value for p in result.index_param_list}
    assert names == {"alpha", "centroid_count"}
    assert values == {"0.5", "100"}


# ---------------------------------------------------------------------------
# HTTP SDK: database_http.create_index() — python/infinity_sdk/infinity/infinity_http.py:689
# ---------------------------------------------------------------------------


def _make_table_http():
    """Create a `table_http` with `net=None`. The `create_index` method
    iterates `index_info.params.items()` and runs the new key guard BEFORE
    any `self.net.*` call (network round-trip), so a non-string key raises
    InfinityException before the HTTP call would be made.
    """
    from infinity.infinity_http import table_http
    return table_http(None, "default_db", "t")


@pytest.mark.parametrize("bad_key", HASHABLE_NON_STRING_KEYS)
def test_http_create_index_rejects_non_string_param_key(bad_key):
    """table_http.create_index() must raise InfinityException(
    ErrorCode.INVALID_INDEX_PARAM, ...) for non-string params dict keys,
    not silently stringify via json.dumps (which would produce {"1": "alpha"}
    and send the wrong key to the server)."""
    from infinity.index import IndexInfo, IndexType

    table_http_inst = _make_table_http()
    info = IndexInfo("col", IndexType.IVF, {bad_key: "alpha"})

    with pytest.raises(InfinityException) as exc_info:
        table_http_inst.create_index("idx", info)

    assert exc_info.value.error_code == ErrorCode.INVALID_INDEX_PARAM
    assert "Invalid index param key" in exc_info.value.error_msg
    assert repr(bad_key) in exc_info.value.error_msg
    assert f"got {type(bad_key).__name__}" in exc_info.value.error_msg


def test_http_create_index_value_validation_still_active():
    """The pre-existing isinstance(value, str) check must still fire for a
    string key but non-string value at the same site."""
    from infinity.index import IndexInfo, IndexType

    table_http_inst = _make_table_http()
    info = IndexInfo("col", IndexType.IVF, {"alpha": 1})  # int value, not int key

    with pytest.raises(InfinityException) as exc_info:
        table_http_inst.create_index("idx", info)

    assert exc_info.value.error_code == ErrorCode.INVALID_INDEX_PARAM
    assert "parameter value" in exc_info.value.error_msg


# ---------------------------------------------------------------------------
# Source-level guard: lock the 3 production sites so a future refactor that
# removes any of the 3 guards will fail this test.
# ---------------------------------------------------------------------------


def test_source_guards_lock_3_sites():
    """Asserts that each of the 3 sites has an isinstance(key, str) guard
    immediately before the isinstance(value, str) check / InitParameter
    construction. Locks the fix in place.

    The 3 sites are distinguished by (file, error_code_suffix_in_msg) tuples
    because the parameter variable is consistently `key` in all 3 cases:

    - python/infinity_sdk/infinity/index.py (Thrift IndexInfo.to_ttype)
    - python/infinity_embedded/index.py (embedded IndexInfo.to_local_type)
    - python/infinity_sdk/infinity/infinity_http.py (HTTP create_index)
    """
    import re

    # Match the guard pattern (single-line raise on the line after the if):
    #   if not isinstance(key, str):
    #       raise InfinityException(ErrorCode.INVALID_INDEX_PARAM, f"Invalid index param key: ..." )
    pattern = re.compile(
        r"if not isinstance\(key, str\):\s*\n"
        r'\s*raise InfinityException\(ErrorCode\.INVALID_INDEX_PARAM,\s*f"Invalid index param key: ',
        re.MULTILINE,
    )

    expected_paths = [
        os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..", "..", "python", "infinity_sdk", "infinity", "index.py",
            )
        ),
        os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..", "..", "python", "infinity_embedded", "index.py",
            )
        ),
        os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..", "..", "python", "infinity_sdk", "infinity", "infinity_http.py",
            )
        ),
    ]

    for path in expected_paths:
        with open(path) as f:
            src = f.read()
        count = len(pattern.findall(src))
        assert count == 1, (
            f"Expected 1 'isinstance(key, str)' guard in {path}, "
            f"found {count}. The cycle 84 fix added 1 such guard per file. "
            f"A future refactor that removes the guard would silently allow "
            f"non-string keys through to ttypes.InitParameter / "
            f"LocalInitParameter / json.dumps, masking user intent."
        )

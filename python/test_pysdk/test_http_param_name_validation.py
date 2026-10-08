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

"""Regression tests for issue #146.

The HTTP SDK's `create_table` (in `database_http`) and `add_columns`
(in `table_http`) iterate user-supplied `columns_definition` dicts and
call `str.lower()` on the parameter name without any
`isinstance(..., str)` guard. A non-string `param_name` raised a raw
Python `AttributeError` instead of a typed `InfinityException`. This
test file pins the typed-exception behavior at both sites using the
parametrized rejection set established by PRs #121, #123, #140, #145.
"""
import pytest
from infinity.common import InfinityException
from infinity.errors import ErrorCode

# Hashable subset for dict-key tests (Python dicts cannot contain dict/set/list as keys).
HASHABLE_NON_STRING_KEYS = [None, True, False, 1, 1.5, (1, 2)]


# ---------------------------------------------------------------------------
# HTTP SDK: create_table (in database_http) and add_columns (in table_http)
# ---------------------------------------------------------------------------
# Both methods iterate `columns_definition[col]` and call
# `param_name.lower()` before any network call. The validation is the
# first thing the loop does, so a non-string `param_name` raises
# InfinityException before any `self.net.*` call. The tests use
# `net=None` to confirm this — no live server needed.

def _make_database_http():
    """Create a `database_http` with `net=None`. Iteration logic runs
    before any `self.net.*` call, so a non-string `param_name` raises
    before `net` is touched."""
    from infinity.infinity_http import database_http
    return database_http(None, "default_db")


def _make_table_http():
    """Create a `table_http` with `net=None`. Same as above."""
    from infinity.infinity_http import table_http
    return table_http(None, "default_db", "t")


def test_http_create_table_rejects_non_string_param_name():
    """database_http.create_table must raise InfinityException(
    ErrorCode.INVALID_DATA_TYPE, ...) for non-string column-parameter keys,
    not AttributeError."""
    db_http = _make_database_http()
    for bad_key in HASHABLE_NON_STRING_KEYS:
        columns_definition = {"c1": {bad_key: "int"}}
        with pytest.raises(InfinityException) as exc_info:
            db_http.create_table("t", columns_definition=columns_definition)
        assert exc_info.value.error_code == ErrorCode.INVALID_DATA_TYPE
        assert "Invalid column parameter key" in exc_info.value.error_msg


def test_http_add_columns_rejects_non_string_param_name():
    """table_http.add_columns must raise InfinityException(
    ErrorCode.INVALID_DATA_TYPE, ...) for non-string column-parameter keys,
    not AttributeError."""
    table_http_inst = _make_table_http()
    for bad_key in HASHABLE_NON_STRING_KEYS:
        columns_definition = {"c1": {bad_key: "int"}}
        with pytest.raises(InfinityException) as exc_info:
            table_http_inst.add_columns(columns_definition)
        assert exc_info.value.error_code == ErrorCode.INVALID_DATA_TYPE
        assert "Invalid column parameter key" in exc_info.value.error_msg


def test_http_create_table_error_message_mentions_actual_type():
    """The InfinityException error message must mention the actual
    bad-key type (e.g. "got int"), so a user can debug their input."""
    db_http = _make_database_http()
    for bad_key, expected_type_name in [(None, "NoneType"), (1, "int"), (1.5, "float"),
                                          (True, "bool"), (False, "bool"), ((1, 2), "tuple")]:
        columns_definition = {"c1": {bad_key: "int"}}
        with pytest.raises(InfinityException) as exc_info:
            db_http.create_table("t", columns_definition=columns_definition)
        assert f"got {expected_type_name}" in exc_info.value.error_msg

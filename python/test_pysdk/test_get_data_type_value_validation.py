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

"""Regression tests for issue #148.

`get_data_type` in both Thrift and embedded SDK paths calls
`column_info["type"].lower()` without any `isinstance(..., str)` guard.
A non-string "type" value raises a raw Python `AttributeError` instead
of a typed `InfinityException`. This test file pins the typed-exception
behavior at both sites using the parametrized rejection set established
by PRs #121, #123, #140, #145, #147.
"""
import pytest
from infinity.common import InfinityException
from infinity.errors import ErrorCode

# Parametrized rejection set: covers None, bool, int, float, dict, list, tuple, set.
NON_STRING_VALUES = [None, True, False, 1, 1.5, {"a": "b"}, [1, 2], (1, 2), {1, 2}]


# The embedded SDK depends on a C extension (infinity_embedded.embedded_infinity_ext)
# that is only built when the C++ engine is compiled. On a Python-only dev machine
# (Apple Clang 17 here, project needs Clang 20+), the extension is unavailable.
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

def test_thrift_get_data_type_rejects_non_string_type_value():
    """get_data_type (Thrift) must raise InfinityException(
    ErrorCode.INVALID_DATA_TYPE, ...) for non-string "type" values, not AttributeError."""
    from infinity.remote_thrift.utils import get_data_type
    for bad_type in NON_STRING_VALUES:
        column_info = {"type": bad_type}
        with pytest.raises(InfinityException) as exc_info:
            get_data_type(column_info)
        assert exc_info.value.error_code == ErrorCode.INVALID_DATA_TYPE
        assert "Invalid column type" in exc_info.value.error_msg


def test_thrift_get_data_type_accepts_string_type_value():
    """Existing positive case: a string "type" like 'int' or 'float' is
    accepted. We assert that the result is a Thrift DataType object (i.e. the
    function ran to completion without raising)."""
    from infinity.remote_thrift.utils import get_data_type
    # Use simple, unparameterized type names — get_data_type treats commas
    # as a separator and would split "varchar(64)" into ["varchar(64)"]
    # which then fails the type-name lookup. We only need to verify the
    # `isinstance` guard is bypassed for valid strings.
    for good_type in ["int", "float", "bool", "varchar"]:
        column_info = {"type": good_type}
        result = get_data_type(column_info)
        # The result is a ttypes.DataType; we just check it ran without raising.
        assert result is not None


# ---------------------------------------------------------------------------
# Embedded SDK
# ---------------------------------------------------------------------------

@skip_if_no_embedded
def test_embedded_get_data_type_rejects_non_string_type_value():
    """get_data_type (embedded) must raise InfinityException(
    ErrorCode.INVALID_DATA_TYPE, ...) for non-string "type" values, not AttributeError."""
    from infinity_embedded.local_infinity.utils import get_data_type
    for bad_type in NON_STRING_VALUES:
        column_info = {"type": bad_type}
        with pytest.raises(InfinityException) as exc_info:
            get_data_type(column_info)
        assert exc_info.value.error_code == ErrorCode.INVALID_DATA_TYPE
        assert "Invalid column type" in exc_info.value.error_msg


@skip_if_no_embedded
def test_embedded_get_data_type_accepts_string_type_value():
    """Existing positive case: a string "type" is accepted (embedded)."""
    from infinity_embedded.local_infinity.utils import get_data_type
    for good_type in ["int", "float", "bool", "varchar"]:
        column_info = {"type": good_type}
        result = get_data_type(column_info)
        assert result is not None

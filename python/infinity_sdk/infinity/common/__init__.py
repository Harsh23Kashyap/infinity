# Copyright(C) 2023 InfiniFlow, Inc. All rights reserved.
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

"""Public common helpers for the infinity SDK.

This package was previously a single file (`infinity/common.py`). It is now
a package so we can add focused submodules (e.g. `bf16`, `thrift_codec`)
without breaking the public `from infinity.common import X` surface that
the rest of the SDK and downstream code already uses.
"""

from infinity.common.legacy import (  # noqa: F401  -- re-exports
    NetworkAddress,
    SparseVector,
    Array,
    FDE,
    URI,
    VEC,
    INSERT_DATA,
    LOCAL_HOST,
    LOCAL_INFINITY_PATH,
    ConflictType,
    SortType,
    InfinityException,
    UnsupportedColumnTypeError,
    DEFAULT_MATCH_VECTOR_TOPN,
    DEFAULT_MATCH_SPARSE_TOPN,
)

# Expose the new bf16 helper at the package level so callers can use
# ``infinity.common.bf16_bytes_to_float32_list`` as a shortcut.  The full
# module is still importable via ``infinity.common.bf16``.
from infinity.common import bf16 as _bf16  # noqa: F401  -- also sets pkg.bf16
bf16_bytes_to_float32_list = _bf16.bf16_bytes_to_float32_list

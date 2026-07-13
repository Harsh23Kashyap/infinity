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

"""BFloat16 byte-level helpers shared by the SDK and embedded runtimes.

The two Python packages (``infinity_sdk`` and ``infinity_embedded``) each shipped
an identical ``bf16_bytes_to_float32_list`` helper. This module is the single
source of truth. Both packages re-export ``bf16_bytes_to_float32_list`` from
``infinity.common.bf16`` so callers keep working with the existing import
path (``from infinity.remote_thrift.types import bf16_bytes_to_float32_list``
or ``from infinity_embedded.local_infinity.types import bf16_bytes_to_float32_list``).
"""

from typing import List

import numpy as np


def bf16_bytes_to_float32_list(column_vector: bytes) -> List[float]:
    """Decode a column of little-endian BFloat16 values to a list of Python floats."""
    if not column_vector:
        return []
    tmp_u16 = np.frombuffer(column_vector, dtype="<i2")
    result_arr = np.zeros(2 * len(tmp_u16), dtype="<i2")
    result_arr[1::2] = tmp_u16
    return list(result_arr.view("<f4"))

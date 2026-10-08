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

"""Unit tests for the shared Thrift type-decoder helpers.

The codec helpers used to live in two near-identical copies (one in the
``infinity_sdk`` package, one in ``infinity_embedded``).  This test
exercises the shared implementations through a path that does not require
the full ``infinity`` packages to be importable — we load the new
``infinity.common.bf16`` and ``infinity_embedded.bf16`` modules directly
from source so the tests run even when the Thrift C-extension dependency
is not installed.
"""

import importlib.util
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module_from(name: str, file_path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, file_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sdk_common_pkg() -> types.ModuleType:
    """Load the SDK ``infinity.common`` package as a stand-alone module.

    The package layout is::

        python/infinity_sdk/infinity/common/__init__.py   (re-exports legacy)
        python/infinity_sdk/infinity/common/legacy.py     (the old common.py)
        python/infinity_sdk/infinity/common/bf16.py       (new shared helper)

    Loading them with importlib avoids executing ``infinity/__init__.py``
    which depends on the Thrift C-extension.
    """
    common_pkg_root = REPO_ROOT / "python/infinity_sdk/infinity/common"
    # We treat the package as a real package so the relative ``from
    # infinity.common.legacy import ...`` inside ``__init__.py`` resolves
    # correctly.  The fake parent package's ``__path__`` only points at the
    # ``common/`` subdirectory, and we never go up to ``infinity/__init__``.
    pkg_name = "infinity.common"
    parent_name = "infinity"
    # Pre-create the parent and the package, then load submodules.
    if parent_name not in sys.modules:
        parent_pkg = types.ModuleType(parent_name)
        parent_pkg.__path__ = []  # do not search siblings
        sys.modules[parent_name] = parent_pkg
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(common_pkg_root)]
    sys.modules[pkg_name] = pkg
    legacy_mod = _load_module_from(
        f"{pkg_name}.legacy",
        common_pkg_root / "legacy.py",
    )
    bf16_mod = _load_module_from(
        f"{pkg_name}.bf16",
        common_pkg_root / "bf16.py",
    )
    # Bind submodules as attributes on the parent package so that
    # ``infinity.common.bf16`` resolves after the package init runs.
    pkg.legacy = legacy_mod  # type: ignore[attr-defined]
    pkg.bf16 = bf16_mod  # type: ignore[attr-defined]
    # Run the package __init__ in the *same* module object so that
    # ``from infinity.common import bf16 as _bf16`` inside the init
    # ends up setting ``bf16`` on the package we return.
    init_spec = importlib.util.spec_from_file_location(
        pkg_name, common_pkg_root / "__init__.py"
    )
    init_module = importlib.util.module_from_spec(init_spec)
    init_module.__dict__.update(pkg.__dict__)
    sys.modules[pkg_name] = init_module
    init_spec.loader.exec_module(init_module)
    return init_module


@pytest.fixture(scope="module")
def embedded_bf16() -> types.ModuleType:
    return _load_module_from(
        "_test_embedded_bf16",
        REPO_ROOT / "python/infinity_embedded/bf16.py",
    )


class TestUnsupportedColumnTypeError:
    def test_subclass_of_infinity_exception(self, sdk_common_pkg):
        cls = sdk_common_pkg.UnsupportedColumnTypeError
        assert issubclass(cls, sdk_common_pkg.InfinityException)
        # Stays a regular Exception for non-typed callers too.
        assert issubclass(cls, Exception)

    def test_carries_ttype_and_message(self, sdk_common_pkg):
        err = sdk_common_pkg.UnsupportedColumnTypeError("MyFancyType")
        assert err.ttype == "MyFancyType"
        # The error_msg is consumed by callers that read err.error_msg.
        assert err.error_msg == "Unsupported column type 'MyFancyType'"
        # Subclassing the SDK error means the Infinity error-code / -msg
        # fields stay populated for existing error-handling code paths.
        assert isinstance(err, sdk_common_pkg.InfinityException)


class TestBf16BytesToFloat32:
    """``bf16_bytes_to_float32_list`` is the truly-portable part of the
    previous duplication — both packages now wrap the same logic."""

    def test_empty_input(self, sdk_common_pkg, embedded_bf16):
        assert sdk_common_pkg.bf16.bf16_bytes_to_float32_list(b"") == []
        assert embedded_bf16.bf16_bytes_to_float32_list(b"") == []

    @pytest.mark.parametrize(
        "raw, expected",
        [
            # bf16(0.5) = 0x3f00 (little-endian: 00 3f)
            (b"\x00\x3f", 0.5),
            # bf16(1.0) = 0x3f80
            (b"\x80\x3f", 1.0),
            # bf16(-1.0) = 0xbf80
            (b"\x80\xbf", -1.0),
            # bf16(0.0) = 0x0000
            (b"\x00\x00", 0.0),
        ],
    )
    def test_single_value_round_trip(self, sdk_common_pkg, embedded_bf16, raw, expected):
        for mod in (sdk_common_pkg.bf16, embedded_bf16):
            out = mod.bf16_bytes_to_float32_list(raw)
            assert len(out) == 1
            # Convert through float to avoid numpy-version-specific
            # element-type comparison noise.
            assert math.isclose(float(out[0]), expected, rel_tol=1e-6, abs_tol=1e-9)

    def test_multi_value_round_trip_matches_numpy(self, sdk_common_pkg):
        raw = (
            b"\x00\x3f"  # 0.5
            b"\x80\x3f"  # 1.0
            b"\x80\xbf"  # -1.0
            b"\x00\x00"  # 0.0
        )
        sdk = list(sdk_common_pkg.bf16.bf16_bytes_to_float32_list(raw))
        # Cross-check: parse the same buffer via a numpy view.
        u16 = np.frombuffer(raw, dtype="<i2")
        arr = np.zeros(2 * len(u16), dtype="<i2")
        arr[1::2] = u16
        np_out = arr.view("<f4")
        assert len(sdk) == len(np_out)
        for got, want in zip(sdk, np_out):
            assert math.isclose(float(got), float(want), rel_tol=1e-6, abs_tol=1e-9)

    def test_embedded_and_sdk_produce_identical_results(self, sdk_common_pkg, embedded_bf16):
        # Same input -> same output from both packages.
        raw = b"\x00\x3f\x80\x3f\x80\xbf\x00\x00\xcd\xcc\x0c\x40"
        sdk = list(sdk_common_pkg.bf16.bf16_bytes_to_float32_list(raw))
        emb = list(embedded_bf16.bf16_bytes_to_float32_list(raw))
        assert len(sdk) == len(emb)
        for got, want in zip(sdk, emb):
            assert math.isclose(float(got), float(want), rel_tol=1e-6, abs_tol=1e-9)


class TestLegacyReExports:
    """The new package ``infinity.common`` re-exports every public name
    that the old single-file ``infinity.common`` exposed, so existing
    imports like ``from infinity.common import InfinityException`` keep
    working."""

    def test_legacy_names_available(self, sdk_common_pkg):
        for name in (
            "NetworkAddress",
            "SparseVector",
            "Array",
            "FDE",
            "URI",
            "VEC",
            "INSERT_DATA",
            "LOCAL_HOST",
            "LOCAL_INFINITY_PATH",
            "ConflictType",
            "SortType",
            "InfinityException",
            "UnsupportedColumnTypeError",
            "DEFAULT_MATCH_VECTOR_TOPN",
            "DEFAULT_MATCH_SPARSE_TOPN",
        ):
            assert hasattr(sdk_common_pkg, name), name

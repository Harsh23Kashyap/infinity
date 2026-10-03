import struct

import pytest


# Regression tests for the `case LogicalType.kRowID:` arm added in PR #129
# (Fix #128 / upstream #3517) at the top of `parse_single_array_bytes` in both
# the embedded and Thrift Python SDKs. The arm sets `single_pod_element_size = 8`
# (and `tmp_column_type = ttypes.ColumnType.ColumnInt64` on the Thrift side),
# mirroring the existing `kBigInt` arm. The engine stores RowID as 8-byte
# unsigned integers per `src/function/builtin_functions_impl.cpp:208`
# (`std::make_shared<SpecialFunction>("ROW_ID", DataType(LogicalType::kBigInt), 1, SpecialType::kRowID)`).
#
# Pre-fix on `main` @ `b1083912a`: raises `NotImplementedError: Unexpected type
# <WrapDataType logical_type=LogicalType.kRowID ...>` from the `case _:` default.
# Post-fix: returns `(decoded_list, final_offset)` where `decoded_list` is the
# list of BigInt-equivalent row IDs and `final_offset == 4 + N * 8`.
#
# These tests do not require a live Infinity server; they construct the
# `WrapDataType` / `ttypes.DataType` directly and call the parser. Same shape
# as `test_query_builder_optional_clause_none_clear_clause` (PR #127) and
# `test_array_of_embedding_parse` (PR #3455).


@pytest.mark.parametrize("element_count", [0, 1, 3])
def test_parse_single_array_bytes_handles_rowid_element_embedded(element_count):
    """Embedded SDK `parse_single_array_bytes` decodes `array,rowid` columns as
    BigInt-equivalent row IDs (single_pod_element_size = 8)."""
    embedded_types = pytest.importorskip("infinity_embedded.local_infinity.types")
    embedded_utils = pytest.importorskip("infinity_embedded.local_infinity.utils")
    LogicalType = embedded_utils.LogicalType

    column_type = embedded_utils.WrapDataType()
    column_type.logical_type = LogicalType.kArray
    column_type.array_type = embedded_utils.WrapDataType()
    column_type.array_type.logical_type = LogicalType.kRowID

    # Synthesize a byte stream: 4-byte count + N * 8 bytes for the row IDs.
    row_ids = list(range(100, 100 + element_count))
    bytes_data = struct.pack("<I", element_count) + struct.pack(f"<{element_count}q", *row_ids)
    # Plus 4 trailing bytes that should NOT be consumed.
    bytes_data += struct.pack("<I", 0xDEADBEEF)

    array_data, offset = embedded_types.parse_single_array_bytes(column_type, bytes_data, 0)

    assert array_data == row_ids, (
        f"Expected decoded list {row_ids}, got {array_data}"
    )
    assert offset == 4 + element_count * 8, (
        f"Expected offset {4 + element_count * 8}, got {offset}"
    )
    # Trailing bytes should be preserved for the caller.
    assert bytes_data[offset:] == struct.pack("<I", 0xDEADBEEF)


@pytest.mark.parametrize("element_count", [0, 1, 3])
def test_parse_single_array_bytes_handles_rowid_element_thrift(element_count):
    """Thrift SDK `parse_single_array_bytes` decodes `array,rowid` columns as
    BigInt-equivalent row IDs (single_pod_element_size = 8,
    tmp_column_type = ColumnInt64)."""
    # The Thrift types module is exposed as `infinity.remote_thrift.types` (the
    # HTTP import path) and also as `infinity_sdk.infinity.remote_thrift.types`
    # (the SDK package import path). Try the HTTP path first (matches the
    # rest of test_pysdk/), then fall back to the SDK package path.
    ttypes = None
    parser_fn = None
    try:
        from infinity.remote_thrift import types as remote_thrift_types
        ttypes = remote_thrift_types.ttypes
        parser_fn = remote_thrift_types.parse_single_array_bytes
    except ImportError:
        from infinity_sdk.infinity.remote_thrift import types as remote_thrift_types
        ttypes = remote_thrift_types.ttypes
        parser_fn = remote_thrift_types.parse_single_array_bytes

    column_type = ttypes.DataType()
    column_type.logic_type = ttypes.LogicType.Array
    column_type.physical_type = ttypes.PhysicalDataType()
    column_type.physical_type.array_type = ttypes.ArrayInfo()
    column_type.physical_type.array_type.element_data_type = ttypes.DataType()
    column_type.physical_type.array_type.element_data_type.logic_type = ttypes.LogicType.RowID

    row_ids = list(range(100, 100 + element_count))
    bytes_data = struct.pack("<I", element_count) + struct.pack(f"<{element_count}q", *row_ids)
    bytes_data += struct.pack("<I", 0xDEADBEEF)

    array_data, offset = parser_fn(column_type, bytes_data, 0)

    assert array_data == row_ids, (
        f"Expected decoded list {row_ids}, got {array_data}"
    )
    assert offset == 4 + element_count * 8, (
        f"Expected offset {4 + element_count * 8}, got {offset}"
    )
    assert bytes_data[offset:] == struct.pack("<I", 0xDEADBEEF)
import infinity
import pytest
from infinity import index
from infinity.common import ConflictType
from infinity.errors import ErrorCode
from infinity.infinity_http import infinity_http
from infinity.remote_thrift.client import ThriftInfinityClient
from infinity.remote_thrift.db import RemoteDatabase
from infinity.remote_thrift.query_builder import InfinityThriftQueryBuilder
from infinity.remote_thrift.table import RemoteTable

from common import common_values


@pytest.fixture(scope="class")
def http(request):
    return request.config.getoption("--http")


@pytest.fixture(scope="class")
def setup_class(request, http):
    if http:
        uri = common_values.TEST_LOCAL_HOST
        request.cls.infinity_obj = infinity_http()
    else:
        uri = common_values.TEST_LOCAL_HOST
        request.cls.infinity_obj = infinity.connect(uri)
    request.cls.uri = uri
    yield
    request.cls.infinity_obj.disconnect()


@pytest.mark.usefixtures("setup_class")
@pytest.mark.ubsan
class TestInfinity:
    @pytest.mark.usefixtures("skip_if_local_infinity")
    @pytest.mark.usefixtures("skip_if_http")
    def test_query(self):
        conn = ThriftInfinityClient(common_values.TEST_LOCAL_HOST)
        db = RemoteDatabase(conn, "default_db")
        db.drop_table("my_table", conflict_type=ConflictType.Ignore)
        db.create_table(
            "my_table", {
                "num": {"type": "integer"}, "body": {"type": "varchar"}, "vec": {"type": "vector,5,float"}},
            ConflictType.Error)

        table = RemoteTable(conn, "default_db", "my_table")
        res = table.insert(
            [{"num": 1, "body": "undesirable, unnecessary, and harmful", "vec": [1.0] * 5}])
        assert res.error_code == ErrorCode.OK
        res = table.insert(
            [{"num": 2, "body": "publisher=US National Office for Harmful Algal Blooms", "vec": [4.0] * 5}])
        assert res.error_code == ErrorCode.OK
        res = table.insert(
            [{"num": 3, "body": "in the case of plants, growth and chemical", "vec": [7.0] * 5}])
        assert res.error_code == ErrorCode.OK

        res = table.create_index("my_index",
                                 index.IndexInfo("body",
                                                 index.IndexType.FullText),
                                 ConflictType.Error)
        assert res.error_code == ErrorCode.OK

        # Create a query builder
        query_builder = InfinityThriftQueryBuilder(table)
        query_builder.output(["num", "body"])
        query_builder.match_dense('vec', [3.0] * 5, 'float', 'ip', 2)
        query_builder.match_text('body', 'harmful', 2, None)
        query_builder.fusion(method='rrf', topn=10, fusion_params=None)
        res, extra_result = query_builder.to_df()
        print(res)
        res = table.drop_index("my_index", ConflictType.Error)
        assert res.error_code == ErrorCode.OK

        res = db.drop_table("my_table", ConflictType.Error)
        assert res.error_code == ErrorCode.OK

        res = conn.disconnect()
        assert res.error_code == ErrorCode.OK

    @pytest.mark.usefixtures("skip_if_http")
    def test_query_builder(self):
        # connect
        db_obj = self.infinity_obj.get_database("default_db")
        db_obj.drop_table("test_query_builder",
                          conflict_type=ConflictType.Ignore)
        table_obj = db_obj.create_table(
            "test_query_builder", {"c1": {"type": "int"}}, ConflictType.Error)
        query_builder = table_obj.query_builder
        res = query_builder.output(["*"]).to_df()
        print(res)

        res = db_obj.drop_table("test_query_builder", ConflictType.Error)
        assert res.error_code == ErrorCode.OK


# Regression tests for PR #121 (Fix #120): match_dense must reject non-string
# knn_params values with InfinityException(INVALID_PARAMETER_VALUE), not
# AttributeError from calling .lower() on a non-string. Server-free: the
# isinstance guard fires before any actual server round-trip. Same pattern as
# test_query_builder_optional_clause_none_clear_clause (PR #127) and
# test_parse_single_array_bytes_handles_rowid_element_* (PR #131).
#
# The key "threshold" is chosen to avoid the filter-pop branch in
# get_search_optional_filter_from_opt_params (which only fires on key "filter"
# and has its own isinstance check at infinity_embedded/local_infinity/utils.py:266
# and the Thrift SDK's mirror at infinity_sdk/infinity/remote_thrift/utils.py).
@pytest.mark.parametrize("builder_type", ["thrift", "local"])
@pytest.mark.parametrize("opt_value", [None, True, False, 0, 0.5, [1, 2, 3], {"a": "b"}])
def test_match_dense_knn_opt_param_rejects_non_string_value(builder_type, opt_value):
    """Non-string `knn_params` values must surface
    InfinityException(INVALID_PARAMETER_VALUE), not AttributeError."""
    if builder_type == "local":
        local_query_builder = pytest.importorskip("infinity_embedded.local_infinity.query_builder")
        query_builder = local_query_builder.InfinityLocalQueryBuilder(None)
    else:
        query_builder = InfinityThriftQueryBuilder(None)

    with pytest.raises(InfinityException) as exc:
        query_builder.match_dense(
            "vec", [1.0], "l2", 0, knn_params={"threshold": opt_value}
        )
    assert exc.value.error_code == ErrorCode.INVALID_PARAMETER_VALUE


@pytest.mark.parametrize("builder_type", ["thrift", "local"])
def test_match_dense_knn_opt_param_accepts_string_value(builder_type):
    """Happy path for PR #121 (Fix #120). String `knn_params` values must not
    trigger the isinstance guard, and the call must complete (no
    INVALID_PARAMETER_VALUE for a valid string)."""
    if builder_type == "local":
        local_query_builder = pytest.importorskip("infinity_embedded.local_infinity.query_builder")
        query_builder = local_query_builder.InfinityLocalQueryBuilder(None)
    else:
        query_builder = InfinityThriftQueryBuilder(None)

    # String values must NOT raise INVALID_PARAMETER_VALUE. (We do not assert
    # full to_df() success here because the query_builder's match_dense path
    # needs additional state for the actual search, which is server-side; we
    # only verify the isinstance guard does not fire.)
    try:
        query_builder.match_dense(
            "vec", [1.0], "l2", 0, knn_params={"threshold": "0.5"}
        )
    except InfinityException as exc:
        # The only acceptable exception is INVALID_EMBEDDING_DATA_TYPE (the
        # helper that interprets [1.0] as a Tensor of float may surface it
        # before reaching the search path); INVALID_PARAMETER_VALUE would mean
        # the isinstance guard mis-fired on a valid string.
        assert exc.error_code != ErrorCode.INVALID_PARAMETER_VALUE, (
            "isinstance guard fired on a valid string value"
        )

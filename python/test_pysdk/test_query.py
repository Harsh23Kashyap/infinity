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


# Regression tests for PR #123 (Fix #122): table_http_result.to_result() must
# reject non-string, non-primitive row values with InfinityException
# (INVALID_DATA_TYPE), not AttributeError from calling .lower() on a
# non-string. Server-free: the test mocks table_http.show_columns_type() and
# pre-populates output_res so to_result() skips its select() round-trip. Same
# pytest-importorskip pattern as PR #127 / PR #131 / PR #133.
#
# The dispatch chain in to_result() checks isinstance(v, (int, float)),
# is_list(v), is_date(v)/is_time(v)/is_datetime(v), is_sparse(v), and only
# falls into the `else` (where the PR #123 isinstance guard fires) when
# none of those match. None/int/float are caught by the first branch. So
# the parametrized set is restricted to types that bypass the existing
# branches and hit the isinstance guard: bool (True/False) and dict.
@pytest.mark.parametrize("col_value", [True, False, {"a": "b"}])
def test_to_result_row_value_rejects_non_string_non_primitive_value(col_value):
    """Non-string, non-primitive values in `output_res` must surface
    InfinityException(INVALID_DATA_TYPE), not AttributeError."""
    try:
        from infinity_sdk.infinity.infinity_http import table_http_result as table_http_result_cls
    except ImportError:
        # The HTTP SDK is exposed at two import paths depending on whether the
        # test environment has `infinity_sdk` installed as a package or via
        # the legacy `infinity` import path. Fall back to the legacy path.
        from infinity.infinity_http import table_http_result as table_http_result_cls

    class _MockTableHttp:
        def show_columns_type(self):
            return {"col": "varchar"}

    result = table_http_result_cls(output=["col"], table_http=_MockTableHttp())
    # Pre-populate output_res so to_result() skips its select() round-trip.
    result.output_res = [{"col": col_value}]

    with pytest.raises(InfinityException) as exc:
        result.to_result()
    assert exc.value.error_code == ErrorCode.INVALID_DATA_TYPE


def test_to_result_row_value_accepts_string_value():
    """Happy path for PR #123 (Fix #122). String values that hit the `else`
    branch must NOT raise INVALID_DATA_TYPE — they go through the v.lower()
    'true' / 'false' / 'none' dispatch correctly."""
    try:
        from infinity_sdk.infinity.infinity_http import table_http_result as table_http_result_cls
    except ImportError:
        from infinity.infinity_http import table_http_result as table_http_result_cls

    class _MockTableHttp:
        def show_columns_type(self):
            return {"col": "varchar"}

    result = table_http_result_cls(output=["col"], table_http=_MockTableHttp())
    result.output_res = [{"col": "true"}]

    # String "true" must route through the lower() path and NOT raise
    # INVALID_DATA_TYPE. (We don't assert the full to_df() output because
    # the test focuses on the isinstance guard's behavior.)
    result.to_result()

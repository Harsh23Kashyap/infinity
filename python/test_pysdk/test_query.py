import infinity
import pytest
from infinity import index
from infinity.common import ConflictType, InfinityException
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


# Regression tests for PR #139 (Fix #139): InfinityThriftQueryBuilder.group_by
# must reject non-string column values with InfinityException(INVALID_DATA_TYPE),
# not AttributeError from calling .lower() on a non-string. Server-free: the
# validation runs at the top of group_by() before any other call, so a Mock
# table passed to the builder is sufficient.
#
# Cycle 72 lesson: the same family of v.lower()-without-isinstance guards
# already shipped for delimiter (PR #113), file_type (PR #119), KNN opt-param
# (PR #121), and parse_df (PR #123). PR #139 closes the gap at InfinityThriftQueryBuilder.group_by.
#
# IMPORTANT: in Python 3, `bool` is a subclass of `int`, so
# `isinstance(True, str) == False`. That means True/False are correctly
# rejected by the new guard. The minimum parametrized set that actually
# exercises the new guard is: `int`, `None`, `dict`, `list`, `bool`.
@pytest.mark.parametrize("bad_value", [1, None, True, False, {"a": "b"}, [1, 2]])
def test_thrift_group_by_rejects_non_string_list_element(bad_value):
    """Non-string elements in a group_by list must surface
    InfinityException(INVALID_DATA_TYPE), not AttributeError."""
    builder = InfinityThriftQueryBuilder(table=None)
    with pytest.raises(InfinityException) as exc:
        builder.group_by(["id", bad_value])
    assert exc.value.error_code == ErrorCode.INVALID_DATA_TYPE


@pytest.mark.parametrize("bad_value", [1, None, True, False, {"a": "b"}, [1, 2]])
def test_thrift_group_by_rejects_non_string_scalar(bad_value):
    """A non-string scalar passed to group_by must surface
    InfinityException(INVALID_DATA_TYPE), not AttributeError."""
    builder = InfinityThriftQueryBuilder(table=None)
    with pytest.raises(InfinityException) as exc:
        builder.group_by(bad_value)
    assert exc.value.error_code == ErrorCode.INVALID_DATA_TYPE


def test_thrift_group_by_accepts_string_list():
    """Happy path for PR #139. A list of strings must NOT raise
    INVALID_DATA_TYPE — case normalization proceeds correctly."""
    builder = InfinityThriftQueryBuilder(table=None)
    # Should not raise; just verify the builder state was updated.
    builder.group_by(["ID", "name"])
    assert builder._groupby is not None
    assert len(builder._groupby) == 2


def test_thrift_group_by_accepts_string_scalar():
    """Happy path for PR #139, scalar form."""
    builder = InfinityThriftQueryBuilder(table=None)
    builder.group_by("id")
    assert builder._groupby is not None
    assert len(builder._groupby) == 1

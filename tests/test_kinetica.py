"""
Tests for the Kinetica Ray Data integration.

These tests use mocks to verify the KineticaDatasource and KineticaDatasink
work correctly without requiring a running Kinetica server.
"""

import json
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pytest
from gpudb import GPUdb
from kinetica_ray.datasink import (
    KineticaDatasink,
    KineticaSinkMode,
    KineticaTableSettings,
)
from kinetica_ray.datasource import (
    KineticaDatasource,
    _has_balanced_quotes,
    _is_filter_safe,
)
from ray.data._internal.execution.interfaces.task_context import TaskContext

# ============================================================================
# Fixtures for Mocking GPUdb Client
# ============================================================================


@pytest.fixture
def mock_gpudb_client():
    """Mock GPUdb client for datasource tests.

    KineticaDatasource wraps a real GPUdbTable to read table type info and
    row samples. GPUdbTable's real constructor expects a fully-functional,
    server-shaped response protocol (AttrDict-style responses, not plain
    dicts) that a bare client mock can't satisfy. So gpudb.GPUdbTable is
    patched here to return a fake table exposing a real GPUdbRecordType and
    canned records -- isolating these tests to KineticaDatasource's own
    logic rather than gpudb's internal wire protocol. The fake table is
    exposed as client.gpudb_table for tests that need to assert on it.
    """
    from gpudb import GPUdbRecordColumn, GPUdbRecordType

    client = MagicMock(spec=GPUdb)

    # Mock show_table response, used directly by _get_table_info() for the
    # unfiltered row count.
    client.show_table.return_value = {
        "sizes": [100],
        "total_size": 100,
    }

    # Mock get_records response, used directly by _get_table_info() for the
    # filtered row count.
    client.get_records.return_value = {
        "total_number_of_records": 100,
    }

    record_type = GPUdbRecordType(
        columns=[
            GPUdbRecordColumn(
                name="id",
                column_type=GPUdbRecordColumn._ColumnType.LONG,
                column_properties=[],
            ),
            GPUdbRecordColumn(
                name="name",
                column_type=GPUdbRecordColumn._ColumnType.STRING,
                column_properties=[],
            ),
            GPUdbRecordColumn(
                name="value",
                column_type=GPUdbRecordColumn._ColumnType.DOUBLE,
                column_properties=[],
            ),
        ],
        label="test_table",
    )
    sample_records = [
        {"id": 1, "name": "Alice", "value": 100.5},
        {"id": 2, "name": "Bob", "value": 200.75},
    ]

    fake_table = MagicMock()
    fake_table.gpudbrecord_type = record_type
    fake_table.get_records.return_value = sample_records
    fake_table.get_records_by_column.return_value = sample_records
    client.gpudb_table = fake_table

    with patch("gpudb.GPUdbTable", return_value=fake_table):
        yield client


@pytest.fixture
def mock_gpudb_sink_client():
    """Mock GPUdb client for datasink tests."""
    # spec=GPUdb so isinstance(client, GPUdb) passes inside GPUdbTable.__init__.
    client = MagicMock(spec=GPUdb)

    # Mock table existence check
    client.has_table.return_value = {"table_exists": False}

    # Mock insert_records response
    client.insert_records.return_value = {
        "count_inserted": 3,
        "count_updated": 0,
        "info": {},
    }

    # Mock show_table response
    client.show_table.return_value = {
        "type_ids": ["type_123"],
        "type_schemas": [
            json.dumps(
                {
                    "fields": [
                        {"name": "id", "type": "long"},
                        {"name": "name", "type": "string"},
                    ]
                }
            )
        ],
        "properties": [{"id": [], "name": []}],
    }

    return client


@pytest.fixture(autouse=True)
def patch_gpudb():
    """Automatically patch GPUdb for all tests."""
    with patch("gpudb.GPUdb") as mock_gpudb_class:
        mock_instance = MagicMock()
        mock_gpudb_class.return_value = mock_instance
        yield mock_instance


# ============================================================================
# Filter Safety Tests
# ============================================================================


class TestFilterSafety:
    """Tests for filter expression safety validation."""

    @pytest.mark.parametrize(
        "filter_expr, is_safe",
        [
            # Safe expressions
            ("id > 100", True),
            ("name = 'Alice' AND value > 50", True),
            ("id > 100 AND name IS NOT NULL", True),
            # Keywords inside string literals are allowed (not injection)
            ("city = 'Union City'", True),
            ("status = 'DROP_PENDING'", True),
            ("name = 'Delete Me'", True),
            ('comment = "ALTER this later"', True),
            # Escaped quotes in strings
            ("name = 'O''Brien'", True),
            # Unsafe expressions - keywords outside strings
            ("id = 1; DROP TABLE test;", False),
            ("id > 100; SELECT * FROM users", False),
            ("id IN {1, 2, 3}", False),
            # Keywords outside of string context
            ("id = 1 UNION SELECT * FROM secrets", False),
            ("id = 1 -- comment injection", False),
            ("id = 1 /* block comment */", False),
            # Unclosed string literals (could bypass stripping)
            ("x = ''' ; DROP TABLE t", False),
            ('x = """ ; DROP TABLE t', False),
            ("name = 'value", False),
            ('comment = "test', False),
        ],
    )
    def test_is_filter_safe(self, filter_expr, is_safe):
        """Test filter safety validation."""
        assert _is_filter_safe(filter_expr) == is_safe

    def test_balanced_quotes_detection(self):
        """Test the _has_balanced_quotes helper function."""
        # Balanced quotes
        assert _has_balanced_quotes("name = 'Alice'") is True
        assert _has_balanced_quotes('status = "active"') is True
        assert _has_balanced_quotes("name = 'O''Brien'") is True
        assert _has_balanced_quotes('msg = "He said ""hi"""') is True
        assert _has_balanced_quotes("id = 1") is True

        # Unbalanced quotes
        assert _has_balanced_quotes("name = 'Alice") is False
        assert _has_balanced_quotes('status = "active') is False
        assert _has_balanced_quotes("x = '''") is False


# ============================================================================
# KineticaDatasource Tests
# ============================================================================


class TestKineticaDatasource:
    """Tests for KineticaDatasource."""

    @pytest.fixture
    def datasource(self):
        """Create a KineticaDatasource with test parameters."""
        with patch("kinetica_ray.datasource.KineticaDatasource._init_client"):
            ds = KineticaDatasource(
                url="http://localhost:9191",
                table_name="test_table",
                username="admin",
                password="password",
                columns=["id", "name"],
                filter_expression="id > 100",
                batch_size=5000,
            )
            return ds

    def test_init(self, datasource):
        """Test datasource initialization."""
        assert datasource._url == "http://localhost:9191"
        assert datasource._table_name == "test_table"
        assert datasource._username == "admin"
        assert datasource._columns == ["id", "name"]
        assert datasource._filter_expression == "id > 100"
        assert datasource._batch_size == 5000

    def test_default_batch_size(self):
        """Test default batch size."""
        with patch("kinetica_ray.datasource.KineticaDatasource._init_client"):
            ds = KineticaDatasource(
                url="http://localhost:9191",
                table_name="test_table",
            )
            assert ds._batch_size == 10000

    def test_unsafe_filter_rejected(self):
        """Test that unsafe filter expressions are rejected without leaking input."""
        with pytest.raises(ValueError, match="unsafe patterns") as exc_info:
            KineticaDatasource(
                url="http://localhost:9191",
                table_name="test_table",
                filter_expression="id = 1; DROP TABLE test;",
            )
        # Verify the error message does NOT include the raw user input
        # (to prevent log injection)
        assert "DROP TABLE" not in str(exc_info.value)

    def test_get_name(self, datasource):
        """Test datasource name generation."""
        assert datasource.get_name() == "Kinetica(test_table)"

    @patch.object(KineticaDatasource, "_init_client")
    def test_get_table_info(self, mock_init_client, mock_gpudb_client):
        """Test _get_table_info method."""
        mock_init_client.return_value = mock_gpudb_client

        ds = KineticaDatasource(
            url="http://localhost:9191",
            table_name="test_table",
        )

        total_count, arrow_schema = ds._get_table_info(mock_gpudb_client)

        assert total_count == 100
        assert arrow_schema is not None
        assert len(arrow_schema) == 3

    @patch.object(KineticaDatasource, "_init_client")
    def test_get_table_info_with_filter(self, mock_init_client, mock_gpudb_client):
        """Test _get_table_info with filter expression."""
        mock_init_client.return_value = mock_gpudb_client

        ds = KineticaDatasource(
            url="http://localhost:9191",
            table_name="test_table",
            filter_expression="id > 50",
        )

        total_count, arrow_schema = ds._get_table_info(mock_gpudb_client)

        assert total_count >= 0
        assert arrow_schema is not None

    @patch.object(KineticaDatasource, "_init_client")
    def test_estimate_row_size(self, mock_init_client, mock_gpudb_client):
        """Test _estimate_row_size method."""
        mock_init_client.return_value = mock_gpudb_client

        ds = KineticaDatasource(
            url="http://localhost:9191",
            table_name="test_table",
            columns=["id", "name"],
        )

        row_size = ds._estimate_row_size(mock_gpudb_client, sample_size=100)

        assert row_size > 0
        # ds has columns=["id", "name"] set, so the table wrapper's
        # column-selecting variant is the one actually used.
        mock_gpudb_client.gpudb_table.get_records_by_column.assert_called()

    @patch.object(KineticaDatasource, "_init_client")
    @pytest.mark.parametrize("parallelism", [1, 2, 4])
    def test_get_read_tasks(self, mock_init_client, mock_gpudb_client, parallelism):
        """Test get_read_tasks with different parallelism levels."""
        mock_init_client.return_value = mock_gpudb_client

        ds = KineticaDatasource(
            url="http://localhost:9191",
            table_name="test_table",
        )

        read_tasks = ds.get_read_tasks(parallelism)

        assert len(read_tasks) <= parallelism
        assert all(task.metadata.num_rows > 0 for task in read_tasks)

    @patch.object(KineticaDatasource, "_init_client")
    def test_get_read_tasks_empty_table(self, mock_init_client, mock_gpudb_client):
        """Test get_read_tasks with empty table."""
        mock_init_client.return_value = mock_gpudb_client
        mock_gpudb_client.show_table.return_value = {
            "type_schemas": [json.dumps({"fields": []})],
            "properties": [{}],
            "sizes": [0],
            "total_size": 0,
        }

        ds = KineticaDatasource(
            url="http://localhost:9191",
            table_name="empty_table",
        )

        read_tasks = ds.get_read_tasks(parallelism=2)

        assert len(read_tasks) == 0

    @patch.object(KineticaDatasource, "_init_client")
    def test_estimate_inmemory_data_size(self, mock_init_client, mock_gpudb_client):
        """Test estimate_inmemory_data_size method."""
        mock_init_client.return_value = mock_gpudb_client

        ds = KineticaDatasource(
            url="http://localhost:9191",
            table_name="test_table",
        )

        size = ds.estimate_inmemory_data_size()

        assert size is not None
        assert size > 0


# ============================================================================
# KineticaDatasink Tests
# ============================================================================


class TestKineticaDatasink:
    """Tests for KineticaDatasink."""

    @pytest.fixture
    def datasink(self):
        """Create a KineticaDatasink with test parameters."""
        with patch("kinetica_ray.datasink.KineticaDatasink._init_client"):
            ds = KineticaDatasink(
                url="http://localhost:9191",
                table_name="test_table",
                username="admin",
                password="password",
                mode=KineticaSinkMode.APPEND,
                batch_size=5000,
            )
            return ds

    def test_init(self, datasink):
        """Test datasink initialization."""
        assert datasink._url == "http://localhost:9191"
        assert datasink._table_name == "test_table"
        assert datasink._username == "admin"
        assert datasink._mode == KineticaSinkMode.APPEND
        assert datasink._batch_size == 5000

    def test_string_mode(self):
        """Test datasink accepts string mode values."""
        schema = pa.schema([pa.field("id", pa.int64())])
        with patch("kinetica_ray.datasink.KineticaDatasink._init_client"):
            ds = KineticaDatasink(
                url="http://localhost:9191",
                table_name="test_table",
                mode="overwrite",
                schema=schema,
            )
            assert ds._mode == KineticaSinkMode.OVERWRITE

    def test_table_settings(self):
        """Test KineticaTableSettings configuration."""
        settings = KineticaTableSettings(
            is_replicated=True,
            chunk_size=1000000,
            ttl=60,
            primary_keys=["id"],
            shard_keys=["region"],
        )

        with patch("kinetica_ray.datasink.KineticaDatasink._init_client"):
            ds = KineticaDatasink(
                url="http://localhost:9191",
                table_name="test_table",
                table_settings=settings,
            )

            assert ds._table_settings.is_replicated is True
            assert ds._table_settings.chunk_size == 1000000
            assert ds._table_settings.ttl == 60
            assert ds._table_settings.primary_keys == ["id"]
            assert ds._table_settings.shard_keys == ["region"]

    def test_get_name(self, datasink):
        """Test datasink name generation."""
        assert datasink.get_name() == "Kinetica(test_table)"

    def test_supports_distributed_writes(self, datasink):
        """Test that distributed writes are always supported.

        Table DDL (CREATE/DROP) is performed in on_write_start() before any
        writes begin, so all modes have a known table structure by the time
        distributed writes start.
        """
        # Distributed writes are always supported since on_write_start
        # runs before any write() calls
        assert datasink.supports_distributed_writes is True

    def test_min_rows_per_write(self, datasink):
        """Test min_rows_per_write property (used by Ray Data framework)."""
        assert datasink.min_rows_per_write == 5000

    @patch.object(KineticaDatasink, "_init_client")
    def test_table_exists(self, mock_init_client, mock_gpudb_sink_client):
        """Test _table_exists method."""
        mock_init_client.return_value = mock_gpudb_sink_client
        mock_gpudb_sink_client.has_table.return_value = {"table_exists": True}

        ds = KineticaDatasink(
            url="http://localhost:9191",
            table_name="test_table",
        )

        exists = ds._table_exists(mock_gpudb_sink_client)

        assert exists is True
        mock_gpudb_sink_client.has_table.assert_called_once()

    @patch.object(KineticaDatasink, "_init_client")
    def test_drop_table(self, mock_init_client, mock_gpudb_sink_client):
        """Test _drop_table method uses no_error_if_not_exists option."""
        mock_init_client.return_value = mock_gpudb_sink_client

        ds = KineticaDatasink(
            url="http://localhost:9191",
            table_name="test_table",
        )

        ds._drop_table(mock_gpudb_sink_client)

        mock_gpudb_sink_client.clear_table.assert_called_once_with(
            table_name="test_table",
            options={"no_error_if_not_exists": "true"},
        )

    @patch.object(KineticaDatasink, "_init_client")
    @patch("kinetica_ray.type_utils.arrow_schema_to_kinetica_columns")
    def test_create_table(
        self, mock_arrow_to_kinetica, mock_init_client, mock_gpudb_sink_client
    ):
        """Test _create_table method."""
        from gpudb import GPUdbRecordColumn, GPUdbRecordType

        mock_init_client.return_value = mock_gpudb_sink_client

        # Mock columns
        mock_columns = [
            GPUdbRecordColumn(
                name="id",
                column_type=GPUdbRecordColumn._ColumnType.LONG,
                column_properties=[],
                is_nullable=False,
            ),
        ]

        # Mock record type
        mock_record_type = MagicMock(spec=GPUdbRecordType)
        mock_record_type.create_type.return_value = "type_123"
        mock_record_type.schema_string = "schema_string"
        mock_record_type.column_properties = {}

        with patch("gpudb.GPUdbRecordType", return_value=mock_record_type):
            ds = KineticaDatasink(
                url="http://localhost:9191",
                table_name="test_table",
            )

            ds._create_table(mock_gpudb_sink_client, mock_columns)

            mock_gpudb_sink_client.create_table.assert_called_once()

    @patch.object(KineticaDatasink, "_init_client")
    @pytest.mark.parametrize(
        "mode, table_exists, should_create",
        [
            (KineticaSinkMode.CREATE, False, True),
            (KineticaSinkMode.APPEND, False, True),
            (KineticaSinkMode.APPEND, True, False),
            (KineticaSinkMode.OVERWRITE, False, True),
            (KineticaSinkMode.OVERWRITE, True, True),
        ],
    )
    def test_on_write_start_modes(
        self,
        mock_init_client,
        mock_gpudb_sink_client,
        mode,
        table_exists,
        should_create,
    ):
        """Test on_write_start with different modes."""
        mock_init_client.return_value = mock_gpudb_sink_client
        mock_gpudb_sink_client.has_table.return_value = {"table_exists": table_exists}

        schema = pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field("name", pa.string()),
            ]
        )

        with (
            patch.object(KineticaDatasink, "_create_table") as mock_create,
            patch.object(KineticaDatasink, "_drop_table") as mock_drop,
            patch.object(
                KineticaDatasink, "_get_existing_record_type"
            ) as mock_get_type,
            patch(
                "kinetica_ray.type_utils.arrow_schema_to_kinetica_columns"
            ) as mock_arrow_to_kinetica,
        ):
            mock_arrow_to_kinetica.return_value = []

            # Mock existing record type
            mock_record_type = MagicMock()
            mock_record_type.columns = []
            mock_get_type.return_value = mock_record_type

            ds = KineticaDatasink(
                url="http://localhost:9191",
                table_name="test_table",
                mode=mode,
                schema=schema,
            )

            ds.on_write_start(schema)

            if mode == KineticaSinkMode.OVERWRITE and table_exists:
                mock_drop.assert_called_once()

            if should_create:
                if mode == KineticaSinkMode.APPEND and table_exists:
                    mock_get_type.assert_called_once()
                else:
                    # CREATE or OVERWRITE should create table
                    if mode != KineticaSinkMode.APPEND or not table_exists:
                        mock_create.assert_called()

    @patch.object(KineticaDatasink, "_init_client")
    def test_on_write_start_create_existing_table_fails(
        self, mock_init_client, mock_gpudb_sink_client
    ):
        """Test that CREATE mode fails if table already exists."""
        from gpudb import GPUdbException

        mock_init_client.return_value = mock_gpudb_sink_client
        mock_gpudb_sink_client.has_table.return_value = {"table_exists": True}

        schema = pa.schema([pa.field("id", pa.int64())])

        ds = KineticaDatasink(
            url="http://localhost:9191",
            table_name="test_table",
            mode=KineticaSinkMode.CREATE,
            schema=schema,
        )

        with pytest.raises(GPUdbException, match="already exists"):
            ds.on_write_start(schema)

    @patch.object(KineticaDatasink, "_init_client")
    @patch("kinetica_ray.type_utils.arrow_schema_to_kinetica_columns")
    @patch("kinetica_ray.type_utils.convert_arrow_batch_to_records")
    def test_write(
        self,
        mock_convert_batch,
        mock_arrow_to_kinetica,
        mock_init_client,
        mock_gpudb_sink_client,
    ):
        """Test write method."""
        from gpudb import GPUdbRecordColumn

        mock_init_client.return_value = mock_gpudb_sink_client
        mock_gpudb_sink_client.has_table.return_value = {"table_exists": False}

        # Mock conversion functions
        mock_arrow_to_kinetica.return_value = []
        mock_convert_batch.return_value = [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
        ]

        schema = pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field("name", pa.string()),
            ]
        )

        # Create test data
        rb = pa.record_batch(
            [pa.array([1, 2]), pa.array(["Alice", "Bob"])],
            names=["id", "name"],
        )
        block_data = pa.Table.from_batches([rb])

        # write() wraps a real GPUdbTable for multihead ingestion. GPUdbTable's
        # real constructor expects a fully-functional, server-shaped response
        # protocol that a bare client mock can't satisfy, so it's patched here
        # -- isolating this test to KineticaDatasink's own write() logic.
        fake_gpudb_table = MagicMock()
        fake_gpudb_table.insert_records.return_value = {"info": {}}
        fake_gpudb_table.total_inserted = 2
        fake_gpudb_table.total_updated = 0

        with (
            patch.object(KineticaDatasink, "_create_table"),
            patch("gpudb.GPUdbTable", return_value=fake_gpudb_table),
        ):
            ds = KineticaDatasink(
                url="http://localhost:9191",
                table_name="test_table",
                mode=KineticaSinkMode.CREATE,
                schema=schema,
            )
            # A real (non-empty) column list: GPUdbTable rejects an empty one.
            ds._column_defs = [
                {
                    "name": "id",
                    "column_type": GPUdbRecordColumn._ColumnType.LONG,
                    "column_properties": [],
                    "is_nullable": True,
                },
                {
                    "name": "name",
                    "column_type": GPUdbRecordColumn._ColumnType.STRING,
                    "column_properties": [],
                    "is_nullable": True,
                },
            ]

            ctx = TaskContext(task_idx=0, op_name="test_write")
            result = ds.write([block_data], ctx=ctx)

            assert result["num_inserted"] == 2
            assert result["num_updated"] == 0
            fake_gpudb_table.flush_data_to_server.assert_called_once()


# ============================================================================
# Type Utils Tests
# ============================================================================


class TestKineticaTypeUtils:
    """Tests for type conversion utilities."""

    def test_arrow_schema_conversion(self):
        """Test converting Arrow schema to Kinetica columns."""
        from kinetica_ray.type_utils import (
            arrow_schema_to_kinetica_columns,
        )

        schema = pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field("name", pa.string()),
                pa.field("value", pa.float64()),
                pa.field("active", pa.bool_()),
            ]
        )

        columns = arrow_schema_to_kinetica_columns(schema)

        assert len(columns) == 4
        assert columns[0].name == "id"
        assert columns[1].name == "name"
        assert columns[2].name == "value"
        assert columns[3].name == "active"

    def test_arrow_schema_with_keys(self):
        """Test converting Arrow schema with primary/shard keys."""
        from gpudb import GPUdbColumnProperty
        from kinetica_ray.type_utils import (
            arrow_schema_to_kinetica_columns,
        )

        schema = pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field("region", pa.string()),
                pa.field("value", pa.float64()),
            ]
        )

        columns = arrow_schema_to_kinetica_columns(
            schema,
            primary_keys=["id"],
            shard_keys=["region"],
        )

        assert GPUdbColumnProperty.PRIMARY_KEY in columns[0].column_properties
        assert GPUdbColumnProperty.SHARD_KEY in columns[1].column_properties

    def test_arrow_schema_with_invalid_keys_rejected(self):
        """Test that invalid primary/shard keys raise ValueError."""
        from kinetica_ray.type_utils import (
            arrow_schema_to_kinetica_columns,
        )

        schema = pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field("name", pa.string()),
            ]
        )

        # Invalid primary key
        with pytest.raises(ValueError, match="non-existent columns"):
            arrow_schema_to_kinetica_columns(
                schema,
                primary_keys=["nonexistent_column"],
            )

        # Invalid shard key
        with pytest.raises(ValueError, match="non-existent columns"):
            arrow_schema_to_kinetica_columns(
                schema,
                shard_keys=["also_nonexistent"],
            )

        # Both invalid
        with pytest.raises(ValueError, match="non-existent columns"):
            arrow_schema_to_kinetica_columns(
                schema,
                primary_keys=["bad_pk"],
                shard_keys=["bad_sk"],
            )

    def test_arrow_to_kinetica_integer_types(self):
        """Test Arrow integer types convert correctly to Kinetica."""
        from gpudb import GPUdbColumnProperty
        from kinetica_ray.type_utils import (
            arrow_to_kinetica_type,
        )

        # Signed integers
        assert arrow_to_kinetica_type(pa.int8()) == (
            "int",
            [GPUdbColumnProperty.INT8],
        )
        assert arrow_to_kinetica_type(pa.int16()) == (
            "int",
            [GPUdbColumnProperty.INT16],
        )
        assert arrow_to_kinetica_type(pa.int32()) == ("int", [])
        assert arrow_to_kinetica_type(pa.int64()) == ("long", [])

        # Unsigned integers
        assert arrow_to_kinetica_type(pa.uint8()) == (
            "int",
            [GPUdbColumnProperty.INT16],
        )
        assert arrow_to_kinetica_type(pa.uint16()) == ("int", [])
        assert arrow_to_kinetica_type(pa.uint32()) == ("long", [])
        assert arrow_to_kinetica_type(pa.uint64()) == (
            "string",
            [GPUdbColumnProperty.ULONG],
        )

    def test_arrow_to_kinetica_float_types(self):
        """Test Arrow float types convert correctly to Kinetica."""
        from kinetica_ray.type_utils import (
            arrow_to_kinetica_type,
        )

        assert arrow_to_kinetica_type(pa.float32()) == ("float", [])
        assert arrow_to_kinetica_type(pa.float64()) == ("double", [])

    def test_arrow_to_kinetica_datetime_types(self):
        """Test Arrow date/time types convert correctly to Kinetica."""
        from gpudb import GPUdbColumnProperty
        from kinetica_ray.type_utils import (
            arrow_to_kinetica_type,
        )

        assert arrow_to_kinetica_type(pa.date32()) == (
            "string",
            [GPUdbColumnProperty.DATE],
        )
        assert arrow_to_kinetica_type(pa.date64()) == (
            "string",
            [GPUdbColumnProperty.DATE],
        )
        assert arrow_to_kinetica_type(pa.time32("ms")) == (
            "string",
            [GPUdbColumnProperty.TIME],
        )
        assert arrow_to_kinetica_type(pa.time64("us")) == (
            "string",
            [GPUdbColumnProperty.TIME],
        )
        assert arrow_to_kinetica_type(pa.timestamp("us")) == (
            "string",
            [GPUdbColumnProperty.DATETIME],
        )

    def test_arrow_to_kinetica_other_types(self):
        """Test Arrow boolean, string, binary types convert correctly."""
        from gpudb import GPUdbColumnProperty
        from kinetica_ray.type_utils import (
            arrow_to_kinetica_type,
        )

        assert arrow_to_kinetica_type(pa.bool_()) == (
            "int",
            [GPUdbColumnProperty.BOOLEAN],
        )
        assert arrow_to_kinetica_type(pa.string()) == ("string", [])
        assert arrow_to_kinetica_type(pa.binary()) == ("bytes", [])

    def test_kinetica_to_arrow_case_insensitive(self):
        """Test Kinetica to Arrow conversion is case-insensitive."""
        from gpudb import GPUdbRecordColumn
        from kinetica_ray.type_utils import (
            kinetica_to_arrow_type,
        )

        # Helper to create mock column
        def make_col(col_type, props):
            col = GPUdbRecordColumn(
                name="test",
                column_type=col_type,
                column_properties=props,
            )
            return col

        # Test lowercase properties
        assert (
            kinetica_to_arrow_type(
                make_col(GPUdbRecordColumn._ColumnType.INT, ["boolean"])
            )
            == pa.bool_()
        )
        assert (
            kinetica_to_arrow_type(
                make_col(GPUdbRecordColumn._ColumnType.INT, ["int8"])
            )
            == pa.int8()
        )
        assert (
            kinetica_to_arrow_type(
                make_col(GPUdbRecordColumn._ColumnType.STRING, ["date"])
            )
            == pa.date32()
        )
        assert kinetica_to_arrow_type(
            make_col(GPUdbRecordColumn._ColumnType.STRING, ["time"])
        ) == pa.time64("us")
        assert kinetica_to_arrow_type(
            make_col(GPUdbRecordColumn._ColumnType.STRING, ["datetime"])
        ) == pa.timestamp("us")

        # Test uppercase properties
        assert (
            kinetica_to_arrow_type(
                make_col(GPUdbRecordColumn._ColumnType.INT, ["BOOLEAN"])
            )
            == pa.bool_()
        )
        assert (
            kinetica_to_arrow_type(
                make_col(GPUdbRecordColumn._ColumnType.INT, ["INT8"])
            )
            == pa.int8()
        )
        assert (
            kinetica_to_arrow_type(
                make_col(GPUdbRecordColumn._ColumnType.STRING, ["DATE"])
            )
            == pa.date32()
        )
        assert kinetica_to_arrow_type(
            make_col(GPUdbRecordColumn._ColumnType.STRING, ["DATETIME"])
        ) == pa.timestamp("us")

        # Test mixed case properties
        assert (
            kinetica_to_arrow_type(
                make_col(GPUdbRecordColumn._ColumnType.INT, ["Boolean"])
            )
            == pa.bool_()
        )
        assert kinetica_to_arrow_type(
            make_col(GPUdbRecordColumn._ColumnType.STRING, ["DateTime"])
        ) == pa.timestamp("us")

    def test_convert_arrow_batch_datetime_serialization(self):
        """Test date/time values serialize to ISO format strings."""
        from datetime import date, datetime, time

        from gpudb import GPUdbColumnProperty, GPUdbRecordColumn
        from kinetica_ray.type_utils import (
            convert_arrow_batch_to_records,
        )

        # Create test data with date/time columns
        data = {
            "id": [1, 2],
            "date_col": [date(2024, 1, 15), date(2024, 12, 31)],
            "time_col": [time(10, 30, 45), time(23, 59, 59)],
            "datetime_col": [
                datetime(2024, 1, 15, 10, 30, 45),
                datetime(2024, 12, 31, 23, 59, 59),
            ],
        }
        batch = pa.RecordBatch.from_pydict(data)

        # Create column definitions
        columns = [
            GPUdbRecordColumn(
                name="id",
                column_type=GPUdbRecordColumn._ColumnType.INT,
                column_properties=[],
            ),
            GPUdbRecordColumn(
                name="date_col",
                column_type=GPUdbRecordColumn._ColumnType.STRING,
                column_properties=[GPUdbColumnProperty.DATE],
            ),
            GPUdbRecordColumn(
                name="time_col",
                column_type=GPUdbRecordColumn._ColumnType.STRING,
                column_properties=[GPUdbColumnProperty.TIME],
            ),
            GPUdbRecordColumn(
                name="datetime_col",
                column_type=GPUdbRecordColumn._ColumnType.STRING,
                column_properties=[GPUdbColumnProperty.DATETIME],
            ),
        ]

        records = convert_arrow_batch_to_records(batch, columns)

        # Verify date serialization
        assert records[0]["date_col"] == "2024-01-15"
        assert records[1]["date_col"] == "2024-12-31"

        # Verify time serialization
        assert records[0]["time_col"] == "10:30:45"
        assert records[1]["time_col"] == "23:59:59"

        # Verify datetime serialization (ISO format)
        assert "2024-01-15" in records[0]["datetime_col"]
        assert "10:30:45" in records[0]["datetime_col"]

    def test_convert_arrow_batch_datetime_json_serializable(self):
        """Test that records with date/time are JSON serializable."""
        import json
        from datetime import date, datetime, time

        from gpudb import GPUdbColumnProperty, GPUdbRecordColumn
        from kinetica_ray.type_utils import (
            convert_arrow_batch_to_records,
        )

        data = {
            "id": [1],
            "date_col": [date(2024, 1, 15)],
            "time_col": [time(10, 30, 45)],
            "datetime_col": [datetime(2024, 1, 15, 10, 30, 45)],
        }
        batch = pa.RecordBatch.from_pydict(data)

        columns = [
            GPUdbRecordColumn(
                name="id",
                column_type=GPUdbRecordColumn._ColumnType.INT,
                column_properties=[],
            ),
            GPUdbRecordColumn(
                name="date_col",
                column_type=GPUdbRecordColumn._ColumnType.STRING,
                column_properties=[GPUdbColumnProperty.DATE],
            ),
            GPUdbRecordColumn(
                name="time_col",
                column_type=GPUdbRecordColumn._ColumnType.STRING,
                column_properties=[GPUdbColumnProperty.TIME],
            ),
            GPUdbRecordColumn(
                name="datetime_col",
                column_type=GPUdbRecordColumn._ColumnType.STRING,
                column_properties=[GPUdbColumnProperty.DATETIME],
            ),
        ]

        records = convert_arrow_batch_to_records(batch, columns)

        # This should not raise TypeError
        json_str = json.dumps(records[0])
        parsed = json.loads(json_str)

        assert isinstance(parsed["date_col"], str)
        assert isinstance(parsed["time_col"], str)
        assert isinstance(parsed["datetime_col"], str)

    def test_convert_arrow_batch_null_datetime(self):
        """Test that null date/time values are handled correctly."""
        from datetime import date

        from gpudb import GPUdbColumnProperty, GPUdbRecordColumn
        from kinetica_ray.type_utils import (
            convert_arrow_batch_to_records,
        )

        data = {
            "id": [1, 2],
            "date_col": [date(2024, 1, 15), None],
        }
        batch = pa.RecordBatch.from_pydict(data)

        columns = [
            GPUdbRecordColumn(
                name="id",
                column_type=GPUdbRecordColumn._ColumnType.INT,
                column_properties=[],
            ),
            GPUdbRecordColumn(
                name="date_col",
                column_type=GPUdbRecordColumn._ColumnType.STRING,
                column_properties=[GPUdbColumnProperty.DATE],
            ),
        ]

        records = convert_arrow_batch_to_records(batch, columns)

        assert records[0]["date_col"] == "2024-01-15"
        assert records[1]["date_col"] is None

    def test_convert_records_to_arrow_error_handling(self):
        """Test that type conversion errors include column name."""
        from kinetica_ray.type_utils import (
            convert_records_to_arrow_table,
        )

        schema = pa.schema([pa.field("int_col", pa.int64())])
        bad_records = [{"int_col": "not_an_integer"}]

        with pytest.raises(pa.ArrowTypeError) as exc_info:
            convert_records_to_arrow_table(bad_records, schema)

        error_msg = str(exc_info.value)
        assert "int_col" in error_msg

    def test_vector_bytes_json_serialization(self):
        """Test that vector (bytes) can be JSON serialized via base64."""
        import base64
        import json
        import struct

        # Simulate the custom serializer used in _write_simple
        def json_serializer(obj):
            if isinstance(obj, bytes):
                return base64.b64encode(obj).decode("ascii")
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        # Create a 3D vector
        vector_bytes = struct.pack("3f", 1.0, 2.0, 3.0)
        record = {"id": 1, "embedding": vector_bytes}

        # Should serialize without error
        json_str = json.dumps(record, default=json_serializer)
        parsed = json.loads(json_str)

        assert isinstance(parsed["embedding"], str)

        # Verify we can decode back to original bytes
        decoded = base64.b64decode(parsed["embedding"])
        assert decoded == vector_bytes

    def test_decimal_scale_zero_preserved(self):
        """Test that decimal scale=0 is preserved, not treated as falsy."""
        from gpudb import GPUdbRecordColumn
        from kinetica_ray.type_utils import (
            kinetica_to_arrow_type,
        )

        # Create a decimal column with explicit scale=0. GPUdbRecordColumn
        # has no precision=/scale= constructor kwargs -- it derives them by
        # parsing "decimal(p,s)" out of column_properties.
        col = GPUdbRecordColumn(
            name="amount",
            column_type=GPUdbRecordColumn._ColumnType.STRING,
            column_properties=["decimal(10,0)"],  # Integer decimal (no decimal places)
        )

        arrow_type = kinetica_to_arrow_type(col)

        # Verify it's a decimal type
        assert pa.types.is_decimal(arrow_type), f"Expected decimal, got {arrow_type}"
        # Verify scale is 0, not the default (4)
        assert arrow_type.scale == 0, (
            f"Expected scale=0, got {arrow_type.scale}. "
            "Scale=0 should not be treated as falsy."
        )
        assert arrow_type.precision == 10

    def test_decimal_scale_none_uses_default(self):
        """Test that decimal with scale=None uses the default scale.

        The real GPUdbRecordColumn always eagerly resolves .precision/.scale
        to concrete defaults once is_decimal is True -- it never actually
        leaves them None, so kinetica_to_arrow_type's own defensive
        `is not None` handling for that case can't be exercised through the
        real class. Test it directly with a minimal object exposing the
        same attributes kinetica_to_arrow_type's decimal branch reads.
        """
        from types import SimpleNamespace

        from gpudb import GPUdbRecordColumn
        from kinetica_ray.type_utils import (
            kinetica_to_arrow_type,
        )

        col = SimpleNamespace(
            name="amount",
            column_type=GPUdbRecordColumn._ColumnType.STRING,
            column_properties=["decimal"],
            is_decimal=True,
            precision=18,
            scale=None,  # Should use default
        )

        arrow_type = kinetica_to_arrow_type(col)

        assert pa.types.is_decimal(arrow_type)
        # Default scale is 4
        assert arrow_type.scale == GPUdbRecordColumn.DEFAULT_DECIMAL_SCALE

    def test_fixed_size_list_float_to_vector(self):
        """Test fixed-size list of floats maps to Kinetica vector type."""
        from kinetica_ray.type_utils import (
            arrow_to_kinetica_type,
        )

        # 3D float vector
        arrow_type = pa.list_(pa.float32(), 3)
        kinetica_type, props = arrow_to_kinetica_type(arrow_type)

        assert kinetica_type == "bytes", f"Expected 'bytes', got {kinetica_type}"
        assert "vector(3)" in props, f"Expected 'vector(3)' in props, got {props}"

    def test_fixed_size_list_double_to_vector(self):
        """Test fixed-size list of doubles maps to Kinetica vector type."""
        from kinetica_ray.type_utils import (
            arrow_to_kinetica_type,
        )

        # 128D double vector
        arrow_type = pa.list_(pa.float64(), 128)
        kinetica_type, props = arrow_to_kinetica_type(arrow_type)

        assert kinetica_type == "bytes"
        assert "vector(128)" in props

    def test_fixed_size_list_int_to_array(self):
        """Test fixed-size list of non-floats maps to Kinetica array type."""
        from kinetica_ray.type_utils import (
            arrow_to_kinetica_type,
        )

        # Fixed-size int array
        arrow_type = pa.list_(pa.int32(), 4)
        kinetica_type, props = arrow_to_kinetica_type(arrow_type)

        assert kinetica_type == "string", f"Expected 'string', got {kinetica_type}"
        assert "array(int,4)" in props, f"Expected 'array(int,4)' in props, got {props}"

    def test_variable_list_to_array(self):
        """Test variable-length list maps to Kinetica array type without size."""
        from kinetica_ray.type_utils import (
            arrow_to_kinetica_type,
        )

        # Variable-length list
        arrow_type = pa.list_(pa.float64())
        kinetica_type, props = arrow_to_kinetica_type(arrow_type)

        assert kinetica_type == "string"
        # Should be array(double) without size
        assert "array(double)" in props, (
            f"Expected 'array(double)' in props, got {props}"
        )

    def test_convert_arrow_batch_null_columns(self):
        """Test convert_arrow_batch_to_records handles None columns gracefully."""
        from kinetica_ray.type_utils import (
            convert_arrow_batch_to_records,
        )

        # Create a simple batch
        batch = pa.RecordBatch.from_pydict(
            {
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
            }
        )

        # Should not raise TypeError when columns is None
        records = convert_arrow_batch_to_records(batch, None)

        assert len(records) == 3
        assert records[0]["id"] == 1
        assert records[0]["name"] == "Alice"

    def test_convert_arrow_batch_vector_invalid_values_error(self):
        """Test that vector serialization with invalid values raises ValueError."""
        from gpudb import GPUdbRecordColumn
        from kinetica_ray.type_utils import (
            convert_arrow_batch_to_records,
        )

        # Create a batch with a list that should be treated as a vector
        # but contains non-float values
        batch = pa.RecordBatch.from_pydict(
            {
                "id": [1],
                "embedding": [["not", "floats", "here"]],  # Strings instead of floats
            }
        )

        # Create column definitions with vector type
        columns = [
            GPUdbRecordColumn(
                name="id",
                column_type=GPUdbRecordColumn._ColumnType.INT,
                column_properties=[],
            ),
            GPUdbRecordColumn(
                name="embedding",
                column_type=GPUdbRecordColumn._ColumnType.BYTES,
                column_properties=["vector(3)"],
            ),
        ]

        # Should raise ValueError with helpful message including column name
        with pytest.raises(ValueError, match="embedding"):
            convert_arrow_batch_to_records(batch, columns)


# ============================================================================
# Datasource Validation Tests
# ============================================================================


class TestKineticaDatasourceValidation:
    """Tests for input validation in KineticaDatasource."""

    def test_base_class_attributes_initialized(self):
        """Test that base class mixin attributes are properly initialized.

        KineticaDatasource must call super().__init__() to initialize
        _predicate_expr from _DatasourcePredicatePushdownMixin.
        """
        with patch("kinetica_ray.datasource.KineticaDatasource._init_client"):
            ds = KineticaDatasource(
                url="http://localhost:9191",
                table_name="test_table",
            )

            # These attributes are set by base class __init__
            # If super().__init__() wasn't called, these would raise AttributeError
            assert hasattr(ds, "_predicate_expr")
            assert ds._predicate_expr is None  # Initial value

    def test_invalid_sort_order_rejected(self):
        """Test that invalid sort_order values are rejected."""
        with pytest.raises(ValueError, match="Invalid sort_order"):
            KineticaDatasource(
                url="http://localhost:9191",
                table_name="test_table",
                sort_order="invalid",
            )

    def test_valid_sort_orders_accepted(self):
        """Test that valid sort_order values are accepted."""
        with patch("kinetica_ray.datasource.KineticaDatasource._init_client"):
            # ascending should work
            ds1 = KineticaDatasource(
                url="http://localhost:9191",
                table_name="test_table",
                sort_order="ascending",
            )
            assert ds1._sort_order == "ascending"

            # descending should work
            ds2 = KineticaDatasource(
                url="http://localhost:9191",
                table_name="test_table",
                sort_order="descending",
            )
            assert ds2._sort_order == "descending"


# ============================================================================
# Datasink Validation Tests
# ============================================================================


class TestKineticaDatasinkValidation:
    """Tests for input validation in KineticaDatasink."""

    def test_base_class_attributes_initialized(self):
        """Test that base class is properly initialized.

        KineticaDatasink must call super().__init__() to ensure
        the Datasink base class initializes any required state.
        """
        with patch("kinetica_ray.datasink.KineticaDatasink._init_client"):
            sink = KineticaDatasink(
                url="http://localhost:9191",
                table_name="test_table",
            )

            # Verify the datasink is a proper Datasink instance
            # and that super().__init__() was called (no AttributeError)
            from ray.data.datasource.datasink import Datasink

            assert isinstance(sink, Datasink)


# ============================================================================
# Datasink Serialization Tests
# ============================================================================


class TestKineticaDatasinkSerialization:
    """Tests for column serialization in KineticaDatasink."""

    def test_decimal_columns_preserve_precision_scale(self):
        """Test that decimal column precision/scale survives serialization."""
        from gpudb import GPUdbRecordColumn

        with patch("kinetica_ray.datasink.KineticaDatasink._init_client"):
            ds = KineticaDatasink(
                url="http://localhost:9191",
                table_name="test_table",
            )

            # Create a decimal column with specific precision and scale.
            # GPUdbRecordColumn has no precision=/scale= constructor kwargs --
            # it derives them by parsing "decimal(p,s)" out of column_properties.
            decimal_col = GPUdbRecordColumn(
                name="amount",
                column_type=GPUdbRecordColumn._ColumnType.STRING,
                column_properties=["decimal(10,2)"],
                is_nullable=False,
            )

            # Serialize and deserialize
            dicts = ds._columns_to_dicts([decimal_col])
            restored = ds._dicts_to_columns(dicts)

            # Verify precision and scale are preserved
            assert restored[0].precision == 10
            assert restored[0].scale == 2

    def test_non_decimal_columns_no_precision_scale(self):
        """Test that non-decimal columns don't include precision/scale."""
        from gpudb import GPUdbRecordColumn

        with patch("kinetica_ray.datasink.KineticaDatasink._init_client"):
            ds = KineticaDatasink(
                url="http://localhost:9191",
                table_name="test_table",
            )

            # Create a regular string column
            string_col = GPUdbRecordColumn(
                name="name",
                column_type=GPUdbRecordColumn._ColumnType.STRING,
                column_properties=[],
                is_nullable=True,
            )

            # Serialize
            dicts = ds._columns_to_dicts([string_col])

            # Verify no precision/scale in dict (they weren't set)
            assert "precision" not in dicts[0] or dicts[0].get("precision") is None
            assert "scale" not in dicts[0] or dicts[0].get("scale") is None


# ============================================================================
# GPUdbTable Creation Helper Tests
# ============================================================================


class TestTryCreateGpudbTable:
    """Tests for _try_create_gpudb_table helper method."""

    @patch.object(KineticaDatasink, "_init_client")
    @patch.object(KineticaDatasink, "_create_gpudb_table")
    def test_success_returns_gpudb_table(
        self, mock_create_gpudb_table, mock_init_client
    ):
        """Test that successful creation returns the GPUdbTable."""
        mock_client = MagicMock()
        mock_init_client.return_value = mock_client

        mock_gpudb_table = MagicMock()
        mock_create_gpudb_table.return_value = mock_gpudb_table

        ds = KineticaDatasink(
            url="http://localhost:9191",
            table_name="test_table",
            use_multihead=False,
        )

        result = ds._try_create_gpudb_table(mock_client)

        assert result == mock_gpudb_table
        mock_create_gpudb_table.assert_called_once_with(mock_client, table_exists=False)

    @patch.object(KineticaDatasink, "_init_client")
    @patch.object(KineticaDatasink, "_create_gpudb_table")
    def test_failure_with_multihead_raises(
        self, mock_create_gpudb_table, mock_init_client
    ):
        """Test that failure with multihead=True raises RuntimeError."""
        mock_client = MagicMock()
        mock_init_client.return_value = mock_client

        mock_create_gpudb_table.side_effect = Exception("Connection failed")

        ds = KineticaDatasink(
            url="http://localhost:9191",
            table_name="test_table",
            use_multihead=True,
        )

        with pytest.raises(RuntimeError, match="multihead ingest"):
            ds._try_create_gpudb_table(mock_client)

    @patch.object(KineticaDatasink, "_init_client")
    @patch.object(KineticaDatasink, "_create_gpudb_table")
    def test_failure_without_multihead_returns_none(
        self, mock_create_gpudb_table, mock_init_client
    ):
        """Test that failure with multihead=False returns None and logs warning."""
        mock_client = MagicMock()
        mock_init_client.return_value = mock_client

        mock_create_gpudb_table.side_effect = Exception("Connection failed")

        ds = KineticaDatasink(
            url="http://localhost:9191",
            table_name="test_table",
            use_multihead=False,
        )

        result = ds._try_create_gpudb_table(mock_client)

        assert result is None


# ============================================================================
# Deferred Table Creation Tests
# ============================================================================


class TestKineticaDatasinkTableCreation:
    """Tests for table creation in KineticaDatasink.

    Table DDL (CREATE/DROP) is performed in on_write_start(), following
    Ray's datasink contract where side effects should not occur in __init__.
    """

    @patch.object(KineticaDatasink, "_init_client")
    @patch.object(KineticaDatasink, "_table_exists")
    @patch.object(KineticaDatasink, "_create_table")
    @patch.object(KineticaDatasink, "_create_gpudb_table")
    @patch("kinetica_ray.type_utils.arrow_schema_to_kinetica_columns")
    @patch("kinetica_ray.type_utils.convert_arrow_batch_to_records")
    def test_on_write_start_creates_table(
        self,
        mock_convert,
        mock_arrow_to_kinetica,
        mock_create_gpudb_table,
        mock_create_table,
        mock_table_exists,
        mock_init_client,
    ):
        """Test that on_write_start creates the table when it doesn't exist.

        Table DDL (CREATE/DROP) is performed in on_write_start(), not in
        __init__ or write(), following Ray's datasink contract.
        """
        mock_client = MagicMock()
        mock_init_client.return_value = mock_client
        mock_table_exists.return_value = False  # Table doesn't exist

        mock_arrow_to_kinetica.return_value = []
        mock_convert.return_value = [{"id": 1}]
        mock_create_gpudb_table.return_value = None

        # Create datasink with schema (table creation happens in on_write_start)
        schema = pa.schema([pa.field("id", pa.int64())])
        ds = KineticaDatasink(
            url="http://localhost:9191",
            table_name="test_table",
            mode=KineticaSinkMode.APPEND,
            schema=schema,
            use_multihead=False,
        )

        # Call on_write_start to trigger table creation
        # (This is what Ray Data framework calls before write())
        ds.on_write_start()

        # Verify table was created in on_write_start
        mock_create_table.assert_called_once()

        # Create test data and write
        rb = pa.record_batch([pa.array([1])], names=["id"])
        block_data = pa.Table.from_batches([rb])

        ctx = TaskContext(task_idx=0, op_name="test_write")
        ds.write([block_data], ctx=ctx)

    @patch.object(KineticaDatasink, "_init_client")
    @patch.object(KineticaDatasink, "_table_exists")
    @patch.object(KineticaDatasink, "_create_table")
    @patch("kinetica_ray.type_utils.arrow_schema_to_kinetica_columns")
    def test_on_write_start_error_propagated(
        self,
        mock_arrow_to_kinetica,
        mock_create_table,
        mock_table_exists,
        mock_init_client,
    ):
        """Test that errors during table creation in on_write_start are propagated."""
        mock_client = MagicMock()
        mock_init_client.return_value = mock_client
        mock_table_exists.return_value = False  # Table doesn't exist

        # Simulate a real error during table creation
        mock_create_table.side_effect = Exception("Connection refused")

        mock_arrow_to_kinetica.return_value = []

        # Create datasink with schema (table creation happens in on_write_start)
        schema = pa.schema([pa.field("id", pa.int64())])
        ds = KineticaDatasink(
            url="http://localhost:9191",
            table_name="test_table",
            mode=KineticaSinkMode.APPEND,
            schema=schema,
            use_multihead=False,
        )

        # Should raise the real error when on_write_start tries to create the table
        with pytest.raises(Exception, match="Connection refused"):
            ds.on_write_start()


# ============================================================================
# Module Import Tests
# ============================================================================


class TestModuleImports:
    """Tests for module imports and API exposure."""

    def test_read_kinetica_importable(self):
        """Test read_kinetica is importable from kinetica_ray."""
        from kinetica_ray import read_kinetica

        assert callable(read_kinetica)

    def test_read_kinetica_sql_importable(self):
        """Test read_kinetica_sql is importable from kinetica_ray."""
        from kinetica_ray import read_kinetica_sql

        assert callable(read_kinetica_sql)

    def test_write_kinetica_importable(self):
        """Test write_kinetica is importable from kinetica_ray."""
        from kinetica_ray import write_kinetica

        assert callable(write_kinetica)

    def test_write_kinetica_sql_importable(self):
        """Test write_kinetica_sql is importable from kinetica_ray."""
        from kinetica_ray import write_kinetica_sql

        assert callable(write_kinetica_sql)

    def test_datasource_importable(self):
        """Test KineticaDatasource is importable."""
        from kinetica_ray.datasource import (
            KineticaDatasource,
        )

        assert KineticaDatasource is not None

    def test_datasink_importable(self):
        """Test KineticaDatasink is importable."""
        from kinetica_ray.datasink import (
            KineticaDatasink,
            KineticaSinkMode,
            KineticaTableSettings,
        )

        assert KineticaDatasink is not None
        assert KineticaSinkMode is not None
        assert KineticaTableSettings is not None

    def test_sql_connection_factory_importable(self):
        """Test SQL connection factory is importable."""
        from kinetica_ray.sql_connection import (
            KineticaConnectionFactory,
            create_kinetica_connection_factory,
        )

        assert KineticaConnectionFactory is not None
        assert callable(create_kinetica_connection_factory)

    def test_create_gpudb_client_importable(self):
        """Test create_gpudb_client shared factory is importable."""
        from kinetica_ray.type_utils import (
            create_gpudb_client,
        )

        assert callable(create_gpudb_client)


# ============================================================================
# Client Factory Tests
# ============================================================================


class TestCreateGpudbClient:
    """Tests for the shared create_gpudb_client factory function."""

    def test_create_client_basic(self, patch_gpudb):
        """Test basic client creation with minimal parameters."""
        from kinetica_ray.type_utils import (
            create_gpudb_client,
        )

        client = create_gpudb_client(url="http://localhost:9191")
        assert client is not None

    def test_create_client_with_auth(self, patch_gpudb):
        """Test client creation with authentication."""
        from kinetica_ray.type_utils import (
            create_gpudb_client,
        )

        client = create_gpudb_client(
            url="http://localhost:9191",
            username="admin",
            password="password123",
        )
        assert client is not None

    def test_create_client_with_options(self, patch_gpudb):
        """Test client creation with additional options."""
        from kinetica_ray.type_utils import (
            create_gpudb_client,
        )

        client = create_gpudb_client(
            url="http://localhost:9191",
            username="admin",
            password="password",
            options={"timeout": 30000},
        )
        assert client is not None

    def test_datasource_uses_shared_factory(self):
        """Test that KineticaDatasource uses the shared factory."""
        with patch("kinetica_ray.type_utils.create_gpudb_client") as mock_factory:
            mock_factory.return_value = MagicMock()

            ds = KineticaDatasource(
                url="http://localhost:9191",
                table_name="test_table",
                username="admin",
                password="password",
            )

            ds._init_client()

            mock_factory.assert_called_once_with(
                url="http://localhost:9191",
                username="admin",
                password="password",
                options={},
            )

    def test_datasink_uses_shared_factory(self):
        """Test that KineticaDatasink uses the shared factory."""
        with patch("kinetica_ray.type_utils.create_gpudb_client") as mock_factory:
            mock_factory.return_value = MagicMock()

            ds = KineticaDatasink(
                url="http://localhost:9191",
                table_name="test_table",
                username="admin",
                password="password",
            )

            ds._init_client()

            mock_factory.assert_called_once_with(
                url="http://localhost:9191",
                username="admin",
                password="password",
                options={},
            )


# ============================================================================
# Integration Tests (require Kinetica server)
# ============================================================================


class TestKineticaIntegration:
    """Integration tests requiring a running Kinetica server.

    Point these at a real server with --kinetica-url (and optionally
    --kinetica-username / --kinetica-password), or the KINETICA_URL /
    KINETICA_USER / KINETICA_PASS environment variables. Tests skip at
    runtime if no URL is available either way.
    """

    @pytest.fixture
    def connection_params(self, kinetica_connection_params):
        return kinetica_connection_params

    def test_read_simple_query(self, connection_params):
        """Test reading data from a Kinetica table."""
        import kinetica_ray as kr
        import ray

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        try:
            ds = kr.read_kinetica(
                table_name="ki_home.ki_catalog_ddl",  # System table that should exist
                **connection_params,
                limit=10,
            )

            count = ds.count()
            assert count >= 0  # Table might be empty but query should work

        finally:
            pass

    def test_write_and_read_roundtrip(self, connection_params):
        """Test writing and reading back data."""
        import kinetica_ray as kr
        import ray

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        table_name = "test_ray_roundtrip"

        try:
            # Create test data
            data = [
                {"id": 1, "name": "Alice", "value": 100.5},
                {"id": 2, "name": "Bob", "value": 200.75},
                {"id": 3, "name": "Charlie", "value": 300.25},
            ]
            ds = ray.data.from_items(data)

            # Write to Kinetica
            kr.write_kinetica(
                ds,
                table_name=table_name,
                mode="overwrite",
                **connection_params,
            )

            # Read back
            read_ds = kr.read_kinetica(
                table_name=table_name,
                **connection_params,
            )

            # Verify
            assert read_ds.count() == 3

        finally:
            # Cleanup: drop the test table
            try:
                from gpudb import GPUdb

                client = GPUdb(
                    host=connection_params["url"],
                    username=connection_params.get("username"),
                    password=connection_params.get("password"),
                )
                client.clear_table(table_name=table_name)
            except Exception:
                pass

    def test_read_with_filter(self, connection_params):
        """Test reading with a filter expression."""
        import kinetica_ray as kr
        import ray

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        table_name = "test_ray_filter"

        try:
            # Create test data
            data = [{"id": i, "value": i * 10} for i in range(100)]
            ds = ray.data.from_items(data)

            # Write to Kinetica
            kr.write_kinetica(
                ds,
                table_name=table_name,
                mode="overwrite",
                **connection_params,
            )

            # Read with filter
            read_ds = kr.read_kinetica(
                table_name=table_name,
                filter_expression="value >= 500",
                **connection_params,
            )

            # Verify filter worked
            count = read_ds.count()
            assert count == 50  # ids 50-99 have values >= 500

        finally:
            # Cleanup
            try:
                from gpudb import GPUdb

                client = GPUdb(
                    host=connection_params["url"],
                    username=connection_params.get("username"),
                    password=connection_params.get("password"),
                )
                client.clear_table(table_name=table_name)
            except Exception:
                pass

    def test_read_specific_columns(self, connection_params):
        """Test reading specific columns."""
        import kinetica_ray as kr
        import ray

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        table_name = "test_ray_columns"

        try:
            # Create test data
            data = [
                {"id": 1, "name": "Alice", "value": 100.5, "extra": "data"},
            ]
            ds = ray.data.from_items(data)

            # Write to Kinetica
            kr.write_kinetica(
                ds,
                table_name=table_name,
                mode="overwrite",
                **connection_params,
            )

            # Read specific columns
            read_ds = kr.read_kinetica(
                table_name=table_name,
                columns=["id", "name"],
                **connection_params,
            )

            # Verify only requested columns present
            row = read_ds.take(1)[0]
            assert "id" in row
            assert "name" in row
            assert "value" not in row
            assert "extra" not in row

        finally:
            # Cleanup
            try:
                from gpudb import GPUdb

                client = GPUdb(
                    host=connection_params["url"],
                    username=connection_params.get("username"),
                    password=connection_params.get("password"),
                )
                client.clear_table(table_name=table_name)
            except Exception:
                pass


# ============================================================================
# Full Implementation Integration Test (requires Kinetica server)
# ============================================================================


class TestKineticaFullImplementation:
    """One end-to-end integration test exercising the full kinetica_ray
    public API against a real Kinetica server: write_kinetica (overwrite,
    with table_settings/primary_keys/shard_keys), write_kinetica_sql
    (append via SQL INSERT), read_kinetica (columns, filter_expression,
    sort_by, and hash-partitioned parallel reads via partition_column),
    and read_kinetica_sql (an aggregate query) -- across every Kinetica
    column type kinetica_ray's type_utils supports.
    """

    NUM_ROWS = 40

    @pytest.fixture
    def connection_params(self, kinetica_connection_params):
        return kinetica_connection_params

    @staticmethod
    def _build_source_table():
        """Build a pyarrow table covering every Kinetica-supported type."""
        import datetime as dt
        from decimal import Decimal

        n = TestKineticaFullImplementation.NUM_ROWS

        columns = {
            "id": [i for i in range(n)],
            "region": [f"region_{i % 4}" for i in range(n)],
            "is_active": [i % 2 == 0 for i in range(n)],
            "small_num": [i - 20 for i in range(n)],  # fits int8
            "medium_num": [i * 10 for i in range(n)],  # fits int16
            "count32": [i * 1000 for i in range(n)],
            "big_count": [i * 10_000_000_000 for i in range(n)],
            "ratio32": [i * 0.5 for i in range(n)],
            "ratio64": [i * 0.25 for i in range(n)],
            "price": [Decimal(f"{i}.{i % 100:02d}") for i in range(n)],
            "label": [f"label_{i}" for i in range(n)],
            "event_date": [
                dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(n)
            ],
            "event_time": [
                dt.time(hour=i % 24, minute=(i * 3) % 60, second=(i * 7) % 60)
                for i in range(n)
            ],
            "event_datetime": [
                dt.datetime(2024, 1, 1) + dt.timedelta(hours=i) for i in range(n)
            ],
            "tags": [[f"tag{i}", f"tag{i + 1}"] for i in range(n)],
            "embedding": [
                [float(i), float(i + 1), float(i + 2), float(i + 3)] for i in range(n)
            ],
            "attributes": [{"k1": f"v{i}", "k2": i} for i in range(n)],
        }

        schema = pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field("region", pa.string()),
                pa.field("is_active", pa.bool_()),
                pa.field("small_num", pa.int8()),
                pa.field("medium_num", pa.int16()),
                pa.field("count32", pa.int32()),
                pa.field("big_count", pa.int64()),
                pa.field("ratio32", pa.float32()),
                pa.field("ratio64", pa.float64()),
                pa.field("price", pa.decimal128(10, 2)),
                pa.field("label", pa.string()),
                pa.field("event_date", pa.date32()),
                pa.field("event_time", pa.time64("us")),
                pa.field("event_datetime", pa.timestamp("us")),
                pa.field("tags", pa.list_(pa.string())),
                pa.field("embedding", pa.list_(pa.float32(), 4)),
                pa.field(
                    "attributes",
                    pa.struct([("k1", pa.string()), ("k2", pa.int64())]),
                ),
            ]
        )

        arrays = [pa.array(columns[field.name], type=field.type) for field in schema]
        return pa.Table.from_arrays(arrays, schema=schema)

    def test_full_implementation(self, connection_params):
        """Round-trips a rich schema through every public read/write path."""
        import uuid

        import kinetica_ray as kr
        import ray

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        table_name = f"kinetica_ray_full_integration_{uuid.uuid4().hex[:8]}"
        n = self.NUM_ROWS

        try:
            # --- write_kinetica: create the table via overwrite, with
            # primary/shard keys, then verify a plain read round-trips
            # every column type correctly. ---
            source_table = self._build_source_table()
            ds = ray.data.from_arrow(source_table)

            kr.write_kinetica(
                ds,
                table_name=table_name,
                mode="overwrite",
                table_settings=kr.KineticaTableSettings(
                    primary_keys=["id"],
                    shard_keys=["region"],
                ),
                batch_size=10,
                use_multihead=True,
                **connection_params,
            )

            read_ds = kr.read_kinetica(
                table_name=table_name,
                sort_by="id",
                **connection_params,
            )
            rows = sorted(read_ds.take_all(), key=lambda r: r["id"])
            assert len(rows) == n

            for i, row in enumerate(rows):
                assert row["id"] == i
                assert row["region"] == f"region_{i % 4}"
                assert row["is_active"] == (i % 2 == 0)
                assert row["small_num"] == i - 20
                assert row["medium_num"] == i * 10
                assert row["count32"] == i * 1000
                assert row["big_count"] == i * 10_000_000_000
                assert row["ratio32"] == pytest.approx(i * 0.5)
                assert row["ratio64"] == pytest.approx(i * 0.25)
                assert float(row["price"]) == pytest.approx(i + (i % 100) / 100)
                assert row["label"] == f"label_{i}"
                assert row["tags"] == [f"tag{i}", f"tag{i + 1}"]
                assert list(row["embedding"]) == pytest.approx(
                    [float(i), float(i + 1), float(i + 2), float(i + 3)]
                )
                # struct columns round-trip through Kinetica as JSON text.
                attributes = json.loads(row["attributes"])
                assert attributes == {"k1": f"v{i}", "k2": i}

            # --- read_kinetica: columns + filter_expression + sort_by. ---
            filtered_ds = kr.read_kinetica(
                table_name=table_name,
                columns=["id", "count32"],
                filter_expression="count32 >= 20000",
                sort_by="id",
                **connection_params,
            )
            filtered_rows = filtered_ds.take_all()
            assert len(filtered_rows) == n - 20  # ids 20..39 have count32 >= 20000
            assert all("region" not in row for row in filtered_rows)
            assert all(row["count32"] >= 20000 for row in filtered_rows)

            # --- read_kinetica: hash-partitioned parallel reads. ---
            partitioned_ds = kr.read_kinetica(
                table_name=table_name,
                partition_column="id",
                override_num_blocks=4,
                **connection_params,
            )
            partitioned_ids = sorted(row["id"] for row in partitioned_ds.take_all())
            assert partitioned_ids == list(range(n))  # no dupes, none missing

            # --- write_kinetica_sql: append rows to the existing table. ---
            extra_rows = [
                {
                    "id": n,
                    "region": "region_extra",
                    "count32": 99000,
                },
                {
                    "id": n + 1,
                    "region": "region_extra",
                    "count32": 99000,
                },
            ]
            extra_ds = ray.data.from_items(extra_rows)
            kr.write_kinetica_sql(
                extra_ds,
                sql=f"INSERT INTO {table_name} (id, region, count32) VALUES (?, ?, ?)",
                **connection_params,
            )

            # --- read_kinetica_sql: an aggregate query over everything. ---
            agg_ds = kr.read_kinetica_sql(
                sql=f"SELECT region, COUNT(*) AS cnt, SUM(count32) AS total "
                f"FROM {table_name} GROUP BY region ORDER BY region",
                **connection_params,
            )
            # Normalize key case: SQL drivers vary in whether unquoted
            # identifiers/aliases come back lowercased, uppercased, or as
            # written -- the data values themselves are left untouched.
            agg_rows = {
                {k.lower(): v for k, v in row.items()}["region"]: {
                    k.lower(): v for k, v in row.items()
                }
                for row in agg_ds.take_all()
            }
            assert agg_rows["region_extra"]["cnt"] == 2
            assert agg_rows["region_extra"]["total"] == 99000 * 2
            assert sum(row["cnt"] for row in agg_rows.values()) == n + 2

        finally:
            try:
                from gpudb import GPUdb

                client = GPUdb(
                    host=connection_params["url"],
                    username=connection_params.get("username"),
                    password=connection_params.get("password"),
                )
                client.clear_table(
                    table_name=table_name,
                    options={"no_error_if_not_exists": "true"},
                )
            except Exception:
                pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
I/O operations for Kinetica-Ray integration.
"""

from typing import Any, Dict, List, Optional

from ray.data import Dataset, read_datasource, read_sql

from .datasink import KineticaDatasink, KineticaTableSettings
from .datasource import KineticaDatasource
from .sql_connection import create_kinetica_connection_factory


def read_kinetica(
    table_name: str,
    url: str,
    *,
    username: Optional[str] = None,
    password: Optional[str] = None,
    columns: Optional[List[str]] = None,
    filter_expression: Optional[str] = None,
    sort_by: Optional[str] = None,
    sort_order: str = "ascending",
    limit: Optional[int] = None,
    batch_size: int = 10000,
    use_multihead_io: bool = False,
    convert_special_types: bool = True,
    partition_column: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
    num_cpus: Optional[float] = None,
    num_gpus: Optional[float] = None,
    memory: Optional[float] = None,
    ray_remote_args: Optional[Dict[str, Any]] = None,
    concurrency: Optional[int] = None,
    override_num_blocks: Optional[int] = None,
) -> Dataset:
    """Create a :class:`~ray.data.Dataset` from a Kinetica database table.

    Kinetica is a distributed, in-memory analytical database designed for
    real-time analytics on streaming and historical data. This function reads
    data using Kinetica's native API with parallel pagination.

    Examples:
        >>> import kinetica_ray as kr
        >>> ds = kr.read_kinetica(  # doctest: +SKIP
        ...     table_name="transactions",
        ...     url="http://localhost:9191",
        ...     username="admin",
        ...     password="password",
        ...     filter_expression="amount > 1000",
        ...     columns=["id", "customer", "amount"],
        ... )

    Args:
        table_name: Name of the Kinetica table to read.
        url: URL of the Kinetica server (e.g., "http://localhost:9191").
        username: Authentication username.
        password: Authentication password.
        columns: Specific columns to read. None reads all columns.
        filter_expression: SQL WHERE clause filter (without WHERE keyword).
        sort_by: Column to sort by for consistent pagination.
        sort_order: "ascending" or "descending".
        limit: Maximum rows to read.
        batch_size: Records per API request for pagination. Default is 10,000.
        use_multihead_io: If True, enables multihead I/O for parallel reads
            from multiple Kinetica nodes. Can improve performance for large
            datasets on clustered deployments. Default is False.
        convert_special_types: If True, converts special types (arrays, JSON)
            on retrieval. Default is True.
        partition_column: Column name for hash-based partitioning in parallel
            reads. When specified, rows are deterministically assigned to read
            tasks using MOD(HASH(column), parallelism), guaranteeing each row
            is read by exactly one task. This enables safe parallel reads
            without requiring a unique sort key. Should be a column with good
            value distribution (e.g., primary key). Cannot be used with limit.
        options: Additional GPUdb client options.
        num_cpus: The number of CPUs to reserve for each parallel read worker.
        num_gpus: The number of GPUs to reserve for each parallel read worker.
        memory: The heap memory in bytes to reserve for each parallel read worker.
        ray_remote_args: kwargs passed to :func:`ray.remote` in the read tasks.
        concurrency: The maximum number of Ray tasks to run concurrently.
        override_num_blocks: Override the number of output blocks from all read tasks.

    Returns:
        :class:`~ray.data.Dataset` producing rows from the Kinetica table.
    """
    datasource = KineticaDatasource(
        url=url,
        table_name=table_name,
        username=username,
        password=password,
        columns=columns,
        filter_expression=filter_expression,
        sort_by=sort_by,
        sort_order=sort_order,
        limit=limit,
        batch_size=batch_size,
        use_multihead_io=use_multihead_io,
        convert_special_types=convert_special_types,
        partition_column=partition_column,
        options=options,
    )
    return read_datasource(
        datasource,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        memory=memory,
        ray_remote_args=ray_remote_args,
        concurrency=concurrency,
        override_num_blocks=override_num_blocks,
    )


def read_kinetica_sql(
    sql: str,
    url: str,
    *,
    username: Optional[str] = None,
    password: Optional[str] = None,
    oauth_token: Optional[str] = None,
    default_schema: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
    ray_remote_args: Optional[Dict[str, Any]] = None,
    concurrency: Optional[int] = None,
    override_num_blocks: Optional[int] = None,
) -> Dataset:
    """Create a :class:`~ray.data.Dataset` from a Kinetica SQL query.

    This function uses Kinetica's DB-API 2.0 compliant interface with Ray Data's
    native ``read_sql`` function. It's ideal for complex SQL queries including
    JOINs, aggregations, and subqueries.

    For simple table reads with parallel pagination, consider using
    :func:`read_kinetica` instead which uses Kinetica's native API.

    Examples:
        >>> import kinetica_ray as kr
        >>> ds = kr.read_kinetica_sql(  # doctest: +SKIP
        ...     sql="SELECT t.id, t.customer, SUM(t.amount) as total "
        ...         "FROM transactions t "
        ...         "JOIN customers c ON t.customer_id = c.id "
        ...         "GROUP BY t.id, t.customer",
        ...     url="http://localhost:9191",
        ...     username="admin",
        ...     password="password",
        ... )

    Args:
        sql: SQL query to execute.
        url: URL of the Kinetica server (e.g., "http://localhost:9191").
        username: Authentication username.
        password: Authentication password.
        oauth_token: OAuth token for authentication (alternative to username/password).
        default_schema: Default schema to use for queries.
        options: Additional GPUdb client options.
        ray_remote_args: kwargs passed to :func:`ray.remote` in the read tasks.
        concurrency: The maximum number of Ray tasks to run concurrently.
        override_num_blocks: Override the number of output blocks.

    Returns:
        :class:`~ray.data.Dataset` producing rows from the query results.
    """
    connection_factory = create_kinetica_connection_factory(
        url=url,
        username=username,
        password=password,
        oauth_token=oauth_token,
        default_schema=default_schema,
        options=options,
    )

    read_kwargs: Dict[str, Any] = {
        "sql": sql,
        "connection_factory": connection_factory,
    }

    if ray_remote_args is not None:
        read_kwargs["ray_remote_args"] = ray_remote_args

    if concurrency is not None:
        read_kwargs["concurrency"] = concurrency

    if override_num_blocks is not None:
        read_kwargs["override_num_blocks"] = override_num_blocks

    return read_sql(**read_kwargs)


def write_kinetica(
    ds: Dataset,
    table_name: str,
    url: str,
    *,
    username: Optional[str] = None,
    password: Optional[str] = None,
    mode: str = "append",
    table_settings: Optional[KineticaTableSettings] = None,
    batch_size: int = 10000,
    use_multihead: bool = True,
    options: Optional[Dict[str, Any]] = None,
    ray_remote_args: Optional[Dict[str, Any]] = None,
    concurrency: Optional[int] = None,
) -> None:
    """Write a :class:`~ray.data.Dataset` to a Kinetica database table.

    Kinetica is a distributed, in-memory analytical database. This function
    uses Kinetica's multihead ingestion for optimal write performance when
    writing to a distributed cluster.

    Examples:
        >>> import ray
        >>> import kinetica_ray as kr
        >>> ds = ray.data.from_items([  # doctest: +SKIP
        ...     {"id": 1, "name": "Alice", "amount": 100.0},
        ...     {"id": 2, "name": "Bob", "amount": 200.0},
        ... ])
        >>> kr.write_kinetica(  # doctest: +SKIP
        ...     ds,
        ...     table_name="transactions",
        ...     url="http://localhost:9191",
        ...     username="admin",
        ...     password="password",
        ...     mode="append",
        ... )

    Args:
        ds: The dataset to write.
        table_name: Target table name.
        url: Kinetica server URL (e.g., "http://localhost:9191").
        username: Authentication username.
        password: Authentication password.
        mode: Write mode - "create", "append", or "overwrite".
        table_settings: Kinetica-specific table options (KineticaTableSettings).
        batch_size: Records per ingestion batch. Default is 10,000.
        use_multihead: Enable multihead ingestion for parallelism.
        options: Additional GPUdb client options.
        ray_remote_args: Keyword arguments passed to :func:`ray.remote`.
        concurrency: Maximum concurrent write tasks.
    """
    # KineticaDatasink expects a PyArrow schema, not a Ray Data Schema.
    ray_schema = ds.schema()
    pa_schema = (
        ray_schema.base_schema if hasattr(ray_schema, "base_schema") else ray_schema
    )

    datasink = KineticaDatasink(
        url=url,
        table_name=table_name,
        username=username,
        password=password,
        mode=mode,
        schema=pa_schema,
        table_settings=table_settings,
        batch_size=batch_size,
        use_multihead=use_multihead,
        options=options,
    )
    ds.write_datasink(
        datasink,
        ray_remote_args=ray_remote_args,
        concurrency=concurrency,
    )


def write_kinetica_sql(
    ds: Dataset,
    sql: str,
    url: str,
    *,
    username: Optional[str] = None,
    password: Optional[str] = None,
    oauth_token: Optional[str] = None,
    default_schema: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
    ray_remote_args: Optional[Dict[str, Any]] = None,
    concurrency: Optional[int] = None,
) -> None:
    """Write a :class:`~ray.data.Dataset` to Kinetica using SQL INSERT.

    This function uses Kinetica's DB-API 2.0 interface with Ray Data's native
    ``write_sql`` method. The target table must already exist.

    For automatic table creation and multihead ingestion, use
    :func:`write_kinetica` instead.

    Examples:
        >>> import ray
        >>> import kinetica_ray as kr
        >>> ds = ray.data.from_items([  # doctest: +SKIP
        ...     {"id": 1, "name": "Alice", "value": 100.0},
        ...     {"id": 2, "name": "Bob", "value": 200.0},
        ... ])
        >>> kr.write_kinetica_sql(  # doctest: +SKIP
        ...     ds,
        ...     sql="INSERT INTO my_table (id, name, value) VALUES (?, ?, ?)",
        ...     url="http://localhost:9191",
        ...     username="admin",
        ...     password="password",
        ... )

    Args:
        ds: The dataset to write.
        sql: SQL INSERT statement with parameter placeholders.
            Use '?' for placeholders (qmark paramstyle).
        url: Kinetica server URL (e.g., "http://localhost:9191").
        username: Authentication username.
        password: Authentication password.
        oauth_token: OAuth token for authentication (alternative to username/password).
        default_schema: Default schema for queries.
        options: Additional GPUdb client options.
        ray_remote_args: Keyword arguments passed to :func:`ray.remote`.
        concurrency: Maximum concurrent write tasks.
    """
    connection_factory = create_kinetica_connection_factory(
        url=url,
        username=username,
        password=password,
        oauth_token=oauth_token,
        default_schema=default_schema,
        options=options,
    )

    write_kwargs: Dict[str, Any] = {
        "sql": sql,
        "connection_factory": connection_factory,
    }

    if ray_remote_args is not None:
        write_kwargs["ray_remote_args"] = ray_remote_args

    if concurrency is not None:
        write_kwargs["concurrency"] = concurrency

    ds.write_sql(**write_kwargs)

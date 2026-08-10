# Kinetica-Ray

A Python library that provides integration between [Ray](https://ray.io/) and
[Kinetica](https://www.kinetica.com/) for distributed data processing.

Kinetica is a distributed, in-memory analytical database. This package lets
Ray Data read from and write to Kinetica tables in parallel, without requiring
any changes to Ray itself.

## Installation

```bash
pip install -e .
```

## Usage

```python
import ray
import kinetica_ray as kr

# Read from a Kinetica table.
ds = kr.read_kinetica(
    table_name="transactions",
    url="http://localhost:9191",
    username="admin",
    password="password",
    filter_expression="amount > 1000",
)

# Write a dataset to a Kinetica table.
ds = ray.data.from_items([{"id": 1, "name": "Alice", "amount": 100.0}])
kr.write_kinetica(
    ds,
    table_name="transactions",
    url="http://localhost:9191",
    username="admin",
    password="password",
    mode="append",
)
```

For SQL-based access (arbitrary queries, JOINs, existing-table INSERTs), use
`read_kinetica_sql` / `write_kinetica_sql`, which go through Kinetica's DB-API
2.0 interface and Ray Data's native `read_sql` / `write_sql`.

## Development

```bash
pip install -e ".[dev]"
pytest tests/
```

Most tests run against mocks and need no live server. A handful of
integration tests (including one comprehensive end-to-end test covering
every read/write path and column type) require a real Kinetica server and
are skipped unless you point them at one, either via CLI options:

```bash
pytest tests/ --kinetica-url=http://localhost:9191 \
    --kinetica-username=admin --kinetica-password=secret
```

or environment variables:

```bash
KINETICA_URL=http://localhost:9191 KINETICA_USER=admin KINETICA_PASS=secret \
    pytest tests/
```

<h3 align="center" style="margin:0px">
	<img width="200" src="https://kinetica-web-assets.s3.us-east-1.amazonaws.com/assets/kinetica_logo_gray.svg" alt="Kinetica Logo"/>
</h3>
<h5 align="center" style="margin:0px">
	<a href="https://www.kinetica.com/">Website</a>
	|
	<a href="https://docs.kinetica.com/">Docs</a>
	|
	<a href="https://join.slack.com/t/kinetica-community/shared_invite/zt-1bt9x3mvr-uMKrXlSDXfy3oU~sKi84qg">Community Slack</a>   
</h5>


## Contents ##

* [Overview](#overview)
* [Build/Run](#build-and-run)
* [Example](#example)
* [Development](#development)
* [Support](#support)
* [Contact Us](#contact-us)


## Overview ##

Kinetica-Ray provides integration between [Ray](https://ray.io/) and
[Kinetica](https://www.kinetica.com/) for distributed data processing.

Kinetica is a distributed, in-memory analytical database. This package lets
Ray Data read from and write to Kinetica tables in parallel, without
requiring any changes to Ray itself.

* `read_kinetica` / `write_kinetica` use Kinetica's native API (parallel
  pagination on read, multihead ingestion on write).
* `read_kinetica_sql` / `write_kinetica_sql` go through Kinetica's DB-API
  2.0 interface and Ray Data's native `read_sql` / `write_sql`, for
  arbitrary queries, JOINs, and inserting into an existing table.


## Build and Run ##

```bash
pip install -e .
```


## Example ##

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

See [`examples/basic_usage.py`](examples/basic_usage.py) for a complete
write-then-read example.


## Development ##

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


## Support ##

For bugs, please submit an
[issue on Github](https://github.com/kineticadb/kinetica-ray/issues).

For support, you can post on
[stackoverflow](https://stackoverflow.com/questions/tagged/kinetica) under the
``kinetica`` tag or
[Slack](https://join.slack.com/t/kinetica-community/shared_invite/zt-1bt9x3mvr-uMKrXlSDXfy3oU~sKi84qg).


## Contact Us ##

* Ask a question on Slack:
  [Slack](https://join.slack.com/t/kinetica-community/shared_invite/zt-1bt9x3mvr-uMKrXlSDXfy3oU~sKi84qg)
* Follow on GitHub:
  [Follow @kineticadb](https://github.com/kineticadb) 
* Email us:  <support@kinetica.com>
* Visit:  <https://www.kinetica.com/contact/>

"""
Pytest configuration for kinetica_ray tests.

Adds command-line options for pointing the integration tests at a real
Kinetica server, e.g.:

    pytest tests/ --kinetica-url=http://localhost:9191 \
        --kinetica-username=admin --kinetica-password=secret
"""

import os

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--kinetica-url",
        action="store",
        default=None,
        help="URL of a running Kinetica server, e.g. http://localhost:9191. "
        "Required to run the Kinetica integration tests. Falls back to the "
        "KINETICA_URL environment variable if not given.",
    )
    parser.addoption(
        "--kinetica-username",
        action="store",
        default=None,
        help="Username for the Kinetica server. Falls back to KINETICA_USER.",
    )
    parser.addoption(
        "--kinetica-password",
        action="store",
        default=None,
        help="Password for the Kinetica server. Falls back to KINETICA_PASS.",
    )


@pytest.fixture(scope="session")
def kinetica_connection_params(request):
    """Connection parameters for the Kinetica integration tests.

    Resolved from --kinetica-url/--kinetica-username/--kinetica-password
    command-line options, falling back to the KINETICA_URL/KINETICA_USER/
    KINETICA_PASS environment variables. Skips the requesting test if no
    URL was provided by either mechanism.
    """
    url = request.config.getoption("--kinetica-url") or os.environ.get("KINETICA_URL")
    if not url:
        pytest.skip(
            "Kinetica integration tests require a server: pass "
            "--kinetica-url (and optionally --kinetica-username / "
            "--kinetica-password), or set KINETICA_URL (and optionally "
            "KINETICA_USER / KINETICA_PASS)."
        )

    username = request.config.getoption("--kinetica-username") or os.environ.get(
        "KINETICA_USER", "admin"
    )
    password = request.config.getoption("--kinetica-password") or os.environ.get(
        "KINETICA_PASS", ""
    )

    return {"url": url, "username": username, "password": password}

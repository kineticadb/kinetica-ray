"""
Kinetica-Ray: Ray integration for the Kinetica database.

This package provides integration between Ray Data and Kinetica, enabling
efficient parallel reads and writes between Ray datasets and Kinetica tables.
"""

__version__ = "0.1.0"
__author__ = "Kinetica"
__email__ = "support@kinetica.com"

from .datasink import KineticaDatasink, KineticaSinkMode, KineticaTableSettings
from .datasource import KineticaDatasource
from .io import read_kinetica, read_kinetica_sql, write_kinetica, write_kinetica_sql
from .sql_connection import KineticaConnectionFactory, create_kinetica_connection_factory

__all__ = [
    "read_kinetica",
    "read_kinetica_sql",
    "write_kinetica",
    "write_kinetica_sql",
    "KineticaDatasource",
    "KineticaDatasink",
    "KineticaSinkMode",
    "KineticaTableSettings",
    "KineticaConnectionFactory",
    "create_kinetica_connection_factory",
]

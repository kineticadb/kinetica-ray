"""
Basic usage example for Kinetica-Ray integration.

This example demonstrates how to:
1. Write a Ray dataset to a Kinetica table.
2. Read Kinetica data back into Ray.
"""

import kinetica_ray as kr
import ray


def main():
    """Main example function."""
    ray.init(ignore_reinit_error=True)

    print("Kinetica-Ray Integration Example")
    print("=" * 40)

    connection_params = {
        "url": "http://localhost:9191",
        "username": "admin",
        "password": "password",
    }
    table_name = "kinetica_ray_example"

    print("\n1. Creating sample data...")
    ds = ray.data.from_items(
        [{"id": i, "name": f"User_{i}", "score": 50.0 + i} for i in range(100)]
    )

    print("\n2. Writing dataset to Kinetica...")
    kr.write_kinetica(
        ds,
        table_name=table_name,
        mode="overwrite",
        **connection_params,
    )

    print("\n3. Reading dataset back from Kinetica...")
    read_ds = kr.read_kinetica(
        table_name=table_name,
        filter_expression="score >= 100",
        **connection_params,
    )

    print(f"Read {read_ds.count()} rows with score >= 100")
    print(read_ds.take(5))


if __name__ == "__main__":
    main()

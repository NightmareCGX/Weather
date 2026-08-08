import os
import sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Ensure absolute path to services/api/src is added. This must happen before
# the api.* imports below, so Alembic can resolve the application models no
# matter what working directory it is invoked from. The imports are therefore
# deliberately not at module top level (ruff E402 is suppressed for them).
current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from api.core.database import Base  # noqa: E402
from api.models import *  # noqa: E402,F403  # noqa: E402 - import ordering; F403 - star import required for Alembic metadata

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def include_object(
    object: object,  # noqa: A002 - alembic's callback signature
    name: str,
    type_: str,
    reflected: bool,
    compare_to: object | None,
) -> bool:
    """Exclude PostGIS-owned objects from autogenerate comparisons.

    ``CREATE EXTENSION postgis`` installs its own tables (e.g.
    ``spatial_ref_sys``) in the ``public`` schema. They are not part of the
    application metadata, so Alembic autogenerate/``check`` would otherwise
    always report them as orphaned/removed tables and fail CI. The extension
    is created by the migration itself (``001_initial_schema``), so these
    tables are expected and must be ignored when comparing metadata.
    """
    if type_ == "table" and name == "spatial_ref_sys":
        return False
    return True


def get_url():
    return os.getenv(
        "DATABASE_URL",
        "postgresql://weather_user:weather_password@localhost:5432/weather_db",
    )


def run_migrations_offline() -> None:
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        include_object=include_object,
        dialect_opts={"server_version_info": (16, 0)},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = get_url()
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            include_object=include_object,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

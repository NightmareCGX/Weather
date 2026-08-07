"""Database engine and declarative base for the ingestion catalog writer.

The catalog ORM models (``ingestion.core.catalog``) subclass
:data:`CatalogBase`. It lives in its own module so the models can ``import``
the base rather than assign it inline, matching the API service pattern
(``services/api/src/api/core/database.py``) and keeping mypy happy.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from ingestion.core.config import settings

#: Declarative base for the ingestion catalog tables.
CatalogBase = declarative_base()

engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

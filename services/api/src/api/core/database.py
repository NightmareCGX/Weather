from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from api.core.config import settings

# Pool tuning (P1-2): pool_size=20, max_overflow=20, pool_timeout=10.
# SQLite in-memory connections use SingletonThreadPool which does not accept QueuePool arguments.
if str(settings.DATABASE_URL).startswith("sqlite"):
    engine = create_engine(settings.DATABASE_URL, echo=False, pool_pre_ping=True)
else:
    engine = create_engine(
        settings.DATABASE_URL,
        echo=False,
        pool_pre_ping=True,
        pool_size=20,
        max_overflow=20,
        pool_timeout=10,
    )
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

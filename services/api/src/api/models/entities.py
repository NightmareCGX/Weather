from datetime import datetime, timezone
from geoalchemy2 import Geometry
from sqlalchemy import (
    Column,
    String,
    Float,
    Integer,
    Boolean,
    DateTime,
    ForeignKey,
    UniqueConstraint,
    Index,
)
from sqlalchemy.orm import relationship

from api.core.database import Base


class ForecastCenter(Base):
    __tablename__ = "forecast_centers"

    id = Column(String, primary_key=True)
    center_id = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    country = Column(String, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    models = relationship(
        "Model", back_populates="center", cascade="all, delete-orphan"
    )


class Model(Base):
    __tablename__ = "models"

    id = Column(String, primary_key=True)
    model_id = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    center_id = Column(String, ForeignKey("forecast_centers.center_id"), nullable=False)
    is_ensemble = Column(Boolean, default=False, nullable=False)
    resolution_km = Column(Float, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    center = relationship("ForecastCenter", back_populates="models")
    versions = relationship(
        "ModelVersion", back_populates="model", cascade="all, delete-orphan"
    )


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(String, primary_key=True)
    model_id = Column(String, ForeignKey("models.model_id"), nullable=False)
    version_string = Column(String, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint("model_id", "version_string", name="uq_model_version"),
    )

    model = relationship("Model", back_populates="versions")
    runs = relationship(
        "ModelRun", back_populates="model_version", cascade="all, delete-orphan"
    )


class ModelRun(Base):
    __tablename__ = "model_runs"

    id = Column(String, primary_key=True)
    model_version_id = Column(String, ForeignKey("model_versions.id"), nullable=False)
    cycle_time = Column(DateTime(timezone=True), nullable=False)
    status = Column(String, nullable=False, default="processing")
    zarr_store_path = Column(String, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint("model_version_id", "cycle_time", name="uq_model_run_cycle"),
        Index("idx_model_runs_cycle", "model_version_id", cycle_time.desc()),
    )

    model_version = relationship("ModelVersion", back_populates="runs")
    ensemble_members = relationship(
        "EnsembleMember", back_populates="run", cascade="all, delete-orphan"
    )
    forecast_products = relationship(
        "ForecastProduct", back_populates="run", cascade="all, delete-orphan"
    )


class EnsembleMember(Base):
    __tablename__ = "ensemble_members"

    id = Column(String, primary_key=True)
    run_id = Column(String, ForeignKey("model_runs.id"), nullable=False)
    member_index = Column(Integer, nullable=False)
    member_name = Column(String, nullable=False)

    __table_args__ = (
        UniqueConstraint("run_id", "member_index", name="uq_ensemble_member_index"),
    )

    run = relationship("ModelRun", back_populates="ensemble_members")


class ForecastVariable(Base):
    __tablename__ = "forecast_variables"

    id = Column(String, primary_key=True)
    variable_code = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    unit = Column(String, nullable=False)


class ForecastGrid(Base):
    __tablename__ = "forecast_grids"

    id = Column(String, primary_key=True)
    grid_code = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    resolution_km = Column(Float, nullable=False)


class ForecastProduct(Base):
    __tablename__ = "forecast_products"

    id = Column(String, primary_key=True)
    run_id = Column(String, ForeignKey("model_runs.id"), nullable=False)
    variable_id = Column(
        String, ForeignKey("forecast_variables.variable_code"), nullable=False
    )
    grid_id = Column(String, ForeignKey("forecast_grids.grid_code"), nullable=False)
    product_type = Column(String, nullable=False)
    lead_time_hours = Column(Integer, nullable=False)
    zarr_chunk_path = Column(String, nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "run_id",
            "variable_id",
            "grid_id",
            "product_type",
            "lead_time_hours",
            name="uq_forecast_product_coords",
        ),
        Index("idx_forecast_products_catalog", "run_id", "variable_id", "grid_id"),
    )

    run = relationship("ModelRun", back_populates="forecast_products")
    variable = relationship("ForecastVariable")
    grid = relationship("ForecastGrid")


class Station(Base):
    __tablename__ = "stations"

    id = Column(String, primary_key=True)
    station_code = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    elevation_m = Column(Float, nullable=False)
    geom = Column(
        Geometry(geometry_type="POINT", srid=4326, spatial_index=False), nullable=False
    )

    __table_args__ = (Index("idx_stations_geom", "geom", postgresql_using="gist"),)

    observations = relationship(
        "VerificationObservation",
        back_populates="station",
        cascade="all, delete-orphan",
    )


class City(Base):
    __tablename__ = "cities"

    id = Column(String, primary_key=True)
    city_name = Column(String, nullable=False, index=True)
    region = Column(String, nullable=False)
    country = Column(String, nullable=False)
    population = Column(Integer, nullable=True)
    geom = Column(
        Geometry(geometry_type="POINT", srid=4326, spatial_index=False), nullable=False
    )

    __table_args__ = (Index("idx_cities_geom", "geom", postgresql_using="gist"),)


class SkiResort(Base):
    __tablename__ = "ski_resorts"

    id = Column(String, primary_key=True)
    resort_name = Column(String, nullable=False, index=True)
    region = Column(String, nullable=False)
    country = Column(String, nullable=False)
    summit_elevation_m = Column(Float, nullable=False)
    # Note: SkiResort geom is represented as a Point geometry for MVP point queries and spatial lookups.
    geom = Column(
        Geometry(geometry_type="POINT", srid=4326, spatial_index=False), nullable=False
    )

    __table_args__ = (Index("idx_ski_resorts_geom", "geom", postgresql_using="gist"),)


class VerificationObservation(Base):
    __tablename__ = "verification_observations"

    id = Column(String, primary_key=True)
    station_id = Column(String, ForeignKey("stations.station_code"), nullable=False)
    valid_time = Column(DateTime(timezone=True), nullable=False)
    variable_code = Column(String, nullable=False)
    observed_value = Column(Float, nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "station_id", "valid_time", "variable_code", name="uq_station_observation"
        ),
        Index("idx_verification_station_time", "station_id", valid_time.desc()),
    )

    station = relationship("Station", back_populates="observations")


class PointQueryFallbackAudit(Base):
    __tablename__ = "point_query_fallback_audit"

    cache_key = Column(String, primary_key=True)
    query_params = Column(String, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    expires_at = Column(DateTime(timezone=True), nullable=False)
    fallback_reason = Column(String, nullable=False)
    #: Cumulative number of fallbacks recorded for this key. Concurrency-safe
    #: upserts increment this counter instead of dropping a row on a PK
    #: conflict, so the audit ledger never loses a fallback event.
    fallback_count = Column(
        Integer, nullable=False, server_default="1", default=1
    )

    __table_args__ = (Index("idx_point_query_fallback_expires", "expires_at"),)

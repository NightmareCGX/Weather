"""Initial database schema with PostGIS support and timezone-aware timestamps.

Revision ID: 001_initial_schema
Revises:
Create Date: 2026-07-31 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
import geoalchemy2

# revision identifiers, used by Alembic.
revision = '001_initial_schema'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Ensure PostGIS extension exists
    op.execute("CREATE EXTENSION IF NOT EXISTS postgis;")

    # 1. forecast_centers
    op.create_table(
        'forecast_centers',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('center_id', sa.String(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('country', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('center_id')
    )
    op.create_index(op.f('ix_forecast_centers_center_id'), 'forecast_centers', ['center_id'], unique=True)

    # 2. models
    op.create_table(
        'models',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('model_id', sa.String(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('center_id', sa.String(), nullable=False),
        sa.Column('is_ensemble', sa.Boolean(), nullable=False),
        sa.Column('resolution_km', sa.Float(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['center_id'], ['forecast_centers.center_id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('model_id')
    )
    op.create_index(op.f('ix_models_model_id'), 'models', ['model_id'], unique=True)

    # 3. model_versions
    op.create_table(
        'model_versions',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('model_id', sa.String(), nullable=False),
        sa.Column('version_string', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['model_id'], ['models.model_id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('model_id', 'version_string', name='uq_model_version')
    )

    # 4. model_runs
    op.create_table(
        'model_runs',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('model_version_id', sa.String(), nullable=False),
        sa.Column('cycle_time', sa.DateTime(timezone=True), nullable=False),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('zarr_store_path', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['model_version_id'], ['model_versions.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('model_version_id', 'cycle_time', name='uq_model_run_cycle')
    )
    op.create_index('idx_model_runs_cycle', 'model_runs', ['model_version_id', sa.text('cycle_time DESC')], unique=False)

    # 5. ensemble_members
    op.create_table(
        'ensemble_members',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('run_id', sa.String(), nullable=False),
        sa.Column('member_index', sa.Integer(), nullable=False),
        sa.Column('member_name', sa.String(), nullable=False),
        sa.ForeignKeyConstraint(['run_id'], ['model_runs.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('run_id', 'member_index', name='uq_ensemble_member_index')
    )

    # 6. forecast_variables
    op.create_table(
        'forecast_variables',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('variable_code', sa.String(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('unit', sa.String(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('variable_code')
    )
    op.create_index(op.f('ix_forecast_variables_variable_code'), 'forecast_variables', ['variable_code'], unique=True)

    # 7. forecast_grids
    op.create_table(
        'forecast_grids',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('grid_code', sa.String(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('resolution_km', sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('grid_code')
    )
    op.create_index(op.f('ix_forecast_grids_grid_code'), 'forecast_grids', ['grid_code'], unique=True)

    # 8. forecast_products
    op.create_table(
        'forecast_products',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('run_id', sa.String(), nullable=False),
        sa.Column('variable_id', sa.String(), nullable=False),
        sa.Column('grid_id', sa.String(), nullable=False),
        sa.Column('product_type', sa.String(), nullable=False),
        sa.Column('lead_time_hours', sa.Integer(), nullable=False),
        sa.Column('zarr_chunk_path', sa.String(), nullable=True),
        sa.ForeignKeyConstraint(['grid_id'], ['forecast_grids.grid_code'], ),
        sa.ForeignKeyConstraint(['run_id'], ['model_runs.id'], ),
        sa.ForeignKeyConstraint(['variable_id'], ['forecast_variables.variable_code'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('run_id', 'variable_id', 'grid_id', 'product_type', 'lead_time_hours', name='uq_forecast_product_coords')
    )
    op.create_index('idx_forecast_products_catalog', 'forecast_products', ['run_id', 'variable_id', 'grid_id'], unique=False)

    # 9. stations
    op.create_table(
        'stations',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('station_code', sa.String(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('elevation_m', sa.Float(), nullable=False),
        sa.Column('geom', geoalchemy2.types.Geometry(geometry_type='POINT', srid=4326, spatial_index=False, from_text='ST_GeomFromEWKT', name='geometry'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('station_code')
    )
    op.create_index(op.f('ix_stations_station_code'), 'stations', ['station_code'], unique=True)
    op.create_index('idx_stations_geom', 'stations', ['geom'], unique=False, postgresql_using='gist')

    # 10. cities
    op.create_table(
        'cities',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('city_name', sa.String(), nullable=False),
        sa.Column('region', sa.String(), nullable=False),
        sa.Column('country', sa.String(), nullable=False),
        sa.Column('population', sa.Integer(), nullable=True),
        sa.Column('geom', geoalchemy2.types.Geometry(geometry_type='POINT', srid=4326, spatial_index=False, from_text='ST_GeomFromEWKT', name='geometry'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_cities_city_name'), 'cities', ['city_name'], unique=False)
    op.create_index('idx_cities_geom', 'cities', ['geom'], unique=False, postgresql_using='gist')

    # 11. ski_resorts
    op.create_table(
        'ski_resorts',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('resort_name', sa.String(), nullable=False),
        sa.Column('region', sa.String(), nullable=False),
        sa.Column('country', sa.String(), nullable=False),
        sa.Column('summit_elevation_m', sa.Float(), nullable=False),
        sa.Column('geom', geoalchemy2.types.Geometry(geometry_type='POINT', srid=4326, spatial_index=False, from_text='ST_GeomFromEWKT', name='geometry'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_ski_resorts_resort_name'), 'ski_resorts', ['resort_name'], unique=False)
    op.create_index('idx_ski_resorts_geom', 'ski_resorts', ['geom'], unique=False, postgresql_using='gist')

    # 12. verification_observations
    op.create_table(
        'verification_observations',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('station_id', sa.String(), nullable=False),
        sa.Column('valid_time', sa.DateTime(timezone=True), nullable=False),
        sa.Column('variable_code', sa.String(), nullable=False),
        sa.Column('observed_value', sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(['station_id'], ['stations.station_code'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('station_id', 'valid_time', 'variable_code', name='uq_station_observation')
    )
    op.create_index('idx_verification_station_time', 'verification_observations', ['station_id', sa.text('valid_time DESC')], unique=False)

    # 13. point_query_fallback_audit
    op.create_table(
        'point_query_fallback_audit',
        sa.Column('cache_key', sa.String(), nullable=False),
        sa.Column('query_params', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('fallback_reason', sa.String(), nullable=False),
        sa.PrimaryKeyConstraint('cache_key')
    )
    op.create_index('idx_point_query_fallback_expires', 'point_query_fallback_audit', ['expires_at'], unique=False)


def downgrade() -> None:
    op.drop_index('idx_point_query_fallback_expires', table_name='point_query_fallback_audit')
    op.drop_table('point_query_fallback_audit')
    op.drop_index('idx_verification_station_time', table_name='verification_observations')
    op.drop_table('verification_observations')
    op.drop_index('idx_ski_resorts_geom', table_name='ski_resorts', postgresql_using='gist')
    op.drop_index(op.f('ix_ski_resorts_resort_name'), table_name='ski_resorts')
    op.drop_table('ski_resorts')
    op.drop_index('idx_cities_geom', table_name='cities', postgresql_using='gist')
    op.drop_index(op.f('ix_cities_city_name'), table_name='cities')
    op.drop_table('cities')
    op.drop_index('idx_stations_geom', table_name='stations', postgresql_using='gist')
    op.drop_index(op.f('ix_stations_station_code'), table_name='stations')
    op.drop_table('stations')
    op.drop_index('idx_forecast_products_catalog', table_name='forecast_products')
    op.drop_table('forecast_products')
    op.drop_index(op.f('ix_forecast_grids_grid_code'), table_name='forecast_grids')
    op.drop_table('forecast_grids')
    op.drop_index(op.f('ix_forecast_variables_variable_code'), table_name='forecast_variables')
    op.drop_table('forecast_variables')
    op.drop_table('ensemble_members')
    op.drop_index('idx_model_runs_cycle', table_name='model_runs')
    op.drop_table('model_runs')
    op.drop_table('model_versions')
    op.drop_index(op.f('ix_models_model_id'), table_name='models')
    op.drop_table('models')
    op.drop_index(op.f('ix_forecast_centers_center_id'), table_name='forecast_centers')
    op.drop_table('forecast_centers')

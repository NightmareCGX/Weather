"""Service layer for the API service.

Services coordinate PostGIS queries, Zarr slicing, interpolation, units
conversion, and caching. FastAPI routers stay thin (ENGINEERING_CONTRACT
section 2): they validate request parameters, call these services, and
serialize results.
"""

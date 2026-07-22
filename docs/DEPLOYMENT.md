# Deployment Architecture & Operations

The platform is designed for cloud-native containerized deployment across multi-region cloud environments (AWS / GCP / Azure).

---

## 1. Local Development (`Docker Compose`)
For local development, all auxiliary services are spun up via Docker Compose:
- **PostgreSQL 16 + PostGIS**: Spatial relational database.
- **Redis**: Caching and Celery message broker.
- **MinIO**: S3-compatible object storage for local Zarr and GRIB2 testing.

---

## 2. Production Cloud Architecture
- **Container Orchestration**: Kubernetes (EKS / GKE) running stateless microservices (`services/ingestion`, `services/processing`, `services/api`).
- **GPU Inference Node Pools**: Dedicated GPU node groups (NVIDIA T4 / A10G) running PyTorch inference workers for AI downscaling.
- **Object Storage**: AWS S3 with lifecycle policies for hot/cold Zarr raster storage.
- **Database**: Managed PostgreSQL (AWS RDS / GCP Cloud SQL) with PostGIS extension enabled and read replicas for serving.
- **CDN & Edge**: Cloudflare / AWS CloudFront caching API responses and map tiles at the edge.

---

## 3. CI/CD Pipelines (`GitHub Actions`)
- **Linting & Type Checking**: Ruff, Black, MyPy, ESLint.
- **Automated Testing**: Pytest for Python services, Jest/Playwright for Next.js frontend.
- **Container Builds**: Automated multi-architecture Docker image builds pushed to container registries on semantic release tags.
- **Infrastructure as Code**: Terraform for provisioning cloud networking, S3 buckets, RDS instances, and Kubernetes clusters.

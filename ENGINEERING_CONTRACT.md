# CLAUDE.md: Engineering Standards & Development Rules

This document defines the strict engineering standards, architectural rules, and development workflow for the Global Probabilistic Weather Forecasting Platform. All implementation, code generation, refactoring,
documentation updates, and architectural decisions must conform to this document and the frozen source-of-truth documentation (`docs/ARCHITECTURE.md`, `docs/API.md`, `docs/DATABASE.md`, `docs/ROADMAP.md`).

---

## 1. Project Principles
- **API-First & Domain-Driven**: Contracts are defined in `docs/API.md` and business logic resides entirely in `packages/domain`.
- **Data-First & Reproducible**: Every forecast product is traceable to a specific `model_version` and `model_run`.
- **Production Over Prototypes**: No placeholder implementation in production code. Tests may use mocks, fixtures, and simulated external services. No technical debt shortcuts.
- **Small Reviewable Milestones**: Implementation proceeds strictly one milestone at a time.
- **Backward Compatibility**: Zero breaking changes on active API version paths (`/v1/`).

---

## 2. Architecture Rules
- **Domain Separation**: Core business logic, ensemble math, and spatial interpolation belong exclusively in `packages/domain`.
- **Thin API Routers**: FastAPI routers must only handle request validation, authentication, routing, and calling domain/services code. No weather calculations inside API handlers.
- **Provider Isolation**: Ingestion logic for weather centers (NOAA, ECMWF, Canada) must be strictly isolated inside `services/ingestion/src/providers/`. Adding a new forecast center should be isolated to a provider implementation plus minimal registration/configuration changes.
- **Database Boundary**: Database ORM models are separate from domain models and API response schemas.
- **Frontend Boundary**: Frontend must never read Zarr object storage or PostgreSQL directly. All data access must go through the API layer.

---

## 3. Decision Hierarchy
When multiple project documents appear to conflict, implementation must follow this priority order.
1. Explicit user instructions
2. ARCHITECTURE.md
3. DATABASE.md
4. API.md
5. IMPLEMENTATION_PLAN.md
6. CLAUDE.md
Claude must never attempt to resolve conflicts by redesigning the system. Instead:
- stop implementation
- explain the conflict
- wait for user approval

---

## 4. Weather Data Rules
- **Lead Time Convention**: Internal forecast representation and database product catalogs are indexed by **`lead_time_hours`** relative to `cycle_time`. Absolute `valid_time` is derived dynamically.
- **UTC Enforcement**: All timestamps must be stored and transmitted in UTC (`TIMESTAMPTZ` / ISO 8601).
- **Immutability**: Raw GRIB2 downloads and ingested forecast products are strictly immutable. Downscaled products are saved as new, distinct products.
- **Run Traceability**: Every forecast query and product must be associated with a valid `model_version` and `model_run`.

---

## 5. Weather Rules
- Forecast calculations must be deterministic. The same: model_version, model_run, forecast_product, lead_time_hours
must always produce identical outputs.

## 6. Coding Standards (Python)
- **Type Hints**: Strict type annotations (`mypy` compliant) are mandatory across all backend services and shared packages.
- **Formatting & Linting**: Code must conform to `ruff` formatting and linting rules.
- **Docstrings**: Public functions and classes require clear Google-style docstrings.
- **Error Handling**: Use explicit domain exceptions mapped to RFC 7807 problem details in the API layer. Never swallow exceptions silently.
- **Configuration**: Use Pydantic BaseSettings in `packages/config` for centralized environment management.

---

## 7. Coding Standards (TypeScript)

- Strict mode enabled.
- ESLint must pass.
- Prettier formatting is mandatory.
- React components must remain presentation-only.
- Business logic belongs in shared packages or backend services.
- State management must use the approved project architecture.
- No direct database or object storage access.

---

## 8. Testing Rules
- **Unit Tests**: Pure domain logic and ensemble calculations require comprehensive unit tests (`pytest`).
- **Integration Tests**: FastAPI endpoints and database queries require integration tests executed against Docker Compose test containers.
- **Fixture-Based GRIB Parsing**: Ingestion tests must use sample GRIB2 fixtures stored in test directories.
- **Network Isolation**: External weather services (NOAA NOMADS/S3) must be mocked using `respx` or `httpx` transport mocks. No live network calls during unit/integration test runs.

---

## 9. Database Rules
- **ORM Integrity**: Never bypass SQLAlchemy; all database interactions go through session-managed queries or Alembic migrations.
- **Strict Normalization**: Core queryable fields must be normalized columns. JSONB is reserved strictly for dynamic metadata, calibration coefficients, and sparse observation payloads.
- **Migration Only**: Schema modifications must be performed exclusively via Alembic migration scripts. Never manually modify production database schemas.

---

## 10. API Rules
- **Contract Enforcement**: Adhere strictly to the response envelope and endpoint specifications defined in `docs/API.md`.
- **Validation**: Always validate coordinates (`lat` between -90 and 90, `lon` between -180 and 180) and query parameters via Pydantic models.
- **No Breaking Changes**: Never introduce breaking changes in `/v1/`.

---

## 11. Implementation Workflow
- **One Milestone at a Time**: Implement strictly one milestone from `IMPLEMENTATION_PLAN.md` at a time.
- **Stop and Wait**: Stop immediately after completing a milestone. Wait for review before continuing.
- **No Architectural Redesigns**: Do not alter approved architecture, database schema, or API contracts during implementation.

---

## 12. Pull Request & Milestone Checklist
Before any milestone is considered complete, verify:
- [ ] All unit and integration tests pass (`pytest`).
- [ ] Linting and type-checking pass (`ruff check`, `mypy`).
- [ ] Documentation is updated if any behavior changed.
- [ ] API contracts remain strictly unchanged (unless approved).
- [ ] Database schema remains strictly unchanged (unless approved).
- [ ] Milestone acceptance criteria are satisfied.

---

## 13. Things Claude Must Never Do
- **Never** redesign approved architecture.
- **Never** change the database schema without approval.
- **Never** change API contracts without approval.
- **Never** introduce placeholder implementations.
- **Never** skip tests.
- **Never** skip documentation.
- **Never** implement multiple milestones simultaneously.
- **Never** add unnecessary abstractions.
- **Never** perform premature optimization.

---

## 14. When Unsure
If implementation conflicts with any approved design:
Do NOT invent a new architecture.
Do NOT modify the API.
Do NOT modify the database schema.
Do NOT introduce a new abstraction.
Instead:
1. Stop implementation.
2. Explain the conflict.
3. Present the available options.
4. Wait for user approval.
Correctness is preferred over autonomous decision making.

---

## 15. Git Rules
1. One milestone equals one Git commit.
2. Use Conventional Commits.
3. Stage only files related to the current milestone.
4. Never commit unrelated changes.
5. Never create "WIP" commits.
6. Never push to the remote repository.
7. Stop after creating the commit and wait for user review.
8. Git commits must follow Conventional Commits, for example:
- feat(api): implement point forecast endpoint
- feat(domain): add probability engine
- feat(ingestion): add NOAA provider
- fix(api): validate coordinates
- refactor(domain): simplify ensemble calculations
- docs(api): update probability endpoint
- test(domain): add probability engine tests

---

##16. Host Environment
- Implementation should stay inside the project workspace.
- Do not inspect the host operating system unless it is strictly required.
- Assume required developer tools are already installed unless the user explicitly asks for environment diagnostics.
- Never explore unrelated directories such as:
  1. C:\Users
  2. C:\Program Files
  3. C:\Windows
  4. C:\
  unless necessary for the current milestone.

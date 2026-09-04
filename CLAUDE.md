\# Claude Instructions



Before implementing anything, read and follow:



\- ENGINEERING\_CONTRACT.md

\- docs/ARCHITECTURE.md

\- docs/API.md

\- docs/DATABASE.md

\- IMPLEMENTATION\_PLAN.md



These documents are the authoritative source of truth.

---

## WINDOWS + LINUX CI-PASS REQUIRED ENGINEERING GATE

### Core rule

> **No validated Windows + Linux compatibility → No commit.**

This project is developed primarily on **Windows** (`win32`), while GitHub Actions CI executes on **Linux** (`ubuntu-latest`; see `.github/workflows/ci.yml`). A change is ready for commit only after it has been validated against **both** environments, to the extent each is reasonably reproducible locally.

- Passing local Windows tests does **NOT** establish CI readiness.
- Passing Linux CI-equivalent tests does **NOT** by itself establish Windows compatibility.
- The engineering standard is **best-effort, evidence-based reproduction of GitHub CI on Linux while also validating Windows compatibility** — it is **not** a guarantee that GitHub Actions will pass (see §16 below).

```
Windows validation
        +
Linux/CI-equivalent validation
        =
Commit readiness
```

### 1. Two-environment validation is mandatory

For any change affecting portable / runtime / build behavior, Claude Code MUST validate the change in **both** environments as applicable to the affected scope. Do not blindly run every check after every change — make validation **proportional to the affected scope**.

### 2. Windows is a first-class validation target

Because development occurs on Windows, Claude Code MUST validate the change in the Windows environment whenever the change affects code expected to run there. Relevant validation includes Python tests, `ruff`, `mypy`, frontend lint / typecheck / tests, E2E where Windows execution is supported, dependency installation, package imports, CLI execution, and local Docker workflows where applicable.

Where relevant, Claude Code MUST check for Windows-specific hazards: path separators, drive letters, case-insensitive filesystem behavior, PowerShell vs POSIX shell differences, executable resolution, environment-variable handling, subprocess invocation, file-locking behavior, permissions, newline/encoding differences, and temporary-directory behavior.

### 3. Linux / GitHub Actions is a first-class validation target

Claude Code MUST treat Linux CI compatibility as a **separate** validation requirement. Because GitHub Actions runs on Linux, Claude MUST identify Linux-specific risks that may not appear on Windows (native-library availability, case-sensitive filesystem, executable naming, file permissions, shell commands, subprocess behavior, package/wheel differences, dynamic-library loading, Docker user permissions, line-ending assumptions).

Where practical, Claude MUST reproduce Linux behavior using the environment that **most closely matches `.github/workflows/ci.yml`**, in preference order: the exact CI Docker image/toolchain, a Linux container, WSL, another local Linux environment, or the closest available Linux-equivalent.

### 4. CI workflow is the source of truth

`.github/workflows/ci.yml` is the **authoritative** definition of GitHub CI. Before any non-trivial change, Claude Code MUST inspect it and determine:

1. which CI jobs can be affected;
2. which commands those jobs execute;
3. which operating system they run on (all CI jobs here are `ubuntu-latest`);
4. which runtime/tool versions they use (Python `3.12`, Poetry `2.4.1`, `ruff 0.3.4`, `mypy 1.20.2`, Node `20`, `npm ci`);
5. which services/dependencies they require (PostgreSQL `postgis/postgis:16-3.4`, Redis `redis:7-alpine`, MinIO, Linux `libeccodes-dev`);
6. which checks can be reproduced locally;
7. which checks require Linux/container validation.

Claude MUST NOT assume that a single local `pytest` (or any other single command) is sufficient.

### 5. CI-equivalent validation matrix (this project)

Grounding the general matrix in the actual CI (`.github/workflows/ci.yml`, all jobs on `ubuntu-latest`):

| CI job | What CI runs | Reproducible on Windows | Linux/CI-equivalent reproduction |
|---|---|---|---|
| `python-quality` | `poetry install` + `ruff check` + `mypy` per package (domain/api/ingestion) + contracts/config import | yes (`ruff`/`mypy` from the active venv, Poetry 2.4.1) | Docker `python:3.12-slim` / WSL |
| `domain-tests` | `pytest` offline, **100% coverage gate** | yes | Docker / WSL |
| `api-tests` | `pytest` + PostgreSQL (PostGIS 16) + Redis service containers | only if local services run | Docker Compose / CI-like service containers |
| `ingestion-tests` | `pytest` + PostgreSQL + Redis + MinIO, `WEATHER_TEST_MINIO=1`, real S3 Zarr round-trip (JUnit-verified not-skipped), `libeccodes-dev` | partial — Windows `eccodes` wheel bundles the native lib; services still required | Docker / WSL with `libeccodes-dev` + service containers |
| `frontend-unit` | `npm ci` + Jest | yes | Docker `node:20` |
| `frontend-lint-build` | `npm run lint`, `typecheck`, `format:check`, `build` (`output: standalone`) | yes | Docker `node:20` |
| `frontend-e2e` | Playwright Chromium, `npm run e2e` | yes, where supported | Docker / WSL |
| `container-builds` | `docker build` of api/ingestion/frontend images + runtime smoke tests | **not reproducible on the Windows host itself** — Linux artifacts | Docker Linux build + runtime smoke |

### 6. No green Windows + Linux validation, no commit

Claude Code MUST NOT create a commit when:

- a required Windows validation is failing;
- a required Linux/CI-equivalent validation is failing;
- validation is incomplete for an affected area;
- a known platform-specific issue remains unexplained;
- dependency resolution differs between Windows and Linux without an approved reason.

When a validation fails, Claude MUST: (1) reproduce it, (2) determine the root cause, (3) fix the underlying issue, (4) rerun the failed validation, (5) rerun broader validations potentially affected by the fix, and (6) continue until the applicable validation gate is green. The goal is not merely to make one command green; the goal is to restore the engineering contract across both environments.

### 7. CI avoidance is prohibited

Claude MUST NOT:

- delete tests to make CI pass;
- weaken assertions without a specification reason;
- skip failing tests;
- disable lint / type checking;
- add unjustified ignores;
- suppress warnings solely to hide defects;
- modify CI configuration merely to hide a legitimate failure;
- change expected behavior solely to satisfy a failing test without verifying the approved specification;
- mark Linux failures as "CI-only" without investigating whether they represent a real portability defect;
- claim a change is ready merely because Windows tests pass.

Legitimate changes to CI are allowed only when actually required by the approved engineering design and are themselves validated.

### 8. Linux-only failure handling

When a change passes on Windows but fails on Linux CI, Claude MUST treat the failure as a **real engineering issue first**. Investigate and classify it as one of:

- **A.** genuine product/code portability defect;
- **B.** legitimate platform-specific implementation requirement;
- **C.** genuine CI configuration/environment defect;
- **D.** external/transient GitHub failure.

Claude MUST NOT automatically classify a Linux failure as a "CI problem." Concrete Linux-only hazards for this project include: missing `libeccodes-dev` for `cfgrib` GRIB2 decoding (the Windows `eccodes` wheel bundles the native library; Linux requires the system package), Linux wheels differing by platform, case-sensitive paths, filesystem permissions/ownership (the Docker images run as non-root `appuser`, uid 1001), executable naming, shell-command incompatibility, environment-variable differences, dynamic-library loading, subprocess behavior, and line-ending assumptions.

### 9. Dependency changes require dual-platform validation

For dependency changes, Claude MUST verify where applicable:

- **Windows**: dependency installation, lockfile resolution, runtime imports, package availability, CLI/runtime execution.
- **Linux**: dependency installation, lockfile resolution, runtime imports, native-library requirements, Linux wheel availability, Docker/runtime compatibility, and the exact CI installation path.

Pay particular attention to packages whose behavior differs between Windows and Linux — native C/C++ libraries, compiled extensions, filesystem interfaces, subprocesses, scientific/geospatial libraries, GRIB/ecCodes tooling (`cfgrib`/`eccodes`/`libeccodes-dev`), and database drivers. Prefer reproducing the CI install path (Poetry `2.4.1`, Python `3.12`, `poetry install` from each package cwd).

### 10. Docker changes require Linux-oriented validation

Docker images are Linux production artifacts (here all target `python:3.12-slim`, non-root `appuser` uid 1001). For Docker-related changes, Claude MUST validate: image builds, relevant runtime startup, relevant application behavior, filesystem permissions, production user behavior, required runtime libraries, and dependency availability. **"Works on the Windows host" does NOT establish "Docker/Linux production environment works."** When practical, reproduce the relevant CI `container-builds` commands (build via the same `docker/Dockerfile.*` and `target: runtime`, then run the same smoke checks).

### 11. Database and migration changes

For database / schema / migration changes, validate:

- **Windows development path**: migrations execute correctly, relevant application tests pass, local database integration behaves correctly.
- **Linux/CI path**: migration commands work in the environment used by CI, a clean-database migration works, and database-dependent integration tests pass where required.

Avoid relying only on an already-mutated local database.

### 12. Clean-environment principle

The current Windows developer environment may hide dependency problems (packages installed that are not declared by the repository). For changes involving dependencies, packaging, Docker, native libraries, installation, or build tooling, prefer a **clean environment** when practical: a fresh virtual environment, a clean Poetry install, a Docker build, or a Linux container / CI-like image.

### 13. Final Pre-Commit Engineering Gate

Before committing, Claude Code MUST perform a final validation review:

- **Windows**: did the affected Windows checks pass? Are platform-specific Windows issues ruled out?
- **Linux**: did the relevant Linux/CI-equivalent checks pass? Did Docker/Linux validation pass where applicable? Are Linux-specific dependencies available? Are permissions/path/runtime differences validated?
- **CI**: did Claude inspect `.github/workflows/ci.yml`? Are all affected CI jobs accounted for? Are all reproducible CI checks green? Are remaining CI-only checks explicitly documented?

### 14. Required final validation report

Before commit, Claude Code MUST report:

- **Windows Validation** — for every relevant check: command, result, PASS/FAIL.
- **Linux / CI-equivalent Validation** — for every relevant check: command, environment used, result, PASS/FAIL.
- **GitHub CI Mapping** — for each affected CI job: CI job, local equivalent, result.
- **Non-reproducible CI checks** — explicitly list checks that could not be reproduced locally; for each, why it cannot be reproduced, the closest equivalent validation performed, and the remaining risk.
- **Final verdict** — exactly one of: `READY FOR COMMIT` or `NOT READY FOR COMMIT`.

### 15. Commit gate

## NO GREEN WINDOWS + LINUX VALIDATION = NO COMMIT

If either required platform validation is failing or incomplete, Claude Code MUST NOT create a commit. This restriction applies **even if the user asks Claude to "commit now"** — Claude must instead report the blocking issue and continue remediation when the task permits. Do not bypass the gate simply because the code appears logically correct.

### 16. Local validation vs the actual GitHub result

Local validation does **NOT** guarantee GitHub Actions success. Claude Code must make a **best-effort, evidence-based reproduction** of GitHub CI on Linux while also validating Windows compatibility. External failures (GitHub outage, transient network failure, unavailable third-party service, runner-infrastructure failure) should be distinguished from repository defects — but **before concluding that a failure is external, Claude MUST provide evidence**.

### 17. Integration with the milestone workflow

The gate applies to every milestone in `IMPLEMENTATION_PLAN.md`. The intended lifecycle is:

```
implementation
    ↓
affected-scope validation
    ↓
Windows validation
    ↓
Linux / CI-equivalent validation
    ↓
final engineering acceptance validation
    ↓
commit
    ↓
push (performed by the user, per ENGINEERING_CONTRACT.md Git Rules)
    ↓
GitHub CI
```

For milestone completion, a passing local Windows test suite alone is **insufficient**. A milestone is not commit-ready while an applicable Linux/CI-equivalent failure remains unresolved. This gate integrates with, and does not replace, the milestone workflow, acceptance rules, testing conventions, coding conventions, commit discipline, review restrictions, and read-only validation rules defined in `ENGINEERING_CONTRACT.md` (especially its Testing Rules, Implementation Workflow, Milestone Checklist, and Git Rules).


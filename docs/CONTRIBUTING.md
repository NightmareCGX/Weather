# Contributing Guidelines

Thank you for contributing to the Global Probabilistic Weather Platform.

---

## 1. Code Standards & Style
- **Python**: Follow PEP 8. Use `ruff` for linting and formatting. Type hints (`mypy`) are strictly required across all backend services (`services/ingestion`, `services/processing`, `services/api`).
- **TypeScript / React**: Use TypeScript with strict mode enabled. Follow Tailwind CSS conventions and Next.js App Router patterns.

---

## 2. Pull Request Workflow
1. Create a feature branch from `main` (`feat/ingestion-noaa`, `fix/zarr-chunking`).
2. Ensure all tests pass (`pytest` for Python, `npm test` for frontend).
3. Open a Pull Request with a clear description of architectural changes, test coverage, and performance impact.
4. Require at least one review from a senior staff engineer before merging.

# Contributing Guidelines

Thank you for contributing to the Global Probabilistic Weather Platform.

---

## 1. Code Standards & Style
- **Python**: Follow PEP 8. Use `ruff` for linting and formatting. Type hints (`mypy`) are strictly required across all backend services (`services/ingestion`, `services/api`).
- **TypeScript / React**: Use TypeScript with strict mode enabled. Follow Tailwind CSS conventions and Next.js App Router patterns.

---

## 2. Pull Request Workflow
1. Create a feature branch from `main` (`feat/ingestion-noaa`, `fix/zarr-chunking`).
2. Ensure all tests pass (`pytest` for Python, `npm test` for frontend).
3. Open a Pull Request with a clear description of architectural changes, test coverage, and performance impact.
4. Require at least one review from a senior staff engineer before merging.

---

## 3. Release Branch Sync

`release/1.0` is kept in lockstep with `main` by [`release-sync.yml`](../.github/workflows/release-sync.yml): every push to `main` fast-forwards the release branch to that same commit. Nothing has to be run or remembered by hand while a release is open.

**To freeze the release at a chosen commit**, add that commit's hash as the first non-comment line of `.github/release-sync/release-1.0.cutoff`:

```
# Last main commit allowed to reach release/1.0. Delete this file to follow main again.
beea0e089765c1743b7a539c3b3ee2772c466779
```

The next sync run advances `release/1.0` to exactly that commit and then stops — permanently. The cutoff is a hard ceiling: no run, including a manual `workflow_dispatch`, can move the release branch past it.

What makes this safe to run unattended:

* The push is **fast-forward only**. The workflow never force-pushes, so it cannot discard release-only commits.
* A release-only commit on the branch (a hotfix) is itself a freeze signal — the branch stops being a fast-forward of `main`, and the automatic sync switches off on its own.
* To take one specific `main` fix into a frozen release, `git cherry-pick <sha>` onto a branch cut from `release/1.0` and open a PR. Do not merge `main` into a frozen release branch.
* A manual run may pass `sync_to` to advance the branch only as far as a given commit; it is rejected if that commit is past the cutoff.
* Deleting the cutoff file resumes following `main`, but only while the release branch is still a fast-forward of it.
* If the release branch is covered by a ruleset, the Actions app must be a **bypass actor**, or the sync push will be rejected.

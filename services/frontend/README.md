# Weather Platform Frontend

Next.js (App Router) + TypeScript + Tailwind CSS client for the Global
Probabilistic Weather Forecasting Platform. Milestone 12 foundation: a
MapLibre GL JS map with base (OSM) tiles, a weather raster layer configured
from `/v1/maps` metadata, and a model/variable/lead-time control surface.

## Prerequisites

- Node.js >= 18.17 (Next.js 14 and the pinned `maplibre-gl@4.1.0`).
- The FastAPI backend running on `127.0.0.1:8000` (uvicorn default). Start it
  from `services/api` with `uvicorn api.main:app`. Backend container services
  (PostgreSQL, Redis, MinIO) must be up via `docker compose up -d`.

## Getting started

```sh
npm install
npm run dev
```

Open http://localhost:3000. Requests to `/v1/*` are proxied to the backend by
a Next.js rewrite (see `next.config.mjs`). Override the backend target with
the `API_PROXY_TARGET` environment variable if it is not on `127.0.0.1:8000`.

## Scripts

| Script                 | Purpose                                     |
| ---------------------- | ------------------------------------------- |
| `npm run dev`          | Start the Next.js development server        |
| `npm run build`        | Production build (runs ESLint + type-check) |
| `npm start`            | Start the production server                 |
| `npm run lint`         | ESLint (`next lint`)                        |
| `npm run typecheck`    | TypeScript type-check (`tsc --noEmit`)      |
| `npm test`             | Jest + React Testing Library (offline)      |
| `npm run format`       | Prettier write                              |
| `npm run format:check` | Prettier check                              |

## Weather layer note

`/v1/maps` is a metadata-only endpoint: it returns the tile template, zoom
range, and legend, but the backend does not serve tile imagery yet. The map
configures a weather raster source/layer from that metadata; until the backend
serves tiles, requests for weather tiles may 404 and the base map keeps
rendering. This graceful degradation is expected, not a blocker.

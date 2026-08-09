import { defineConfig } from "@playwright/test";

/**
 * Playwright E2E configuration for the weather platform frontend.
 *
 * E2E tests use *mocked API routes* (`page.route("/v1/**", …)`) so they are
 * deterministic, offline, and require no Docker/PostgreSQL — matching the
 * repository's network-isolation conventions. The `webServer` boots the Next.js
 * production build, and the `E2E_LIVE=1` environment variable can switch the
 * tests to hit a real backend by disabling the API route mocks.
 */
export default defineConfig({
  testDir: "./e2e",
  // MapLibre uses WebGL/canvas which is resource-heavy in headless Chromium;
  // running tests serially keeps selection/marker rendering stable.
  fullyParallel: false,
  workers: 1,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  reporter: process.env.CI ? "line" : "list",
  use: {
    baseURL: "http://127.0.0.1:3000",
    trace: "on-first-retry",
  },
  projects: [
    {
      name: "chromium",
      use: {
        browserName: "chromium",
        launchOptions: {
          args: process.env.CI
            ? ["--use-gl=swiftshader", "--disable-gpu"]
            : ["--use-gl=swiftshader", "--disable-gpu"],
        },
      },
    },
  ],
  webServer: {
    command: "npm run build && npm run start",
    url: "http://127.0.0.1:3000",
    reuseExistingServer: !process.env.CI,
    timeout: 180_000,
  },
});

import { expect, test } from "@playwright/test";

import { installApiFailureMocks, installApiMocks } from "./mock-api";

/**
 * Milestone 13 E2E scenarios. All API traffic is mocked deterministically;
 * the browser is real Chromium against the Next.js production build.
 *
 * The search-option locators are scoped to the search listbox (`role="listbox"`
 * with `name="Search results"`) because the page also contains native
 * `<select>` `<option>` elements for the model/variable/lead-time controls.
 */

/** The search autocomplete listbox containing location results. */
const searchResults = (page: import("@playwright/test").Page) =>
  page.getByRole("listbox", { name: "Search results" });

test.beforeEach(async ({ page }) => {
  await installApiMocks(page);
});

test("search → forecast: select a city and render the point forecast dashboard", async ({
  page,
}) => {
  await page.goto("/");

  // The search combobox is present.
  const input = page.getByLabel(/Search for a city/);
  await expect(input).toBeVisible();

  await input.fill("Aspen");
  // Debounced autocomplete returns the city result in the search listbox.
  const option = searchResults(page).getByRole("option", { name: /Aspen/ }).first();
  await expect(option).toBeVisible();

  await option.click();

  // Dashboard appears with the location summary and meteograms. The chart
  // aria-label uses the catalog variable name ("2-Meter Temperature").
  await expect(page.getByText("Hourly Forecast")).toBeVisible();
  await expect(page.getByText("Aspen", { exact: true })).toBeVisible();
  await expect(
    page.getByRole("img", { name: /2-Meter Temperature hourly forecast over lead time/ })
  ).toBeVisible();
});

test("map click → forecast: selecting a coordinate opens the dashboard", async ({ page }) => {
  await page.goto("/");

  // Click the map canvas. The map is created client-side (dynamic import) and
  // only enables click selection after its `load` event, so retry the click a
  // few times until the coordinate selection registers and the dashboard
  // appears (guards the load-event race under mocked tiles).
  const map = page.getByTestId("weather-map");
  await expect(map).toBeVisible();

  for (let attempt = 0; attempt < 3; attempt += 1) {
    await map.click({ position: { x: 320, y: 240 } });
    const dashboardVisible = await page
      .getByText("Hourly Forecast")
      .isVisible()
      .catch(() => false);
    if (dashboardVisible) {
      break;
    }
  }

  // A coordinate selection renders a summary with the clicked coordinates
  // (the exact lat/lon depends on the map viewport, so match any coordinate
  // pair) and fetches /v1/points.
  await expect(page.getByText("Hourly Forecast")).toBeVisible();
  await expect(page.getByText(/-?\d+\.\d+, -?\d+\.\d+/)).toBeVisible();
});

test("ensemble statistics: deterministic selected model shows ensemble empty state", async ({
  page,
}) => {
  await page.goto("/");

  const input = page.getByLabel(/Search for a city/);
  await input.fill("Aspen");
  await searchResults(page).getByRole("option", { name: /Aspen/ }).first().click();

  // The availability mock's default selected model is GFS (deterministic), so
  // the ensemble panel shows the honest empty state instead of requesting a
  // hard-coded ensemble model that may not exist in the database.
  await expect(page.getByText(/Ensemble Statistics \(GFS\)/)).toBeVisible();
  await expect(
    page.getByText("No ensemble data available for the selected forecast.")
  ).toBeVisible();
});

test("selecting an ensemble model renders the percentile fan and member histogram", async ({
  page,
}) => {
  await page.goto("/");

  // Select the GEFS model (present in the availability mock, is_ensemble=true).
  const modelSelect = page.getByLabel("Model");
  await modelSelect.selectOption("gefs");

  const input = page.getByLabel(/Search for a city/);
  await input.fill("Aspen");
  await searchResults(page).getByRole("option", { name: /Aspen/ }).first().click();

  // The ensemble statistics section renders the fan chart for the selected
  // ensemble model.
  await expect(page.getByText(/Ensemble Statistics \(GEFS\)/)).toBeVisible();
  await expect(
    page.getByRole("img", { name: /ensemble percentile fan over lead time/ })
  ).toBeVisible();

  // The mock returns members for /v1/ensembles, so the Distribution View shows
  // a genuine member histogram.
  await expect(page.getByText(/Member distribution/)).toBeVisible();
  await expect(page.getByRole("img", { name: /Histogram of 5 ensemble members/ })).toBeVisible();
});

test("empty search: coherent empty state, app stays usable", async ({ page }) => {
  await page.goto("/");

  const input = page.getByLabel(/Search for a city/);
  await input.fill("zzzznomatch");
  await expect(page.getByText("No matching locations.")).toBeVisible();

  // The map still works after an empty search.
  await expect(page.getByTestId("weather-map")).toBeVisible();
});

test("api failure: useful error state and graceful degradation", async ({ page }) => {
  // Re-install with failure mocks BEFORE navigation.
  await page.unroute("**/v1/**");
  await installApiFailureMocks(page);
  await page.goto("/");

  const input = page.getByLabel(/Search for a city/);
  await input.fill("Aspen");
  // The search request fails; the autocomplete shows an inline alert.
  await expect(page.getByText("Backend unavailable")).toBeVisible();

  // The application remains usable (the map and header are still present).
  await expect(page.getByTestId("weather-map")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Weather Platform" })).toBeVisible();
});

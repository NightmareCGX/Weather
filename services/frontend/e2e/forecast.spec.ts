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

test("place search → resolution → forecast: selecting a place suggestion resolves coordinates and opens dashboard", async ({
  page,
}) => {
  await page.goto("/");

  const input = page.getByLabel(/Search for a city/);
  await expect(input).toBeVisible();

  await input.fill("Boulder");
  const option = searchResults(page)
    .getByRole("option", { name: /Boulder/ })
    .first();
  await expect(option).toBeVisible();

  await option.click();

  // Selecting a place triggers /v1/search/places/:id resolution to coordinates (40.0150, -105.2705)
  await expect(page.getByText("Hourly Forecast")).toBeVisible();
  await expect(page.getByText("Boulder, CO", { exact: true })).toBeVisible();
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

test("ensemble statistics: deterministic selected model shows no ensemble panel", async ({
  page,
}) => {
  await page.goto("/");

  const input = page.getByLabel(/Search for a city/);
  await input.fill("Aspen");
  await searchResults(page).getByRole("option", { name: /Aspen/ }).first().click();

  // The availability mock's default selected model is GFS (deterministic).
  // Per the approved remediation (Issue 5), a deterministic model is NOT an
  // ensemble product, so the UI must render the deterministic point forecast
  // and must NOT render a misleading "Ensemble Statistics" panel at all —
  // neither the heading nor a deterministic-model empty-state message.
  await expect(page.getByText("Hourly Forecast")).toBeVisible();
  await expect(
    page.getByRole("img", { name: /2-Meter Temperature hourly forecast over lead time/ })
  ).toBeVisible();
  await expect(page.getByText(/Ensemble Statistics/)).toHaveCount(0);
  await expect(page.getByText("No ensemble data available for the selected forecast.")).toHaveCount(
    0
  );
});

test("selecting an ensemble model renders the percentile fan and member distribution", async ({
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

  // The mock returns members and pdf for /v1/ensembles, so the Distribution View
  // renders both the histogram bars and the continuous PDF line.
  await expect(page.getByText(/Member distribution/)).toBeVisible();
  const distribution = page.getByRole("img", {
    name: /Histogram and PDF of 5 ensemble members/,
  });
  await expect(distribution).toBeVisible();
  await expect(distribution.locator(".recharts-rectangle").first()).toBeVisible();
  await expect(distribution.locator(".recharts-line-curve")).toBeVisible();
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

test("forecast map transition: selecting B while A tiles are in flight immediately dispatches B", async ({
  page,
}) => {
  const dispatchedLeads: string[] = [];
  let delayedResolve: (() => void) | null = null;
  const holdPromise = new Promise<void>((resolve) => {
    delayedResolve = resolve;
  });

  // Intercept tile requests with controlled delay for lead_time_hours=6
  await page.route("**/v1/maps/**/*.png*", async (route) => {
    const url = new URL(route.request().url());
    const lead = url.searchParams.get("lead_time_hours") ?? "unknown";
    dispatchedLeads.push(lead);

    if (lead === "6") {
      // Hold lead=6 requests in flight until released
      await holdPromise;
    }
    await route.fulfill({
      status: 200,
      contentType: "image/png",
      headers: { "Cache-Control": "no-cache" },
      body: Buffer.alloc(0),
    });
  });

  await page.goto("/");
  await expect(page.getByTestId("weather-map")).toBeVisible();

  // Clear initial 0h requests
  dispatchedLeads.length = 0;

  // 1. User selects +6h (tiles will be held in flight)
  const leadSelect = page.getByLabel("Lead time");
  await leadSelect.selectOption("6");

  // Verify lead=6 tile requests are dispatched and currently in flight
  await expect.poll(() => dispatchedLeads.includes("6")).toBe(true);

  // 2. While lead=6 tiles are in flight, user switches to +12h
  await leadSelect.selectOption("12");

  // Invariant: Lead=12 requests MUST be dispatched immediately without waiting for lead=6
  await expect.poll(() => dispatchedLeads.includes("12")).toBe(true);

  // Release the held lead=6 requests
  if (delayedResolve) {
    (delayedResolve as () => void)();
  }

  // The selector value remains authoritative at +12h
  await expect(leadSelect).toHaveValue("12");
});

test("forecast map rapid transition: A -> B -> C rapidly switches and C is authoritative", async ({
  page,
}) => {
  const dispatchedLeads: string[] = [];

  await page.route("**/v1/maps/**/*.png*", async (route) => {
    const url = new URL(route.request().url());
    const lead = url.searchParams.get("lead_time_hours") ?? "unknown";
    dispatchedLeads.push(lead);
    // Simulate realistic network delay
    await new Promise((r) => setTimeout(r, 100));
    await route.fulfill({
      status: 200,
      contentType: "image/png",
      headers: { "Cache-Control": "no-cache" },
      body: Buffer.alloc(0),
    });
  });

  await page.goto("/");
  await expect(page.getByTestId("weather-map")).toBeVisible();

  dispatchedLeads.length = 0;
  const leadSelect = page.getByLabel("Lead time");

  // Rapidly switch through 6h -> 12h -> 18h
  await leadSelect.selectOption("6");
  await leadSelect.selectOption("12");
  await leadSelect.selectOption("18");

  // All transitions should have triggered without dropping
  await expect.poll(() => dispatchedLeads.includes("18")).toBe(true);
  await expect(leadSelect).toHaveValue("18");
});

test("state sync regression: GFS precipitation -> GEFS switch queries temperature without stale precipitation error", async ({
  page,
}) => {
  const ensembleRequests: { model: string; variable: string; lead: string }[] = [];

  // Track all /v1/ensembles requests
  await page.route("**/v1/ensembles?*", async (route) => {
    const url = new URL(route.request().url());
    const model = url.searchParams.get("model") ?? "unknown";
    const variable = url.searchParams.get("variable") ?? "unknown";
    const lead = url.searchParams.get("lead_time_hours") ?? "0";
    ensembleRequests.push({ model, variable, lead });

    // Fallback or continue default mock
    await route.fallback();
  });

  // Delay /v1/points responses by 200ms to realistically exercise the async transition window
  await page.route("**/v1/points?*", async (route) => {
    await new Promise((r) => setTimeout(r, 200));
    await route.fallback();
  });

  await page.goto("/");

  // 1. Select a location to open the dashboard
  const input = page.getByLabel(/Search for a city/);
  await input.fill("Aspen");
  await searchResults(page).getByRole("option", { name: /Aspen/ }).first().click();
  await expect(page.getByText("Hourly Forecast")).toBeVisible();

  // 2. Select Precipitation Rate on GFS
  const variableSelect = page.getByLabel("Variable");
  await variableSelect.selectOption("precipitation_rate");
  await expect(variableSelect).toHaveValue("precipitation_rate");

  // Clear any previous ensemble requests
  ensembleRequests.length = 0;

  // 3. Switch model to GEFS
  // ForecastSelectionProvider normalizes GEFS to default variable: temperature_2m
  const modelSelect = page.getByLabel("Model");
  await modelSelect.selectOption("gefs");

  // 4. Verify authoritative selection and UI agreement
  await expect(modelSelect).toHaveValue("gefs");
  await expect(variableSelect).toHaveValue("temperature_2m");

  // 5. Ensemble Statistics (GEFS) panel must appear and render without error
  await expect(page.getByText(/Ensemble Statistics \(GEFS\)/)).toBeVisible();
  await expect(
    page.getByRole("img", { name: /ensemble percentile fan over lead time/ })
  ).toBeVisible();

  // 6. Assert that ALL ensemble requests dispatched for GEFS used temperature_2m
  await expect.poll(() => ensembleRequests.length > 0).toBe(true);
  for (const req of ensembleRequests) {
    if (req.model === "gefs") {
      expect(req.variable).toBe("temperature_2m");
    }
  }

  // Ensure NO request ever went out requesting GEFS + precipitation_rate
  const invalidRequests = ensembleRequests.filter(
    (req) => req.model === "gefs" && req.variable === "precipitation_rate"
  );
  expect(invalidRequests).toHaveLength(0);

  // 7. Ensure no application error alert exists on the page
  const appAlerts = page.locator('[role="alert"]:not(#__next-route-announcer__)');
  await expect(appAlerts).toHaveCount(0);
  await expect(page.getByText(/Failed to load/i)).toHaveCount(0);
});

test("phase 1a variable expansion: switching through all Phase 1A variables updates map and legend", async ({
  page,
}) => {
  await page.goto("/");
  await expect(page.getByTestId("weather-map")).toBeVisible();

  const variableSelect = page.getByLabel("Variable");
  const legend = page.getByTestId("legend-gradient");

  // 1. Relative humidity
  await variableSelect.selectOption("relative_humidity_2m");
  await expect(variableSelect).toHaveValue("relative_humidity_2m");
  await expect(legend).toBeVisible();

  // 2. Wind gust
  await variableSelect.selectOption("wind_gust");
  await expect(variableSelect).toHaveValue("wind_gust");
  await expect(legend).toBeVisible();

  // 3. Visibility
  await variableSelect.selectOption("visibility");
  await expect(variableSelect).toHaveValue("visibility");
  await expect(legend).toBeVisible();

  // 4. Snow depth
  await variableSelect.selectOption("snow_depth");
  await expect(variableSelect).toHaveValue("snow_depth");
  await expect(legend).toBeVisible();

  // Ensure no error alert was produced during rapid switching
  const appAlerts = page.locator('[role="alert"]:not(#__next-route-announcer__)');
  await expect(appAlerts).toHaveCount(0);
});

test("phase 1a gefs variable selection: selecting GEFS relative humidity updates ensemble panel", async ({
  page,
}) => {
  await page.goto("/");

  const modelSelect = page.getByLabel("Model");
  await modelSelect.selectOption("gefs");

  const variableSelect = page.getByLabel("Variable");
  await variableSelect.selectOption("relative_humidity_2m");

  const input = page.getByLabel(/Search for a city/);
  await input.fill("Aspen");
  await searchResults(page).getByRole("option", { name: /Aspen/ }).first().click();

  await expect(page.getByText(/Ensemble Statistics \(GEFS\)/)).toBeVisible();
  await expect(
    page.getByRole("img", { name: /ensemble percentile fan over lead time/ })
  ).toBeVisible();
});

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

test("phase 1b wind product: selecting Wind updates map, meteogram, and ensemble Wind Rose", async ({
  page,
}) => {
  await page.goto("/");
  await expect(page.getByTestId("weather-map")).toBeVisible();

  const variableSelect = page.getByLabel("Variable");

  // 1. Verify raw U/V are not present in variable options
  const options = await variableSelect.locator("option").allTextContents();
  expect(options).not.toContain("10-Meter U Wind Component");
  expect(options).not.toContain("10-Meter V Wind Component");
  expect(options).toContain("10-Meter Wind");

  // 2. Select Wind product
  await variableSelect.selectOption("wind_10m");
  await expect(variableSelect).toHaveValue("wind_10m");
  await expect(page.getByTestId("legend-gradient")).toBeVisible();

  // 3. Search and select a city to view the point forecast
  const input = page.getByLabel(/Search for a city/);
  await input.fill("Aspen");
  await searchResults(page).getByRole("option", { name: /Aspen/ }).first().click();

  // 4. Verify meteograms include 10-Meter Wind
  await expect(page.getByText("Hourly Forecast")).toBeVisible();
  await expect(
    page.getByRole("img", { name: /10-Meter Wind hourly forecast over lead time/ })
  ).toBeVisible();

  // 5. Switch to GEFS model to test Ensemble Wind Rose
  const modelSelect = page.getByLabel("Model");
  await modelSelect.selectOption("gefs");
  await variableSelect.selectOption("wind_10m");

  // 6. Verify Ensemble Statistics and Wind Rose appear
  await expect(page.getByText(/Ensemble Statistics \(GEFS\)/)).toBeVisible();
  await expect(page.getByText(/Wind Direction & Speed Distribution \(Wind Rose\)/)).toBeVisible();
  await expect(page.getByRole("img", { name: /ensemble wind rose chart/i })).toBeVisible();
  await expect(page.getByText("CALM")).toBeVisible();
});

test("phase 1b.3 animated wind map: progressive rendering, lead switching, consensus flow, and reduced motion", async ({
  page,
}) => {
  const vectorRequests: { model: string; lead: string }[] = [];

  await page.route("**/v1/maps/**/vector-field*", async (route) => {
    const url = new URL(route.request().url());
    const lead = url.searchParams.get("lead_time_hours") ?? "0";
    const isGefs = url.pathname.includes("/gefs/");
    vectorRequests.push({ model: isGefs ? "gefs" : "gfs", lead });
    await route.fallback();
  });

  await page.goto("/");
  await expect(page.getByTestId("weather-map")).toBeVisible();

  const variableSelect = page.getByLabel("Variable");
  const leadSelect = page.getByLabel("Lead time");
  const modelSelect = page.getByLabel("Model");
  const canvas = page.getByTestId("wind-particle-canvas");

  // 1. Select Wind product -> Stage A (scalar raster) and Stage B (particle canvas)
  await variableSelect.selectOption("wind_10m");
  await expect(variableSelect).toHaveValue("wind_10m");
  await expect(page.getByTestId("legend-gradient")).toBeVisible();
  await expect(canvas).toBeVisible();

  // Vector field request dispatched for GFS lead 0
  await expect.poll(() => vectorRequests.some((r) => r.model === "gfs")).toBe(true);

  // 2. Scrub through lead times (0h -> 6h -> 12h)
  await leadSelect.selectOption("6");
  await expect.poll(() => vectorRequests.some((r) => r.lead === "6")).toBe(true);
  await expect(leadSelect).toHaveValue("6");

  await leadSelect.selectOption("12");
  await expect.poll(() => vectorRequests.some((r) => r.lead === "12")).toBe(true);
  await expect(leadSelect).toHaveValue("12");

  // 3. Switch GFS -> GEFS consensus flow
  await modelSelect.selectOption("gefs");
  await variableSelect.selectOption("wind_10m");
  await expect(modelSelect).toHaveValue("gefs");
  await expect(variableSelect).toHaveValue("wind_10m");
  await expect.poll(() => vectorRequests.some((r) => r.model === "gefs")).toBe(true);

  // 4. Switch Wind -> Temperature -> particle canvas inactive
  await variableSelect.selectOption("temperature_2m");
  await expect(variableSelect).toHaveValue("temperature_2m");

  // 5. Reduced motion: simulate prefers-reduced-motion: reduce
  await page.emulateMedia({ reducedMotion: "reduce" });
  await variableSelect.selectOption("wind_10m");
  await expect(variableSelect).toHaveValue("wind_10m");
  // Raster remains functional
  await expect(page.getByTestId("legend-gradient")).toBeVisible();
  await expect(page.getByTestId("weather-map")).toBeVisible();
});

test("phase 1c.3 3-hour precipitation: amount, phase evolution, GEFS 100% phase support, and map UX", async ({
  page,
}) => {
  await page.goto("/");
  await expect(page.getByTestId("weather-map")).toBeVisible();

  const variableSelect = page.getByLabel("Variable");

  // 1. Invariant: Raw categorical flags (crain, csnow, cfrzr, cicep) are NEVER present in selectors
  const options = await variableSelect.locator("option").allTextContents();
  expect(options).not.toContain("crain");
  expect(options).not.toContain("csnow");
  expect(options).not.toContain("cfrzr");
  expect(options).not.toContain("cicep");
  expect(options).toContain("3-Hour Precipitation");

  // 2. Select 3-Hour Precipitation on GFS
  await variableSelect.selectOption("precipitation_amount_3h");
  await expect(variableSelect).toHaveValue("precipitation_amount_3h");
  await expect(page.getByTestId("legend-gradient")).toBeVisible();
  await expect(page.getByText("3-Hour Precipitation (mm)")).toBeVisible();

  // 3. Search and select a city to inspect Point Forecast meteogram
  const input = page.getByLabel(/Search for a city/);
  await input.fill("Aspen");
  await searchResults(page).getByRole("option", { name: /Aspen/ }).first().click();

  await expect(page.getByText("Hourly Forecast")).toBeVisible();
  await expect(
    page.getByRole("img", { name: /3-Hour Precipitation hourly forecast over lead time/ })
  ).toBeVisible();

  // Verify Phase legend badges in meteogram
  await expect(page.getByText("Phases:")).toBeVisible();
  await expect(page.getByText("Rain", { exact: true })).toBeVisible();
  await expect(page.getByText("Snow", { exact: true })).toBeVisible();
  await expect(page.getByText("Freezing Rain", { exact: true })).toBeVisible();
  await expect(page.getByText("Ice Pellets", { exact: true })).toBeVisible();
  await expect(page.getByText("Mixed", { exact: true })).toBeVisible();

  // Scroll meteogram into view and screenshot
  const precipMeteogram = page.getByRole("img", {
    name: /3-Hour Precipitation hourly forecast over lead time/,
  });
  await precipMeteogram.scrollIntoViewIfNeeded();
  await page.screenshot({ path: "e2e/screenshots/1c3-gfs-precipitation.png" });

  // 4. Switch to GEFS ensemble model with 3-Hour Precipitation
  const modelSelect = page.getByLabel("Model");
  await modelSelect.selectOption("gefs");
  await variableSelect.selectOption("precipitation_amount_3h");

  // 5. Verify GEFS Ensemble Phase Support 100% composition visualization
  await expect(page.getByText(/Ensemble Statistics \(GEFS\)/)).toBeVisible();
  await expect(page.getByText(/Ensemble Phase Support/)).toBeVisible();
  const phaseChart = page.getByRole("img", {
    name: /ensemble phase support composition/i,
  });
  await expect(phaseChart).toBeVisible();

  // Scroll phase support into view and screenshot
  await phaseChart.scrollIntoViewIfNeeded();
  await page.screenshot({ path: "e2e/screenshots/1c3-gefs-phase-support.png" });

  // Verify all 6 physical phase breakdown values and no mixed segment
  await expect(page.getByTestId("phase-segment-dry")).toBeVisible();
  await expect(page.getByTestId("phase-segment-rain")).toBeVisible();
  await expect(page.getByTestId("phase-segment-snow")).toBeVisible();
  await expect(page.getByTestId("phase-segment-freezing_rain")).toBeVisible();
  await expect(page.getByTestId("phase-segment-ice_pellets")).toBeVisible();
  await expect(page.getByTestId("phase-segment-unknown")).toBeVisible();
  await expect(page.locator('[data-testid="phase-segment-mixed"]')).toHaveCount(0);

  // Verify percentages
  await expect(page.getByText("52%").first()).toBeVisible(); // Rain
  await expect(page.getByText("26%").first()).toBeVisible(); // Snow
  await expect(page.getByText("10%").first()).toBeVisible(); // Dry
  await expect(page.getByText("8%").first()).toBeVisible(); // Freezing Rain
  await expect(page.getByText("3%").first()).toBeVisible(); // Ice Pellets
  await expect(page.getByText("1%").first()).toBeVisible(); // Unknown

  // Verify secondary transition frequency
  await expect(page.getByText(/Member Phase Transitions/)).toBeVisible();
  await expect(page.getByText("Rain → Snow")).toBeVisible();
  await expect(page.getByText("· 27%")).toBeVisible();

  // Ensure no error alert exists
  const appAlerts = page.locator('[role="alert"]:not(#__next-route-announcer__)');
  await expect(appAlerts).toHaveCount(0);
});

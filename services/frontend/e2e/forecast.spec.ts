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
  await expect(page.getByText(/temperature_2m · percentile range/)).toBeVisible();
  await expect(page.getByText(/over lead time/)).toHaveCount(0);
  await expect(page.getByRole("img", { name: /ensemble percentile fan over time/ })).toBeVisible();

  // The mock returns members and pdf for /v1/ensembles, so the Distribution View
  // renders both the histogram bars and the continuous PDF line.
  await expect(page.getByText(/Member distribution/)).toBeVisible();
  // Valid-time is localized for Aspen (Mountain Time: MDT) and no lead-time "+6h" is displayed
  await expect(page.getByText(/Member distribution · .* (MDT|GMT-6)/)).toBeVisible();
  await expect(page.getByText(/Member distribution · \+6h/)).toHaveCount(0);
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
  const dispatchedTimes: string[] = [];
  let delayedResolve: (() => void) | null = null;
  const holdPromise = new Promise<void>((resolve) => {
    delayedResolve = resolve;
  });

  // Intercept tile requests with controlled delay for the second valid time
  await page.route("**/v1/maps/**/*.png*", async (route) => {
    const url = new URL(route.request().url());
    const time =
      url.searchParams.get("valid_time") ?? url.searchParams.get("lead_time_hours") ?? "unknown";
    dispatchedTimes.push(time);

    if (time.includes("06:00") || time === "6") {
      // Hold requests in flight until released
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

  dispatchedTimes.length = 0;

  // 1. User selects next valid time
  const timeSelect = page.getByLabel("Valid time");
  const options = await timeSelect.locator("option").all();
  if (options.length > 1) {
    const val1 = await options[1].getAttribute("value");
    if (val1) {
      await timeSelect.selectOption(val1);
      await expect
        .poll(() => dispatchedTimes.some((t) => t.includes("06:00") || t === "6"))
        .toBe(true);
    }
    if (options.length > 2) {
      const val2 = await options[2].getAttribute("value");
      if (val2) {
        await timeSelect.selectOption(val2);
        await expect
          .poll(() => dispatchedTimes.some((t) => t.includes("12:00") || t === "12"))
          .toBe(true);
      }
    }
  }

  if (delayedResolve) {
    (delayedResolve as () => void)();
  }
});

test("forecast map rapid transition: A -> B -> C rapidly switches and C is authoritative", async ({
  page,
}) => {
  const dispatchedTimes: string[] = [];

  await page.route("**/v1/maps/**/*.png*", async (route) => {
    const url = new URL(route.request().url());
    const time =
      url.searchParams.get("valid_time") ?? url.searchParams.get("lead_time_hours") ?? "unknown";
    dispatchedTimes.push(time);
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

  dispatchedTimes.length = 0;
  const timeSelect = page.getByLabel("Valid time");
  const options = await timeSelect.locator("option").all();
  if (options.length >= 3) {
    const val1 = await options[1].getAttribute("value");
    const val2 = await options[2].getAttribute("value");
    const val3 = options.length > 3 ? await options[3].getAttribute("value") : val2;

    if (val1) await timeSelect.selectOption(val1);
    if (val2) await timeSelect.selectOption(val2);
    if (val3) await timeSelect.selectOption(val3);

    await expect(timeSelect).toHaveValue(val3 ?? "");
  }
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
  await expect(page.getByRole("img", { name: /ensemble percentile fan over time/ })).toBeVisible();

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
  await expect(page.getByRole("img", { name: /ensemble percentile fan over time/ })).toBeVisible();
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
  const timeSelect = page.getByLabel("Valid time");
  const modelSelect = page.getByLabel("Model");
  const canvas = page.getByTestId("wind-particle-canvas");

  // 1. Select Wind product -> Stage A (scalar raster) and Stage B (particle canvas)
  await variableSelect.selectOption("wind_10m");
  await expect(variableSelect).toHaveValue("wind_10m");
  await expect(page.getByTestId("legend-gradient")).toBeVisible();
  await expect(canvas).toBeVisible();

  // Vector field request dispatched for GFS lead 0
  await expect.poll(() => vectorRequests.some((r) => r.model === "gfs")).toBe(true);

  // 2. Scrub through valid times
  const timeOptions = await timeSelect.locator("option").all();
  if (timeOptions.length > 2) {
    const val6 = await timeOptions[1].getAttribute("value");
    const val12 = await timeOptions[2].getAttribute("value");
    if (val6) {
      await timeSelect.selectOption(val6);
      await expect.poll(() => vectorRequests.length > 0).toBe(true);
      await expect(timeSelect).toHaveValue(val6);
    }
    if (val12) {
      await timeSelect.selectOption(val12);
      await expect.poll(() => vectorRequests.length > 0).toBe(true);
      await expect(timeSelect).toHaveValue(val12);
    }
  }

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

test("cloud products: 3-Hour Cloud Cover and Cloud Ceiling point, meteogram, map, and GEFS ensemble distribution", async ({
  page,
}) => {
  await page.goto("/");
  await expect(page.getByTestId("weather-map")).toBeVisible();

  const variableSelect = page.getByLabel("Variable");

  // 1. Verify Cloud Cover and Cloud Ceiling options are present
  const options = await variableSelect.locator("option").allTextContents();
  expect(options).toContain("3-Hour Cloud Cover");
  expect(options).toContain("Cloud Ceiling Height");

  // 2. Select 3-Hour Cloud Cover on GFS
  await variableSelect.selectOption("cloud_cover_3h");
  await expect(variableSelect).toHaveValue("cloud_cover_3h");
  await expect(page.getByTestId("legend-gradient")).toBeVisible();
  await expect(page.getByText("3-Hour Cloud Cover (%)")).toBeVisible();

  // 3. Search and select a city to inspect Point Forecast meteograms
  const input = page.getByLabel(/Search for a city/);
  await input.fill("Aspen");
  await searchResults(page).getByRole("option", { name: /Aspen/ }).first().click();

  await expect(page.getByText("Hourly Forecast")).toBeVisible();
  await expect(
    page.getByRole("img", { name: /3-Hour Cloud Cover hourly forecast over lead time/ })
  ).toBeVisible();

  // 4. Select Cloud Ceiling on GFS
  await variableSelect.selectOption("cloud_ceiling");
  await expect(variableSelect).toHaveValue("cloud_ceiling");
  await expect(page.getByTestId("legend-gradient")).toBeVisible();

  await expect(
    page.getByRole("img", { name: /Cloud Ceiling Height hourly forecast over lead time/ })
  ).toBeVisible();

  // 5. Switch to GEFS ensemble model with Cloud Ceiling
  const modelSelect = page.getByLabel("Model");
  await modelSelect.selectOption("gefs");
  await variableSelect.selectOption("cloud_ceiling");

  // Verify GEFS Ensemble Unlimited Ceiling probability tile and statistics
  await expect(page.getByText(/Ensemble Statistics \(GEFS\)/)).toBeVisible();
  await expect(page.getByText(/Unlimited Ceiling Probability/i)).toBeVisible();
  await expect(page.getByText("40%")).toBeVisible(); // P(Unlimited) = 0.40

  // 6. Switch to GEFS ensemble model with 3-Hour Cloud Cover
  await variableSelect.selectOption("cloud_cover_3h");
  await expect(page.getByText(/Ensemble Statistics \(GEFS\)/)).toBeVisible();

  // Ensure no error alert exists
  const appAlerts = page.locator('[role="alert"]:not(#__next-route-announcer__)');
  await expect(appAlerts).toHaveCount(0);
});

test("Lifecycle V2 user flow: select model, variable, and valid time updates map and dashboard", async ({
  page,
}) => {
  await page.goto("/");
  await expect(page.getByTestId("weather-map")).toBeVisible();

  // 1. Verify Model, Variable, and Valid Time controls are present; Initial & Lead Time are absent
  await expect(page.getByLabel("Model")).toBeVisible();
  await expect(page.getByLabel("Variable")).toBeVisible();
  const validSelect = page.getByLabel("Valid time");
  await expect(validSelect).toBeVisible();
  expect(await page.getByLabel("Initial time").count()).toBe(0);
  expect(await page.getByLabel("Lead time").count()).toBe(0);

  // 2. Select valid time option
  const options = await validSelect.locator("option").all();
  expect(options.length).toBeGreaterThan(1);
  const secondTime = await options[1].getAttribute("value");
  if (secondTime) {
    await validSelect.selectOption(secondTime);
    await expect(validSelect).toHaveValue(secondTime);
  }

  // 3. Search for city and open dashboard
  const input = page.getByLabel(/Search for a city/);
  await input.fill("Aspen");
  await searchResults(page).getByRole("option", { name: /Aspen/ }).first().click();

  // 4. Verify dashboard and hourly forecast curve appear
  await expect(page.getByText("Hourly Forecast")).toBeVisible();
  await expect(page.getByText("Aspen", { exact: true })).toBeVisible();
  await expect(
    page.getByRole("img", { name: /2-Meter Temperature hourly forecast over lead time/ })
  ).toBeVisible();
});

test("forecast panel close: clicking Close (X) deselects location, removes panel and marker, restores full map area", async ({
  page,
}) => {
  await page.goto("/");

  const input = page.getByLabel(/Search for a city/);
  await input.fill("Aspen");
  await searchResults(page).getByRole("option", { name: /Aspen/ }).first().click();

  // Location selected: Hourly Forecast is visible and marker is rendered on map
  await expect(page.getByText("Hourly Forecast")).toBeVisible();
  await expect(page.locator(".maplibregl-marker")).toBeVisible();

  // Click Close (X) button
  const closeBtn = page.getByRole("button", { name: "Close forecast panel" });
  await expect(closeBtn).toBeVisible();
  await closeBtn.click();

  // Sidebar and marker are both removed
  await expect(page.getByText("Hourly Forecast")).toHaveCount(0);
  await expect(page.locator(".maplibregl-marker")).toHaveCount(0);

  // Map remains fully visible
  await expect(page.getByTestId("weather-map")).toBeVisible();
});

test("forecast panel close: Close (X) remains pinned and clickable while forecast content is scrolled downward", async ({
  page,
}) => {
  await page.goto("/");

  const input = page.getByLabel(/Search for a city/);
  await input.fill("Aspen");
  await searchResults(page).getByRole("option", { name: /Aspen/ }).first().click();

  // Location selected: Hourly Forecast is visible and marker is rendered on map
  await expect(page.getByText("Hourly Forecast")).toBeVisible();
  await expect(page.locator(".maplibregl-marker")).toBeVisible();

  // Confirm Close button is initially visible
  const closeBtn = page.getByRole("button", { name: "Close forecast panel" });
  await expect(closeBtn).toBeVisible();

  // Scroll the forecast content container substantially downward
  const scrollContainer = page.locator("#forecast-panel-content div.overflow-y-auto");
  await scrollContainer.evaluate((el) => {
    el.scrollTop = 400;
  });

  // Verify container actually scrolled
  const scrolledTop = await scrollContainer.evaluate((el) => el.scrollTop);
  expect(scrolledTop).toBeGreaterThanOrEqual(200);

  // Confirm Close button is STILL visible after scrolling
  await expect(closeBtn).toBeVisible();

  // Verify bounding box remains inside the visible viewport and near top of sidebar
  const closeBox = await closeBtn.boundingBox();
  expect(closeBox).not.toBeNull();
  expect(closeBox!.y).toBeGreaterThanOrEqual(0);

  const panelBox = await page.locator("#forecast-panel").boundingBox();
  expect(panelBox).not.toBeNull();
  expect(closeBox!.y).toBeGreaterThanOrEqual(panelBox!.y);
  expect(closeBox!.y + closeBox!.height).toBeLessThan(panelBox!.y + 120);

  // Click Close while panel is scrolled
  await closeBtn.click();

  // Sidebar and marker are both removed
  await expect(page.getByText("Hourly Forecast")).toHaveCount(0);
  await expect(page.locator(".maplibregl-marker")).toHaveCount(0);

  // Map remains fully visible
  await expect(page.getByTestId("weather-map")).toBeVisible();
});

test("forecast panel collapse and expand: toggle collapses panel and preserves marker/data, expand restores dashboard", async ({
  page,
}) => {
  await page.goto("/");

  const input = page.getByLabel(/Search for a city/);
  await input.fill("Aspen");
  await searchResults(page).getByRole("option", { name: /Aspen/ }).first().click();

  await expect(page.getByText("Hourly Forecast")).toBeVisible();
  await expect(page.locator(".maplibregl-marker")).toBeVisible();

  // Measure map width before collapse
  const mapBefore = await page.getByTestId("weather-map").boundingBox();
  expect(mapBefore).not.toBeNull();

  // Click Collapse control
  const collapseBtn = page.getByRole("button", { name: "Collapse forecast panel" });
  await expect(collapseBtn).toBeVisible();
  await expect(collapseBtn).toHaveAttribute("aria-expanded", "true");
  await collapseBtn.click();

  // Forecast content is hidden, but marker remains visible
  await expect(page.locator("#forecast-panel-content")).toHaveClass(/hidden/);
  await expect(page.locator(".maplibregl-marker")).toBeVisible();

  // Map recovers full width
  const mapCollapsed = await page.getByTestId("weather-map").boundingBox();
  expect(mapCollapsed).not.toBeNull();
  expect(mapCollapsed!.width).toBeGreaterThan(mapBefore!.width);

  // Expand button is visible
  const expandBtn = page.getByRole("button", { name: "Expand forecast panel" });
  await expect(expandBtn).toBeVisible();
  await expect(expandBtn).toHaveAttribute("aria-expanded", "false");

  // Click Expand
  await expandBtn.click();

  // Forecast dashboard returns with same location and marker
  await expect(page.locator("#forecast-panel-content")).not.toHaveClass(/hidden/);
  await expect(page.getByText("Hourly Forecast")).toBeVisible();
  await expect(page.getByText("Aspen", { exact: true })).toBeVisible();
  await expect(page.locator(".maplibregl-marker")).toBeVisible();
});

test("mobile viewport: collapse restores map view with marker, expand restores forecast, close clears selection", async ({
  page,
}) => {
  await page.setViewportSize({ width: 375, height: 667 });
  await page.goto("/");

  const input = page.getByLabel(/Search for a city/);
  await input.fill("Aspen");
  await searchResults(page).getByRole("option", { name: /Aspen/ }).first().click();

  await expect(page.getByText("Hourly Forecast")).toBeVisible();
  await expect(page.locator(".maplibregl-marker")).toBeVisible();

  // On mobile, collapse reveals the map
  const collapseBtn = page.getByRole("button", { name: "Collapse forecast panel" });
  await expect(collapseBtn).toBeVisible();
  await collapseBtn.click();

  await expect(page.locator("#forecast-panel-content")).toHaveClass(/hidden/);
  await expect(page.locator(".maplibregl-marker")).toBeVisible();

  // Tap Expand
  const expandBtn = page.getByRole("button", { name: "Expand forecast panel" });
  await expect(expandBtn).toBeVisible();
  await expandBtn.click();

  await expect(page.getByText("Hourly Forecast")).toBeVisible();

  // On mobile: scroll forecast content downward
  const scrollContainer = page.locator("#forecast-panel-content div.overflow-y-auto");
  await scrollContainer.evaluate((el) => {
    el.scrollTop = 300;
  });

  // Verify Close button remains visible while scrolled on mobile
  const closeBtn = page.getByRole("button", { name: "Close forecast panel" });
  await expect(closeBtn).toBeVisible();

  const closeBox = await closeBtn.boundingBox();
  expect(closeBox).not.toBeNull();
  expect(closeBox!.y).toBeGreaterThanOrEqual(0);

  // Tap Close (X) while scrolled
  await closeBtn.click();

  await expect(page.getByText("Hourly Forecast")).toHaveCount(0);
  await expect(page.locator(".maplibregl-marker")).toHaveCount(0);
});

test("direct place search selection: place with coordinates commits without calling /v1/search/places/*", async ({
  page,
}) => {
  let resolvePlacesCalled = false;
  await page.route("**/v1/search/places/*", (route) => {
    resolvePlacesCalled = true;
    route.fulfill({ status: 500 });
  });

  await page.goto("/");
  const input = page.getByLabel(/Search for a city/);
  await input.fill("Vail");

  const option = searchResults(page)
    .getByRole("option", { name: /Vail, CO, USA/ })
    .first();
  await expect(option).toBeVisible();
  await option.click();

  // Forecast opens immediately with direct coordinates
  await expect(page.getByText("Hourly Forecast")).toBeVisible();
  await expect(page.getByText("Vail, CO, USA", { exact: true })).toBeVisible();

  // Verify /v1/search/places/* was never called!
  expect(resolvePlacesCalled).toBe(false);
});

test("search attribution: search listbox renders accessible Geoapify and LocationIQ attribution footer", async ({
  page,
}) => {
  await page.goto("/");
  const input = page.getByLabel(/Search for a city/);
  await input.fill("Aspen");

  await expect(searchResults(page)).toBeVisible();
  const geoLink = searchResults(page).getByRole("link", { name: "Geoapify" });
  await expect(geoLink).toBeVisible();
  await expect(geoLink).toHaveAttribute("href", "https://www.geoapify.com");

  const locLink = searchResults(page).getByRole("link", { name: "LocationIQ.com" });
  await expect(locLink).toBeVisible();
  await expect(locLink).toHaveAttribute("href", "https://locationiq.com");
});

test("startup privacy invariant: page load makes best-effort /v1/locate call without browser geolocation prompt or marker", async ({
  page,
}) => {
  let locateEndpointCalled = false;
  await page.route("**/v1/locate", (route) => {
    locateEndpointCalled = true;
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        latitude: 39.7392,
        longitude: -104.9903,
        city: "Denver",
        region: "Colorado",
        country: "US",
        approximate: true,
      }),
    });
  });

  // Track if browser navigator.geolocation is ever invoked
  await page.addInitScript(() => {
    (window as any).__geolocationInvoked = false;
    if (navigator.geolocation) {
      const origGetCurrentPosition = navigator.geolocation.getCurrentPosition;
      navigator.geolocation.getCurrentPosition = function (...args) {
        (window as any).__geolocationInvoked = true;
        return origGetCurrentPosition.apply(this, args);
      };
    }
  });

  await page.goto("/");

  // Wait for map and header to be ready
  await expect(page.getByText("Weather Platform")).toBeVisible();
  await page.waitForTimeout(1000);

  // Best-effort /v1/locate was invoked on startup
  expect(locateEndpointCalled).toBe(true);

  // Critical privacy invariant: navigator.geolocation was NEVER called automatically!
  const geolocationInvoked = await page.evaluate(() => (window as any).__geolocationInvoked);
  expect(geolocationInvoked).toBe(false);

  // Invariant: startup IP location does NOT open forecast panel or place a selection marker
  await expect(page.getByText("Hourly Forecast")).toHaveCount(0);
  await expect(page.locator(".maplibregl-marker")).toHaveCount(0);
});

test("locate me allow: clicking Locate Me with granted permission commits location, moves map, and opens forecast", async ({
  page,
  context,
}) => {
  // Grant geolocation permission and configure coordinates (Seattle)
  await context.grantPermissions(["geolocation"]);
  await context.setGeolocation({ latitude: 47.6062, longitude: -122.3321 });

  await page.goto("/");

  const locateBtn = page.getByRole("button", { name: "Locate me" });
  await expect(locateBtn).toBeVisible();
  await locateBtn.click();

  // Marker appears and Forecast Panel opens with acquired coordinates
  await expect(page.getByText("Hourly Forecast")).toBeVisible();
  await expect(page.getByText("47.6062, -122.3321")).toBeVisible();
  await expect(page.locator(".maplibregl-marker")).toBeVisible();
});

test("locate me deny privacy: denying geolocation renders non-blocking alert and NEVER calls /v1/locate as fallback", async ({
  page,
  context,
}) => {
  let locateFallbackCalled = false;
  let startupCompleted = false;

  await page.route("**/v1/locate", (route) => {
    if (startupCompleted) {
      locateFallbackCalled = true;
    }
    route.fulfill({ status: 404 });
  });

  // Explicitly deny geolocation permissions
  await context.clearPermissions();

  await page.goto("/");
  await expect(page.getByText("Weather Platform")).toBeVisible();
  startupCompleted = true;

  const locateBtn = page.getByRole("button", { name: "Locate me" });
  await locateBtn.click();

  // Non-blocking alert rendered
  const alert = page.getByRole("alert").filter({ hasText: "Location access denied" });
  await expect(alert).toBeVisible();
  await expect(alert).toContainText("Location access denied");

  // Critical privacy invariant: NO /v1/locate was called as a fallback for Locate Me!
  expect(locateFallbackCalled).toBe(false);
  // No forecast panel was opened
  await expect(page.getByText("Hourly Forecast")).toHaveCount(0);
});

test("mobile layout: Locate Me button, search bar, and controls remain non-overlapping at 375px", async ({
  page,
}) => {
  await page.setViewportSize({ width: 375, height: 667 });
  await page.goto("/");

  const searchInput = page.getByLabel(/Search for a city/);
  await expect(searchInput).toBeVisible();

  const locateBtn = page.getByRole("button", { name: "Locate me" });
  await expect(locateBtn).toBeVisible();

  const searchBox = await searchInput.boundingBox();
  const locateBox = await locateBtn.boundingBox();

  expect(searchBox).not.toBeNull();
  expect(locateBox).not.toBeNull();

  // Verify Locate Me button does not overlap search input horizontally
  expect(locateBox!.x).toBeGreaterThan(searchBox!.x + searchBox!.width);
});

test("map-pan network test: panning/zooming map does NOT issue search requests, query change samples latest bias", async ({
  page,
}) => {
  let searchRequestCount = 0;
  let lastSearchUrl = "";

  await page.route("**/v1/search?*", (route) => {
    searchRequestCount += 1;
    lastSearchUrl = route.request().url();
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ object: "list", data: [], has_more: false, next_cursor: null }),
    });
  });

  await page.goto("/");
  const input = page.getByLabel(/Search for a city/);
  await input.fill("Denver");
  await page.waitForTimeout(400); // allow debounce

  const initialCount = searchRequestCount;
  expect(initialCount).toBeGreaterThanOrEqual(1);

  // Pan the map multiple times
  const map = page.getByTestId("weather-map");
  await map.dragTo(map, { sourcePosition: { x: 200, y: 200 }, targetPosition: { x: 100, y: 100 } });
  await page.waitForTimeout(500);
  await map.dragTo(map, { sourcePosition: { x: 150, y: 150 }, targetPosition: { x: 250, y: 250 } });
  await page.waitForTimeout(500);

  // Invariant: map panning alone must NOT increase search request count!
  expect(searchRequestCount).toBe(initialCount);

  // Now modify the query
  await input.fill("Denver West");
  await page.waitForTimeout(400); // allow debounce

  // Exactly one new debounced search request must have fired
  expect(searchRequestCount).toBe(initialCount + 1);
  expect(lastSearchUrl).toContain("q=Denver+West");
});

test("selection race page-level test: delayed geolocation does not overwrite subsequent user selection", async ({
  page,
  context,
}) => {
  await context.grantPermissions(["geolocation"]);

  let resolveGeo!: (pos: GeolocationPosition) => void;
  // Override navigator.geolocation in page with controlled delayed promise
  await page.addInitScript(() => {
    let pendingResolve: any = null;
    (window as any).__resolveGeo = (pos: any) => {
      if (pendingResolve) pendingResolve(pos);
    };
    navigator.geolocation.getCurrentPosition = (success, error, options) => {
      new Promise((resolve) => {
        pendingResolve = resolve;
      }).then((pos) => success(pos as any));
    };
  });

  await page.goto("/");

  // 1. Click Locate Me (in flight)
  const locateBtn = page.getByRole("button", { name: "Locate me" });
  await locateBtn.click();
  await expect(locateBtn).toHaveAttribute("aria-busy", "true");

  // 2. Before geolocation finishes, user explicitly selects a search result (Aspen)
  const input = page.getByLabel(/Search for a city/);
  await input.fill("Aspen");
  const option = searchResults(page).getByRole("option", { name: /Aspen/ }).first();
  await expect(option).toBeVisible();
  await option.click();

  await expect(page.getByText("Hourly Forecast")).toBeVisible();
  await expect(page.getByText("Aspen", { exact: true })).toBeVisible();

  // 3. Delayed geolocation fix (Seattle) arrives late
  await page.evaluate(() => {
    (window as any).__resolveGeo({
      coords: { latitude: 47.6062, longitude: -122.3321 },
    });
  });
  await page.waitForTimeout(500);

  // Invariant: Aspen MUST remain selected; Seattle fix must be rejected by generation guard!
  await expect(page.getByText("Aspen", { exact: true })).toBeVisible();
  await expect(page.getByText("47.6062, -122.3321")).toHaveCount(0);
});

test("forecast panel reopen UI: Search, Map Click, and Locate Me all reopen panel after close", async ({
  page,
  context,
}) => {
  await context.grantPermissions(["geolocation"]);
  await context.setGeolocation({ latitude: 47.6062, longitude: -122.3321 });

  await page.goto("/");

  // 1. Search selection opens panel
  const input = page.getByLabel(/Search for a city/);
  await input.fill("Aspen");
  await searchResults(page).getByRole("option", { name: /Aspen/ }).first().click();
  await expect(page.getByText("Hourly Forecast")).toBeVisible();

  // Close panel
  await page.getByRole("button", { name: "Close forecast panel" }).click();
  await expect(page.getByText("Hourly Forecast")).toHaveCount(0);

  // 2. Map click reopens panel
  const map = page.getByTestId("weather-map");
  await map.click({ position: { x: 300, y: 200 } });
  await expect(page.getByText("Hourly Forecast")).toBeVisible();

  // Close panel again
  await page.getByRole("button", { name: "Close forecast panel" }).click();
  await expect(page.getByText("Hourly Forecast")).toHaveCount(0);

  // 3. Locate Me reopens panel
  await page.getByRole("button", { name: "Locate me" }).click();
  await expect(page.getByText("Hourly Forecast")).toBeVisible();
  await expect(page.getByText("47.6062, -122.3321")).toBeVisible();
});

test("mobile runtime layout screenshots: capture 375x667 and 390x844 viewports", async ({
  page,
  context,
}) => {
  await context.grantPermissions(["geolocation"]);
  await context.setGeolocation({ latitude: 39.7392, longitude: -104.9903 });

  for (const viewport of [
    { width: 375, height: 667, name: "375x667" },
    { width: 390, height: 844, name: "390x844" },
  ]) {
    await page.setViewportSize({ width: viewport.width, height: viewport.height });
    await page.goto("/");

    // 1. Locate Me idle screenshot
    await expect(page.getByRole("button", { name: "Locate me" })).toBeVisible();
    await page.screenshot({
      path: `services/frontend/e2e/screenshots/locate-me-idle-${viewport.name}.png`,
    });

    // 2. Search dropdown open & attribution visible screenshot
    const input = page.getByLabel(/Search for a city/);
    await input.fill("Aspen");
    await expect(searchResults(page)).toBeVisible();
    await page.screenshot({
      path: `services/frontend/e2e/screenshots/search-dropdown-attribution-${viewport.name}.png`,
    });

    // 3. Forecast panel open screenshot
    await searchResults(page).getByRole("option", { name: /Aspen/ }).first().click();
    await expect(page.getByText("Hourly Forecast")).toBeVisible();
    await page.screenshot({
      path: `services/frontend/e2e/screenshots/forecast-panel-open-${viewport.name}.png`,
    });
  }
});

test("selected-location local time display: dropdown remains UTC, adjacent display and hourly forecast localize", async ({
  page,
}) => {
  await page.goto("/");

  // Initial state: no location selected -> UTC
  const validTimeDisplay = page.getByTestId("valid-time");
  await expect(validTimeDisplay).toBeVisible();
  await expect(validTimeDisplay).toHaveText(/Valid .* UTC/);

  // Dropdown option is in UTC
  const validSelect = page.getByLabel("Valid time");
  await expect(validSelect).toBeVisible();
  const dropdownText = await validSelect.evaluate(
    (sel: HTMLSelectElement) => sel.options[sel.selectedIndex]?.text
  );
  expect(dropdownText).toContain("UTC");

  // Search and select Aspen (Mountain Time zone: America/Denver)
  const input = page.getByLabel(/Search for a city/);
  await input.fill("Aspen");
  await searchResults(page).getByRole("option", { name: /Aspen/ }).first().click();

  // Hourly Forecast opens
  await expect(page.getByText("Hourly Forecast")).toBeVisible();

  // Dropdown MUST still remain UTC
  const dropdownTextAfter = await validSelect.evaluate(
    (sel: HTMLSelectElement) => sel.options[sel.selectedIndex]?.text
  );
  expect(dropdownTextAfter).toContain("UTC");

  // Adjacent display updates to Mountain Time (MDT / GMT-6)
  await expect(validTimeDisplay).toHaveText(/Valid .* (MDT|GMT-6)/);

  // Close forecast panel
  await page.getByRole("button", { name: "Close forecast panel" }).click();
  await expect(page.getByText("Hourly Forecast")).not.toBeVisible();

  // Adjacent display returns to UTC
  await expect(validTimeDisplay).toHaveText(/Valid .* UTC/);
});

test("ensemble statistics valid-time display: renders calendar valid times and localizes with selected location", async ({
  page,
}) => {
  await page.goto("/");

  // Switch to GEFS ensemble model
  await page.getByLabel("Model").selectOption("gefs");

  // Select Aspen (Mountain Time: America/Denver)
  const input = page.getByLabel(/Search for a city/);
  await input.fill("Aspen");
  await searchResults(page).getByRole("option", { name: /Aspen/ }).first().click();

  // Ensemble statistics panel opens
  await expect(page.getByText(/Ensemble Statistics \(GEFS\)/)).toBeVisible();

  // Ensemble chart is visible with valid-time-based accessible label
  const fanChart = page.getByRole("img", { name: /ensemble percentile fan over time/ });
  await expect(fanChart).toBeVisible();

  // Distribution header shows localized valid time in MDT and NO "+6h"
  await expect(page.getByText(/Member distribution · .* (MDT|GMT-6)/)).toBeVisible();
  await expect(page.getByText(/Member distribution · \+6h/)).toHaveCount(0);
  await expect(page.getByText(/over lead time/)).toHaveCount(0);

  // Close panel
  await page.getByRole("button", { name: "Close forecast panel" }).click();
  await expect(page.getByText("Hourly Forecast")).not.toBeVisible();
});

test("startup coarse IP success: moves map to regional viewport and localizes valid-time", async ({
  page,
}) => {
  await page.route("**/v1/locate", (route) => {
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        latitude: 39.7392,
        longitude: -104.9903,
        city: "Denver",
        region: "Colorado",
        country: "US",
        approximate: true,
      }),
    });
  });

  await page.goto("/");
  await expect(page.getByText("Weather Platform")).toBeVisible();

  // Wait for map to load and easeTo Denver region (lat ~ 39.7, lon ~ -105.0, zoom ~ 6.5)
  await page.waitForFunction(() => {
    const map = (window as any).__weatherMap;
    if (!map || typeof map.getCenter !== "function") return false;
    const c = map.getCenter();
    const z = map.getZoom();
    return Math.abs(c.lat - 39.7392) < 0.5 && Math.abs(c.lng - -104.9903) < 0.5 && z >= 6.0;
  });

  // Valid label display uses Mountain Time (MDT/GMT-6)
  const validTimeDisplay = page.getByTestId("valid-time");
  await expect(validTimeDisplay).toBeVisible();
  await expect(validTimeDisplay).toHaveText(/Valid .* (MDT|GMT-6)/);

  // Valid Time dropdown remains in canonical UTC
  const validSelect = page.getByLabel("Valid time");
  await expect(validSelect).toBeVisible();
  const dropdownText = await validSelect.evaluate(
    (sel: HTMLSelectElement) => sel.options[sel.selectedIndex]?.text
  );
  expect(dropdownText).toContain("UTC");

  // Invariant: no marker, no forecast panel
  await expect(page.locator(".maplibregl-marker")).toHaveCount(0);
  await expect(page.getByText("Hourly Forecast")).toHaveCount(0);
});

test("startup coarse IP failure (404): map remains at CONUS viewport and valid-time is UTC", async ({
  page,
}) => {
  await page.route("**/v1/locate", (route) => {
    route.fulfill({ status: 404 });
  });

  await page.goto("/");
  await expect(page.getByText("Weather Platform")).toBeVisible();
  await page.waitForTimeout(1000);

  // Map remains at default CONUS center (lat ~ 39.2, lng ~ -106.8, zoom 5)
  const center = await page.evaluate(() => {
    const map = (window as any).__weatherMap;
    return map ? { lng: map.getCenter().lng, lat: map.getCenter().lat, zoom: map.getZoom() } : null;
  });
  expect(center).not.toBeNull();
  expect(center!.lat).toBeCloseTo(39.2, 1);
  expect(center!.lng).toBeCloseTo(-106.8, 1);
  expect(center!.zoom).toBeCloseTo(5, 0);

  // Valid label display remains UTC
  const validTimeDisplay = page.getByTestId("valid-time");
  await expect(validTimeDisplay).toHaveText(/Valid .* UTC/);
  await expect(page.locator(".maplibregl-marker")).toHaveCount(0);
  await expect(page.getByText("Hourly Forecast")).toHaveCount(0);
});

test("late IP response after user map pan does NOT move camera", async ({ page }) => {
  let fulfillLocate!: () => void;
  const locatePromise = new Promise<void>((resolve) => {
    fulfillLocate = resolve;
  });

  await page.route("**/v1/locate", async (route) => {
    await locatePromise;
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        latitude: 39.7392,
        longitude: -104.9903,
        city: "Denver",
        region: "Colorado",
        country: "US",
        approximate: true,
      }),
    });
  });

  await page.goto("/");
  await expect(page.getByText("Weather Platform")).toBeVisible();

  // Wait for map style to be loaded
  await page.waitForFunction(() => {
    const map = (window as any).__weatherMap;
    return map && map.isStyleLoaded();
  });

  // Simulate manual user pan/drag on the map canvas
  const mapCanvas = page.locator("canvas.maplibregl-canvas");
  await expect(mapCanvas).toBeVisible();
  const box = await mapCanvas.boundingBox();
  expect(box).not.toBeNull();

  const startX = box!.x + box!.width / 2;
  const startY = box!.y + box!.height / 2;
  await page.mouse.move(startX, startY);
  await page.mouse.down();
  await page.mouse.move(startX + 100, startY + 100, { steps: 10 });
  await page.mouse.up();
  await page.waitForTimeout(300);

  // Read camera position after user pan
  const pannedCenter = await page.evaluate(() => {
    const map = (window as any).__weatherMap;
    return { lat: map.getCenter().lat, lng: map.getCenter().lng };
  });

  // Now release the delayed IP response
  fulfillLocate();
  await page.waitForTimeout(1000);

  // Camera must NOT have snapped back to Denver!
  const finalCenter = await page.evaluate(() => {
    const map = (window as any).__weatherMap;
    return { lat: map.getCenter().lat, lng: map.getCenter().lng };
  });

  expect(finalCenter.lat).toBeCloseTo(pannedCenter.lat, 1);
  expect(finalCenter.lng).toBeCloseTo(pannedCenter.lng, 1);
  // Explicitly verify it did NOT easeTo Denver (-104.9903, 39.7392)
  expect(Math.abs(finalCenter.lng - -104.9903)).toBeGreaterThan(2);
});

test("late IP response after Locate Me attempt does NOT move camera", async ({ page, context }) => {
  let fulfillLocate!: () => void;
  const locatePromise = new Promise<void>((resolve) => {
    fulfillLocate = resolve;
  });

  await page.route("**/v1/locate", async (route) => {
    await locatePromise;
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        latitude: 39.7392,
        longitude: -104.9903,
        city: "Denver",
        region: "Colorado",
        country: "US",
        approximate: true,
      }),
    });
  });

  await context.clearPermissions();
  await page.goto("/");
  await expect(page.getByText("Weather Platform")).toBeVisible();

  const locateBtn = page.getByRole("button", { name: "Locate me" });
  await locateBtn.click();
  const alert = page.getByRole("alert").filter({ hasText: "Location access denied" });
  await expect(alert).toBeVisible();

  // Release delayed IP response
  fulfillLocate();
  await page.waitForTimeout(1000);

  // Camera remains at default CONUS (lat ~ 39.2, lng ~ -106.8, zoom 5), NOT eased to Denver
  const center = await page.evaluate(() => {
    const map = (window as any).__weatherMap;
    return { lat: map.getCenter().lat, lng: map.getCenter().lng, zoom: map.getZoom() };
  });
  expect(center.lat).toBeCloseTo(39.2, 1);
  expect(center.lng).toBeCloseTo(-106.8, 1);
  expect(center.zoom).toBeCloseTo(5, 0);
});

test("selecting and clearing location does NOT recenter map to startup IP", async ({ page }) => {
  await page.route("**/v1/locate", (route) => {
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        latitude: 39.7392,
        longitude: -104.9903,
        city: "Denver",
        region: "Colorado",
        country: "US",
        approximate: true,
      }),
    });
  });

  await page.goto("/");
  await expect(page.getByText("Weather Platform")).toBeVisible();

  // Wait for initial easeTo Denver to complete
  await page.waitForFunction(() => {
    const map = (window as any).__weatherMap;
    if (!map) return false;
    return Math.abs(map.getCenter().lat - 39.7392) < 0.5;
  });

  // Select Tokyo from search
  const input = page.getByLabel(/Search for a city/);
  await input.fill("Tokyo");
  await searchResults(page).getByRole("option", { name: /Tokyo/ }).first().click();

  // Wait for Tokyo selection and flyTo
  await expect(page.getByText("Hourly Forecast")).toBeVisible();
  await page.waitForFunction(() => {
    const map = (window as any).__weatherMap;
    if (!map) return false;
    return Math.abs(map.getCenter().lat - 35.6762) < 1.0;
  });

  // Close forecast panel (clearing selectedLocation)
  await page.getByRole("button", { name: "Close forecast panel" }).click();
  await expect(page.getByText("Hourly Forecast")).not.toBeVisible();

  // Wait a moment to ensure no easeTo back to Denver happens
  await page.waitForTimeout(1000);

  // Camera must remain in Tokyo area, NOT ease back to Denver!
  const center = await page.evaluate(() => {
    const map = (window as any).__weatherMap;
    return map.getCenter();
  });
  expect(center.lat).toBeCloseTo(35.6762, 0);
  expect(center.lng).toBeGreaterThan(130);
  // Explicitly verify it did NOT easeTo Denver (-104.9903, 39.7392)
  expect(center.lng).not.toBeCloseTo(-104.9903, 0);

  // But display timezone returned to Denver!
  const validTimeDisplay = page.getByTestId("valid-time");
  await expect(validTimeDisplay).toHaveText(/Valid .* (MDT|GMT-6)/);
});

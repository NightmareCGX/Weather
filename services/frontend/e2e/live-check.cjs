/* eslint-disable @typescript-eslint/no-var-requires */
const { chromium } = require("@playwright/test");

/**
 * True live full-stack verification against the real backend (no API mocks).
 *
 * Requires: API on :8000 (with the real DB), frontend dev server on :3000
 * proxying /v1/* to the API.
 */
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();

  // --- Step 1: load the app, wait for the availability-driven selectors ---
  const errors = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") errors.push(msg.text());
  });
  await page.goto("http://127.0.0.1:3000/", { waitUntil: "networkidle" });

  // The Model dropdown should list only what the real DB has.
  const modelOptions = await page.$$eval(
    "#\\:r\\: \\> option, select[aria-label='Model'] option",
    (opts) => opts.map((o) => o.textContent)
  );
  const modelSelect = page.getByLabel("Model");
  await modelSelect.waitFor({ timeout: 10000 });
  const models = await modelSelect.evaluate((sel) =>
    Array.from(sel.options).map((o) => ({ value: o.value, text: o.textContent }))
  );
  console.log("MODELS:", JSON.stringify(models));

  const variableSelect = page.getByLabel("Variable");
  const variables = await variableSelect.evaluate((sel) =>
    Array.from(sel.options).map((o) => ({ value: o.value, text: o.textContent }))
  );
  console.log("VARIABLES:", JSON.stringify(variables));

  const initialSelect = page.getByLabel("Initial time");
  const initialTimes = await initialSelect.evaluate((sel) =>
    Array.from(sel.options).map((o) => ({ value: o.value, text: o.textContent }))
  );
  console.log("INITIAL TIMES:", JSON.stringify(initialTimes));

  const leadSelect = page.getByLabel("Lead time");
  const leads = await leadSelect.evaluate((sel) =>
    Array.from(sel.options).map((o) => ({ value: o.value, text: o.textContent }))
  );
  console.log("LEAD TIMES:", JSON.stringify(leads));

  // Valid time display.
  const validTime = await page
    .getByTestId("valid-time")
    .textContent()
    .catch(() => null);
  console.log("VALID TIME:", validTime);

  // The map should show a legend (meaning the metadata/tile layer is active).
  await page.waitForTimeout(2000);
  const legendUnit = await page
    .getByTestId("legend-gradient")
    .isVisible()
    .catch(() => false);
  console.log("LEGEND VISIBLE:", legendUnit);

  // --- Step 2: verify no console errors about model not found ---
  console.log("CONSOLE ERRORS:", JSON.stringify(errors));

  await browser.close();
  console.log("LIVE VERIFICATION DONE");
})();

/* eslint-disable @typescript-eslint/no-var-requires */
const { chromium } = require("@playwright/test");

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  await page.goto("http://127.0.0.1:3000/", { waitUntil: "networkidle" });
  await page.waitForTimeout(3000);
  await page.screenshot({ path: "live-fullstack.png" });
  await browser.close();
  console.log("screenshot saved");
})();

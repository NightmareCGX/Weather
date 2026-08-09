const nextJest = require("next/jest");

const createJestConfig = nextJest({
  dir: "./",
});

const customJestConfig = {
  testEnvironment: "jsdom",
  testMatch: ["<rootDir>/src/**/*.{test,spec}.{ts,tsx}"],
  setupFilesAfterEnv: ["<rootDir>/jest.setup.ts"],
  moduleNameMapper: {
    "^maplibre-gl$": "<rootDir>/__mocks__/maplibre-gl.ts",
    "\\.(css|less|scss|sass)$": "identity-obj-proxy",
  },
  clearMocks: true,
};

module.exports = createJestConfig(customJestConfig);

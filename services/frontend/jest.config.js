const nextJest = require("next/jest");

const createJestConfig = nextJest({
  dir: "./",
});

const customJestConfig = {
  testEnvironment: "jsdom",
  setupFilesAfterEnv: ["<rootDir>/jest.setup.ts"],
  moduleNameMapper: {
    "^maplibre-gl$": "<rootDir>/__mocks__/maplibre-gl.ts",
    "\\.(css|less|scss|sass)$": "identity-obj-proxy",
  },
  clearMocks: true,
};

module.exports = createJestConfig(customJestConfig);

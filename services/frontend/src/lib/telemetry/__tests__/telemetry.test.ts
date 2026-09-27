import { onLCP } from "web-vitals";

import {
  flush,
  initClientTelemetry,
  isClientTelemetryEnabled,
  queuedEventCountForTests,
  resetClientTelemetryForTests,
  TELEMETRY_ENDPOINT,
} from "../telemetry";

jest.mock("web-vitals", () => ({
  onCLS: jest.fn(),
  onFCP: jest.fn(),
  onINP: jest.fn(),
  onLCP: jest.fn(),
  onTTFB: jest.fn(),
}));

const fetchMock = jest.fn<Promise<Response>, [RequestInfo | URL, RequestInit?]>();
const sendBeaconMock = jest.fn<boolean, [string, BodyInit]>();

function jsonResponse(status: number): Response {
  // jsdom has no Response constructor; the beacon only checks failure states,
  // so a minimal status-bearing stub matches the project's test convention.
  return {
    ok: status >= 200 && status < 300,
    status,
  } as Response;
}

beforeEach(() => {
  resetClientTelemetryForTests();
  jest.clearAllMocks();
  fetchMock.mockResolvedValue(jsonResponse(202));
  // jsdom ships no fetch implementation; assign the mock directly.
  global.fetch = fetchMock as unknown as typeof fetch;
  Object.defineProperty(window.navigator, "sendBeacon", {
    value: sendBeaconMock,
    configurable: true,
  });
  delete process.env.NEXT_PUBLIC_CLIENT_TELEMETRY_ENABLED;
});

afterEach(() => {
  jest.restoreAllMocks();
});

describe("initClientTelemetry", () => {
  it("registers error listeners and web-vitals observers once", () => {
    const addSpy = jest.spyOn(window, "addEventListener");

    initClientTelemetry();
    initClientTelemetry();

    // Idempotent: exactly one listener per tracked event, one observer per vital.
    expect(addSpy).toHaveBeenCalledTimes(3); // error, unhandledrejection, pagehide
    expect(onLCP).toHaveBeenCalledTimes(1);
  });

  it("does nothing when NEXT_PUBLIC_CLIENT_TELEMETRY_ENABLED=0", () => {
    process.env.NEXT_PUBLIC_CLIENT_TELEMETRY_ENABLED = "0";
    const addSpy = jest.spyOn(window, "addEventListener");

    initClientTelemetry();

    expect(isClientTelemetryEnabled()).toBe(false);
    expect(addSpy).not.toHaveBeenCalled();
    expect(onLCP).not.toHaveBeenCalled();
  });
});

describe("error reporting", () => {
  it("queues a window error event and posts the batch on flush", async () => {
    initClientTelemetry();
    window.dispatchEvent(
      new ErrorEvent("error", { message: "boom", error: new TypeError("boom") })
    );
    expect(queuedEventCountForTests()).toBe(1);

    flush();
    await Promise.resolve();

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(TELEMETRY_ENDPOINT);
    expect(init?.method).toBe("POST");
    expect(init?.keepalive).toBe(true);
    const body = JSON.parse(String(init?.body)) as {
      events: Array<{ type: string; name: string; message: string; page_url: string }>;
    };
    expect(body.events).toHaveLength(1);
    expect(body.events[0]).toMatchObject({ type: "error", name: "TypeError", message: "boom" });
    expect(body.events[0].page_url).toContain("http://localhost");
  });

  it("queues unhandled rejections with the rejection reason", () => {
    initClientTelemetry();
    const rejection = Object.assign(new Event("unhandledrejection"), {
      reason: new Error("async boom"),
    });
    window.dispatchEvent(rejection);

    expect(queuedEventCountForTests()).toBe(1);

    flush();
    const body = JSON.parse(String(fetchMock.mock.calls[0][1]?.body)) as {
      events: Array<{ message: string }>;
    };
    expect(body.events[0].message).toBe("async boom");
  });

  it("truncates oversized stacks to the API limit", () => {
    initClientTelemetry();
    const error = new Error("huge");
    error.stack = `at f()\n${"x".repeat(5000)}`;
    window.dispatchEvent(new ErrorEvent("error", { message: "huge", error }));

    flush();
    const body = JSON.parse(String(fetchMock.mock.calls[0][1]?.body)) as {
      events: Array<{ stack: string }>;
    };
    expect(body.events[0].stack).toHaveLength(2048);
  });

  it("flushes immediately when the batch reaches the server-side cap", () => {
    initClientTelemetry();
    for (let index = 0; index < 20; index += 1) {
      window.dispatchEvent(
        new ErrorEvent("error", { message: `boom ${index}`, error: new Error(`e${index}`) })
      );
    }

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const body = JSON.parse(String(fetchMock.mock.calls[0][1]?.body)) as { events: unknown[] };
    expect(body.events).toHaveLength(20);
    expect(queuedEventCountForTests()).toBe(0);
  });
});

describe("web vitals reporting", () => {
  it("reports the LCP metric through the registered observer", async () => {
    initClientTelemetry();
    const onLCPMock = jest.mocked(onLCP);
    const observer = onLCPMock.mock.calls[0][0];
    observer({ name: "LCP", value: 1234.5, rating: "good" } as Parameters<typeof observer>[0]);

    flush();
    await Promise.resolve();

    const body = JSON.parse(String(fetchMock.mock.calls[0][1]?.body)) as {
      events: Array<{ type: string; name: string; value: number; rating: string }>;
    };
    expect(body.events[0]).toMatchObject({
      type: "web_vital",
      name: "LCP",
      value: 1234.5,
      rating: "good",
    });
  });
});

describe("page-exit flush", () => {
  it("uses navigator.sendBeacon when the page is hidden", () => {
    initClientTelemetry();
    window.dispatchEvent(new ErrorEvent("error", { message: "boom", error: new Error("boom") }));
    sendBeaconMock.mockReturnValue(true);

    Object.defineProperty(document, "visibilityState", { value: "hidden", configurable: true });
    document.dispatchEvent(new Event("visibilitychange"));

    expect(sendBeaconMock).toHaveBeenCalledTimes(1);
    expect(sendBeaconMock.mock.calls[0][0]).toBe(TELEMETRY_ENDPOINT);
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

describe("failure isolation", () => {
  it("never surfaces a failing endpoint to the application", async () => {
    initClientTelemetry();
    window.dispatchEvent(new ErrorEvent("error", { message: "boom", error: new Error("boom") }));
    fetchMock.mockRejectedValue(new Error("network down"));

    expect(() => flush()).not.toThrow();
    await Promise.resolve();
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("drops events when the beacon rejects and falls back to fetch", () => {
    initClientTelemetry();
    window.dispatchEvent(new ErrorEvent("error", { message: "boom", error: new Error("boom") }));
    sendBeaconMock.mockReturnValue(false);

    Object.defineProperty(document, "visibilityState", { value: "hidden", configurable: true });
    document.dispatchEvent(new Event("visibilitychange"));

    expect(sendBeaconMock).toHaveBeenCalledTimes(1);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});

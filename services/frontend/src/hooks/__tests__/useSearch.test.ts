import { act, renderHook, waitFor } from "@testing-library/react";
import { useSearch } from "@/hooks/useSearch";

const mockFetch = jest.fn<Promise<Response>, [RequestInfo | URL, RequestInit?]>();

function jsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: () => Promise.resolve(body),
  } as Response;
}

function envelopeList(data: unknown) {
  return jsonResponse({ object: "list", data, has_more: false, next_cursor: null });
}

beforeEach(() => {
  mockFetch.mockReset();
  globalThis.fetch = mockFetch as unknown as typeof fetch;
  jest.useFakeTimers();
});

afterEach(() => {
  jest.useRealTimers();
});

function advanceDebounce() {
  act(() => {
    jest.advanceTimersByTime(300);
  });
}

describe("useSearch", () => {
  it("does not fire for a query shorter than the minimum length", () => {
    renderHook(() => useSearch(""));
    advanceDebounce();
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it("debounces and fires /v1/search after the delay", async () => {
    mockFetch.mockResolvedValueOnce(envelopeList([]));
    renderHook(() => useSearch("Aspen"));

    expect(mockFetch).not.toHaveBeenCalled();
    advanceDebounce();
    expect(mockFetch).toHaveBeenCalledWith(
      "/v1/search?q=Aspen&type=all&limit=20",
      expect.any(Object)
    );
  });

  it("supersedes stale requests when the query changes quickly", async () => {
    // A deferred promise for the first query that we will resolve later.
    let resolveFirst!: (value: Response) => void;
    mockFetch.mockImplementationOnce(
      () => new Promise<Response>((resolve) => (resolveFirst = resolve))
    );
    mockFetch.mockResolvedValueOnce(
      envelopeList([{ id: "city_aspen", object: "city", name: "Aspen" }])
    );

    const { rerender } = renderHook(({ q }) => useSearch(q), { initialProps: { q: "As" } });
    advanceDebounce();

    // User types more before the first response resolves.
    rerender({ q: "Aspen" });
    advanceDebounce();

    // Resolve the stale response for "As" late; it must be ignored.
    await act(async () => {
      resolveFirst(envelopeList([{ id: "city_aspen", object: "city", name: "Aspen" }]));
    });

    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledTimes(2);
    });
  });

  it("surfaces an error status on API failure", async () => {
    mockFetch.mockRejectedValueOnce(new TypeError("Failed to fetch"));
    const { result } = renderHook(() => useSearch("Aspen"));
    advanceDebounce();

    await waitFor(() => {
      expect(result.current.status).toBe("error");
    });
    expect(result.current.error).not.toBeNull();
  });

  it("ignores aborted (superseded) requests without an error", async () => {
    mockFetch.mockImplementationOnce(() =>
      Promise.reject(new DOMException("Aborted", "AbortError"))
    );
    const { result } = renderHook(() => useSearch("Aspen"));
    advanceDebounce();

    // The aborted request is silent: no error state is ever set.
    await act(async () => {
      await Promise.resolve();
    });
    expect(result.current.error).toBeNull();
  });
});

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

  it("debounces and fires /v1/search (place autocomplete) after the delay", async () => {
    mockFetch.mockResolvedValueOnce(envelopeList([]));
    renderHook(() => useSearch("Aspen"));

    expect(mockFetch).not.toHaveBeenCalled();
    advanceDebounce();
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining("/v1/search?q=Aspen&type=place&limit=20"),
      expect.any(Object)
    );
  });

  it("supersedes stale requests when the query changes quickly", async () => {
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

    await act(async () => {
      await Promise.resolve();
    });
    expect(result.current.error).toBeNull();
  });

  it("A -> B -> A: prevents stale A1 request from committing over newer A2 request", async () => {
    let resolveA1!: (value: Response) => void;
    let resolveA2!: (value: Response) => void;

    mockFetch
      // Request A1 ("Denver")
      .mockImplementationOnce(() => new Promise<Response>((resolve) => (resolveA1 = resolve)))
      // Request B ("Boulder")
      .mockResolvedValueOnce(
        envelopeList([{ id: "city_boulder", object: "city", name: "Boulder City" }])
      )
      // Request A2 ("Denver")
      .mockImplementationOnce(() => new Promise<Response>((resolve) => (resolveA2 = resolve)));

    const { result, rerender } = renderHook(({ q }) => useSearch(q), {
      initialProps: { q: "Denver" },
    });
    advanceDebounce();
    expect(mockFetch).toHaveBeenCalledTimes(1);

    // Switch to Boulder
    rerender({ q: "Boulder" });
    advanceDebounce();

    // Switch back to Denver
    rerender({ q: "Denver" });
    advanceDebounce();
    expect(mockFetch).toHaveBeenCalledTimes(3);

    // Stale A1 resolves with stale data
    await act(async () => {
      resolveA1(
        envelopeList([{ id: "city_denver_stale", object: "city", name: "Denver Stale A1" }])
      );
    });

    // A1 must NOT commit
    expect(result.current.results).not.toEqual([
      expect.objectContaining({ name: "Denver Stale A1" }),
    ]);

    // A2 resolves with fresh data
    await act(async () => {
      resolveA2(
        envelopeList([{ id: "city_denver_fresh", object: "city", name: "Denver Fresh A2" }])
      );
    });

    // A2 commits
    expect(result.current.results).toEqual([expect.objectContaining({ name: "Denver Fresh A2" })]);
    expect(result.current.status).toBe("success");
  });

  it("stale autocomplete failure: old rejected request does not replace current results or error", async () => {
    let rejectA1!: (error: unknown) => void;
    mockFetch
      // Request A1 ("Denver")
      .mockImplementationOnce((_, init) => {
        return new Promise<Response>((_, reject) => {
          rejectA1 = reject;
          init?.signal?.addEventListener("abort", () => {
            // Signal aborted
          });
        });
      })
      // Request B ("Boulder")
      .mockResolvedValueOnce(
        envelopeList([{ id: "city_boulder", object: "city", name: "Boulder City" }])
      );

    const { result, rerender } = renderHook(({ q }) => useSearch(q), {
      initialProps: { q: "Denver" },
    });
    advanceDebounce();

    // Switch to Boulder and let B succeed
    rerender({ q: "Boulder" });
    advanceDebounce();

    await waitFor(() => {
      expect(result.current.status).toBe("success");
      expect(result.current.results).toHaveLength(1);
      expect(result.current.results[0].name).toBe("Boulder City");
    });

    // Stale A1 rejects with non-abort error
    await act(async () => {
      rejectA1(new TypeError("Network error on old query"));
    });

    // Current state for Boulder remains intact
    expect(result.current.status).toBe("success");
    expect(result.current.results[0].name).toBe("Boulder City");
    expect(result.current.error).toBeNull();
  });

  it("unmount: pending autocomplete request does not commit state after cleanup", async () => {
    let resolveA!: (value: Response) => void;
    mockFetch.mockImplementationOnce(
      () => new Promise<Response>((resolve) => (resolveA = resolve))
    );

    const { result, unmount } = renderHook(() => useSearch("Denver"));
    advanceDebounce();

    unmount();

    await act(async () => {
      resolveA(envelopeList([{ id: "city_denver", object: "city", name: "Denver" }]));
    });

    // No error or unexpected commit
    expect(result.current.results).toEqual([]);
  });
});

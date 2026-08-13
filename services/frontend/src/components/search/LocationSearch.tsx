"use client";

import { useEffect, useId, useRef, useState } from "react";

import { useSearch } from "@/hooks/useSearch";
import type { SelectedLocation } from "@/lib/api/types";
import { searchResultToSelectedLocation } from "@/lib/forecast/selection";

interface LocationSearchProps {
  onSelect: (location: SelectedLocation) => void;
  disabled?: boolean;
  placeholder?: string;
}

/**
 * Accessible location search combobox backed by `/v1/search`.
 *
 * The component owns only local UI state (query, open, highlighted index);
 * selection is reported up through {@link onSelect} as a
 * {@link SelectedLocation}. Keyboard navigation, focus management, and ARIA
 * combobox/listbox semantics are handled here so the rest of the dashboard
 * stays presentation-only.
 */
export function LocationSearch({ onSelect, disabled = false, placeholder }: LocationSearchProps) {
  const id = useId();
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [highlighted, setHighlighted] = useState(-1);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  const { results, status, error } = useSearch(query);

  const listboxId = `${id}-listbox`;
  const optionId = (index: number) => `${id}-option-${index}`;

  const showListbox = open && query.trim().length > 0;

  const selectIndex = (index: number) => {
    const result = results[index];
    if (result === undefined) return;
    onSelect(searchResultToSelectedLocation(result));
    setQuery("");
    setOpen(false);
    setHighlighted(-1);
  };

  const onKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (!showListbox || results.length === 0) return;
    switch (event.key) {
      case "ArrowDown":
        event.preventDefault();
        setHighlighted((index) => (index + 1) % results.length);
        break;
      case "ArrowUp":
        event.preventDefault();
        setHighlighted((index) => (index <= 0 ? results.length - 1 : index - 1));
        break;
      case "Enter":
        event.preventDefault();
        if (highlighted >= 0) {
          selectIndex(highlighted);
        }
        break;
      case "Escape":
        event.preventDefault();
        setOpen(false);
        setHighlighted(-1);
        break;
      default:
        break;
    }
  };

  // Close the listbox when the user clicks outside the whole component. The
  // check is scoped to the container (not just the input) so clicking a result
  // option inside the dropdown does not close it before `onMouseDown` fires
  // `selectIndex` on the option.
  useEffect(() => {
    if (!showListbox) return;
    const onPointerDown = (event: PointerEvent) => {
      if (containerRef.current !== null && !containerRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("pointerdown", onPointerDown);
    return () => document.removeEventListener("pointerdown", onPointerDown);
  }, [showListbox]);

  return (
    <div ref={containerRef} className="relative w-full">
      <label htmlFor={id} className="sr-only">
        Search for a city, ski resort, or station
      </label>
      <input
        ref={inputRef}
        id={id}
        type="search"
        value={query}
        disabled={disabled}
        role="combobox"
        aria-expanded={showListbox}
        aria-controls={listboxId}
        aria-autocomplete="list"
        aria-activedescendant={highlighted >= 0 ? optionId(highlighted) : undefined}
        aria-label="Search for a city, ski resort, or station"
        placeholder={placeholder ?? "Search cities, ski resorts, stations…"}
        className="w-full rounded border border-slate-300 bg-white px-3 py-2 text-sm text-slate-800 focus:border-slate-500 focus:outline-none focus:ring-1 focus:ring-slate-500 disabled:opacity-60"
        onChange={(event) => {
          setQuery(event.target.value);
          setOpen(true);
          setHighlighted(-1);
        }}
        onFocus={() => setOpen(true)}
        onKeyDown={onKeyDown}
      />

      {showListbox && (
        <ul
          id={listboxId}
          role="listbox"
          aria-label="Search results"
          className="absolute z-30 mt-1 max-h-72 w-full overflow-auto rounded border border-slate-200 bg-white shadow-lg"
        >
          {status === "loading" && (
            <li
              role="option"
              aria-disabled
              aria-selected="false"
              className="px-3 py-2 text-sm text-slate-500"
            >
              Searching…
            </li>
          )}
          {error !== null && (
            <li role="alert" className="px-3 py-2 text-sm text-red-700">
              {error}
            </li>
          )}
          {status === "success" && results.length === 0 && (
            <li className="px-3 py-2 text-sm text-slate-500">No matching locations.</li>
          )}
          {results.map((result, index) => (
            <li
              key={`${result.object}-${result.id}`}
              id={optionId(index)}
              role="option"
              aria-selected={highlighted === index}
              className={`flex cursor-pointer items-center justify-between gap-3 px-3 py-2 text-sm ${
                highlighted === index ? "bg-slate-100" : ""
              }`}
              onMouseDown={(event) => {
                // Keep focus on the input; use onMouseDown so it fires before blur.
                event.preventDefault();
                selectIndex(index);
              }}
              onMouseEnter={() => setHighlighted(index)}
            >
              <span className="truncate">
                <span className="text-slate-900">{result.name}</span>
                {result.region !== null && (
                  <span className="ml-2 text-xs text-slate-500">
                    {result.region}
                    {result.country !== null ? `, ${result.country}` : ""}
                  </span>
                )}
              </span>
              <span className="shrink-0 rounded bg-slate-100 px-1.5 py-0.5 text-xs text-slate-600">
                {result.object === "city"
                  ? "City"
                  : result.object === "ski_resort"
                    ? "Ski resort"
                    : "Station"}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

/**
 * Time formatting helpers for forecast timestamps.
 *
 * All timestamps from the backend are ISO 8601 UTC strings ending in `Z`
 * (API.md section 2.6). Formatting always renders in UTC using the native
 * `Intl.DateTimeFormat` — no date/time dependency is introduced.
 */

const UTC_TIME_FORMATTER = new Intl.DateTimeFormat("en", {
  timeZone: "UTC",
  hour: "2-digit",
  minute: "2-digit",
  hour12: false,
});

const UTC_DAY_HOUR_FORMATTER = new Intl.DateTimeFormat("en", {
  timeZone: "UTC",
  month: "short",
  day: "numeric",
  hour: "2-digit",
  minute: "2-digit",
  hour12: false,
});

const UTC_FULL_FORMATTER = new Intl.DateTimeFormat("en", {
  timeZone: "UTC",
  year: "numeric",
  month: "short",
  day: "numeric",
  hour: "2-digit",
  minute: "2-digit",
  hour12: false,
});

/** Format a valid time as a short UTC time-of-day, e.g. `"06:00"`. */
export function formatTimeUtc(iso: string): string {
  return UTC_TIME_FORMATTER.format(new Date(iso));
}

/** Format a valid time as a UTC day + time, e.g. `"Jul 21, 06:00"`. */
export function formatDayHourUtc(iso: string): string {
  return UTC_DAY_HOUR_FORMATTER.format(new Date(iso));
}

/** Format a valid time as a full UTC timestamp for tooltips, e.g. `"Jul 21, 2026, 06:00"`. */
export function formatFullUtc(iso: string): string {
  return UTC_FULL_FORMATTER.format(new Date(iso));
}

/** Format a lead offset as a compact label, e.g. `"+6h"` / `"0h"`. */
export function formatLeadTimeHours(lead: number): string {
  return `+${lead}h`;
}

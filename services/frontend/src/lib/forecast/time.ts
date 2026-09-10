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

// ---------------------------------------------------------------------------
// Timezone-aware display formatters
// ---------------------------------------------------------------------------

const UTC_DAY_HOUR_WITH_TZ_FORMATTER = new Intl.DateTimeFormat("en", {
  timeZone: "UTC",
  month: "short",
  day: "numeric",
  hour: "2-digit",
  minute: "2-digit",
  hour12: false,
  hourCycle: "h23",
  timeZoneName: "short",
});

const dayHourFormatters = new Map<string, Intl.DateTimeFormat>();
const dayHourWithTzFormatters = new Map<string, Intl.DateTimeFormat>();

/** Test whether an IANA timezone identifier is valid and supported by Intl. */
export function isValidTimeZone(timeZone: string): boolean {
  try {
    new Intl.DateTimeFormat(undefined, { timeZone });
    return true;
  } catch {
    return false;
  }
}

function getDayHourFormatter(timeZone: string): Intl.DateTimeFormat {
  let formatter = dayHourFormatters.get(timeZone);
  if (!formatter) {
    try {
      formatter = new Intl.DateTimeFormat("en", {
        timeZone,
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
        hour12: false,
        hourCycle: "h23",
      });
    } catch {
      formatter = UTC_DAY_HOUR_FORMATTER;
    }
    dayHourFormatters.set(timeZone, formatter);
  }
  return formatter;
}

function getDayHourWithTzFormatter(timeZone: string): Intl.DateTimeFormat {
  let formatter = dayHourWithTzFormatters.get(timeZone);
  if (!formatter) {
    try {
      formatter = new Intl.DateTimeFormat("en", {
        timeZone,
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
        hour12: false,
        hourCycle: "h23",
        timeZoneName: "short",
      });
    } catch {
      formatter = UTC_DAY_HOUR_WITH_TZ_FORMATTER;
    }
    dayHourWithTzFormatters.set(timeZone, formatter);
  }
  return formatter;
}

/**
 * Format an ISO 8601 UTC timestamp in a specific timezone (or UTC if null/invalid)
 * as day + time, e.g. `"Sep 9, 18:00"` or `"Sep 10, 00:00"`.
 *
 * Uses explicit 24-hour cycle (`h23`) so midnight formats consistently as `00:00`.
 */
export function formatDayHourInTimeZone(iso: string, timeZone?: string | null): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) {
    return iso;
  }
  const tz = timeZone && isValidTimeZone(timeZone) ? timeZone : "UTC";
  try {
    return getDayHourFormatter(tz).format(date);
  } catch {
    return UTC_DAY_HOUR_FORMATTER.format(date);
  }
}

/**
 * Format an ISO 8601 UTC timestamp in a specific timezone (or UTC if null/invalid)
 * with a short timezone abbreviation/offset, e.g. `"Sep 9, 18:00 MDT"` or `"Sep 10, 00:00 UTC"`.
 *
 * Used for the adjacent Valid Time display and Meteogram tooltip header.
 */
export function formatDayHourWithTimeZone(iso: string, timeZone?: string | null): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) {
    return iso;
  }
  const tz = timeZone && isValidTimeZone(timeZone) ? timeZone : "UTC";
  try {
    return getDayHourWithTzFormatter(tz).format(date);
  } catch {
    return UTC_DAY_HOUR_WITH_TZ_FORMATTER.format(date);
  }
}

"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { useSelectedLocation } from "@/context/selected-location";
import { getApproximateLocation, RequestAbortedError } from "@/lib/api/client";
import {
  canonicalizeLongitude,
  formatCoordinates,
  isValidCoordinate,
} from "@/lib/forecast/selection";
import type { SelectedLocation } from "@/lib/api/types";

export interface GeolocationNotice {
  type: "status" | "alert";
  message: string;
}

export interface UseGeolocationOptions {
  onLocateSuccess?: (location: SelectedLocation) => void;
}

export interface UseGeolocationResult {
  isLocating: boolean;
  notice: GeolocationNotice | null;
  clearNotice: () => void;
  locateMe: () => void;
}

export const GEOLOCATION_OPTIONS: PositionOptions = {
  enableHighAccuracy: false,
  timeout: 10000,
  maximumAge: 300000,
};

export function useGeolocation(options?: UseGeolocationOptions): UseGeolocationResult {
  const { beginAsyncSelection, commitAsyncSelection } = useSelectedLocation();
  const [isLocating, setIsLocating] = useState(false);
  const [notice, setNotice] = useState<GeolocationNotice | null>(null);

  const onLocateSuccessRef = useRef(options?.onLocateSuccess);
  onLocateSuccessRef.current = options?.onLocateSuccess;

  const activeTokenRef = useRef<number | null>(null);
  const ipAbortRef = useRef<AbortController | null>(null);
  const noticeTimerRef = useRef<number | null>(null);

  const showNotice = useCallback((type: "status" | "alert", message: string) => {
    if (noticeTimerRef.current !== null) {
      window.clearTimeout(noticeTimerRef.current);
    }
    setNotice({ type, message });
    noticeTimerRef.current = window.setTimeout(() => {
      setNotice(null);
      noticeTimerRef.current = null;
    }, 6000);
  }, []);

  const clearNotice = useCallback(() => {
    if (noticeTimerRef.current !== null) {
      window.clearTimeout(noticeTimerRef.current);
      noticeTimerRef.current = null;
    }
    setNotice(null);
  }, []);

  useEffect(() => {
    return () => {
      if (noticeTimerRef.current !== null) {
        window.clearTimeout(noticeTimerRef.current);
      }
      ipAbortRef.current?.abort();
    };
  }, []);

  const handleIpFallback = useCallback(
    async (token: number) => {
      ipAbortRef.current?.abort();
      const controller = new AbortController();
      ipAbortRef.current = controller;

      try {
        const approx = await getApproximateLocation({ signal: controller.signal });
        if (activeTokenRef.current !== token) {
          // A newer action intervened while IP lookup was in flight
          return;
        }

        if (approx !== null) {
          const canonicalLon = canonicalizeLongitude(approx.longitude);
          if (isValidCoordinate(approx.latitude, canonicalLon)) {
            const displayName =
              [approx.city, approx.region, approx.country].filter(Boolean).join(", ") ||
              formatCoordinates(approx.latitude, canonicalLon);

            const location: SelectedLocation = {
              name: displayName,
              object: "coordinates",
              latitude: approx.latitude,
              longitude: canonicalLon,
              elevation_m: null,
              region: approx.region,
              country: approx.country,
              id: null,
              resolvedVia: "coordinates",
            };

            const committed = commitAsyncSelection(location, token);
            if (committed) {
              showNotice("status", "Using approximate location");
              onLocateSuccessRef.current?.(location);
              return;
            }
          }
        }

        showNotice("alert", "Location unavailable. Please search for a city or click the map.");
      } catch (err) {
        if (err instanceof RequestAbortedError) return;
        showNotice("alert", "Location unavailable. Please search for a city or click the map.");
      } finally {
        if (activeTokenRef.current === token) {
          setIsLocating(false);
          activeTokenRef.current = null;
        }
      }
    },
    [commitAsyncSelection, showNotice]
  );

  const locateMe = useCallback(() => {
    if (typeof navigator === "undefined" || !navigator.geolocation) {
      // Browser does not support geolocation; try coarse IP fallback
      const token = beginAsyncSelection();
      activeTokenRef.current = token;
      setIsLocating(true);
      clearNotice();
      handleIpFallback(token);
      return;
    }

    const token = beginAsyncSelection();
    activeTokenRef.current = token;
    setIsLocating(true);
    clearNotice();

    navigator.geolocation.getCurrentPosition(
      (position) => {
        if (activeTokenRef.current !== token) {
          // Newer selection superseded this fix
          return;
        }
        setIsLocating(false);
        activeTokenRef.current = null;

        const rawLat = position.coords.latitude;
        const rawLon = position.coords.longitude;
        const canonicalLon = canonicalizeLongitude(rawLon);

        if (!isValidCoordinate(rawLat, canonicalLon)) {
          showNotice("alert", "Received invalid location coordinates from device.");
          return;
        }

        const location: SelectedLocation = {
          name: formatCoordinates(rawLat, canonicalLon),
          object: "coordinates",
          latitude: rawLat,
          longitude: canonicalLon,
          elevation_m: null,
          region: null,
          country: null,
          id: null,
          resolvedVia: "coordinates",
        };

        const committed = commitAsyncSelection(location, token);
        if (committed) {
          clearNotice();
          onLocateSuccessRef.current?.(location);
        }
      },
      (error) => {
        if (activeTokenRef.current !== token) {
          return;
        }

        if (error.code === error.PERMISSION_DENIED) {
          // Strict privacy invariant: NEVER fall back to IP geolocation after explicit denial
          setIsLocating(false);
          activeTokenRef.current = null;
          showNotice(
            "alert",
            "Location access denied. Please enable location permissions in your browser."
          );
          return;
        }

        if (error.code === error.POSITION_UNAVAILABLE || error.code === error.TIMEOUT) {
          // May use coarse infrastructure IP fallback
          handleIpFallback(token);
          return;
        }

        // Unknown or unexpected error: safe non-blocking feedback
        setIsLocating(false);
        activeTokenRef.current = null;
        showNotice("alert", "Unable to retrieve your location. Please try again.");
      },
      GEOLOCATION_OPTIONS
    );
  }, [beginAsyncSelection, clearNotice, commitAsyncSelection, handleIpFallback, showNotice]);

  return {
    isLocating,
    notice,
    clearNotice,
    locateMe,
  };
}

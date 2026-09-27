import type { Metadata } from "next";

import "maplibre-gl/dist/maplibre-gl.css";
import "./globals.css";

import ClientTelemetry from "@/components/telemetry/ClientTelemetry";
import { ForecastSelectionProvider } from "@/context/forecast-selection";
import { SelectedLocationProvider } from "@/context/selected-location";

export const metadata: Metadata = {
  title: "Zeus Wx",
  description: "Global probabilistic weather forecasting",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <ClientTelemetry />
        <ForecastSelectionProvider>
          <SelectedLocationProvider>{children}</SelectedLocationProvider>
        </ForecastSelectionProvider>
      </body>
    </html>
  );
}

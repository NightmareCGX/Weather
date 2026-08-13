import type { Metadata } from "next";

import "maplibre-gl/dist/maplibre-gl.css";
import "./globals.css";

import { ForecastSelectionProvider } from "@/context/forecast-selection";
import { SelectedLocationProvider } from "@/context/selected-location";

export const metadata: Metadata = {
  title: "Weather Platform",
  description: "Global probabilistic weather forecasting",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <ForecastSelectionProvider>
          <SelectedLocationProvider>{children}</SelectedLocationProvider>
        </ForecastSelectionProvider>
      </body>
    </html>
  );
}

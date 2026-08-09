import type { Metadata } from "next";

import "maplibre-gl/dist/maplibre-gl.css";
import "./globals.css";

import { MapConfigProvider } from "@/context/map-config";

export const metadata: Metadata = {
  title: "Weather Platform",
  description: "Global probabilistic weather forecasting",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <MapConfigProvider>{children}</MapConfigProvider>
      </body>
    </html>
  );
}

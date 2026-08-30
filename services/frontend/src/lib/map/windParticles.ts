import type { Map as MapLibreMap } from "maplibre-gl";
import type { VectorFieldData } from "@/lib/api/types";

export const CALM_WIND_THRESHOLD_MPS = 0.5;
export const EARTH_RADIUS_METERS = 6371000.0;
export const DEFAULT_LAT_CLAMP = 85.0;

/**
 * Bilinear interpolation of U and V velocity components at (lat, lon).
 */
export function sampleVectorBilinear(
  field: VectorFieldData,
  lat: number,
  lon: number
): [number, number] {
  if (Number.isNaN(lat) || Number.isNaN(lon)) {
    return [NaN, NaN];
  }

  const { meta, u, v } = field;

  // Row fractional index
  let row_f = (lat - meta.lat_start) / meta.lat_step;
  row_f = Math.max(0, Math.min(meta.lat_count - 1, row_f));

  // Longitude alignment into grid convention
  const lon_norm = ((lon % 360.0) + 360.0) % 360.0;
  const aligned_lon =
    meta.lon_start >= 0.0 ? lon_norm : ((((lon + 180.0) % 360.0) + 360.0) % 360.0) - 180.0;

  let col_f = ((aligned_lon - meta.lon_start) / meta.lon_step) % meta.lon_count;
  if (col_f < 0) {
    col_f += meta.lon_count;
  }

  const r0 = Math.floor(row_f);
  const r1 = Math.min(meta.lat_count - 1, r0 + 1);
  const dr = row_f - r0;

  const c0 = Math.floor(col_f);
  const c1 = (c0 + 1) % meta.lon_count;
  const dc = col_f - c0;

  const idx00 = r0 * meta.lon_count + c0;
  const idx01 = r0 * meta.lon_count + c1;
  const idx10 = r1 * meta.lon_count + c0;
  const idx11 = r1 * meta.lon_count + c1;

  const w00 = (1.0 - dr) * (1.0 - dc);
  const w01 = (1.0 - dr) * dc;
  const w10 = dr * (1.0 - dc);
  const w11 = dr * dc;

  const u_val = w00 * u[idx00] + w01 * u[idx01] + w10 * u[idx10] + w11 * u[idx11];
  const v_val = w00 * v[idx00] + w01 * v[idx01] + w10 * v[idx10] + w11 * v[idx11];

  return [u_val, v_val];
}

/**
 * Advect a geographic point given local (u, v) wind velocity in m/s.
 *
 * Accounts for Earth spherical geometry and latitude convergence.
 */
export function advectParticle(
  lat: number,
  lon: number,
  u: number,
  v: number,
  dtSeconds = 60.0,
  latClamp = DEFAULT_LAT_CLAMP,
  earthRadiusM = EARTH_RADIUS_METERS
): [number, number] {
  if (Number.isNaN(lat) || Number.isNaN(lon) || Number.isNaN(u) || Number.isNaN(v)) {
    return [NaN, NaN];
  }

  const clampedLat = Math.max(-latClamp, Math.min(latClamp, lat));
  const radLat = clampedLat * (Math.PI / 180.0);
  const cosLat = Math.max(Math.cos(radLat), Math.cos(latClamp * (Math.PI / 180.0)));

  const dlatDeg = ((v * dtSeconds) / earthRadiusM) * (180.0 / Math.PI);
  const dlonDeg = ((u * dtSeconds) / (earthRadiusM * cosLat)) * (180.0 / Math.PI);

  const newLat = Math.max(-latClamp, Math.min(latClamp, lat + dlatDeg));
  const newLon = lon + dlonDeg;

  let normLon = ((((newLon + 180.0) % 360.0) + 360.0) % 360.0) - 180.0;
  if (normLon === -180.0 && newLon > 0) {
    normLon = 180.0;
  }

  return [newLat, normLon];
}

export function isCalmWind(u: number, v: number, threshold = CALM_WIND_THRESHOLD_MPS): boolean {
  if (Number.isNaN(u) || Number.isNaN(v)) {
    return true;
  }
  return Math.hypot(u, v) < threshold;
}

export interface Particle {
  lat: number;
  lon: number;
  age: number;
  maxAge: number;
  prevX: number | null;
  prevY: number | null;
  visible: boolean;
}

export interface ParticleAnimationOptions {
  maxParticles?: number;
  speedScale?: number;
  fadeOpacity?: number;
  color?: string;
  lineWidth?: number;
}

/**
 * Manages the HTML5 Canvas 2D overlay particle system synchronized with MapLibre.
 */
export class WindParticleAnimation {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D | null;
  private map: MapLibreMap;
  private field: VectorFieldData | null = null;
  private particles: Particle[] = [];
  private isRunning = false;
  private isHidden = false;
  private isReducedMotion = false;
  private rafId: number | null = null;
  private lastTimestamp = 0;
  private dpr = 1;
  private options: Required<ParticleAnimationOptions>;

  constructor(canvas: HTMLCanvasElement, map: MapLibreMap, options: ParticleAnimationOptions = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d", { willReadFrequently: false });
    this.map = map;
    this.options = {
      maxParticles: options.maxParticles ?? 3000,
      speedScale: options.speedScale ?? 1.0,
      fadeOpacity: options.fadeOpacity ?? 0.94,
      color: options.color ?? "rgba(255, 255, 255, 0.75)",
      lineWidth: options.lineWidth ?? 1.2,
    };

    this.checkReducedMotion();
    this.resize();
    this.bindEvents();
  }

  private checkReducedMotion(): void {
    if (typeof window !== "undefined" && window.matchMedia) {
      this.isReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    }
  }

  private bindEvents(): void {
    if (typeof document !== "undefined") {
      document.addEventListener("visibilitychange", this.handleVisibilityChange);
    }
    this.map.on("move", this.handleMapMove);
    this.map.on("resize", this.handleMapResize);
  }

  private unbindEvents(): void {
    if (typeof document !== "undefined") {
      document.removeEventListener("visibilitychange", this.handleVisibilityChange);
    }
    this.map.off("move", this.handleMapMove);
    this.map.off("resize", this.handleMapResize);
  }

  private handleVisibilityChange = (): void => {
    if (typeof document === "undefined") return;
    this.isHidden = document.hidden;
    if (this.isHidden) {
      this.stop();
    } else if (this.field !== null && !this.isReducedMotion) {
      this.start();
    }
  };

  private handleMapMove = (): void => {
    // Reset particle trail continuity on map movement to prevent long streaks across screen
    for (const p of this.particles) {
      p.prevX = null;
      p.prevY = null;
    }
  };

  private handleMapResize = (): void => {
    this.resize();
  };

  public resize(): void {
    if (!this.canvas) return;
    const rect = this.canvas.getBoundingClientRect();
    const dpr = typeof window !== "undefined" ? Math.min(window.devicePixelRatio || 1, 2) : 1;
    this.dpr = dpr;
    const width = Math.max(1, Math.round(rect.width * dpr));
    const height = Math.max(1, Math.round(rect.height * dpr));

    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
    this.initParticles();
  }

  public setField(field: VectorFieldData | null): void {
    this.field = field;
    if (field === null) {
      this.stop();
      this.clear();
      return;
    }

    this.initParticles();
    if (!this.isReducedMotion && !this.isHidden) {
      this.start();
    }
  }

  private initParticles(): void {
    if (!this.canvas) return;
    const rect = this.canvas.getBoundingClientRect();
    const area = rect.width * rect.height;
    // Adaptive particle density based on screen area
    const targetCount = Math.min(this.options.maxParticles, Math.max(500, Math.round(area / 350)));

    this.particles = [];
    for (let i = 0; i < targetCount; i++) {
      this.particles.push(this.createRandomParticle(true));
    }
  }

  private createRandomParticle(randomAge = false): Particle {
    const bounds = this.map.getBounds();
    const south = Math.max(-85, bounds.getSouth());
    const north = Math.min(85, bounds.getNorth());
    const west = bounds.getWest();
    const east = bounds.getEast();

    const lat = south + Math.random() * (north - south);
    let lon = west + Math.random() * (east - west);
    lon = ((((lon + 180.0) % 360.0) + 360.0) % 360.0) - 180.0;

    const maxAge = 40 + Math.floor(Math.random() * 60);
    const age = randomAge ? Math.floor(Math.random() * maxAge) : 0;

    return {
      lat,
      lon,
      age,
      maxAge,
      prevX: null,
      prevY: null,
      visible: true,
    };
  }

  public start(): void {
    if (this.isRunning || this.field === null || this.isReducedMotion || this.isHidden) {
      return;
    }
    this.isRunning = true;
    this.lastTimestamp = performance.now();
    this.loop(this.lastTimestamp);
  }

  public stop(): void {
    this.isRunning = false;
    if (this.rafId !== null) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
  }

  public clear(): void {
    if (!this.ctx || !this.canvas) return;
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    for (const p of this.particles) {
      p.prevX = null;
      p.prevY = null;
    }
  }

  private loop = (timestamp: number): void => {
    if (!this.isRunning) return;

    const elapsedMs = Math.min(timestamp - this.lastTimestamp, 100);
    this.lastTimestamp = timestamp;

    this.renderFrame(elapsedMs / 1000.0);
    this.rafId = requestAnimationFrame(this.loop);
  };

  private renderFrame(dtSec: number): void {
    if (!this.ctx || !this.canvas || this.field === null) return;
    const ctx = this.ctx;
    const width = this.canvas.width;
    const height = this.canvas.height;
    const dpr = this.dpr;

    // Trail fading using destination-in / destination-out blend
    ctx.save();
    ctx.globalCompositeOperation = "destination-in";
    ctx.fillStyle = `rgba(0, 0, 0, ${this.options.fadeOpacity})`;
    ctx.fillRect(0, 0, width, height);
    ctx.restore();

    ctx.save();
    ctx.globalCompositeOperation = "source-over";
    ctx.strokeStyle = this.options.color;
    ctx.lineWidth = this.options.lineWidth * dpr;
    ctx.lineCap = "round";
    ctx.beginPath();

    const dt = Math.max(0.016, dtSec);
    const bounds = this.map.getBounds();
    const south = Math.max(-85, bounds.getSouth());
    const north = Math.min(85, bounds.getNorth());

    for (let i = 0; i < this.particles.length; i++) {
      const p = this.particles[i];

      if (p.age >= p.maxAge || p.lat < south || p.lat > north || !p.visible) {
        this.particles[i] = this.createRandomParticle(false);
        continue;
      }

      const [u, v] = sampleVectorBilinear(this.field, p.lat, p.lon);
      const spd = Math.hypot(u, v);

      if (spd < CALM_WIND_THRESHOLD_MPS || Number.isNaN(spd)) {
        // Calm wind: fade trail and respawn
        p.visible = false;
        p.prevX = null;
        p.prevY = null;
        continue;
      }

      // Convert current geographic coordinate to screen space
      const screenPt = this.map.project([p.lon, p.lat]);
      const currX = screenPt.x * dpr;
      const currY = screenPt.y * dpr;

      if (
        p.prevX !== null &&
        p.prevY !== null &&
        currX >= 0 &&
        currX <= width &&
        currY >= 0 &&
        currY <= height &&
        Math.hypot(currX - p.prevX, currY - p.prevY) < 150 * dpr
      ) {
        ctx.moveTo(p.prevX, p.prevY);
        ctx.lineTo(currX, currY);
      }

      // Advection step with visual speed mapping
      // Visual speed compression: spd^0.65 ensures light winds are visible while high winds are smooth
      const visualSpeedScale = (Math.pow(spd, 0.65) / spd) * 75.0 * this.options.speedScale;
      const [nextLat, nextLon] = advectParticle(
        p.lat,
        p.lon,
        u * visualSpeedScale,
        v * visualSpeedScale,
        dt
      );

      p.lat = nextLat;
      p.lon = nextLon;
      p.prevX = currX;
      p.prevY = currY;
      p.age += 1;
    }

    ctx.stroke();
    ctx.restore();
  }

  public destroy(): void {
    this.stop();
    this.unbindEvents();
    this.clear();
    this.field = null;
    this.particles = [];
  }
}

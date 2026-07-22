# AI Downscaling Architecture & Plan

This document defines the architecture, training strategy, and inference pipeline for our custom AI downscaling engine.

---

## 1. Objective
Bridge coarse global NWP model outputs (~25km GFS/ECMWF/GDPS) down to high-resolution local weather fields (3km / 1km) capturing localized topographic effects, coastal phenomena, microclimates, and terrain-induced precipitation shadows.

---

## 2. Input & Conditioning Tensors

### 2.1 Dynamic Prognostic Inputs (Coarse ~25km Grids)
- 2-meter Temperature ($T_2$)
- U and V Wind Components ($U_{10}, V_{10}$)
- Surface Pressure ($P_{sfc}$)
- Specific / Relative Humidity ($RH$)
- Accumulated Precipitation ($Precip$)
- Geopotential Height & Temperature at standard pressure levels (850hPa, 500hPa).

### 2.2 Static Conditioning Tensors (High-Resolution 3km/1km Grids)
- **Digital Elevation Model (DEM)**: High-resolution terrain height.
- **Slope & Aspect**: Derived terrain inclination and orientation.
- **Land-Use / Land-Cover (LULC)**: Vegetation type, urban canopy fraction, water bodies.
- **Distance to Coast**: Proximity to large water bodies for marine boundary layer effects.

---

## 3. Neural Architecture
- **Base Architecture**: Conditional Super-Resolution Convolutional Neural Network (SRCNN) / Multiscale UNet with Feature-wise Linear Modulation (FiLM) layers for static terrain conditioning.
- **Loss Function**: Combined L1 Loss (for pixel accuracy) + Gradient Difference Loss (for sharp spatial gradients) + Multi-scale perceptual loss.

---

## 4. Training Data Pipeline
- **Ground Truth / High-Resolution Target**: High-res regional reanalysis datasets (e.g., HRRR over North America, ERA5-Land, or station observations).
- **Coarse Input Generation**: Aggressive spatial degradation (averaging/pooling) of high-res fields to simulate ~25km operational model grids.
- **Dataset Storage**: Historical pairs stored in chunked Zarr datasets for efficient batch loading during PyTorch training epochs.

---

## 5. Production Inference Lifecycle
1. **Ingestion**: Raw GRIB2 files are downloaded and parsed into standardized Zarr stores.
2. **Preprocessing**: Coarse model fields are sliced alongside static terrain tensors for the target region.
3. **Inference Workers**: Celery workers dispatch spatial tiles to GPU inference nodes running PyTorch / ONNX Runtime.
4. **Persistence**: High-resolution downscaled outputs (3km/1km) are written to production Zarr storage, ready for MME calibration and API serving.

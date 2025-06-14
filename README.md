# DPCD-Net

This is the code repository for DPCD Net

## Dataset

This project uses two main datasets:

### 1. CMA-BST Dataset

The China Meteorological Administration (CMA) Tropical Cyclone Best Track Dataset provides 6-hourly tropical cyclone track and intensity data for the western North Pacific region (including the South China Sea) since 1949. For typhoons making landfall in China, the temporal resolution is increased to 3-hourly during the 24 hours before landfall and during their time over land.

Dataset Source: [CMA-BST Dataset](https://tcdata.typhoon.org.cn/zjljsjj.html)

### 2. ERA5 Dataset

ERA5 is the fifth generation ECMWF atmospheric reanalysis dataset, providing hourly estimates of atmospheric, ocean-wave, and land-surface quantities. The dataset covers the period from 1940 to present with a spatial resolution of 0.25° x 0.25° for atmospheric variables.

Dataset Source: [ERA5 Dataset](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview)

## Reqirements

To run this project, you will need the following:

- **Python**: 3.6.15
- **CUDA**: 12.2
- **PyTorch**: 1.10.0

Make sure to install the correct versions of these dependencies to ensure compatibility and optimal performance.

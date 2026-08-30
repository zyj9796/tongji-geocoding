# 地理编码

本目录管理 Tongji SAR 地理编码主流程及共享数据。`code/` 包含 GAMMA/DSM 地形地理编码、严格三角面投影和结果制图代码；`data/` 是公共输入；`results/` 是本类任务的正式输出。

## 主要流程

```text
data/RE_SLAVES + data/tongji_dsm_1m.tif
        ↓
dem_import → multi_look → gc_map2 → geocode_back → data2geotiff
        ↓
results/outputs/rasters + results/picall/主流程
```

GDAL 只用于将 GAMMA GeoTIFF 重投影到 EPSG:4326，不参与几何拟合。`siwei_image1_volume_geocode/` 是四维高景单景体地理编码与 GAMMA 基准精化工作包，复用本目录的 `data/` 和 `results/`。

## 运行

在工作区根目录执行：

```bash
bash geocoding/run.sh
bash geocoding/run_full_area.sh
bash geocoding/run_registered_full_area_geocode.sh
bash geocoding/run_strict_triangle_projection.sh
bash geocoding/run_strict_triangle_registration.sh
```

主输出位于 `geocoding/results/outputs/`，扁平图件集位于 `geocoding/results/picall/主流程/`。

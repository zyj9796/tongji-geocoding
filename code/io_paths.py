from __future__ import annotations

from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent
# ``geocoding`` is a category directory inside ``a_geo_tongji``.  The shared
# geometry modules still live in the parent ``geo_bc/src`` repository.
WORKSPACE_ROOT = PROJECT_DIR.parent
REPO_ROOT = PROJECT_DIR.parents[1]

DATA_DIR = PROJECT_DIR / "data"
RSLC_DIR = DATA_DIR / "RE_SLAVES"
BUILDINGS_SHP = DATA_DIR / "shp" / "tongji_clip_rslc_extent_equal_height_clean.shp"
DSM_SAR_EXTENT_TIF = DATA_DIR / "tongji_dsm_sar_extent.tif"
PS_POINTS_CSV = DATA_DIR / "ps_points_all.csv"


def _discover_dsm() -> Path:
    if DSM_SAR_EXTENT_TIF.exists():
        return DSM_SAR_EXTENT_TIF
    preferred = DATA_DIR / "tongji_dsm.tif"
    if preferred.exists():
        return preferred
    candidates = sorted(DATA_DIR.glob("*dsm*.tif")) + sorted(DATA_DIR.glob("*DSM*.tif")) + sorted(DATA_DIR.glob("*.tif"))
    for path in candidates:
        if path.is_file() and path.name != DSM_SAR_EXTENT_TIF.name:
            return path
    return preferred


DSM_TIF = _discover_dsm()

RESULTS_DIR = PROJECT_DIR / "results"
OUTPUTS_DIR = RESULTS_DIR / "outputs"
TABLE_ROOT = OUTPUTS_DIR / "tables"
TABLE_DIR = TABLE_ROOT / "main"
FULL_AREA_TABLE_DIR = TABLE_ROOT / "full_area"
SAME_PIXEL_TABLE_DIR = TABLE_ROOT / "psinsar_same_pixel"
TYPICAL_TABLE_DIR = TABLE_ROOT / "typical_buildings"
LEGACY_TABLE_DIR = TABLE_ROOT / "legacy"

GEOJSON_ROOT = OUTPUTS_DIR / "geodata"
GEOJSON_DIR = GEOJSON_ROOT / "main"
FULL_AREA_GEOJSON_DIR = GEOJSON_ROOT / "full_area"

RASTER_ROOT = OUTPUTS_DIR / "rasters"
TIF_DIR = RASTER_ROOT / "main"
FULL_AREA_RASTER_DIR = RASTER_ROOT / "full_area"

LOG_ROOT = OUTPUTS_DIR / "logs"
LOG_DIR = LOG_ROOT / "main"
FULL_AREA_LOG_DIR = LOG_ROOT / "full_area"

SUMMARY_DIR = OUTPUTS_DIR / "summaries"
PIC_ALL_DIR = RESULTS_DIR / "pic_all"
WORK_DIR = OUTPUTS_DIR / "work"

IMAGE_DIR = PIC_ALL_DIR
PIC_ALL_KEEP_MANIFEST = OUTPUTS_DIR / "pic_all_keep.json"
MAIN_IMAGE_DIR = PIC_ALL_DIR
FULL_AREA_IMAGE_DIR = PIC_ALL_DIR
SAME_PIXEL_IMAGE_DIR = PIC_ALL_DIR
PPT_IMAGE_DIR = PIC_ALL_DIR
TYPICAL_IMAGE_DIR = PIC_ALL_DIR
LEGACY_IMAGE_DIR = PIC_ALL_DIR

FULL_AREA_DIR = FULL_AREA_TABLE_DIR
SAME_PIXEL_DIR = SAME_PIXEL_TABLE_DIR
PPT_DIR = OUTPUTS_DIR / "psinsar_same_pixel_ppt_package"
PPT_CSV_DIR = PPT_DIR / "csv"
PPT_DOC_DIR = PPT_DIR / "docs"
TYPICAL_DIR = TYPICAL_TABLE_DIR
TRASH_DIR = LEGACY_TABLE_DIR

PPT_ZIP = OUTPUTS_DIR / "psinsar_same_pixel_ppt_package.zip"


def ensure_core_output_dirs() -> None:
    for path in [
        RESULTS_DIR,
        OUTPUTS_DIR,
        TABLE_ROOT,
        TABLE_DIR,
        FULL_AREA_TABLE_DIR,
        SAME_PIXEL_TABLE_DIR,
        GEOJSON_ROOT,
        GEOJSON_DIR,
        FULL_AREA_GEOJSON_DIR,
        RASTER_ROOT,
        TIF_DIR,
        FULL_AREA_RASTER_DIR,
        LOG_ROOT,
        LOG_DIR,
        FULL_AREA_LOG_DIR,
        SUMMARY_DIR,
        IMAGE_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def full_area_file(date: str, suffix: str) -> Path:
    return FULL_AREA_DIR / f"{date}_{suffix}"


def full_area_image(date: str, suffix: str) -> Path:
    return FULL_AREA_IMAGE_DIR / f"{date}_{suffix}"


def typical_image(date: str, suffix: str) -> Path:
    return TYPICAL_IMAGE_DIR / f"{date}_{suffix}"

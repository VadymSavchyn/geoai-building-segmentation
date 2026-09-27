# YOLO to Shapefile: GeoAI Building Segmentation Pipeline

An end-to-end Python pipeline that bridges Computer Vision (YOLOv8 segmentation) and Geographic Information Systems (GIS). It takes raw, unreferenced YOLO predictions, smooths jagged polygon boundaries, and exports them into valid, properly georeferenced ESRI Shapefiles.

## 🔍 The Problem It Solves
Standard CV platforms (like Roboflow) strip spatial metadata (CRS, affine transformations) during dataset export. YOLO outputs relative normalized coordinates (`0 to 1`), making direct GIS integration impossible (predictions often map to "Null Island" near Africa). 
This project solves this by programmatically mapping pixel predictions back to the original georeferenced raster tiles using `rasterio` and `geopandas`.

## 🛠️ Tech Stack
* **Python**
* **Ultralytics YOLOv8** (Instance Segmentation)
* **Rasterio** (Spatial metadata & Affine transformations)
* **OpenCV (`cv2.approxPolyDP`)** (Ramer-Douglas-Peucker polygon smoothing)
* **Geopandas / Shapely** (Vector data manipulation & Shapefile export)

## 📂 Project Structure
```text
├── tif_tiles/              # Georeferenced raster tiles with preserved CRS
├── scripts/
│   ├── 01_slice_raster.py  # Slices large orthoimage into tiles keeping affine matrix
│   ├── 02_predict.py       # Runs YOLOv8 inference and saves raw .txt labels
│   └── 03_vectorize.py     # Smooths polygons and exports to .shp
├── requirements.txt
└── README.md
 

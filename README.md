# YOLO to Shapefile: GeoAI Building Segmentation Pipeline

An end-to-end Python pipeline that bridges Computer Vision (YOLOv8 Instance Segmentation) and Geographic Information Systems (GIS). It slices large orthophotos into memory-safe tiles, runs deep learning inference, reconstructs spatial coordinates from normalized labels, regularizes jagged polygon boundaries, and exports topology-valid ESRI Shapefiles.

## The Problem It Solves
Standard Computer Vision frameworks and annotation platforms (like Roboflow) strip spatial metadata (CRS and affine transformations) during image processing. As a result, YOLO outputs relative normalized pixel coordinates (`0.0` to `1.0`), making direct GIS integration impossible — unreferenced predictions simply map to `(0, 0)` ("Null Island"). 

This pipeline solves the issue by preserving tile-level affine transformation matrices during preprocessing and programmatically projecting pixel-space segmentation masks back into real-world geographic coordinates using `rasterio` and `geopandas`.

## Key Technical Features
- **Memory-Efficient Tiling:** Uses `rasterio.windows.Window` to slice massive GeoTIFF orthophotos in chunks without loading the entire raster into RAM.
- **Coordinate Reconstruction:** Maps normalized YOLO segmentation masks (`[0, 1]`) back to pixel coordinates and applies the $3 \times 3$ affine matrix (`transform * (px, py)`) to restore native CRS coordinates.
- **Polygon Regularization:** Applies the Ramer-Douglas-Peucker algorithm (`cv2.approxPolyDP`) to simplify staircase pixel artifacts into clean building footprints.
- **Topology Healing:** Automatically detects and repairs self-intersecting polygons ("bowties") using `shapely` zero-buffering (`poly.buffer(0)`) to ensure full compatibility with ArcGIS Pro and QGIS.

## Tech Stack
- **Python 3.10+**
- **Ultralytics YOLOv8** (Instance Segmentation)
- **Rasterio** (Windowed raster I/O, CRS management, affine transformations)
- **GeoPandas & Shapely** (Vector geometry creation, topology validation, Shapefile export)
- **OpenCV & NumPy** (Coordinate array manipulation & Ramer-Douglas-Peucker simplification)

## Project Structure
```text
├── tif_tiles/              # Georeferenced raster tiles with preserved CRS
├── scripts/
│   ├── 01_slice_raster.py  # Slices large GeoTIFF into 640x640 tiles keeping affine matrix
│   ├── 02_predict.py       # Runs YOLOv8-seg inference and exports normalized .txt masks
│   └── 03_vectorize.py     # Reprojects coordinates, simplifies polygons, and exports to .shp
├── requirements.txt
└── README.md

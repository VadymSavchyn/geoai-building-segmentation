import os
import cv2
import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import Polygon
from pathlib import Path

# --- 1. Настройки путей ---
INPUT_LABELS = Path("runs/segment/predict/labels")
INPUT_TIFFS = Path("tif_tiles")
OUTPUT_SHP_DIR = Path("shapefile_final")

os.makedirs(OUTPUT_SHP_DIR, exist_ok=True)
OUTPUT_SHP = os.path.join(OUTPUT_SHP_DIR, "buildings_smooth.shp")

epsilon_factor = 0.015 

geometries = []
data_records = []
original_crs = None 
total_processed = 0

print("Начинаю обработку...")

# --- 2. Обработка файлов ---
for txt_path in Path(INPUT_LABELS).glob('*.txt'):
    full_name = txt_path.stem 
    
    # Универсальная обрезка любых суффиксов YOLO / Roboflow
    base_name = full_name.split('_png')[0].split('_jpg')[0].split('.rf.')[0] 
    tif_path = os.path.join(INPUT_TIFFS, f"{base_name}.tif")
            
    if not os.path.exists(tif_path):
        continue
        
    with rasterio.open(tif_path) as src:
        transform = src.transform
        width = src.width
        height = src.height
        
        if original_crs is None:
            original_crs = src.crs 
        
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
            
        class_id = int(parts[0])
        
        norm_coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
        pixel_coords = np.zeros_like(norm_coords)
        pixel_coords[:, 0] = norm_coords[:, 0] * width
        pixel_coords[:, 1] = norm_coords[:, 1] * height 
        
        # Сглаживание Дугласом-Пекером
        pixel_coords_cv = pixel_coords.reshape(-1, 1, 2)
        perimeter = cv2.arcLength(pixel_coords_cv, True)
        epsilon = epsilon_factor * perimeter
        smoothed_cv = cv2.approxPolyDP(pixel_coords_cv, epsilon, True)
        smoothed_pixels = smoothed_cv.reshape(-1, 2)
        
        # Защита: если после сглаживания осталось < 3 точек, берем оригинальные пиксели
        if len(smoothed_pixels) < 3:
            smoothed_pixels = pixel_coords.reshape(-1, 2)
            
        # Умножение пикселей на пространственную матрицу
        geo_coords = []
        for px, py in smoothed_pixels:
            lon, lat = transform * (px, py)
            geo_coords.append((lon, lat))
            
        poly = Polygon(geo_coords)
        if not poly.is_valid:
            poly = poly.buffer(0) 

        geometries.append(poly)
        data_records.append({"class_id": class_id, "source": base_name})
        total_processed += 1

# --- 3. Сохранение ---
print(f"Собрано геометрий: {total_processed}")

if not geometries:
    print("Ошибка: Полигоны не собраны. Проверьте пути.")
else:
    gdf = gpd.GeoDataFrame(data_records, geometry=geometries)
    
    # Назначаем родную систему координат из TIF
    if original_crs is not None:
        print(f"Задана система координат: {original_crs}")
        gdf.set_crs(original_crs, inplace=True)
    else:
        print("Внимание: CRS не прочиталась, ставим неизвестную локальную.")
        
    gdf.to_file(OUTPUT_SHP, driver="ESRI Shapefile", encoding='utf-8')
    print(f"✅ Успешно! Файл сохранен в: {OUTPUT_SHP_DIR}")
import os
import rasterio
from rasterio.windows import Window

input_tiff = r'Newkiev.tif'
output_folder = 'tif_tiles'
os.makedirs(output_folder, exist_ok=True)

tile_size = 640

with rasterio.open(input_tiff) as src:
    meta = src.meta.copy()
    
    tile_count = 0
    # Вычисляем, сколько целых тайлов поместится по ширине и высоте
    rows = src.height // tile_size
    cols = src.width // tile_size
    
    # Режем строго по целым кускам, игнорируя остатки по краям
    for j in range(rows):
        for i in range(cols):
            window = Window(i * tile_size, j * tile_size, tile_size, tile_size)
            transform = src.window_transform(window)
            
            meta.update({
                'height': window.height,
                'width': window.width,
                'transform': transform,
                'driver': 'GTiff'
            })
            
            out_path = os.path.join(output_folder, f"tile_{tile_count}.tif")
            
            with rasterio.open(out_path, 'w', **meta) as dest:
                dest.write(src.read(window=window))
            
            tile_count += 1

print(f"Готово! Нарезано TIF-тайлов: {tile_count}")

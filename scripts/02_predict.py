from ultralytics import YOLO
from pathlib import Path

MODEL_PATH = Path('models/best.pt')  
TILES_DIR = Path('tif_tiles')        

model = YOLO(MODEL_PATH)

if __name__ == '__main__':
    results = model.predict(
        source=TILES_DIR,         
        save=True,                
        save_txt=True,            
        conf=0.5,       
        show_labels=False, 
        show_conf=False
    )

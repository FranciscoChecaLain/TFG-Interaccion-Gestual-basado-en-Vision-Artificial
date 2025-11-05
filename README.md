# Proyecto de Detección de Poses 3D

Este proyecto permite detectar y estimar poses 3D a partir de múltiples cámaras, utilizando YOLOv11 para la detección de personas y st-gcn++ para la estimación de poses. Además, genera los archivos de calibración de las cámaras para poder reutilizar la calibración.

---

## Estructura del proyecto

```
.
│   README.md
│   requirements.txt
│
├───camera_calibration
└───PoseDetection
    │   main.py
    │
    └───Models
        ├───st-gcn++
        │       best_top1_acc_epoch_32.pth
        │       config_inference.py
        │       visualize_training.py
        │
        └───Yolo11-Pose
                yolo11m-pose.pt
```
---

## Descripción de carpetas y archivos

- **README.md**: documentación del proyecto.  
- **requirements.txt**: dependencias de Python necesarias.  
- **camera_calibration/**: carpeta donde se generan los archivos `.pkl` con matrices de proyección y parámetros de las cámaras.  
- **PoseDetection/main.py**: script principal que ejecuta la detección de personas, triangulación de articulaciones y estimación de poses 3D.  
- **PoseDetection/Models/**: contiene los pesos y configuraciones de los modelos preentrenados:
  - `st-gcn++`: modelo para estimación de poses 3D.  
  - `Yolo11-Pose`: modelo YOLOv11 para detección de personas y keypoints.  

---

## Requisitos

- Python 3.9 o versiones compatibles con las dependencias.  
- Dependencias listadas en `requirements.txt` (instalarlas con:  
  ```pip install -r requirements.txt```
)

---

## Uso

1. Clonar o descomprimir el proyecto.  
2. Instalar dependencias:  
   ```pip install -r requirements.txt```
3. Ejecutar el script principal:  
   ```python PoseDetection/main.py```
4. Los archivos de calibración se generarán automáticamente en `camera_calibration/`.  
5. Los modelos preentrenados deben permanecer en `PoseDetection/Models/` para que `main.py` funcione correctamente.  

---

## Notas importantes

- Todas las rutas en `main.py` son **relativas**, por lo que el proyecto es portátil.  
- Los archivos `.pkl` en `camera_calibration/` se generan automáticamente al ejecutar y calibrar las cámaras con `main.py`.   

---

## Contacto / Autor

- Francisco Checa Laín 
- fchecalain@correo.ugr.es

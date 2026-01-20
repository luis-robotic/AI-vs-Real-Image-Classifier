# AI-vs-Real-Image-Classifier
El objetivo de este proyecto es desarrollar un modelo de Inteligencia Artificial capaz de clasificar imágenes en dos categorías: imágenes generadas por IA e imágenes no generadas por IA (reales).

## NOTAS Github
url: https://github.com/luis-robotic/AI-vs-Real-Image-Classifier.git

Cada vez que hagas cambios:

git add .
git commit -m "Descripción del cambio"
git push origin main

#### Descargar el dataset:

import kagglehub

<!-- Download latest version -->
path = kagglehub.dataset_download("philosopher0808/real-vs-ai-generated-faces-dataset")

print("Path to dataset files:", path)


#### Ruta:
C:\Users\Usuario\.cache\kagglehub\datasets\
└── philosopher0808\
    └── real-vs-ai-generated-faces-dataset\
        └── versions\
            └── 1\
                └── dataset\
                    └── dataset\

##### Estructura completa del dataset

dataset/
├── train/
│   ├── 0/
│   │   ├── img_0001.jpg
│   │   ├── img_0002.jpg
│   │   └── ...
│   │
│   └── 1/
│       ├── img_0001.jpg
│       ├── img_0002.jpg
│       └── ...
│
├── validate/
│   ├── 0/
│   │   ├── img_0001.jpg
│   │   ├── img_0002.jpg
│   │   └── ...
│   │
│   └── 1/
│       ├── img_0001.jpg
│       ├── img_0002.jpg
│       └── ...
│
└── test/
    ├── 0/
    │   ├── img_0001.jpg
    │   ├── img_0002.jpg
    │   └── ...
    │
    └── 1/
        ├── img_0001.jpg
        ├── img_0002.jpg
        └── ...




# Real vs AI-Generated Faces Classification

## Descripción del Proyecto
Este proyecto tiene como objetivo desarrollar un **clasificador binario de imágenes** que pueda distinguir entre **rostros reales** y **rostros generados por inteligencia artificial**. El proyecto utiliza **Deep Learning** y arquitecturas de redes convolucionales (CNN) para identificar patrones y artefactos presentes en imágenes sintéticas, incluyendo generadas por GANs, diffusion models y técnicas de faceswap.

---

## Motivación
Con la proliferación de generadores de imágenes basados en IA, detectar imágenes sintéticas se ha vuelto crítico en áreas como:

- Seguridad y detección de deepfakes
- Periodismo y verificación de medios
- Investigación en visión por computadora

El proyecto busca explorar **cómo las redes CNN pueden aprender a diferenciar imágenes reales de las generadas**, evaluando precisión, recall, F1-score y otras métricas relevantes.

---

## Dataset

### Fuente
- Kaggle: [Real vs AI Generated Faces Dataset](https://www.kaggle.com/datasets/philosopher0808/real-vs-ai-generated-faces-dataset)
- El dataset contiene imágenes de rostros reales (FFHQ) y rostros generados por IA de múltiples fuentes (StyleGAN, Stable Diffusion, faceswap, ThisPersonDoesNotExist).

### Estructura
El dataset descargado ya viene organizado en:

dataset/
├── train/
│ ├── 0/ ← imágenes reales
│ └── 1/ ← imágenes generadas por IA
├── validate/
│ ├── 0/
│ └── 1/
└── test/
├── 0/
└── 1/


- Cada subcarpeta contiene imágenes `.jpg`
- La separación `train`, `validate` y `test` permite entrenar y evaluar sin necesidad de dividir manualmente los datos.

---

## Preprocesamiento de Imágenes
Para asegurar consistencia durante el entrenamiento:

- Todas las imágenes se cargan en **RGB**
- Se redimensionan a **128×128 píxeles** (configurable)
- Se normalizan los valores de píxel a `[0,1]`
- Data augmentation opcional:
  - Flip horizontal
  - Rotación ligera
  - Ajuste de brillo / contraste

---

## Arquitectura del Modelo

- Se puede usar una **CNN desde cero**:
  - Varias capas `Conv2D` + `ReLU` + `MaxPooling`
  - Flatten → Dense → Output `sigmoid`
- O **Transfer Learning** con modelos preentrenados como:
  - ResNet50
  - MobileNetV2
  - EfficientNet
- Función de pérdida: **Binary Cross-Entropy**
- Optimizador: **Adam**
- Métricas:
  - Accuracy
  - Precision / Recall
  - F1-score
  - ROC-AUC

---

## Entrenamiento

- Split original del dataset: `train / validate / test`
- Batch size recomendado: 16–32
- Early stopping basado en pérdida de validación
- Learning rate scheduler opcional para mejorar convergencia
- Entrenamiento en GPU recomendado si se dispone

---

## Evaluación

- Se analiza la **matriz de confusión** para identificar falsos positivos y falsos negativos
- Se calcula **accuracy, F1-score y ROC-AUC**
- Se pueden usar herramientas como **Grad-CAM** para visualizar qué áreas de la imagen influyen en la decisión del modelo

---

## Uso del Proyecto

1. Clonar repositorio:
<!-- ```bash -->
git clone <REPO_URL>


2. Crear entorno virtual (recomendado):

conda create -n ia_faces python=3.10
conda activate ia_faces


3. Instalar dependencias:

pip install -r requirements.txt


4. Ejecutar notebooks:

01_exploracion_dataset.ipynb

02_preprocesado.ipynb

03_entrenamiento.ipynb


IA_faces_project/
├── data/                     # Dataset descargado
│   ├── train/
│   ├── validate/
│   └── test/
├── notebooks/
│   ├── 01_exploracion_dataset.ipynb
│   ├── 02_preprocesado.ipynb
│   └── 03_entrenamiento.ipynb
├── src/                      # Scripts de carga de datos y modelos
├── requirements.txt
└── README.md


Consideraciones y Limitaciones

El modelo puede detectar patrones específicos de los generadores incluidos en el dataset, pero puede fallar en nuevos generadores no vistos.

Sensible a postprocesado (blur, resize, filtros)

Se recomienda evaluar con datasets externos para validar generalización

Extensiones Futuras

Clasificación multiclase según tipo de generador

Uso de análisis en frecuencia (Fourier) para detectar artefactos

Ensembles de CNNs para mejorar precisión

Visualización de áreas críticas con Grad-CAM o saliency maps

Referencias

FFHQ: https://github.com/NVlabs/ffhq-dataset

StyleGAN / GANs: Karras et al., 2019

Stable Diffusion: https://stability.ai/blog/stable-diffusion-public-release

Kaggle Dataset: Real vs AI Generated Faces Dataset





# w-------------------------------------------------------------------

🎭 Real vs AI-Generated Faces Classification
Mostrar imagen
Mostrar imagen
Mostrar imagen

Un clasificador binario basado en Deep Learning para distinguir entre rostros reales y rostros generados por inteligencia artificial.

📋 Tabla de Contenidos
Descripción
Motivación
Dataset
Instalación
Estructura del Proyecto
Preprocesamiento
Arquitectura del Modelo
Entrenamiento
Evaluación
Resultados
Limitaciones
Trabajo Futuro
Referencias
Licencia
🎯 Descripción
Este proyecto implementa un clasificador binario de imágenes utilizando redes neuronales convolucionales (CNN) para identificar patrones y artefactos presentes en rostros sintéticos generados por:

GANs (StyleGAN, ThisPersonDoesNotExist)
Modelos de difusión (Stable Diffusion)
Técnicas de faceswap
💡 Motivación
La detección de imágenes sintéticas es crítica en múltiples áreas:

🔒 Seguridad: Detección de deepfakes y fraude de identidad
📰 Periodismo: Verificación de autenticidad de medios
🔬 Investigación: Avances en visión por computadora y detección de manipulación
Este proyecto explora cómo las CNNs pueden aprender a diferenciar rostros reales de generados, evaluando métricas como accuracy, precision, recall, F1-score y ROC-AUC.

📊 Dataset
Fuente
Kaggle: Real vs AI Generated Faces Dataset

El dataset combina:

Rostros reales: FFHQ (Flickr-Faces-HQ)
Rostros sintéticos: StyleGAN, Stable Diffusion, faceswap, ThisPersonDoesNotExist
Estructura
dataset/
├── train/
│   ├── 0/          # Imágenes reales
│   └── 1/          # Imágenes generadas por IA
├── validate/
│   ├── 0/
│   └── 1/
└── test/
    ├── 0/
    └── 1/
Características:

Formato: .jpg
División predefinida en train/validate/test
Clases balanceadas
🚀 Instalación
Requisitos Previos
Python 3.10 o superior
GPU recomendada para entrenamiento (opcional)
Pasos
Clonar el repositorio:
bash
git clone https://github.com/tu-usuario/ia-faces-classification.git
cd ia-faces-classification
Crear entorno virtual (recomendado):
bash
# Con conda
conda create -n ia_faces python=3.10
conda activate ia_faces

# O con venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
Instalar dependencias:
bash
pip install -r requirements.txt
Descargar el dataset:
bash
# Usando Kaggle API
kaggle datasets download -d philosopher0808/real-vs-ai-generated-faces-dataset
unzip real-vs-ai-generated-faces-dataset.zip -d data/
📁 Estructura del Proyecto
IA_faces_project/
├── data/                           # Dataset (no incluido en repo)
│   ├── train/
│   ├── validate/
│   └── test/
├── notebooks/                      # Jupyter notebooks
│   ├── 01_exploracion_dataset.ipynb
│   ├── 02_preprocesado.ipynb
│   ├── 03_entrenamiento.ipynb
│   └── 04_evaluacion.ipynb
├── src/                            # Código fuente
│   ├── data_loader.py             # Carga y preparación de datos
│   ├── models.py                  # Arquitecturas CNN
│   ├── train.py                   # Script de entrenamiento
│   ├── evaluate.py                # Evaluación del modelo
│   └── utils.py                   # Funciones auxiliares
├── models/                         # Modelos entrenados (checkpoints)
├── results/                        # Métricas y visualizaciones
├── requirements.txt                # Dependencias del proyecto
├── README.md                       # Este archivo
└── LICENSE                         # Licencia del proyecto
🔧 Preprocesamiento
Transformaciones aplicadas a las imágenes:

Conversión a RGB: Normalización del espacio de color
Redimensionamiento: 128×128 píxeles (configurable)
Normalización: Valores de píxel escalados a [0, 1]
Data Augmentation (opcional)
Flip horizontal aleatorio
Rotación ligera (±15°)
Ajuste de brillo y contraste
Zoom aleatorio
python
# Ejemplo de configuración
IMG_SIZE = 128
BATCH_SIZE = 32
AUGMENTATION = True
🧠 Arquitectura del Modelo
Opción 1: CNN Personalizada
Conv2D(32) → ReLU → MaxPooling
Conv2D(64) → ReLU → MaxPooling
Conv2D(128) → ReLU → MaxPooling
Flatten
Dense(256) → ReLU → Dropout(0.5)
Dense(1) → Sigmoid
Opción 2: Transfer Learning
Modelos preentrenados disponibles:

ResNet50: Buena precisión, computacionalmente intensivo
MobileNetV2: Ligero, ideal para deployment
EfficientNetB0: Balance entre precisión y eficiencia
Configuración de Entrenamiento
Función de pérdida: Binary Cross-Entropy
Optimizador: Adam (lr=0.001)
Métricas: Accuracy, Precision, Recall, F1-Score, ROC-AUC
🏋️ Entrenamiento
Ejecución con Script
bash
python src/train.py --model resnet50 --epochs 50 --batch-size 32
Ejecución con Notebooks
Abre y ejecuta secuencialmente:

01_exploracion_dataset.ipynb - Análisis exploratorio
02_preprocesado.ipynb - Preparación de datos
03_entrenamiento.ipynb - Entrenamiento del modelo
04_evaluacion.ipynb - Evaluación y visualización
Hiperparámetros Recomendados
Parámetro	Valor
Batch size	16-32
Épocas	30-50
Learning rate	0.001
Early stopping patience	5-10
Callbacks utilizados:

Early Stopping (monitor: val_loss)
ModelCheckpoint (guarda mejor modelo)
ReduceLROnPlateau (ajuste dinámico de lr)
📈 Evaluación
Métricas Calculadas
Accuracy: Precisión general del modelo
Precision/Recall: Por clase (real/fake)
F1-Score: Media armónica precision-recall
ROC-AUC: Área bajo la curva ROC
Matriz de Confusión: Análisis de errores
Visualizaciones
Curvas de entrenamiento (loss/accuracy)
Matriz de confusión
Curva ROC
Grad-CAM: Mapas de calor de activación
Ejemplos de predicciones correctas/incorrectas
Ejemplo de Evaluación
bash
python src/evaluate.py --model models/best_model.h5 --test-dir data/test/
🎯 Resultados
Nota: Completa esta sección después del entrenamiento

Métrica	Valor
Test Accuracy	TBD
Precision (Real)	TBD
Precision (Fake)	TBD
Recall (Real)	TBD
Recall (Fake)	TBD
F1-Score	TBD
ROC-AUC	TBD
Observaciones
[Incluye análisis de errores comunes]
[Tipos de imágenes más difíciles de clasificar]
[Comparación entre arquitecturas probadas]
⚠️ Limitaciones
Generalización limitada: El modelo puede detectar patrones específicos de los generadores incluidos en el dataset, pero puede fallar con nuevos generadores no vistos durante el entrenamiento
Sensibilidad al postprocesado: El rendimiento puede degradarse con imágenes que han sido modificadas mediante blur, resize, compresión JPEG o aplicación de filtros
Evolución de generadores: Los modelos generativos mejoran constantemente, lo que puede reducir la efectividad del clasificador con el tiempo
Dataset específico: Entrenado principalmente con rostros frontales de alta calidad; el rendimiento puede variar con ángulos diferentes, oclusiones o baja resolución
🔮 Trabajo Futuro
Mejoras Propuestas
 Clasificación multiclase: Identificar el tipo específico de generador (StyleGAN, Stable Diffusion, etc.)
 Análisis en frecuencia: Utilizar transformadas de Fourier para detectar artefactos espectrales
 Ensemble de modelos: Combinar múltiples CNNs para mejorar robustez
 Explainability avanzada: Implementar Grad-CAM++ y saliency maps
 Dataset extendido: Evaluar con datasets externos (CelebA-HQ, Generated Faces)
 Detección en video: Extender a detección de deepfakes en secuencias
 Model deployment: API REST y aplicación web para clasificación en tiempo real
 Adversarial training: Mejorar resistencia a ataques adversarios
📚 Referencias
FFHQ Dataset: NVlabs/ffhq-dataset
StyleGAN: Karras et al. (2019) - "A Style-Based Generator Architecture for Generative Adversarial Networks"
Stable Diffusion: Stability AI Blog
Dataset Original: Kaggle - Real vs AI Generated Faces
Grad-CAM: Selvaraju et al. (2017) - "Grad-CAM: Visual Explanations from Deep Networks"
Artículos Relacionados
Wang et al. (2020) - "CNN-generated images are surprisingly easy to spot... for now"
Gragnaniello et al. (2021) - "GAN-generated faces detection"
📄 Licencia
Este proyecto está bajo la Licencia MIT. Ver el archivo LICENSE para más detalles.

👥 Contribuciones
Las contribuciones son bienvenidas. Por favor:

Fork el proyecto
Crea una rama para tu feature (git checkout -b feature/AmazingFeature)
Commit tus cambios (git commit -m 'Add some AmazingFeature')
Push a la rama (git push origin feature/AmazingFeature)
Abre un Pull Request
📧 Contacto
Tu Nombre - tu-email@example.com

Link del Proyecto: https://github.com/tu-usuario/ia-faces-classification

🙏 Agradecimientos
Dataset proporcionado por philosopher0808 en Kaggle
FFHQ dataset por NVIDIA Research
Comunidad de TensorFlow/PyTorch por recursos educativos
<div align="center"> <p>Hecho con ❤️ para la detección de deepfakes</p> <p>⭐ Si este proyecto te ha sido útil, considera darle una estrella</p> </div>

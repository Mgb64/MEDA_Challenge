import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import medmnist
from medmnist import INFO

# --- Configuración ---
ROOT = "/lustre/proyectos/p032/datasets"
OUTPUT_DIR = ROOT + "/all_medmnist_images"  # Carpeta donde irán todas las imágenes
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Lista de todos los datasets 2D en MedMNIST v2
# Excluimos los 3D ('organmnist_3d', 'nodulemnist_3d', 'adrenalmnist_3d', 'fracturemnist_3d', 'vesselmnist_3d', 'synapsemnist_3d')
DATASETS_2D = [
    "pathmnist",
    "chestmnist",
    "dermamnist",
    "octmnist",
    "pneumoniamnist",
    "retinamnist",
    "bloodmnist",
    "tissuemnist",
    "organmnist_axial",
    "organmnist_coronal",
    "organmnist_sagittal",
]

print(
    f"Iniciando extracción de {len(DATASETS_2D)} datasets 2D a la carpeta '{OUTPUT_DIR}'..."
)
print("Esto puede tardar un buen rato...")

total_images_saved = 0

# --- Bucle Principal de Extracción ---
for dataset_name in DATASETS_2D:
    print(f"\nProcesando: {dataset_name}")

    # Obtener información del dataset (para saber si es RGB o Grayscale)
    info = INFO[dataset_name]
    n_channels = info["n_channels"]

    # Determinar el modo de la imagen para PIL
    image_mode = "RGB" if n_channels == 3 else "L"  # 'L' es para grayscale

    # Cargar los datos usando la biblioteca medmnist
    DataClass = getattr(medmnist, info["python_class"])

    # Iterar sobre todos los splits (train, val, test)
    # Queremos TODAS las imágenes para el SSL
    for split in ["train", "val", "test"]:
        try:
            # Descargar y cargar los datos
            data = DataClass(split=split, download=True, root=ROOT)
            images = data.imgs

            print(f"  -> Extrayendo {len(images)} imágenes del split '{split}'...")

            # Bucle para guardar cada imagen
            for i in tqdm(range(len(images)), desc=f"  Split {split}", leave=False):
                img_array = images[i]

                # Convertir el array de NumPy a una imagen de PIL
                pil_image = Image.fromarray(img_array, mode=image_mode)

                # Crear un nombre de archivo único
                # Ej: pathmnist_train_000001.png
                filename = f"{dataset_name}_{split}_{i:06d}.png"
                filepath = os.path.join(OUTPUT_DIR, filename)

                # Guardar la imagen
                pil_image.save(filepath)
                total_images_saved += 1

        except Exception as e:
            print(f"  -> Error procesando {dataset_name} split {split}: {e}")

print(f"\n--- ¡Proceso Completado! ---")
print(f"Se guardaron un total de {total_images_saved} imágenes en '{OUTPUT_DIR}'")

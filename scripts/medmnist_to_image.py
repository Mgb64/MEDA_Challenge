import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import medmnist
from medmnist import INFO

# --- Configuración temporal ---
# Guardaremos las imágenes en la carpeta actual para no afectar el entorno del clúster
ROOT = "/lustre/proyectos/p032/datasets"
OUTPUT_DIR = ROOT + "/all_medmnist_images2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Lista temporal de datasets ---
# Comentamos todos los datasets excepto el primero (pathmnist)
DATASETS_2D = [
    "pathmnist",
    "chestmnist",
    "dermamnist",
    "octmnist",
    "pneumoniamnist",
    "retinamnist",
    "bloodmnist",
    "tissuemnist",
    "organamnist",
    "organcmnist",
    "organsmnist",
]

print(f"Iniciando extracción de {len(DATASETS_2D)} dataset(s) a la carpeta '{OUTPUT_DIR}'...")
print("Esto puede tardar un rato dependiendo del dataset...")

total_images_saved = 0

# --- Bucle principal de extracción ---
for dataset_name in DATASETS_2D:
    print(f"\nProcesando: {dataset_name}")

    # Obtener información del dataset (RGB o Grayscale)
    info = INFO[dataset_name]
    n_channels = info["n_channels"]
    image_mode = "RGB" if n_channels == 3 else "L"

    # Cargar la clase correspondiente del dataset
    DataClass = getattr(medmnist, info["python_class"])

    # --- Crear estructura de carpetas ---
    dataset_dir = os.path.join(OUTPUT_DIR, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    # Procesar cada split (train, val, test)
    for split in ["train", "val", "test"]:
        try:
            # Crear subcarpeta del split dentro del dataset
            split_dir = os.path.join(dataset_dir, split)
            os.makedirs(split_dir, exist_ok=True)

            # Descargar y cargar los datos
            data = DataClass(split=split, download=True, root=ROOT)
            images = data.imgs

            print(f"  -> Extrayendo {len(images)} imágenes del split '{split}'...")

            # Guardar las imágenes en su carpeta correspondiente
            for i in tqdm(range(len(images)), desc=f"  Split {split}", leave=False):
                img_array = images[i]
                pil_image = Image.fromarray(img_array, mode=image_mode)

                # Nombre del archivo dentro de su split
                filename = f"{dataset_name}_{split}_{i:06d}.png"
                filepath = os.path.join(split_dir, filename)

                pil_image.save(filepath)
                total_images_saved += 1

        except Exception as e:
            print(f"  -> Error procesando {dataset_name} split {split}: {e}")

print(f"\n--- ¡Proceso Completado! ---")
print(f"Se guardaron un total de {total_images_saved} imágenes en '{OUTPUT_DIR}'")
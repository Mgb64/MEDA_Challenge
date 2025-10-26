import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl
import torch.nn.functional as F
import os
import sys

# --- 1. Definiciones Clave (Copiadas de tu script) ---

# Constante necesaria para definir la arquitectura del Jigsaw Head
JIGSAW_N = 2

# Tamaño de imagen usado para crear la entrada 'dummy'
IMG_SIZE = 28

# --- ¡IMPORTANTE! ---
# Actualiza esta ruta al checkpoint que generó tu entrenamiento
CKPT_PATH = "/lustre/home/atorres/compartido/models/multi_pretext_model2.ckpt"

# Nombre del archivo de salida
ONNX_OUTPUT_PATH = "final_model.onnx"

# Configuración de dispositivo
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {DEVICE}")


# --- 2. Definición de la Arquitectura del Modelo ---
# Es OBLIGATORIO redefinir la clase exactamente como estaba
# durante el entrenamiento para que Lightning pueda cargar el checkpoint.

class MultiPretextSSL_Lightning(pl.LightningModule):
    def __init__(self, backbone, learning_rate=1e-4):
        super().__init__()
        # 'learning_rate' se guarda, pero 'backbone' no, 
        # por eso debemos pasarlo al cargar.
        self.save_hyperparameters('learning_rate') 
        self.backbone = backbone
        self.lr = learning_rate
        
        num_features = 512 # Salida de ResNet18
        
        # --- DECODER CORREGIDO PARA 28x28 ---
        decoder_layers_28x28 = [
            nn.ConvTranspose2d(num_features, 256, kernel_size=4, stride=1, padding=0), # 1x1 -> 4x4
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1), # 4x4 -> 7x7
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 7x7 -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),    # 14x14 -> 28x28
            nn.Sigmoid()
        ]
        
        self.color_head = nn.Sequential(*decoder_layers_28x28)
        self.patch_head = nn.Sequential(*decoder_layers_28x28)
        
        # --- JIGSAW HEAD ---
        self.n_patches = JIGSAW_N * JIGSAW_N
        self.jigsaw_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, self.n_patches * self.n_patches)
        )

    def forward(self, x, task="color"):
        # El forward no se usa para exportar el backbone,
        # pero es parte de la definición de la clase.
        feats = self.backbone(x)
        if task == "color":
            return self.color_head(feats)
        elif task == "patch":
            return self.patch_head(feats)
        elif task == "jigsaw":
            return self.jigsaw_head(feats)

    # Las funciones training_step y configure_optimizers
    # no son necesarias para la inferencia o exportación.


# --- 3. Carga del Modelo desde Checkpoint ---

print(f"Cargando la arquitectura base (ResNet18)...")

# 1. Verificar si existe el checkpoint
if not os.path.exists(CKPT_PATH):
    print(f"Error: No se encontró el archivo de checkpoint en:")
    print(f"{CKPT_PATH}")
    print("Por favor, verifica la ruta y vuelve a intentarlo.")
    sys.exit(1) # Termina el script si no se encuentra el archivo

# 2. Crear la ARQUITECTURA del backbone (vacía)
# Esto es necesario porque el __init__ de tu modelo lo requiere como argumento.
resnet = models.resnet18(weights=None) 
backbone_arch = nn.Sequential(*list(resnet.children())[:-1])

print(f"Cargando modelo Lightning desde: {CKPT_PATH}")

try:
    # 3. Cargar el modelo completo desde el checkpoint
    # Lightning reconstruirá la clase y cargará los pesos del 'state_dict'
    # en todos los submódulos (incluyendo 'backbone_arch').
    model = MultiPretextSSL_Lightning.load_from_checkpoint(
        checkpoint_path=CKPT_PATH,
        map_location=DEVICE,
        # Proporcionamos el argumento 'backbone' que __init__ espera
        backbone=backbone_arch 
    )
    print("Modelo cargado exitosamente.")
    
except Exception as e:
    print(f"Ocurrió un error al cargar el checkpoint: {e}")
    print("Asegúrate de que la clase 'MultiPretextSSL_Lightning' y JIGSAW_N = 2 estén definidos correctamente.")
    sys.exit(1)


# --- 4. Exportación a ONNX ---

# El modelo que queremos exportar es el backbone (feature extractor)
backbone_to_export = model.backbone

# Poner el modelo en modo de evaluación (desactiva BatchNorm, Dropout, etc.)
backbone_to_export.eval()
backbone_to_export.to(DEVICE)

# Crear una entrada 'dummy' del tamaño correcto
# (Batch_size=1, Canales=3, Altura=IMG_SIZE, Ancho=IMG_SIZE)
dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)

print(f"Exportando el backbone a ONNX: {ONNX_OUTPUT_PATH}...")

try:
    torch.onnx.export(
        backbone_to_export, # El submódulo 'backbone' de tu modelo
        dummy_input,        # Una entrada de ejemplo
        ONNX_OUTPUT_PATH,   # Ruta del archivo de salida
        input_names=['input'],
        output_names=['features'],
        opset_version=17,   # Usamos la misma versión de tu script
        dynamic_axes={'input': {0: 'batch_size'}, 'features': {0: 'batch_size'}}
    )
    
    print("\n¡Exportación completada exitosamente! 🚀")
    print(f"Modelo guardado en: {os.path.abspath(ONNX_OUTPUT_PATH)}")
    
except Exception as e:
    print(f"Ocurrió un error durante la exportación a ONNX: {e}")
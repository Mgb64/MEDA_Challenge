# %%
## Importando librerias
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
import copy
import onnxruntime as ort

# %%
## Configuracion de Variables
# ========================================================
# NUEVO: Ruta a tu modelo ONNX
# Este modelo actuará como el backbone extractor de características.
PATH_ONNX_MODEL = "/lustre/home/opacheco/MEDA_Challenge/models/modelo_chicas.onnx"
# NUEVO: Define el número de características que produce tu modelo ONNX.
# Por ejemplo, para un ResNet-18, esto sería 512.
ONNX_OUTPUT_FEATURES = 6
# ========================================================


# Apunta al directorio raíz del dataset
PATH_DATASET = "/lustre/home/opacheco/MEDA_Challenge/compartido/datasets/daudon_dataset/test"

# Define el número de clases de tu dataset
NUM_CLASES = 6

# Parámetros para el entrenamiento del cabezal
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 1.0 - TRAIN_RATIO - VAL_RATIO
NOM_MODELO = 'CM'
K_SHOTS = 50
EPOCHS = 200 # Ajusta según sea necesario
BATCH_SIZE = 64 # Ajusta según sea necesario

# %%
# --------------------------------------------------------
# 1. FIJAR SEMILLA ALEATORIA
# --------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# %%
# ========================================================
# NUEVO: Clase Wrapper para el Backbone ONNX
# Esta clase carga el modelo ONNX y lo hace compatible con PyTorch.
# ========================================================
class OnnxBackbone(nn.Module):
    def __init__(self, onnx_path):
        super().__init__()
        print(f"Cargando modelo ONNX desde: {onnx_path}")

        # --- AÑADIDO: Configurar opciones de la sesión ---
        # Esto previene los errores de 'pthread_setaffinity_np'
        options = ort.SessionOptions()
        
        # Al especificar un número, ONNX Runtime deja de intentar
        # gestionar la afinidad de los hilos, lo que silencia los errores.
        # Puedes probar con 1, 2, o 4 para ver qué da mejor rendimiento.
        options.intra_op_num_threads = 1 
        # --------------------------------------------------

        # Pasamos las 'options' al crear la sesión
        self.session = ort.InferenceSession(onnx_path, options, providers=['CPUExecutionProvider'])
        
        # Obtener los nombres de entrada y salida del modelo
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        print(f"✅ Modelo ONNX cargado. Input: '{self.input_name}', Output: '{self.output_name}'")

    def forward(self, x):
        # 1. Convertir el tensor de PyTorch (posiblemente en GPU) a un array de NumPy en CPU
        x_np = x.cpu().numpy()

        # 2. Ejecutar la inferencia con ONNX Runtime
        # La salida es una lista de arrays de NumPy
        output_np = self.session.run([self.output_name], {self.input_name: x_np})[0]

        # 3. Convertir el resultado de NumPy de vuelta a un tensor de PyTorch
        # y moverlo al dispositivo original (CPU o GPU)
        output_tensor = torch.from_numpy(output_np).to(x.device)
        
        return output_tensor

# %%
# --------------------------------------------------------
# 2. DATASET Y DATALOADERS (Sin cambios)
# --------------------------------------------------------
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

try:
    full_dataset = ImageFolder(PATH_DATASET, transform=data_transform)
    print(f"Dataset cargado: {len(full_dataset)} imágenes en {len(full_dataset.classes)} clases.")
    NUM_CLASES = len(full_dataset.classes)
except FileNotFoundError:
    print(f"Error: No se encontró el directorio {PATH_DATASET}")
    raise

# División estratificada del dataset
targets = full_dataset.targets
indices = list(range(len(targets)))
train_indices, val_test_indices, train_targets, val_test_targets = train_test_split(
    indices, targets, train_size=TRAIN_RATIO, stratify=targets, random_state=SEED)
test_split_ratio = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
val_indices, test_indices = train_test_split(
    val_test_indices, test_size=test_split_ratio, stratify=val_test_targets, random_state=SEED)

train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(full_dataset, val_indices)
test_dataset = Subset(full_dataset, test_indices)

val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=1)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=1)

# Creación del dataset Few-Shot
print(f"Creando Support Set de {K_SHOTS}-shot...")
support_indices = []
for c in range(NUM_CLASES):
    class_indices = [i for i, t in zip(train_indices, train_targets) if t == c]
    if len(class_indices) < K_SHOTS:
        k_shot_indices = class_indices
    else:
        k_shot_indices = random.sample(class_indices, K_SHOTS)
    support_indices.extend(k_shot_indices)

few_shot_dataset = Subset(full_dataset, support_indices)
few_shot_train_loader = DataLoader(few_shot_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
print(f"DataLoader Few-Shot creado con {len(few_shot_dataset)} imágenes.")

# %%
# --------------------------------------------------------
# 3. CABEZAL DE CLASIFICACIÓN (Adaptado para recibir cualquier backbone)
# --------------------------------------------------------
class ClassificationModel(nn.Module):
    def __init__(self, backbone, num_features, num_classes):
        super().__init__()
        # El backbone será nuestro modelo ONNX envuelto en la clase OnnxBackbone
        self.backbone = backbone

        # Cabezal de clasificación (solo estas capas serán entrenadas)
        self.linear_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Extraer características con el backbone (SIN calcular gradientes para él)
        with torch.no_grad():
            feats = self.backbone(x)
        
        # Aplanar las características para el cabezal lineal
        feats = feats.view(feats.size(0), -1)
        
        # Pasar las características por el cabezal que sí se entrena
        out = self.linear_head(feats)
        return out
    
# === Configuración del modelo con ONNX ===
# 1. Instanciar el backbone ONNX
onnx_backbone = OnnxBackbone(PATH_ONNX_MODEL)

# 2. Crear el modelo de clasificación completo
model = ClassificationModel(
    backbone=onnx_backbone,
    num_features=ONNX_OUTPUT_FEATURES,
    num_classes=NUM_CLASES
).to(DEVICE)

# 3. Configurar pérdida y optimizador
# El optimizador solo actualizará los pesos del `linear_head`
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.linear_head.parameters(), lr=3e-3, weight_decay=1e-4)

# %%
# --------------------------------------------------------
# 4. ENTRENAMIENTO (Sin cambios en la lógica)
# --------------------------------------------------------
print("\n🚀 Iniciando entrenamiento del cabezal de clasificación con backbone ONNX...")

best_val_acc = 0.0
best_weights = None
patience = 30
wait = 0

for epoch in range(EPOCHS):
    model.train() # Pone en modo de entrenamiento solo el cabezal
    running_loss = 0.0

    for inputs, labels in few_shot_train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(few_shot_train_loader.dataset)

    # Validación
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_acc = 100 * correct / total

    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {epoch_loss:.4f} - Val Acc: {val_acc:.2f}%")

    # Early stopping y guardado del mejor modelo
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_weights = copy.deepcopy(model.state_dict())
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print(f"⏹ Early stopping en epoch {epoch+1}")
            break

print("Entrenamiento finalizado.")
if best_weights:
    model.load_state_dict(best_weights)
    print(f"✅ Mejor modelo cargado con Val Acc: {best_val_acc:.2f}%")

# %%
# --------------------------------------------------------
# 5. EVALUACIÓN FINAL Y MATRIZ DE CONFUSIÓN
# --------------------------------------------------------
print("\n🧪 Evaluando en el set de testeo...")
model.eval()
all_labels, all_preds = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

accuracy = 100 * (np.array(all_preds) == np.array(all_labels)).sum() / len(all_labels)
print("\n========================================================")
print(f"🎉 ¡Evaluación con Backbone ONNX completa! 🎉")
print(f"   Accuracy en test: {accuracy:.2f} %")
print("========================================================")

# Matriz de Confusión
cm = confusion_matrix(all_labels, all_preds)

# ========================================================
# NUEVO: Cálculo de Accuracy por Clase
# ========================================================
print("\n--- Accuracy por Clase ---")
class_names = full_dataset.classes

# Calcula la precisión para cada clase
# cm.diag() son los correctos (la diagonal)
# cm.sum(axis=1) es el total de cada clase (la suma de cada fila)
with np.errstate(divide='ignore', invalid='ignore'):
    per_class_recall = np.diag(cm) / cm.sum(axis=1) # Correctos / Totales Reales

# Maneja divisiones por cero (si una clase no tiene muestras en el test set)
per_class_recall = np.nan_to_num(per_class_recall)

# Imprime los resultados
for i, class_name in enumerate(class_names):
    # Usamos :<20 para alinear los nombres de las clases
    print(f"   {class_name:<20}: {per_class_recall[i] * 100:.2f} %")
print("--------------------------------")
# ========================================================

# ========================================================
# NUEVO: Guardar resultados en archivo
# ========================================================
filename = f"results_{NOM_MODELO}_{K_SHOTS}_{EPOCHS}_{SEED}.txt"
print(f"\nGuardando resultados en: {filename}")

with open(filename, 'w') as f:
    f.write("--- Evaluación con Backbone ONNX ---\n")
    f.write(f"Modelo: {NOM_MODELO}\n")
    f.write(f"K_Shots: {K_SHOTS}\n")
    f.write(f"Epochs: {EPOCHS}\n")
    f.write(f"Seed: {SEED}\n")
    f.write("========================================\n")
    f.write(f"Accuracy General en Test: {accuracy:.2f} %\n")
    f.write("========================================\n")
    f.write("\n--- Recall (Sensibilidad) por Clase ---\n")

    for i, class_name in enumerate(class_names):
        f.write(f"   {class_name:<20}: {per_class_recall[i] * 100:.2f} %\n")

print("✅ Resultados guardados.")
# ========================================================

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=full_dataset.classes, yticklabels=full_dataset.classes)
plt.title('Matriz de Confusión (Backbone ONNX)')
plt.ylabel('Etiqueta Real')
plt.xlabel('Etiqueta Predicha')
plt.savefig(f"conf_matrix_{NOM_MODELO}_{K_SHOTS}_{EPOCHS}_{SEED}.png")

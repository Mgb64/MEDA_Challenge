#!/usr/bin/env python
# coding: utf-8

# # Probando Dominio Adversarial

# chestmnist + pathmnist -> SSL
# chestmnist(etiquetado) + breastmnist -> DANN
# bloodmnist -> inferencia

# # SSL

# In[5]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
from lightly.data import LightlyDataset
from lightly.transforms import SimCLRTransform
from lightly.models.modules import SimCLRProjectionHead
from lightly.loss import NTXentLoss
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedShuffleSplit


# In[7]:


# Cargar datos
torch.set_float32_matmul_precision("high")

color_jitter = transforms.ColorJitter(
    0.5 * 0.8,  # brillo
    0.5 * 0.8,  # contraste
    0.5 * 0.8,  # saturación
    0.2 * 0.8,  # tono
)

transform = transforms.Compose([
    transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([color_jitter], p=0.8),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = LightlyDataset(
    input_dir='/lustre/proyectos/p032/datasets/images/tmp',
    transform=transform)

dataloader = DataLoader(
    dataset,
    batch_size=256,
    shuffle=True,
    drop_last=True,
    num_workers=1,
)


# In[8]:


# Define Modelo

# --- 2. Backbone ---
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
backbone = nn.Sequential(*list(resnet.children())[:-1])  # Quitar la capa final

from copy import deepcopy

class SimCLRProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim): # <-- Recibe 2048
        super().__init__()
        hidden_dim = input_dim // 4 # Ej: 2048 // 4 = 512
        
        # ¡CORRECTO! Usa el 'input_dim' (2048)
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.head(x)

class MoCoLightning(pl.LightningModule):
    def __init__(self, backbone, 
                 lr=0.0003, 
                 temperature=0.1, 
                 momentum=0.999, 
                 queue_size=65536,
                 input_dim=512, 
                 output_dim=128):
        
        super().__init__()
        self.save_hyperparameters('lr', 'temperature', 'momentum', 'queue_size', 'input_dim', 'output_dim')

        # 1. Crear los encoders de Consulta (q) y Clave (k)
        # El encoder_q es el que se entrena con backprop
        self.encoder_q = nn.Sequential(
            backbone,
            nn.Flatten(start_dim=1), # <-- APLANA a (B, 2048)
            SimCLRProjectionHead(self.hparams.input_dim, self.hparams.output_dim)
        )
        
        # El encoder_k es el encoder de momentum
        self.encoder_k = deepcopy(self.encoder_q)

        # Congelar los parámetros del encoder_k. No se entrenan con el optimizador.
        for param in self.encoder_k.parameters():
            param.requires_grad = False

        # 2. Crear la fila (queue)
        # 
        self.register_buffer("queue", torch.randn(self.hparams.output_dim, self.hparams.queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        
        # Puntero para saber dónde insertar en la fila
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """ Actualización de momentum para el encoder_k """
        # 
        m = self.hparams.momentum
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """ Saca el batch más antiguo de la fila y añade el nuevo batch de 'keys' """
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        # Asegurarse de que el batch cabe
        assert self.hparams.queue_size % batch_size == 0 

        # Reemplazar las claves en la fila
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.hparams.queue_size  # Mover el puntero
        self.queue_ptr[0] = ptr

    def forward(self, x):
        # El forward ahora solo se usa para inferencia (ej. clasificación lineal)
        # Devuelve solo las características del backbone
        return self.encoder_q[0](x).flatten(start_dim=1)

    def training_step(self, batch, batch_idx):
        (im_q, im_k), _, _ = batch # (x0, x1) ahora son im_q (consulta) e im_k (clave)
        
        # 1. Computar features de consulta (q)
        q = self.encoder_q(im_q)
        q = F.normalize(q, dim=1)

        # 2. Computar features de clave (k)
        with torch.no_grad():
            # Actualizar el encoder de clave (momentum)
            self._momentum_update_key_encoder()
            
            # Obtener las claves (sin gradiente)
            k = self.encoder_k(im_k)
            k = F.normalize(k, dim=1)

        # 3. Calcular la pérdida
        loss = self.moco_loss(q, k)
        
        # 4. Actualizar la fila
        self._dequeue_and_enqueue(k)
        
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def moco_loss(self, q, k):
        # q: NxC (consultas)
        # k: NxC (claves positivas)
        # queue: CxK (claves negativas)

        # Logits positivos (N, 1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        
        # Logits negativos (N, K)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # Logits totales (N, 1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        
        # Aplicar temperatura
        logits /= self.hparams.temperature

        # Etiquetas (siempre es la primera columna, la positiva)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=self.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss

    def configure_optimizers(self):
        # IMPORTANTE: El optimizador SOLO debe entrenar el encoder_q
        # Los parámetros del encoder_k se actualizan por momentum.
        
        # El paper usó AdamW [cite: 735]
        optimizer = torch.optim.AdamW(
            self.encoder_q.parameters(),
            lr=self.hparams.lr,
            weight_decay=1e-5 # El paper probó 1e-5 [cite: 736]
        )
        return optimizer


# In[9]:


# --- 4. Inicializar modelo Lightning ---
from pytorch_lightning.loggers import CSVLogger

logger = CSVLogger(save_dir="logs", name="mo_co_run")

model = MoCoLightning(
    backbone=backbone,
    lr=0.0003,          # El LR que tenías
    temperature=0.1,    # La temperatura que tenías
    queue_size=8192     # Un valor más pequeño si 65536 da OOM
)

# --- 5. Entrenador Lightning ---
trainer = pl.Trainer(
    max_epochs=10,
    accelerator="gpu",  # detecta GPU automáticamente
    devices=2,           # cambia a 4 si quieres usar todas tus GPUs
    log_every_n_steps=10,
    logger=logger,
)

# --- 6. Entrenamiento ---
trainer.fit(model, dataloader)

# --- 7. Guardar backbone al final ---
torch.save(model.encoder_q[0].state_dict(), "MG_backbone_ssl.pth")
print(f"El log de pérdidas por época se guardó en: {logger.log_dir}/metrics.csv")


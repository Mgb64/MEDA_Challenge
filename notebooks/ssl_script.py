# %% [markdown]
# # Probando Dominio Adversarial

# %% [markdown]
# chestmnist + pathmnist -> SSL
# chestmnist(etiquetado) + breastmnist -> DANN
# bloodmnist -> inferencia

# %% [markdown]
# # SSL

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader
from lightly.data import LightlyDataset
from lightly.transforms import SimCLRTransform
from lightly.models.modules import SimCLRProjectionHead
from lightly.loss import NTXentLoss
import pytorch_lightning as pl

# %%
# Cargar datos
torch.set_float32_matmul_precision("high")

transform = SimCLRTransform(input_size=28)
dataset = LightlyDataset(
    input_dir='/lustre/proyectos/p032/datasets/images/tmp/',
    transform=transform)

dataloader = DataLoader(
    dataset,
    batch_size=256,
    shuffle=True,
    drop_last=True,
    num_workers=1,
)

# %%
# Define Modelo

# --- 2. Backbone ---
resnet = models.resnet18()
backbone = nn.Sequential(*list(resnet.children())[:-1])  # Quitar la capa final

# --- 3. LightningModule SimCLR ---
class SimCLRLightning(pl.LightningModule):
    def __init__(self, backbone, temperature=0.1, lr=0.01):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimCLRProjectionHead(input_dim=512, output_dim=128)
        self.temperature = temperature
        self.lr = lr

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        z0, z1 = self(x0), self(x1)
        loss = self.nt_xent_loss(z0, z1)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def nt_xent_loss(self, z0, z1):
        z0 = F.normalize(z0, dim=1)
        z1 = F.normalize(z1, dim=1)
        batch_size = z0.shape[0]
        sim = torch.mm(z0, z1.t()) / self.temperature
        labels = torch.arange(batch_size).type_as(sim).long()
        loss0 = F.cross_entropy(sim, labels)
        loss1 = F.cross_entropy(sim.t(), labels)
        return (loss0 + loss1) / 2

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=5e-4
        )

# %%
# --- 4. Inicializar modelo Lightning ---
model = SimCLRLightning(backbone=backbone, temperature=0.1, lr=0.01)

# --- 5. Entrenador Lightning ---
trainer = pl.Trainer(
    max_epochs=10,
    accelerator="gpu",  # detecta GPU automáticamente
    devices=1,           # cambia a 4 si quieres usar todas tus GPUs
    log_every_n_steps=10,
)

# --- 6. Entrenamiento ---
trainer.fit(model, dataloader)

# --- 7. Guardar backbone al final ---
torch.save(model.backbone.state_dict(), "mi_backbone_ssl_xd.pth")
print("💾 Backbone guardado en mi_backbone_ssl_xd.pth")
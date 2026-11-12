import torch
from torch import nn
from .layer import ECDTLayer
from .evidence import RELATIONS, edge_attributes

class ECDT(nn.Module):

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(1024, 2048), nn.LayerNorm(2048), nn.GELU(), nn.Linear(2048, 2048), nn.LayerNorm(2048), nn.GELU())
        self.decoder = nn.Linear(2048, 250)
        self.layers = nn.ModuleList([ECDTLayer(width=128, task_dim=64, kappa_max=0.5) for _ in range(3)])

    def forward(self, x, is_real, edge_metadata, spatial_membership, morphology_membership, coordinates=None, batch=None):
        edges = edge_metadata
        if batch is not None:
            for name in RELATIONS:
                src, dst = edges[name]
                if bool((batch[src] != batch[dst]).any()):
                    raise ValueError('Cross-slide edge prohibited')
        attrs = edges.get('_evidence')
        if attrs is None:
            attrs = edge_attributes(x, coordinates, is_real, spatial_membership, morphology_membership, edges, batch)
        z = self.stem(x)
        for layer in self.layers:
            z = layer(z, edges, attrs, self.decoder, is_real, batch)
        with torch.autocast(device_type=z.device.type, enabled=False):
            prediction = self.decoder(z[is_real].float())
        return dict(prediction=prediction)

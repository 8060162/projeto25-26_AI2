"""
shared/device.py
----------------
Responsabilidade única: resolver o dispositivo de inferência disponível.

Módulo partilhado entre db_client.py e ingest.py — evita a duplicação
da lógica de detecção MPS/CPU que anteriormente existia nos dois ficheiros.

Se o projecto evoluir para suportar CUDA, a alteração é feita aqui
e propaga-se automaticamente a todos os consumidores.
"""

import torch


def resolve_device() -> str:
    """
    Devolve o identificador do dispositivo de inferência disponível.

    Ordem de preferência:
      1. MPS  — Apple Silicon (macOS)
      2. CUDA — GPU NVIDIA   (quando suportado, adicionar aqui)
      3. CPU  — fallback universal
    """
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
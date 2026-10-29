"""Doublure minimale du module `folder_paths` de ComfyUI.

Le code de visualbruno/ComfyUI-Trellis2 (licence MIT) l'importe pour retrouver le dossier des
modèles. Ici, il n'y a pas de ComfyUI : on ne fournit que `models_dir`. Le chemin n'est utilisé
que par des chemins de repli (encodeur de forme, DINOv3) que le handler remplace explicitement.
"""
import os

models_dir = os.environ.get("PIXAL3D_MODELS_DIR", "/models")

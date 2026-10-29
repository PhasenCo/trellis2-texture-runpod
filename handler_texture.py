"""
handler_texture.py — Worker RunPod Serverless : retexture d'un maillage EXISTANT avec
Microsoft TRELLIS.2 (pipeline officiel `Trellis2TexturingPipeline`, « shape-conditioned
texture generation »).

Différence avec une génération image → 3D : la géométrie vient de NOTRE maillage. Le
pipeline encode sa forme, génère seulement l'apparence (couleur, rugosité, métal, alpha)
d'après l'image de référence, puis la cuit dans une texture.

⚠️ Piège du code officiel corrigé ici : `preprocess_mesh` reconstruit un Trimesh SANS
ses UV, si bien que `postprocess_mesh` redéplie toujours le maillage (cumesh). On rend
les UV d'entrée au maillage prétraité : la texture est alors cuite dans NOTRE atlas, et
les sommets/faces restent ceux de l'entrée (on y remet les coordonnées d'origine).

Entrée (`input`) :
  mesh             base64 (ou URL http) d'un .glb ; un seul maillage, une couche UV conseillée
  mesh_gzip        true si `mesh` est un gzip
  image            base64, data URL ou URL http de l'image de référence
  output_transport "trellis-files-v1" (obligatoire, comme les autres workers)
  resolution       512 | 1024 (défaut 1024)
  texture_size     1024 | 2048 | 4096 (défaut 2048)
  seed             entier (défaut 42)
  keep_uv          true (défaut) : garder les UV d'entrée ; false : laisser TRELLIS déplier
  steps, guidance_strength (défaut 6,0 ; officiel 1,0), guidance_interval, guidance_rescale
Sortie : un seul fichier « glb » par /stream (protocole trellis-files-v1).
"""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")

import base64
import gzip
import io
import json
import tempfile
import time
import traceback
import urllib.request

import numpy as np
import runpod
import torch
import trimesh
from PIL import Image

# Fonctions du worker de base (téléchargement des poids et de DINOv3, mémoire basse).
from handler import ensure_trellis2_weights, ensure_dino, resolve_low_vram, REMBG_REPO
from result_transport import stream_files

CONFIG = "texturing_pipeline.sb.json"
RESOLUTIONS = (512, 1024)
TEXTURE_SIZES = (1024, 2048, 4096)
GUIDAGE_DEFAUT = 6.0

pipeline = None
pipeline_error = None


def lire_octets(brut, gzip_active=False):
    if brut.startswith("http://") or brut.startswith("https://"):
        with urllib.request.urlopen(brut, timeout=60) as r:
            data = r.read()
    else:
        if brut.startswith("data:"):
            brut = brut.split(",", 1)[1]
        data = base64.b64decode(brut)
    return gzip.decompress(data) if gzip_active else data


def ecrire_config(model_dir, dino_dir):
    """Copie de texturing_pipeline.json pointant sur DINOv3 local et le quita-fond libre."""
    with open(os.path.join(model_dir, "texturing_pipeline.json")) as f:
        config = json.load(f)
    config["args"]["image_cond_model"]["args"]["model_name"] = dino_dir
    config["args"]["rembg_model"]["args"]["model_name"] = REMBG_REPO
    config["args"]["low_vram"] = resolve_low_vram()
    with open(os.path.join(model_dir, CONFIG), "w") as f:
        json.dump(config, f, indent=2)


def get_pipeline():
    global pipeline, pipeline_error
    if pipeline is not None:
        return pipeline, None
    try:
        debut = time.time()
        if not torch.cuda.is_available():
            raise RuntimeError("Pas de GPU CUDA sur ce worker.")
        model_dir = ensure_trellis2_weights()
        dino_dir = ensure_dino()
        ecrire_config(model_dir, dino_dir)

        from trellis2.pipelines import Trellis2TexturingPipeline

        # Rendre les UV d'entrée au maillage prétraité (cf. docstring).
        origine = Trellis2TexturingPipeline.preprocess_mesh

        def preprocess_avec_uv(self, mesh):
            sortie = origine(self, mesh)
            uv = getattr(getattr(mesh, "visual", None), "uv", None)
            if getattr(self, "_garder_uv", True) and uv is not None and len(uv) == len(mesh.vertices):
                sortie.visual = trimesh.visual.TextureVisuals(uv=np.asarray(uv, dtype=np.float64).copy())
            return sortie

        Trellis2TexturingPipeline.preprocess_mesh = preprocess_avec_uv

        # Avec des UV d'entrée, `postprocess_mesh` reprend `mesh.vertices` et
        # `mesh.vertex_normals` de trimesh TELS QUELS — tableaux en lecture seule — puis
        # les modifie sur place pour passer en repère glTF : « assignment destination is
        # read-only ». (Sans UV, le chemin officiel recrée ses tableaux en redépliant et ne
        # voit jamais l'erreur.) On lui passe des copies modifiables.
        post_origine = Trellis2TexturingPipeline.postprocess_mesh

        class _MaillageModifiable:
            def __init__(self, mesh):
                self.vertices = np.array(mesh.vertices, dtype=np.float64, copy=True)
                self.faces = np.array(mesh.faces, dtype=np.int64, copy=True)
                self.vertex_normals = np.array(mesh.vertex_normals, dtype=np.float64, copy=True)
                self.visual = mesh.visual

        def postprocess_modifiable(self, mesh, *args, **kwargs):
            return post_origine(self, _MaillageModifiable(mesh), *args, **kwargs)

        Trellis2TexturingPipeline.postprocess_mesh = postprocess_modifiable

        pipe = Trellis2TexturingPipeline.from_pretrained(model_dir, CONFIG)
        # Mêmes correctifs que le worker de base : BiRefNet en float32, alias des couches DINOv3.
        if getattr(pipe, "rembg_model", None) is not None:
            pipe.rembg_model.model.float()
        dino = pipe.image_cond_model.model
        if not hasattr(dino, "layer") and hasattr(getattr(dino, "model", None), "layer"):
            dino.__dict__["layer"] = dino.model.layer
        pipe.cuda()
        pipeline = pipe
        pipeline_error = None
        print(f"✅ TRELLIS.2 texturage prêt en {time.time() - debut:.0f} s (low_vram={pipe.low_vram})", flush=True)
    except Exception as e:
        pipeline_error = f"{type(e).__name__}: {e}"
        print(f"❌ Chargement impossible : {pipeline_error}", flush=True)
        traceback.print_exc()
    return pipeline, pipeline_error


def handler(job):
    entree = job.get("input", {})
    if "transport_probe_bytes" in entree:
        yield from stream_files({"glb": os.urandom(max(1, min(int(entree["transport_probe_bytes"]), 80 * 1024 * 1024)))})
        return
    if entree.get("output_transport") != "trellis-files-v1":
        yield {"error": "Ce worker livre par fragments uniquement (output_transport: trellis-files-v1)."}
        return
    if not entree.get("mesh") or not entree.get("image"):
        yield {"error": "Il faut 'mesh' (.glb) et 'image'."}
        return
    try:
        resolution = int(entree.get("resolution", 1024))
        texture_size = int(entree.get("texture_size", 2048))
        if resolution not in RESOLUTIONS or texture_size not in TEXTURE_SIZES:
            raise ValueError(f"resolution {RESOLUTIONS}, texture_size {TEXTURE_SIZES}")
        seed = int(entree.get("seed", 42))
        # Réglages de l'échantillonneur de texture (défauts officiels : 12 pas, guidage 1,0
        # sur [0,6 ; 0,9]). Un guidage plus fort fait coller la texture à l'image : 6 par
        # défaut ici (essais du 29/10/2026 sur un pied en marbre : motif fidèle et stable,
        # préféré à Pixal3D et au guidage officiel).
        echantillonneur = {"guidance_strength": GUIDAGE_DEFAUT}
        if entree.get("steps") is not None:
            echantillonneur["steps"] = max(4, min(50, int(entree["steps"])))
        if entree.get("guidance_strength") is not None:
            echantillonneur["guidance_strength"] = max(0.0, min(15.0, float(entree["guidance_strength"])))
        if entree.get("guidance_interval") is not None:
            debut_g, fin_g = entree["guidance_interval"]
            echantillonneur["guidance_interval"] = [float(debut_g), float(fin_g)]
        if entree.get("guidance_rescale") is not None:
            echantillonneur["guidance_rescale"] = float(entree["guidance_rescale"])
        garder_uv = bool(entree.get("keep_uv", True))
        glb = lire_octets(entree["mesh"], bool(entree.get("mesh_gzip")))
        maillage = trimesh.load(io.BytesIO(glb), file_type="glb", force="mesh", process=False)
        image = Image.open(io.BytesIO(lire_octets(entree["image"])))
        image = image.convert("RGBA") if image.mode in ("RGBA", "LA", "P") else image.convert("RGB")
    except Exception as e:
        yield {"error": f"Entrée invalide : {type(e).__name__}: {e}"}
        return

    pipe, err = get_pipeline()
    if pipe is None:
        yield {"error": f"Pipeline TRELLIS.2 indisponible : {err}"}
        return

    try:
        debut = time.time()
        uv = getattr(getattr(maillage, "visual", None), "uv", None)
        a_des_uv = uv is not None and len(uv) == len(maillage.vertices)
        pipe._garder_uv = garder_uv and a_des_uv
        print(f"🎨 Texturage : {len(maillage.faces)} faces, UV d'entrée {'gardées' if pipe._garder_uv else 'non'}, "
              f"résolution {resolution}, texture {texture_size}", flush=True)
        print(f"   échantillonneur : {echantillonneur or 'défauts'}", flush=True)
        sortie = pipe.run(maillage, image, seed=seed, resolution=resolution, texture_size=texture_size,
                          tex_slat_sampler_params=echantillonneur)

        # Mêmes sommets, mêmes faces, mêmes UV : on remet nos coordonnées d'origine
        # (le pipeline les avait normalisées dans le cube unité et permutées Y/Z).
        identique = pipe._garder_uv and len(sortie.vertices) == len(maillage.vertices) and len(sortie.faces) == len(maillage.faces)
        if identique:
            sortie = trimesh.Trimesh(
                vertices=np.asarray(maillage.vertices),
                faces=np.asarray(maillage.faces),
                process=False,
                visual=sortie.visual,
            )
        torch.cuda.empty_cache()
        with tempfile.TemporaryDirectory() as tmp:
            chemin = os.path.join(tmp, "texture.glb")
            sortie.export(chemin, file_type="glb")
            with open(chemin, "rb") as f:
                octets = f.read()
        print(f"✨ GLB {len(octets)} octets en {time.time() - debut:.0f} s (géométrie d'origine : {identique})", flush=True)
        yield from stream_files({"glb": octets})
    except Exception as e:
        traceback.print_exc()
        torch.cuda.empty_cache()
        cadres = traceback.extract_tb(e.__traceback__)[-4:]
        ou = " <- ".join(f"{os.path.basename(c.filename)}:{c.lineno} {c.name}" for c in reversed(cadres))
        yield {"error": f"{type(e).__name__}: {e} [{ou}]"}


if __name__ == "__main__":
    get_pipeline()
    runpod.serverless.start({"handler": handler, "return_aggregate_stream": False})

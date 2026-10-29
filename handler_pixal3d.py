"""
handler_pixal3d.py — Worker RunPod Serverless : texturage d'un maillage EXISTANT avec Pixal3D
(TencentARC, SIGGRAPH 2026), à partir d'UNE vue POSÉE (image + caméra exactement connues).

Attributions
  - Pixal3D : TencentARC/Pixal3D, code et poids sous licence MIT.
  - Fonction `texture_mesh_pixal3d` et code `trellis2/` modifié : visualbruno/ComfyUI-Trellis2
    (MIT, dérivé de Microsoft TRELLIS.2). Le nœud ComfyUI « Trellis2 - Mesh Texturing Pixal3D
    MultiView » appelle la même fonction ; on suit son chemin « image + camera_params ».
  - valeoai/NAF (Apache-2.0) : sur-échantillonneur des caractéristiques DINOv3 (torch.hub).

Principe (différent de TRELLIS.2 / handler_texture.py) : au lieu d'une attention DINO globale,
Pixal3D PROJETTE les pixels de la vue dans la grille 3D du maillage. Le maillage est encodé
dans le latent de forme (grille 64^3 à resolution=1024, 96^3 à 1536), chaque voxel va chercher
les caractéristiques DINOv3 (+ NAF) du pixel où il se projette avec la caméra fournie, puis le
débruiteur de texture génère l'apparence (couleur, rugosité, métal, alpha) et on la cuit dans
la texture. Si la caméra ne colle pas au maillage, la texture est délavée ou presque noire.
MoGe N'EST PAS nécessaire : il ne sert qu'à deviner une caméra à partir d'une photo ; ici la
caméra vient de nous.

Conventions (à respecter côté rendu Blender)
  - Maillage d'entrée : .glb glTF, Y vers le haut (ce que l'export Blender produit). Repère
    monde Blender = (x, -z_glb, y_glb) : Z haut, l'AVANT du modèle est -Y (= +Z glTF).
  - Caméra : sténopé, fov HORIZONTAL `fov_deg`, pixels carrés, point principal au centre,
    visant `target` (par défaut le centre de la boîte englobante du maillage), haut = +Z monde.
    Position de l'œil dans le repère Blender, d = distance :
        eye = target + d * [cos(el)*sin(az), -cos(el)*cos(az), sin(el)]
    az=0, el=0 : caméra sur -Y, vue de face (c'est la pose canonique de Pixal3D).
    az=90 : caméra sur +X ; az=180 : de dos ; az=270 : sur -X ; el>0 : au-dessus.
    Aux pôles (|el|=90) le haut de l'image est +Y (el=+90) ou -Y (el=-90).
  - `distance` est en unités du .glb (mètres si l'export en est en mètres), de l'œil à `target`.
    Sans `distance` (framing "auto") elle est estimée sur le remplissage de l'image par la
    silhouette (alpha) en supposant que son côté le plus long vaut la plus grande dimension de
    la boîte englobante : fiable de face, approximatif sous un autre angle. Mieux : la donner.
  - Image : RGBA (fond transparent d'un rendu Blender « film transparent »). Elle n'est ni
    recadrée ni redimensionnée par le worker (le cadrage EST la caméra) ; elle est seulement
    prémultipliée sur fond noir, comme à l'entraînement, et complétée en carré si besoin.

⚠️ Correctifs de code (mêmes que handler_texture.py) : copies modifiables avant
`postprocess_mesh` (tableaux trimesh en lecture seule) ; UV d'entrée conservées ; en sortie,
on remet exactement les sommets, faces et UV d'origine.

Entrée (`input`) :
  mesh             base64 (ou URL http) d'un .glb ; un seul maillage, UV conseillées
  mesh_gzip        true si `mesh` est un gzip
  image            base64, data URL ou URL http de la vue de référence (RGBA conseillé)
  mask             (optionnel) masque base64/URL de l'objet si `image` est opaque
  allow_opaque     true : accepter une image opaque sans masque (fond pris tel quel)
  camera           { azimuth_deg (0), elevation_deg (0), fov_deg (obligatoire),
                     distance (unités du glb) | framing: "auto" (défaut),
                     target: [x, y, z] en coordonnées glb (défaut : centre de la boîte) }
  output_transport "trellis-files-v1" (obligatoire)
  resolution       1024 | 1536 (défaut 1024 ; pas de 512 chez Pixal3D)
  texture_size     1024 | 2048 | 4096 (défaut 2048)
  seed, steps (12), guidance_strength (1,0), guidance_interval ([0,6 ; 0,9]),
  guidance_rescale (0,0), rescale_t (3,0)
  keep_uv          true (défaut) : garder les UV d'entrée ; false : laisser dérouler (risqué)
  naf_tile_factor  1 (défaut) | 2 | 4 : découpe NAF en tuiles pour économiser la VRAM
Sortie : un seul fichier « glb » par /stream (protocole trellis-files-v1).
"""

import os
import sys

# Le `trellis2` de Pixal3D (copie modifiée de ComfyUI-Trellis2) vit dans un dossier à part et
# passe AVANT le /app/trellis2 de l'image de base. À faire avant tout `import trellis2`.
PIXAL3D_SRC = os.environ.get("PIXAL3D_SRC", "/app/pixal3d_src")
PIXAL3D_STUBS = os.environ.get("PIXAL3D_STUBS", "/app/pixal3d_stubs")
for _chemin in (PIXAL3D_STUBS, PIXAL3D_SRC):
    if os.path.isdir(_chemin) and _chemin not in sys.path:
        sys.path.insert(0, _chemin)
for _nom in [n for n in sys.modules if n == "trellis2" or n.startswith("trellis2.")]:
    del sys.modules[_nom]

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")

import base64
import gzip
import io
import math
import time
import traceback
import urllib.request

import numpy as np
import runpod
import torch
import trimesh
from PIL import Image

# Fonctions du worker de base (DINOv3 à accès restreint via HF_TOKEN, mémoire basse).
from handler import ensure_dino, resolve_low_vram
from result_transport import stream_files

MODEL_DIR = os.environ.get("PIXAL3D_MODEL_DIR", "/models/Pixal3D")
CONFIG = "pipeline.json"
RESOLUTIONS = (1024, 1536)
TEXTURE_SIZES = (1024, 2048, 4096)

# glb (Y haut) -> monde Blender (Z haut) : B = (x, -z, y)
GLB_VERS_BLENDER = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
# Blender -> repère d'entrée du pipeline : preprocess_mesh fait (x, y, z) -> (x, -z, y), puis la
# grille est tournée vers Blender par (x, y, z) -> (x, -z, y) ; l'ensemble vaut diag(1, -1, -1).
BLENDER_VERS_ENTREE = np.diag([1.0, -1.0, -1.0])
# Rotation de la caméra de face (colonnes : droite, haut, arrière) en repère Blender.
ROT_FACE = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])

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


def ensure_pixal3d_weights():
    """Poids baked dans l'image ; à défaut (image légère), téléchargés une fois sur le volume."""
    requis = (
        "pipeline.json",
        "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors",
        "ckpts/tex_dec_next_dc_f16c32_fp16.safetensors",
        "ckpts/shape_enc_next_dc_f16c32_fp16.safetensors",
    )
    if all(os.path.exists(os.path.join(MODEL_DIR, f)) for f in requis):
        return MODEL_DIR
    from huggingface_hub import snapshot_download
    base = "/runpod-volume" if os.path.isdir("/runpod-volume") else "/models"
    cible = os.path.join(base, "Pixal3D")
    os.makedirs(cible, exist_ok=True)
    print(f"⬇️  Poids Pixal3D absents de l'image ; téléchargement vers {cible}...", flush=True)
    snapshot_download("TencentARC/Pixal3D", local_dir=cible, allow_patterns=[
        "pipeline.json", "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.*",
        "ckpts/tex_dec_next_dc_f16c32_fp16.*"])
    snapshot_download("microsoft/TRELLIS.2-4B", local_dir=cible,
                      allow_patterns=["ckpts/shape_enc_next_dc_f16c32_fp16.*"])
    return cible


def get_pipeline():
    global pipeline, pipeline_error
    if pipeline is not None:
        return pipeline, None
    try:
        debut = time.time()
        if not torch.cuda.is_available():
            raise RuntimeError("Pas de GPU CUDA sur ce worker.")
        model_dir = ensure_pixal3d_weights()
        dino_dir = ensure_dino()

        import trellis2
        if not os.path.abspath(trellis2.__file__).startswith(os.path.abspath(PIXAL3D_SRC)):
            raise RuntimeError(f"Mauvais paquet trellis2 chargé : {trellis2.__file__} (attendu sous {PIXAL3D_SRC})")
        from trellis2.pipelines import trellis2_image_to_3d as module_pipeline
        from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline

        # Le modèle de conditionnement image (DINOv3 + NAF) est construit à la demande, après
        # `pipe.to()`, donc sur CPU ; en mode non « low_vram » le pipeline ne le déplace pas. On
        # l'envoie sur le GPU à la construction et on règle la découpe en tuiles de NAF.
        construire_origine = module_pipeline.build_pixal3d_image_cond_model

        def construire_sur_gpu(config):
            modele = construire_origine(config)
            modele.naf_tile_factor = int(os.environ.get("PIXAL3D_NAF_TILE_FACTOR", "1"))
            return modele.to(torch.device("cuda"))

        module_pipeline.build_pixal3d_image_cond_model = construire_sur_gpu

        # `postprocess_mesh` reprend `mesh.vertices` / `vertex_normals` de trimesh tels quels
        # (lecture seule) puis les modifie sur place : on lui passe des copies modifiables.
        # Les UV d'entrée, elles, sont déjà gardées (chemin « mesh.visual.uv » du code Pixal3D).
        post_origine = Trellis2ImageTo3DPipeline.postprocess_mesh

        class _MaillageModifiable:
            def __init__(self, mesh):
                self.vertices = np.array(mesh.vertices, dtype=np.float64, copy=True)
                self.faces = np.array(mesh.faces, dtype=np.int64, copy=True)
                self.vertex_normals = np.array(mesh.vertex_normals, dtype=np.float64, copy=True)
                self.visual = mesh.visual

        def postprocess_modifiable(self, mesh, *args, **kwargs):
            return post_origine(self, _MaillageModifiable(mesh), *args, **kwargs)

        Trellis2ImageTo3DPipeline.postprocess_mesh = postprocess_modifiable

        pipe = Trellis2ImageTo3DPipeline.from_pretrained(
            model_dir, CONFIG, keep_models_loaded=True, isPixal3D=True, isPixal3DMV=False)
        # DINOv3 local (à la place du chemin ComfyUI) pour les quatre étages ; seul « tex_1024 » sert.
        for cfg in pipe.PIXAL3D_IMAGE_COND_CONFIGS.values():
            cfg["model_name"] = dino_dir
        pipe.low_vram = resolve_low_vram()
        pipe.to(torch.device("cuda"))
        pipeline = pipe
        pipeline_error = None
        print(f"✅ Pixal3D texturage prêt en {time.time() - debut:.0f} s (low_vram={pipe.low_vram})", flush=True)
    except Exception as e:
        pipeline_error = f"{type(e).__name__}: {e}"
        print(f"❌ Chargement impossible : {pipeline_error}", flush=True)
        traceback.print_exc()
    return pipeline, pipeline_error


# ---------------------------------------------------------------------------
# Caméra et image
# ---------------------------------------------------------------------------

def rotation_camera(azimut_deg, elevation_deg):
    """Rotation 3x3 (colonnes : droite, haut, arrière) de la caméra en repère Blender.

    Même construction que `blender_c2w_from_azimuth_elevation` de ComfyUI-Trellis2.
    """
    az, el = math.radians(float(azimut_deg)), math.radians(float(elevation_deg))
    oeil = np.array([math.cos(el) * math.sin(az), -math.cos(el) * math.cos(az), math.sin(el)])
    arriere = oeil / (np.linalg.norm(oeil) + 1e-12)
    haut_monde = np.array([0.0, 0.0, 1.0])
    if abs(float(arriere @ haut_monde)) > 0.999:
        haut_monde = np.array([0.0, 1.0 if arriere[2] > 0 else -1.0, 0.0])
    droite = np.cross(haut_monde, arriere)
    droite /= np.linalg.norm(droite) + 1e-12
    haut = np.cross(arriere, droite)
    haut /= np.linalg.norm(haut) + 1e-12
    return np.stack([droite, haut, arriere], axis=1)


def preparer_image(image, masque, autoriser_opaque, fov_h_rad):
    """Image carrée prémultipliée sur fond noir + fov horizontal du carré + remplissage.

    Rien n'est recadré ni mis à l'échelle (le cadrage est la caméra). Une image non carrée est
    complétée de noir, centrée : le point principal reste au centre, et si l'image est plus haute
    que large le fov horizontal change avec la nouvelle largeur (focale en pixels inchangée).
    """
    if masque is not None:
        alpha = masque.convert("L").resize(image.size, Image.Resampling.LANCZOS)
        image = image.convert("RGB")
    elif image.mode in ("RGBA", "LA", "P"):
        rgba = image.convert("RGBA")
        alpha, image = rgba.getchannel(3), rgba.convert("RGB")
        if np.all(np.asarray(alpha) == 255) and not autoriser_opaque:
            raise ValueError("L'image RGBA est entièrement opaque : fournir un fond transparent, "
                             "un 'mask', ou allow_opaque=true.")
    elif autoriser_opaque:
        image = image.convert("RGB")
        alpha = Image.new("L", image.size, 255)
    else:
        raise ValueError("Image opaque sans masque : fournir un RGBA (fond transparent), un 'mask', "
                         "ou allow_opaque=true.")
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    a = np.asarray(alpha, dtype=np.float32)[..., None] / 255.0
    premult = (rgb * a * 255.0 + 0.5).astype(np.uint8)
    h, w = premult.shape[:2]
    cote = max(w, h)
    toile = np.zeros((cote, cote, 3), dtype=np.uint8)
    masque_carre = np.zeros((cote, cote), dtype=np.float32)
    x0, y0 = (cote - w) // 2, (cote - h) // 2
    toile[y0:y0 + h, x0:x0 + w] = premult
    masque_carre[y0:y0 + h, x0:x0 + w] = a[..., 0]
    focale_px = w / (2.0 * math.tan(fov_h_rad / 2.0))
    fov_carre = 2.0 * math.atan(cote / (2.0 * focale_px))
    ys, xs = np.nonzero(masque_carre > 0.8)
    if len(xs) == 0:
        raise ValueError("Le masque de l'objet est vide (alpha < 0,8 partout).")
    remplissage = max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) / cote
    return Image.fromarray(toile, "RGB"), fov_carre, float(remplissage)


def lire_camera(brut):
    cam = brut or {}
    if cam.get("fov_deg") is None:
        raise ValueError("camera.fov_deg est obligatoire (fov horizontal, en degrés).")
    fov = float(cam["fov_deg"])
    if not 1.0 <= fov <= 170.0:
        raise ValueError("camera.fov_deg doit être dans [1 ; 170].")
    distance = cam.get("distance")
    if distance is not None:
        distance = float(distance)
        if distance <= 0:
            raise ValueError("camera.distance doit être > 0.")
    elif cam.get("framing", "auto") != "auto":
        raise ValueError("camera.framing ne vaut que \"auto\" (ou fournir camera.distance).")
    cible = cam.get("target")
    if cible is not None:
        cible = np.asarray(cible, dtype=np.float64).reshape(3)
    return {
        "azimut": float(cam.get("azimuth_deg", 0.0)),
        "elevation": float(cam.get("elevation_deg", 0.0)),
        "fov": math.radians(fov),
        "distance": distance,
        "cible": cible,
    }


# ---------------------------------------------------------------------------
# Maillage : repère du pipeline et boîte épinglée
# ---------------------------------------------------------------------------

class Repere:
    """Transformation glb -> repère d'entrée du pipeline, avec son inverse.

    `preprocess_mesh` recentre sur la boîte englobante et met à l'échelle sa plus grande
    dimension à 1 : la caméra, elle, vise `cible` et tourne autour. Pour que les deux
    coïncident, on tourne le maillage autour de `cible` (la vue devient la vue de face) et on
    ÉPINGLE la boîte de normalisation : deux sommets orphelins (sans face) aux coins d'une boîte
    centrée sur `cible` qui contient le maillage. Ils sont retirés à l'export.
    """

    def __init__(self, sommets_glb, cam):
        mn, mx = sommets_glb.min(axis=0), sommets_glb.max(axis=0)
        self.etendue_max = float((mx - mn).max())
        cible_glb = cam["cible"] if cam["cible"] is not None else (mn + mx) / 2.0
        self.cible_blender = GLB_VERS_BLENDER @ cible_glb
        # Mettre la caméra (az, el) en position de face : v' = Rm (v - cible) + cible
        self.rot = ROT_FACE @ rotation_camera(cam["azimut"], cam["elevation"]).T
        entree = self.vers_entree(sommets_glb)
        self.centre = BLENDER_VERS_ENTREE @ self.cible_blender
        self.demi = np.abs(entree - self.centre).max(axis=0)
        self.echelle = 0.99999 / (2.0 * float(self.demi.max()))  # celle de preprocess_mesh

    def vers_entree(self, v_glb):
        b = v_glb @ GLB_VERS_BLENDER.T
        b = (b - self.cible_blender) @ self.rot.T + self.cible_blender
        return b @ BLENDER_VERS_ENTREE.T

    def depuis_entree(self, v_entree):
        b = v_entree @ BLENDER_VERS_ENTREE
        b = (b - self.cible_blender) @ self.rot + self.cible_blender
        return b @ GLB_VERS_BLENDER

    def depuis_normalise(self, v_norm):
        """Sortie du pipeline (centrée sur la boîte épinglée, côté max = 1) -> repère glb."""
        return self.depuis_entree(v_norm / self.echelle + self.centre)


def construire_entree_pipeline(maillage, repere, avec_uv):
    """Maillage dans le repère du pipeline, boîte épinglée, UV d'entrée si demandées."""
    v = repere.vers_entree(np.asarray(maillage.vertices, dtype=np.float64))
    coin_min, coin_max = repere.centre - repere.demi, repere.centre + repere.demi
    tol = 1e-9 * max(1.0, float(repere.demi.max()))
    deja_epinglee = np.allclose(v.min(axis=0), coin_min, atol=tol) and np.allclose(v.max(axis=0), coin_max, atol=tol)
    uv = getattr(getattr(maillage, "visual", None), "uv", None)
    uv = np.asarray(uv, dtype=np.float64) if avec_uv and uv is not None and len(uv) == len(v) else None
    if not deja_epinglee:
        v = np.vstack([v, coin_min[None], coin_max[None]])
        if uv is not None:
            uv = np.vstack([uv, np.zeros((2, 2))])
    visuel = trimesh.visual.TextureVisuals(uv=uv) if uv is not None else None
    return trimesh.Trimesh(vertices=v, faces=np.asarray(maillage.faces), process=False, visual=visuel), (0 if deja_epinglee else 2)


def handler(job):
    entree = job.get("input", {})
    if "transport_probe_bytes" in entree:
        yield from stream_files({"glb": os.urandom(max(1, min(int(entree["transport_probe_bytes"]), 80 * 1024 * 1024)))})
        return
    if entree.get("output_transport") != "trellis-files-v1":
        yield {"error": "Ce worker livre par fragments uniquement (output_transport: trellis-files-v1)."}
        return
    if not entree.get("mesh") or not entree.get("image"):
        yield {"error": "Il faut 'mesh' (.glb) et 'image' (la vue posée)."}
        return
    try:
        resolution = int(entree.get("resolution", 1024))
        texture_size = int(entree.get("texture_size", 2048))
        if resolution not in RESOLUTIONS or texture_size not in TEXTURE_SIZES:
            raise ValueError(f"resolution {RESOLUTIONS}, texture_size {TEXTURE_SIZES}")
        seed = int(entree.get("seed", 42))
        # Réglages de l'échantillonneur de texture (défauts du pipeline Pixal3D).
        echantillonneur = {}
        if entree.get("steps") is not None:
            echantillonneur["steps"] = max(4, min(50, int(entree["steps"])))
        if entree.get("guidance_strength") is not None:
            echantillonneur["guidance_strength"] = max(0.0, min(15.0, float(entree["guidance_strength"])))
        if entree.get("guidance_interval") is not None:
            debut_g, fin_g = entree["guidance_interval"]
            echantillonneur["guidance_interval"] = [float(debut_g), float(fin_g)]
        if entree.get("guidance_rescale") is not None:
            echantillonneur["guidance_rescale"] = float(entree["guidance_rescale"])
        if entree.get("rescale_t") is not None:
            echantillonneur["rescale_t"] = float(entree["rescale_t"])
        garder_uv = bool(entree.get("keep_uv", True))
        tuiles = int(entree.get("naf_tile_factor", 1))
        if tuiles not in (1, 2, 4):
            raise ValueError("naf_tile_factor : 1, 2 ou 4")
        cam = lire_camera(entree.get("camera"))

        glb = lire_octets(entree["mesh"], bool(entree.get("mesh_gzip")))
        maillage = trimesh.load(io.BytesIO(glb), file_type="glb", force="mesh", process=False)
        uv_origine = getattr(getattr(maillage, "visual", None), "uv", None)
        a_des_uv = uv_origine is not None and len(uv_origine) == len(maillage.vertices)
        if garder_uv and not a_des_uv:
            raise ValueError("Le maillage n'a pas d'UV par sommet. Ajouter des UV (recommandé) ou keep_uv=false "
                             "(le pipeline redéplie alors le maillage avec cumesh : non éprouvé ici).")

        image = Image.open(io.BytesIO(lire_octets(entree["image"])))
        masque = Image.open(io.BytesIO(lire_octets(entree["mask"]))) if entree.get("mask") else None
        vue, fov_carre, remplissage = preparer_image(image, masque, bool(entree.get("allow_opaque")), cam["fov"])

        repere = Repere(np.asarray(maillage.vertices, dtype=np.float64), cam)
        if cam["distance"] is not None:
            distance_monde = cam["distance"]
            origine_distance = "fournie"
        else:
            # d (unité = plus grande dimension de la boîte) tel que la boîte remplit `remplissage`
            d_norm = 0.5 / (math.tan(fov_carre / 2.0) * remplissage)
            distance_monde = d_norm * repere.etendue_max
            origine_distance = f"estimée (remplissage {100 * remplissage:.1f} %)"
        distance_norm = distance_monde * repere.echelle
    except Exception as e:
        yield {"error": f"Entrée invalide : {type(e).__name__}: {e}"}
        return

    pipe, err = get_pipeline()
    if pipe is None:
        yield {"error": f"Pipeline Pixal3D indisponible : {err}"}
        return

    try:
        debut = time.time()
        os.environ["PIXAL3D_NAF_TILE_FACTOR"] = str(tuiles)
        # Un modèle de conditionnement déjà construit garde son ancien réglage de tuiles.
        for attr in ("pixal3d_image_cond_tex_1024",):
            modele = getattr(pipe, attr, None)
            if modele is not None:
                modele.naf_tile_factor = tuiles
        mesh_pipeline, orphelins = construire_entree_pipeline(maillage, repere, garder_uv)
        print(f"🎨 Pixal3D : {len(maillage.faces)} faces, UV {'gardées' if garder_uv else 'non'}, "
              f"résolution {resolution}, texture {texture_size}", flush=True)
        print(f"   caméra : az {cam['azimut']:.1f}°, el {cam['elevation']:.1f}°, fov {math.degrees(fov_carre):.2f}° "
              f"(carré), distance {distance_monde:.4f} {origine_distance} -> {distance_norm:.4f} unités normalisées "
              f"({orphelins} sommets d'épinglage)", flush=True)
        print(f"   échantillonneur : {echantillonneur or 'défauts'}", flush=True)

        sortie, _, _ = pipe.texture_mesh_pixal3d(
            mesh=mesh_pipeline,
            views=None,
            image=vue,
            camera_params={"camera_angle_x": fov_carre, "distance": distance_norm, "mesh_scale": 1.0},
            seed=seed,
            tex_slat_sampler_params=echantillonneur,
            resolution=resolution,
            texture_size=texture_size,
            texture_alpha_mode="OPAQUE",
            double_side_material=False,
            bake_on_vertices=False,
            use_custom_normals=False,
            mesh_orientation="none",  # l'orientation est déjà gérée par `Repere`
        )

        # Mêmes sommets, faces et UV : on remet nos coordonnées d'origine (le pipeline les avait
        # normalisées et permutées) et on retire les sommets d'épinglage.
        identique = garder_uv and len(sortie.vertices) == len(mesh_pipeline.vertices) and len(sortie.faces) == len(maillage.faces)
        materiau = sortie.visual.material
        if identique:
            sortie = trimesh.Trimesh(
                vertices=np.asarray(maillage.vertices),
                faces=np.asarray(maillage.faces),
                process=False,
                visual=trimesh.visual.TextureVisuals(uv=np.asarray(uv_origine), material=materiau),
            )
        else:
            # Sortie redépliée : repasser du repère normalisé du pipeline au repère du glb.
            sortie = trimesh.Trimesh(
                vertices=repere.depuis_normalise(np.asarray(sortie.vertices, dtype=np.float64)),
                faces=np.asarray(sortie.faces),
                process=False,
                visual=sortie.visual,
            )
        torch.cuda.empty_cache()
        with io.BytesIO() as tampon:
            sortie.export(tampon, file_type="glb")
            octets = tampon.getvalue()
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

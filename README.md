# Worker RunPod : retexture TRELLIS.2 (Phasen)

Texture un maillage EXISTANT (`.glb`) à partir d'une image de référence avec le pipeline officiel
`Trellis2TexturingPipeline` de Microsoft TRELLIS.2. La géométrie et, par défaut, les UV d'entrée
sont conservées : la texture est cuite dans l'atlas fourni.

Image : `ghcr.io/phasenco/trellis2-texture-runpod` (construite par GitHub Actions à chaque push).
Base : `ghcr.io/huachuhua/trellis2-runpod` (dépendances CUDA, poids TRELLIS.2), figée par empreinte.

Endpoint RunPod : GPU ≥ 24 Go, disque 60 Go, variable `HF_TOKEN` (compte Hugging Face ayant
accepté la licence de `facebook/dinov3-vitl16-pretrain-lvd1689m`, modèle à accès restreint).

Entrée : `mesh` (base64 ou URL d'un .glb), `image` (base64, data URL ou URL), `output_transport:
"trellis-files-v1"`, `resolution` 512|1024, `texture_size` 1024|2048|4096, `seed`, `keep_uv`, `steps`, `guidance_strength`, `guidance_interval`, `guidance_rescale`.
Sortie : le `.glb` texturé, par `/stream` (protocole trellis-files-v1).

## Second moteur : Pixal3D (`Dockerfile.pixal3d`, `handler_pixal3d.py`)

Texture un maillage existant à partir d'**une vue posée** (image + caméra connues, par exemple un
rendu Blender repeint) avec Pixal3D : les pixels de la vue sont *projetés* dans la grille 3D du
maillage au lieu d'une attention DINO globale. Image : tags `pixal3d-latest` et
`pixal3d-sha-<court>` (même dépôt GHCR, même image de base figée). MoGe n'est pas nécessaire.

Entrée : `mesh` (.glb, UV conservées), `image` (RGBA, fond transparent), `camera`
`{ azimuth_deg, elevation_deg, fov_deg, distance | framing:"auto", target? }`, `output_transport:
"trellis-files-v1"`, `resolution` 1024|1536, `texture_size`, `seed`, `steps`, `guidance_strength`,
`guidance_interval`, `guidance_rescale`, `rescale_t`, `keep_uv`, `naf_tile_factor`. Convention
d'axes et détails : en-tête de `handler_pixal3d.py` (avant du modèle = -Y Blender = +Z glTF ;
œil = cible + d·[cos(el)·sin(az), -cos(el)·cos(az), sin(el)] ; fov horizontal ; `distance` en
unités du glb).

Attributions : [TencentARC/Pixal3D](https://huggingface.co/TencentARC/Pixal3D) (code et poids,
MIT) ; [visualbruno/ComfyUI-Trellis2](https://github.com/visualbruno/ComfyUI-Trellis2) (MIT,
dérivé de Microsoft TRELLIS.2 : `texture_mesh_pixal3d` et le dossier `trellis2/` modifié, épinglé
au commit `b0e84c0`) ; [valeoai/NAF](https://github.com/valeoai/NAF) (Apache-2.0) ;
Microsoft TRELLIS.2 (MIT, encodeur de forme). Le DINOv3 reste à accès restreint (`HF_TOKEN`).

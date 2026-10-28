# Worker RunPod : retexture TRELLIS.2 (Phasen)

Texture un maillage EXISTANT (`.glb`) à partir d'une image de référence avec le pipeline officiel
`Trellis2TexturingPipeline` de Microsoft TRELLIS.2. La géométrie et, par défaut, les UV d'entrée
sont conservées : la texture est cuite dans l'atlas fourni.

Image : `ghcr.io/phasenco/trellis2-texture-runpod` (construite par GitHub Actions à chaque push).
Base : `ghcr.io/huachuhua/trellis2-runpod` (dépendances CUDA, poids TRELLIS.2), figée par empreinte.

Endpoint RunPod : GPU ≥ 24 Go, disque 60 Go, variable `HF_TOKEN` (compte Hugging Face ayant
accepté la licence de `facebook/dinov3-vitl16-pretrain-lvd1689m`, modèle à accès restreint).

Entrée : `mesh` (base64 ou URL d'un .glb), `image` (base64, data URL ou URL), `output_transport:
"trellis-files-v1"`, `resolution` 512|1024, `texture_size` 1024|2048|4096, `seed`, `keep_uv`.
Sortie : le `.glb` texturé, par `/stream` (protocole trellis-files-v1).

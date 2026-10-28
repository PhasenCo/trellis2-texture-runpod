# Worker RunPod : retexture d'un maillage existant avec TRELLIS.2 (pipeline officiel de texturage).
# Part de l'image du worker TRELLIS.2 de huachuhua (dépendances CUDA compilées + poids de
# génération, TRELLIS.2 au commit 75fbf01 qui contient déjà Trellis2TexturingPipeline),
# figée par son empreinte, et n'ajoute que ce qui manque au texturage.
FROM ghcr.io/huachuhua/trellis2-runpod@sha256:13636bf6bc0ea015255e5965dc1e0205153de34d63b1099e8d74130cbc96e1b9

# Encodeur de forme (exclu de l'image de base, qui ne génère pas à partir d'un maillage)
# et configuration du pipeline de texturage. Poids publics (licence MIT).
RUN python -c "from huggingface_hub import snapshot_download; \
snapshot_download('microsoft/TRELLIS.2-4B', local_dir='/models/TRELLIS.2-4B', allow_patterns=[ \
'texturing_pipeline.json', 'ckpts/shape_enc_next_dc_f16c32_fp16.*'])" \
    && rm -rf /models/TRELLIS.2-4B/.cache \
    && test -f /models/TRELLIS.2-4B/texturing_pipeline.json

WORKDIR /app
COPY handler_texture.py ./
CMD ["python", "-u", "handler_texture.py"]

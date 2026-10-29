"""
Correctifs appliqués au code `trellis2/` de ComfyUI-Trellis2 pour l'image de base figée.

1. fdg_vae.py importe `tiled_flexible_dual_grid_to_mesh`, absent de l'o_voxel de l'image de
   base (TRELLIS.2 officiel). Cette variante tuilée ne sert qu'au décodage de FORME
   (`useTiled`) ; le texturage part de notre maillage et ne l'appelle jamais. On rend
   l'import facultatif.
"""

import os
import sys

racine = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PIXAL3D_SRC", "/app/pixal3d_src")
chemin = os.path.join(racine, "trellis2", "models", "sc_vaes", "fdg_vae.py")

AVANT = "from o_voxel.convert import flexible_dual_grid_to_mesh, tiled_flexible_dual_grid_to_mesh\n"
APRES = (
    "from o_voxel.convert import flexible_dual_grid_to_mesh\n"
    "try:\n"
    "    from o_voxel.convert import tiled_flexible_dual_grid_to_mesh\n"
    "except ImportError:  # o_voxel de l'image de base : pas de variante tuilée\n"
    "    tiled_flexible_dual_grid_to_mesh = None\n"
)

with open(chemin, encoding="utf-8") as f:
    code = f.read()
if APRES not in code:
    assert AVANT in code, f"import attendu introuvable dans {chemin}"
    code = code.replace(AVANT, APRES, 1)
    with open(chemin, "w", encoding="utf-8") as f:
        f.write(code)
compile(code, chemin, "exec")
print("correctif fdg_vae.py appliqué")

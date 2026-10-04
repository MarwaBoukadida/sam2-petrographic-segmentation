# -*- coding: utf-8 -*-
"""Load NPZ patches, filter borders and build LOSO splits."""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import os
from collections import defaultdict
from tqdm import tqdm
import logging
import random

logger = logging.getLogger(__name__)

@dataclass
class PatchData:
    """Patch arrays, position, slide identifier and source path."""
    patch_image: np.ndarray
    patch_gt: np.ndarray
    position: Tuple[int, int]
    image_id: str
    path: str

class PatchDataset:
    """Load sorted NPZ files containing patch_image, patch_gt, position and image_id. Border tests use per-slide position bounds. Arrays require prior registration and patch extraction."""
    def __init__(self,
                 patch_dir: str,
                 exclude_borders: bool = True,
                 border_margin: int = 0,
                 border_keep_ratio: float = 0.0,
                 seed: int = 42):
        """Load patches and border settings. border_margin uses stored position units. border_keep_ratio does not reintroduce border patches; seed initializes the stored RNG."""
        logger.info(f"Initialisation dataset depuis: {patch_dir}")
        if not os.path.exists(patch_dir):
            raise FileNotFoundError(f"Répertoire {patch_dir} introuvable")

        self.patch_dir = patch_dir
        self.exclude_borders = exclude_borders
        self.border_margin = max(0, int(border_margin))
        self.border_keep_ratio = max(0.0, min(1.0, float(border_keep_ratio)))
        self.rng = random.Random(seed)

        self.patch_paths = self._discover_patches(patch_dir)
        logger.info(f"Fichiers .npz trouvés: {len(self.patch_paths)}")
        if len(self.patch_paths) == 0:
            raise ValueError("Aucun fichier .npz trouvé dans le répertoire")

        self._validate_sample_files()
        self.metadata = self._load_metadata()
        logger.info(f"Patches valides chargés: {len(self.metadata)}")
        self.grid_info = self._compute_grid_info()
        self.valid_indices, self.border_indices = self._compute_filtered_indices()
        logger.info(
            f"Filtrage bordures (margin={self.border_margin}): "
            f"{len(self.valid_indices)} non-bordures, {len(self.border_indices)} bordures"
        )
        logger.info("Dataset initialisé avec succès")

    def _discover_patches(self, patch_dir: str) -> List[str]:
        patch_paths = []
        for root, _, files in os.walk(patch_dir):
            for file in files:
                if file.endswith('.npz'):
                    patch_paths.append(os.path.join(root, file))
        # Sort paths before building metadata indices.
        return sorted(patch_paths)

    def _validate_sample_files(self):
        test_files = self.patch_paths[:min(3, len(self.patch_paths))]
        for path in test_files:
            try:
                with np.load(path, allow_pickle=True) as data:
                    required = ['patch_image', 'patch_gt', 'position', 'image_id']
                    missing = [k for k in required if k not in data.keys()]
                    if missing:
                        logger.warning(f"Clés manquantes dans {os.path.basename(path)}: {missing}")
            except Exception as e:
                logger.error(f"Erreur lecture {os.path.basename(path)}: {e}")

    def _load_metadata(self) -> List[Dict]:
        metadata = []
        skipped = 0
        for i, path in enumerate(tqdm(self.patch_paths, desc="Chargement métadonnées")):
            try:
                with np.load(path, allow_pickle=True) as data:
                    required = ['patch_image', 'patch_gt', 'position', 'image_id']
                    if not all(k in data.keys() for k in required):
                        skipped += 1
                        continue
                    pos = tuple(data["position"])
                    image_id = str(data["image_id"]).strip()
                    metadata.append({
                        "position": pos,
                        "image_id": image_id,
                        "path": path,
                        # Discovery index; load_patch uses metadata indices.
                        "index": i
                    })
            except Exception as e:
                logger.warning(f"Patch {i} ignoré - erreur: {e}")
                skipped += 1
        if skipped > 0:
            logger.warning(f"Patches ignorés: {skipped}")
        return metadata

    def _compute_grid_info(self) -> Dict:
        patch_positions = defaultdict(list)
        for meta in self.metadata:
            y, x = meta["position"]
            img = meta["image_id"]
            patch_positions[img].append((y, x))

        grid_info = {}
        for img, coords in patch_positions.items():
            ys = [y for y, _ in coords]
            xs = [x for _, x in coords]
            min_y, max_y = min(ys), max(ys)
            min_x, max_x = min(xs), max(xs)
            height = max_y - min_y + 1
            width = max_x - min_x + 1
            coverage = len(coords) / (height * width) * 100 if height * width > 0 else 0
            grid_info[img] = {
                "min_y": min_y, "max_y": max_y,
                "min_x": min_x, "max_x": max_x,
                "height": height, "width": width,
                "n_patches": len(coords), "coverage_pct": coverage
            }
            logger.info(f"Image '{img}': grille {height}x{width}, "
                        f"{len(coords)} patches ({coverage:.1f}% couverture)")
        if len(grid_info) < 1:
            raise ValueError("Aucune image détectée dans le dataset")
        logger.info(f"LOSO: {len(grid_info)} folds possibles")
        return grid_info

    def _is_border(self, meta: Dict) -> bool:
        y, x = meta["position"]
        g = self.grid_info[meta["image_id"]]
        return (
            y <= g["min_y"] + self.border_margin or
            y >= g["max_y"] - self.border_margin or
            x <= g["min_x"] + self.border_margin or
            x >= g["max_x"] - self.border_margin
        )

    def _compute_filtered_indices(self) -> Tuple[List[int], List[int]]:
        valid, border = [], []
        for i, meta in enumerate(self.metadata):
            if self._is_border(meta):
                border.append(i)
            else:
                valid.append(i)
        return valid, border

    def load_patch(self, idx: int) -> Optional[PatchData]:
        if idx >= len(self.metadata):
            logger.warning(f"Index {idx} hors limites")
            return None
        meta = self.metadata[idx]
        try:
            with np.load(meta["path"], allow_pickle=True) as data:
                return PatchData(
                    patch_image=data["patch_image"],
                    patch_gt=data["patch_gt"],
                    position=meta["position"],
                    image_id=meta["image_id"],
                    path=meta["path"]
                )
        except Exception as e:
            logger.error(f"Échec chargement patch {idx}: {e}")
            return None

    def get_filtered_indices(self) -> List[int]:
        """Return eligible metadata indices."""
        if not self.exclude_borders:
            return list(range(len(self.metadata)))
        return self.valid_indices

    def get_border_indices(self) -> List[int]:
        return self.border_indices

    def loso_split(self, use_filtered: bool = True) -> List[Tuple[List[int], List[int]]]:
        """Hold out each sorted image_id in turn, with border filtering."""
        unique_images = sorted(set(m["image_id"] for m in self.metadata))
        folds = []
        for img in unique_images:
            train, val = [], []
            for idx, meta in enumerate(self.metadata):
                if meta["image_id"] == img:
                    val.append(idx)
                else:
                    train.append(idx)
            if use_filtered and self.exclude_borders:
                train = [i for i in train if i in self.valid_indices]
                val = [i for i in val if i in self.valid_indices]

                # Border indices below are not added to the fold lists.
                
                if self.border_keep_ratio > 0:
                    train_b = [i for i in train if i in self.border_indices]
                    val_b = [i for i in val if i in self.border_indices]
            folds.append((train, val))
            logger.info(f"Fold '{img}': {len(val)} val (filtered={use_filtered}), {len(train)} train")
        return folds

    def get_statistics(self) -> Dict:
        unique_images = sorted(set(meta["image_id"] for meta in self.metadata))
        return {
            'total_patches': len(self.metadata),
            'unique_images': len(unique_images),
            'non_border_patches': len(self.valid_indices),
            'border_patches': len(self.border_indices),
            'coverage_rate': len(self.valid_indices) / len(self.metadata) if self.metadata else 0.0,
            'images_list': unique_images,
            'loso_folds': len(unique_images)
        }

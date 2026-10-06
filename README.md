# SAM2 parameter optimization for petrographic segmentation

Code for the paper *Evaluating SAM2 Automatic Mask Generation for Petrographic Thin Sections: Effects of Parameter Tuning*.

The code tunes SAM2 automatic mask generation parameters using leave-one-slide-out
(LOSO) validation. It then compares the selected parameters (BEST/TUNED) with
SAM2 DEFAULT on separate test patches. Model weights are not trained.

## Installation

The recorded environment is Windows with Python 3.12.11, PyTorch 2.7.1+cu126
and CUDA 12.6. Package versions are in `requirements.txt`.
This file includes Windows-specific packages; other platforms need an adapted
environment.

Install [SAM2](https://github.com/facebookresearch/sam2) separately.
See its [installation guide](https://github.com/facebookresearch/sam2/blob/main/INSTALL.md)
for platform requirements. From this project folder, in a separate Python 3.12
environment with compatible NVIDIA hardware, run:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
git clone https://github.com/facebookresearch/sam2.git external/sam2
git -C external/sam2 checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4
python -m pip install -e external/sam2 --no-deps
python -m pip check
```

Download the [SAM2.1 Hiera Large checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt)
and place it at `checkpoints/sam2.1_hiera_large.pt`.
The model configuration is `configs/sam2.1/sam2.1_hiera_l.yaml`, provided by SAM2.
Paths and study settings are in `config.py`.

## Data

The study images, reference phase maps and patches are confidential and are not
included. Reproducing the paper's measurements requires the original data and
experiment environment.

To use your own data, each `.npz` file must contain:

| Key | Contents |
| --- | --- |
| `patch_image` | Optical RGB patch, with shape H × W × 3. |
| `patch_gt` | Co-registered RGB reference phase map of the same size. |
| `position` | Patch position `(y, x)`. |
| `image_id` | Source slide identifier. |

Registration and patch extraction must be done beforehand. The loader searches
folders recursively. Use `patches_data/` for development and a separate folder,
such as `patches_data_final_test/`, for test patches.

## Evaluate the selected parameters

```bash
python evaluate_final.py --patch-dir patches_data_final_test
```

The script reads `best_sam2_parameters.json` automatically and runs BEST/TUNED,
then DEFAULT, on the same eligible patches. Use `--params-file PATH` to choose
another JSON.

Tuned parameters keep `crop_n_points_downscale_factor=1` and
`stability_score_offset=0.95`. DEFAULT uses the installed SAM2 defaults,
including a stability offset of 1.0. Both runs use
`apply_postprocessing=False` and binary masks.

Results are saved in `final_test_evaluation/run_<timestamp>/`: detailed CSVs,
summary JSONs, the applied parameters and a BEST-versus-DEFAULT comparison
table and plot. Statistics are mask-level; IoU, Dice and purity are percentages,
and the weighted score is on a 0–1 scale.

The test folder selects the slides. `--images` only displays identifiers;
it does not filter patches. Border patches are excluded by default.

## Optimize on your own data

Use a separate project copy or working folder for a new study. This prevents
resuming another dataset's database or replacing its selected JSON.

```bash
python main.py
python evaluate_final.py --patch-dir patches_data_final_test
```

When the study has at least one completed trial, `main.py` automatically creates
`best_sam2_parameters.json`, containing the raw and effective parameters,
selected trial number, validation score and study summary. It replaces an
existing JSON in the working folder. The next evaluation uses the new parameters
automatically, with the two fixed tuned settings above. No new JSON is exported
if there are no completed trials.

Search ranges are in `optimizer.py`. Each call requests 30 additional trials;
the configured database is resumed if it exists. Optuna's sampler is not
explicitly seeded, so another search may produce different parameters.

Optimization results are saved in `optuna_results/`, with final reports in
`final_report/`. `analysis.py` produces exploratory optimization plots.

SAM2 source and weights are installed separately and use its
[Apache 2.0 license](https://github.com/facebookresearch/sam2/blob/main/LICENSE).


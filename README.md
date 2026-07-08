# Robot Visual Navigation V3

Streamlit application for route planning, Google Street View acquisition and
robot visual localization.

## V3 matching pipeline

The interface uses the embedded VPR backend stored in this V3 folder and
provides three matching modes:

- `MixVPR`: neural ResNet50 + MixVPR global descriptor with 4096 values.
- `Lightweight VPR`: spatial-pyramid and color descriptors.
- `Hybrid VPR`: rank fusion with 80% MixVPR and 20% lightweight VPR.

Street View references are filtered with ROSA before matching. Video replay
uses the first five references for initialization, then searches only the five
forward references around the previous localization, with a 30 metre limit.

## Historical Street View database

The sidebar can now switch the active localization database between:

- `Current Street View`: the normal images downloaded from the route.
- `Historical Street View`: the historical MixVPR test dataset copied into
  `data/streetview_historical/mixvpr_dataset_by_date/`.

When `Historical Street View` is selected, the interface displays the available
dates and the number of images for each date. The selected date becomes the
active database for both single-image localization and replay localization.

The Street View tab also includes a historical archive browser. It lets the
user select a date, inspect the images in route order, open one image with its
metadata, and quickly compare how the same route looked in the past.

Available historical dates in the included dataset:

```text
2009-05: 42 images
2009-06: 1 image
2011-08: 43 images
2014-05: 43 images
```

The app automatically builds a compatible metadata CSV for the selected date,
so MixVPR keeps GPS, heading, pano ID and candidate-rank information.

The original `Robot_Visual_Navigation_V2` project is not modified.

## macOS setup

The V3 folder is self-contained. It embeds the VPR backend and the local
MixVPR checkpoint in:

```text
Robot_Visual_Navigation_V3/
  vpr_backend/
  weights/resnet50_MixVPR_4096.ckpt
```

Activate the existing Python environment, then run:

```bash
chmod +x scripts/*.sh
./scripts/setup_mac.sh
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Results

The Replay tab displays each robot frame beside its selected Street View
image. It also writes:

```text
data/results/replay_results.csv
data/results/vpr/localization.csv
data/results/vpr/filter_manifest.csv
data/results/vpr/similarity.npy
data/results/vpr/descriptors_robot_mixvpr.npy
data/results/vpr/descriptors_streetview_mixvpr.npy
```

The `.csv` files are intended for direct inspection. The `.npy` files retain
the numerical vectors and matrices for scientific analysis.

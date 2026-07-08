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

## Dynamic historical Street View database

The historical database is generated from the current route CSV. The intended
workflow is:

1. Create a route in the interface.
2. Download the current Street View database.
3. The app writes `data/streetview/metadata.csv`.
4. Click `Download historical Street View for current CSV`.
5. The app searches older panoramas for the same GPS points and rebuilds
   `data/streetview_historical/mixvpr_dataset_by_date/`.

The sidebar can then switch the active localization database between:

- `Current Street View`: the normal images downloaded from the route.
- `Historical Street View`: the historical dataset generated from the current
  `data/streetview/metadata.csv`.

When `Historical Street View` is selected, the interface displays the available
dates and the number of images for each date. The selected date becomes the
active database for both single-image localization and replay localization.

The Street View tab also includes a historical archive browser. It lets the
user select a generated date, inspect the images in route order, open one image
with its metadata, and quickly compare how the same route looked in the past.

The app automatically builds a compatible metadata CSV for the selected date,
so MixVPR keeps GPS, heading, pano ID and candidate-rank information.

Historical image folders and metadata CSV files are generated outputs. They are
not committed to Git because each new route should rebuild them from its own
CSV.

## MixVPR vector export

The Street View tab includes `Build / export MixVPR vectors for active database`.
It creates reusable 4096-D vectors for the active image database:

```text
data/results/vpr/vector_databases/<database_name>/
  descriptors_streetview_mixvpr.npy
  descriptors_streetview_mixvpr_filtered.npy
  image_manifest.csv
  metadata.json
```

The neural descriptors are also cached in `cache/vpr/`. If the images and model
checkpoint have not changed, the app reloads the cached vectors instead of
running MixVPR again.

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

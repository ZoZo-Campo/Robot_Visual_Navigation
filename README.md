# Robot Visual Navigation V3

Streamlit application for route planning, Google Street View acquisition and
robot visual localization.

## V3 matching pipeline

The interface uses the sibling `Dev_Matching_V2` backend and provides three
matching modes:

- `MixVPR`: neural ResNet50 + MixVPR global descriptor with 4096 values.
- `Lightweight VPR`: spatial-pyramid and color descriptors.
- `Hybrid VPR`: rank fusion with 80% MixVPR and 20% lightweight VPR.

Street View references are filtered with ROSA before matching. Video replay
uses the first five references for initialization, then searches only the five
forward references around the previous localization, with a 30 metre limit.

The original `Robot_Visual_Navigation_V2` project is not modified.

## macOS setup

Keep these projects next to each other:

```text
Python/
  Dev_Matching_V2/
  Robot_Visual_Navigation_V3/
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

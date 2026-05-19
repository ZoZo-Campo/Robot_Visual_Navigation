# Robot Visual Navigation V2

Visual localization system based on Google Street View and image matching for mobile robot navigation.

---

# Overview

This project aims to localize a mobile robot using:

- a precomputed GPS route,
- Google Street View images,
- computer vision matching,
- offline replay localization from videos or images.

The system can:

1. Generate a route using OpenStreetMap,
2. Download a Street View database,
3. Match robot images against the database,
4. Estimate GPS position,
5. Replay an entire robot video offline.

---

# Main Features

## Route Planning

- Interactive map
- GPS coordinate input
- Multiple waypoints
- OpenStreetMap routing
- Adjustable GPS sampling step

---

## Street View Database Generation

- Automatic Street View download
- Route-oriented heading
- North-oriented heading
- Metadata generation
- Progress bar and live feedback

---

## Visual Localization

### ORB Matching
- Fast local feature matching
- Binary descriptors
- Real-time friendly

### CNN Matching
- Deep feature extraction
- ResNet18 backbone
- Cosine similarity

### Hybrid Matching
- ORB + CNN fusion
- Better robustness
- Weighted score fusion

---

## Replay Localization

- Video frame extraction
- Offline localization
- Estimated trajectory reconstruction
- CSV export

---

# Project Structure

```text
Robot_Visual_Navigation_V2/
│
├── assets/
│
├── core/
│   ├── frame_extractor.py
│   ├── gps_utils.py
│   ├── replay_localizer.py
│   ├── route_planner.py
│   └── streetview_client.py
│
├── matching/
│   ├── cnn_matcher.py
│   ├── hybrid_matcher.py
│   └── orb_matcher.py
│
├── replay/
│   └── replay_matcher.py
│
├── ui/
│   └── streamlit_app.py
│
├── data/
│   ├── frames/
│   ├── results/
│   ├── robot_inputs/
│   │   ├── photos/
│   │   └── videos/
│   ├── routes/
│   └── streetview/
│       ├── cache/
│       └── images/
│
├── config.py
├── requirements.txt
└── README.md
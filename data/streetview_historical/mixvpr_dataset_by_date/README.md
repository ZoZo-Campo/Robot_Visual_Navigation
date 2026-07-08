# Historical Street View Dataset for MixVPR Tests

This folder contains a small historical Google Street View dataset generated from
the GPS route metadata of `Robot_Visual_Navigation_V3`.

The goal is to create a test database for visual place recognition, especially
for MixVPR, using older Street View images aligned with the robot route.

## Source Data

Input CSV copied from the validated V3 application:

```text
data/input/route_current_streetview_metadata.csv
```

Important columns from the input:

```text
index
target_lat
target_lon
pano_id
pano_lat
pano_lon
distance_to_pano_m
heading
image_file
```

Each row corresponds to one point of the route and to one current Street View
image previously downloaded by the V3 app.

## Step 1: Historical Panorama Search

The script below searches older Street View panoramas around each GPS point:

```bash
python scripts/historical_from_csv.py --candidate-limit 3 --before 2015-01
```

Output:

```text
outputs/from_route_metadata/historical_metadata.csv
```

This file contains up to 3 historical panorama candidates for each route point.

The generated candidates include:

```text
source_index
source_lat
source_lon
source_current_pano_id
candidate_rank
historical_pano_id
historical_date
historical_lat
historical_lon
distance_to_source_m
heading
pitch
roll
search_status
```

## Step 2: Static Image Download

The full 360 panorama tile endpoint is currently blocked by Google with HTTP
403 responses. Because of that, the dataset uses the official Google Street View
Static API with a Google Maps API key.

The API key is loaded from:

```text
.env
```

Expected format:

```text
GOOGLE_MAPS_API_KEY=your_key_here
```

The key is not committed because `.env` is ignored by `.gitignore`.

The command used to download the dataset was:

```bash
python scripts/download_static_from_historical_csv.py \
  --candidate-rank all \
  --limit 0 \
  --heading-source route \
  --sleep 0.15
```

Meaning:

- `--candidate-rank all`: download all historical candidates, not only the best one.
- `--limit 0`: no limit, download every row from the historical CSV.
- `--heading-source route`: orient the downloaded image in the direction of travel.
- `--sleep 0.15`: add a short delay between API requests.

## Direction of Travel

For MixVPR, it is important that the Street View image looks approximately in
the robot's direction of movement.

The downloader therefore computes a route heading from the GPS sequence:

```text
current route point -> next route point
```

This heading is used as the `heading` parameter of the official Street View
Static API.

The output CSV records this value in:

```text
static_heading
heading_source
```

For this route, most images are oriented around heading `311`, with a few images
at the beginning and end of the sequence oriented around `124` and `127`.

## Folder Structure

Images are sorted by historical date:

```text
mixvpr_dataset_by_date/
├── 2009-05/
├── 2009-06/
├── 2011-08/
├── 2014-05/
├── downloaded_static_metadata.csv
└── README.md
```

Each image filename contains:

```text
source index
candidate rank
historical date
route heading
historical pano_id
```

Example:

```text
source_0000_rank_0_2014-05_heading_124_KF_9v-wrbuacv24KU9DGzw.jpg
```

This means:

- `source_0000`: route point 0.
- `rank_0`: closest historical candidate for that point.
- `2014-05`: historical Street View date.
- `heading_124`: image downloaded looking in the route direction.
- `KF_9v...`: Google historical panorama ID.

## Dataset Summary

Generated on: 2026-06-27

```text
Route points processed: 43
Candidates per point: 3
Total images downloaded: 129
Download errors: 0
Image format: JPG
Image size: 640 x 640
```

Date distribution:

```text
2009-05: 42 images
2009-06: 1 image
2011-08: 43 images
2014-05: 43 images
```

Candidate rank distribution:

```text
rank 0: 43 images
rank 1: 43 images
rank 2: 43 images
```

## Metadata File

The main metadata file for this downloaded dataset is:

```text
downloaded_static_metadata.csv
```

It links every image to:

- the original route point;
- the current V3 Street View panorama;
- the historical panorama;
- the historical date;
- the distance from the GPS route point;
- the image path;
- the heading used for download;
- the download status.

This CSV should be used as the index file for MixVPR experiments.

## Notes

This dataset is intended for testing visual place recognition robustness across
time. It is not a perfect reconstruction of full historical panoramas because
the direct tile-based panorama endpoint is blocked. However, it is still useful
for testing because the images are:

- historical;
- geolocated around the same route;
- linked to original GPS points;
- oriented in the direction of travel;
- documented with metadata.

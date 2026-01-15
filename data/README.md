# Data Overview

## Project mission
This project groups California Academy of Sciences botany specimens into
inferred field expeditions. It identifies specimens collected close together in
time and geography and treats each group as a likely expedition. The goal is to
make collecting activity easier to explore, summarize, and communicate.

## Primary dataset (Git LFS)
`data/clustered_expeditions_redacted.csv` is the recommended starting point. It
is a shareable snapshot of clustered results with sensitive records removed.

What the file contains:
- One row per specimen record.
- Collecting event, locality, and geography context to support exploration.
- Cluster identifiers, especially `spatiotemporal_cluster_id`, which represents
  the inferred expedition.
- Records flagged for sensitive locality have been removed.

How it was produced (high level):
1) Extract specimen records from the source database.
2) Normalize dates and location fields.
3) Group records that are close in time and space.
4) Remove records that should not be shared publicly.

## Column guide
IDs (useful for linking to source tables):
- collectingeventid: collecting event (field trip) record ID.
- collectionobjectid: specimen record ID.
- localityid: locality record ID.
- geographyid: geography record ID.

Dates (when the collecting event occurred):
- startdate: start date for the event.
- enddate: end date for the event.

Location text (human-readable context):
- localityname: locality description.
- namedplace: nearby place name.
- commonname / fullname / name: geography names at different levels.

Coordinates (where the event happened):
- latitude1 / longitude1: best-known coordinates.
- centroidlat / centroidlon: fallback "center of area" coordinates when precise
  ones are not available.
Note: coordinates may be missing for some rows and should be treated as
approximate.

Elevation:
- minelevation / maxelevation: elevation range in meters.
- elevationaccuracy: confidence/precision of elevation data.

Notes:
- remarks: collecting event notes.
- text1: additional specimen notes.

Cluster results:
- spatial_cluster_id: groups records that are close together in space.
- temporal_cluster_id: groups records that are close together in time.
- spatiotemporal_cluster_id: final expedition ID (use this for grouping).

## Suggested uses
- Map and compare expeditions across time and regions.
- Count specimens per expedition to find large or dense trips.
- Build timelines of collecting activity.
- Create dashboards or narratives for reporting and storytelling.

## Limitations and cautions
- This is a derived dataset, not the full source database.
- Clusters are best-effort inferences, not authoritative trip boundaries.
- Sensitive localities have been removed; do not attempt to reconstruct them.

## Git LFS download
This file is stored with Git LFS. If it looks tiny or like a pointer, run:

```bash
git lfs pull
```

Then open `data/clustered_expeditions_redacted.csv`.

## Other files in this folder
The `data/` directory also contains raw database dumps and other outputs. They
are large and not needed for most users. The redacted CSV above is the
intended starting point.

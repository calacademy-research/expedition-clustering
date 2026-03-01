# Expedition Clustering Expansion Plan

## Overview

This document describes a multi-phase approach to improve expedition clustering by:
1. Splitting over-merged expeditions at internal gaps
2. Iteratively expanding truncated expeditions
3. Using collector coherence to weight merge decisions
4. Tuning parameters per-collection through automated exploration

The goal is to produce more accurate expedition groupings that reflect real-world collecting trips.

---

## Problem Statement

### Current Limitations

1. **14-day window truncation**: Expeditions hitting the 14-day limit may actually continue, but the algorithm can't see past the boundary.

2. **Over-merging**: The current algorithm may group specimens with large internal gaps (e.g., days 1-3 and days 8-10) into one expedition when they're actually separate trips.

3. **Collector signal inconsistency**: For some collections (ich), collector is a strong grouping signal. For others (iz), multiple collectors naturally work the same sites, making collector less reliable.

4. **No iterative refinement**: Current approach is single-pass with no convergence checking.

### Key Insight

The **maximum internal gap** (e.g., 3 days) should be the primary temporal constraint, not the 14-day window. Real expeditions have continuous or near-continuous collecting; gaps > 3 days usually indicate separate trips.

---

## Algorithm Design

### Phase 0: Preprocessing

Before clustering, extract signals that inform later phases:

#### 0.1 Vessel Detection
- Parse collector strings for vessel patterns: `R/V`, `M/V`, `RV`, `SS`, `HMS`
- Example: "B. Fisher, R/V Dorado" → vessel = "Dorado"
- Vessel-based expeditions may have longer acceptable gaps (transit between stations)

#### 0.2 Collector Team Detection
- Analyze historical co-occurrence of collectors
- Build graph of collectors who frequently appear together
- Example: "Fisher" + "Esteves" appear together in 80% of their records → they're a team
- Team membership informs merge decisions when collectors don't exactly match

#### 0.3 Station/Transect Pattern Detection
- Look for systematic locality naming: "Station 1", "Station 2", "Transect A"
- Sequential patterns suggest planned sampling → keep together despite minor gaps

---

### Phase 1: Initial Clustering (Existing)

Run existing grid-based spatiotemporal clustering:
- Spatial grid: collection-specific (10-15km typical)
- Temporal window: 14 days
- Year as hard boundary (never merge across years)
- Optional collector-aware partitioning

Output: `initial_expeditions` with `spatiotemporal_cluster_id`

---

### Phase 2: Gap-Based Splitting

For each expedition from Phase 1, analyze internal temporal gaps and split where gaps exceed threshold.

#### 2.1 Gap Detection

```
For each expedition:
  Sort specimens by collection date
  For each consecutive pair of specimens:
    gap = date[i+1] - date[i]
    if gap > max_gap_days:
      Mark as split point
```

#### 2.2 Density-Aware Gap Detection (Enhancement)

Not all gaps are equal. A day with 1-2 specimens between days with 30+ specimens is likely rest/travel, not a true expedition boundary.

```
Consider:
  Day 1: 45 specimens
  Day 2: 2 specimens   ← low density, possibly travel
  Day 3: 0 specimens   ← gap starts
  Day 4: 0 specimens
  Day 5: 38 specimens

Density-aware: This is ONE expedition with rest days, not two expeditions.
Simple gap detection: Would split at day 2-3 boundary.
```

Implementation: Weight gaps by surrounding specimen density. High-density days on both sides of a gap make the gap more significant.

#### 2.3 Vessel Exception

If expedition has detected vessel:
- Allow longer gaps (5-7 days) for transit between stations
- Ship-based expeditions have different temporal patterns than land-based

#### 2.4 Fragment Tracking

After splitting:
- Track `original_expedition_id` and `split_index` for each fragment
- Fragments with < min_specimens are candidates for re-merging (Phase 4)
- Don't discard fragments yet; they may merge with something later

---

### Phase 3: Collector Coherence Measurement

Before deciding how to weight collector signal in merge decisions, measure per-collection coherence.

#### 3.1 Coherence Score

```
coherence = (expeditions with single collector) / (total expeditions)

High (> 0.7): Collector is reliable signal (ich, orn)
Medium (0.4-0.7): Mixed patterns
Low (< 0.4): Multi-collector common (iz dive sites)
```

#### 3.2 Team Coherence

For collections with identified collector teams:
```
team_coherence = (expeditions matching team patterns) / (total expeditions)
```

#### 3.3 Coherence Table (To Be Tuned)

| Collection | Expected Coherence | Notes |
|------------|-------------------|-------|
| ich | High | Single collector per expedition typical |
| orn | High | Usually single collector or known team |
| iz | Low | Collaborative dive sites, multi-collector normal |
| botany | Medium | Varies by expedition type |
| antweb | High | Usually single collector (Fisher et al.) |
| mam | Medium | Trap lines may have multiple checkers |
| herp | Medium | Field teams vary |

---

### Phase 4: Iterative Expansion

Merge adjacent expeditions/fragments that likely belong together.

#### 4.1 Candidate Identification

Two expeditions A and B are merge candidates if:
- Same collection
- B starts within `max_merge_gap_days` of A ending
- Spatial distance between centroids < `max_merge_distance_km`
- Same year (hard constraint)

#### 4.2 Merge Scoring

```
merge_score = weighted combination of:
  - temporal_proximity: closer gap = higher score
  - spatial_proximity: closer centroids = higher score
  - collector_overlap: shared collectors = higher score (weighted by coherence)
  - vessel_match: same vessel = strong signal
  - team_match: collectors are known team members
  - trajectory_coherence: spatial movement is smooth/logical
```

Weighting adjusted by collection's collector coherence:
- High coherence → collector overlap weighted heavily
- Low coherence → spatial/temporal weighted more heavily

#### 4.3 Merge Threshold

```
if merge_score > MERGE_THRESHOLD:
  merge(A, B)
  record merge in history
```

Threshold is tunable, default 0.6.

#### 4.4 Iteration Until Convergence

```
iteration = 0
while True:
  iteration += 1
  merges_this_round = 0

  for each candidate pair (A, B):
    if should_merge(A, B):
      merge(A, B)
      merges_this_round += 1

  log(f"Iteration {iteration}: {merges_this_round} merges")

  if merges_this_round == 0:
    break  # Converged
```

Typical convergence: 2-5 iterations.

#### 4.5 Post-Merge Validation

After each merge, verify the result doesn't violate constraints:
- No internal gaps > max_gap_days
- If violation found, reject the merge

---

### Phase 5: Fragment Cleanup

After expansion converges:

#### 5.1 Keep Merged Fragments
- Fragments that successfully merged into larger expeditions are kept
- Track `merged_from` list for provenance

#### 5.2 Discard Orphan Fragments
- Fragments that didn't merge AND have < min_specimens are discarded
- These represent isolated specimens that don't form coherent expeditions

#### 5.3 Re-evaluate Small Expeditions
- Expeditions with min_specimens <= count < 2*min_specimens get reviewed
- If they're isolated (no nearby expeditions), keep them
- If they're adjacent to larger expeditions but didn't meet merge threshold, flag for manual review

---

### Phase 6: Confidence Scoring

Assign confidence level to each final expedition.

#### 6.1 Confidence Factors

```
Positive signals (increase confidence):
  - Single collector
  - Short duration (< 7 days)
  - Tight spatial cluster
  - No merges required
  - Vessel detected
  - Station pattern detected

Negative signals (decrease confidence):
  - Multiple collectors (in high-coherence collection)
  - Long duration (> 10 days)
  - Large spatial spread
  - Multiple merges
  - Internal gaps near threshold
  - Fragment that barely merged
```

#### 6.2 Confidence Levels

```
HIGH:   score > 0.7  - Strong evidence this is a real expedition
MEDIUM: score 0.4-0.7 - Reasonable grouping but some ambiguity
LOW:    score < 0.4  - Uncertain, may be over-merged or under-merged
```

#### 6.3 Output

Add to expedition records:
- `confidence`: HIGH/MEDIUM/LOW
- `confidence_score`: 0.0-1.0
- `confidence_notes`: list of factors affecting score

---

### Phase 7: Parameter Tuning

Automated exploration to find optimal parameters per collection.

#### 7.1 Parameter Space

```
max_gap_days: [2, 3, 4, 5, 7]
max_merge_gap_days: [2, 3, 4, 5]
max_merge_distance_km: [10, 15, 20, 25, 30]
merge_threshold: [0.5, 0.55, 0.6, 0.65, 0.7]
density_aware_gaps: [True, False]
```

#### 7.2 Evaluation Metrics

```
Primary metrics:
  - collector_coherence: % expeditions with single primary collector
  - coverage: % specimens assigned to valid expeditions
  - avg_expedition_size: mean specimens per expedition
  - size_distribution: std dev of expedition sizes (lower = more uniform)

Secondary metrics:
  - avg_duration: mean days per expedition
  - max_duration: longest expedition (flag if > 30 days)
  - orphan_rate: % specimens that ended up in discarded fragments
  - merge_rate: avg merges per expedition
```

#### 7.3 Tuning Process

```
For each collection:
  best_score = 0
  best_params = None

  For each parameter combination:
    Run full pipeline (phases 1-6)
    Calculate evaluation metrics
    composite_score = weighted_combination(metrics)

    if composite_score > best_score:
      best_score = composite_score
      best_params = current_params

  Save best_params to collection config
  Generate tuning report with comparisons
```

#### 7.4 Ground Truth Validation (If Available)

If historical expedition records exist:
- Compare clustered expeditions to known expeditions
- Calculate precision/recall
- Use as primary optimization target

---

## Configuration

### Per-Collection Defaults (Starting Points)

| Collection | max_gap_days | merge_gap | distance_km | Notes |
|------------|--------------|-----------|-------------|-------|
| ich | 3 | 3 | 15 | Fish expeditions usually continuous |
| orn | 5 | 5 | 100 | Island hopping, transit gaps |
| iz | 2 | 2 | 10 | Dive sites close; gaps = different trips |
| botany | 3 | 3 | 10 | Standard field work |
| antweb | 3 | 3 | 10 | Standard field work |
| mam | 4 | 4 | 15 | Trap checking may have gaps |
| herp | 3 | 3 | 10 | Standard field work |
| ent | 3 | 3 | 10 | Standard field work |
| geo | 4 | 4 | 20 | Geological sites may be spread out |

These will be refined through tuning.

### Global Defaults

```
MIN_SPECIMENS_PER_EXPEDITION = 20
MERGE_SCORE_THRESHOLD = 0.6
MAX_ITERATIONS = 10
YEAR_HARD_BOUNDARY = True
```

---

## Output Format

### Extended clustered_expeditions.csv

Add columns to existing format:

```
# Existing columns preserved
spatiotemporal_cluster_id, spid, latitude1, longitude1, ...

# New columns
expedition_id_v2          # Final expedition ID after expansion
original_cluster_id       # Pre-expansion cluster ID
was_split                 # True if this specimen's original cluster was split
was_merged                # True if this expedition resulted from merge
merge_sources             # Comma-separated list of merged cluster IDs
merge_iteration           # Which iteration caused final merge (0 = no merge)
confidence                # HIGH/MEDIUM/LOW
confidence_score          # 0.0-1.0
```

### Tuning Report

For each collection, generate:

```
=== ICH Tuning Report ===

Best Parameters:
  max_gap_days: 3
  max_merge_gap_days: 3
  max_merge_distance_km: 15
  merge_threshold: 0.6

Metrics (before → after expansion):
  Expeditions: 1,247 → 1,089 (-12.7%)
  Avg size: 67.3 → 77.2 specimens
  Collector coherence: 82% → 85%
  Coverage: 84,000 specimens (100% retained)

Confidence Distribution:
  HIGH: 743 (68%)
  MEDIUM: 289 (27%)
  LOW: 57 (5%)

Iteration Summary:
  Phase 2 (split): 312 expeditions split into 498 fragments
  Phase 4 iteration 1: 187 merges
  Phase 4 iteration 2: 23 merges
  Phase 4 iteration 3: 2 merges
  Phase 4 iteration 4: 0 merges (converged)
  Phase 5 (cleanup): 89 orphan fragments discarded

Largest Expeditions:
  1. Expedition 4521: 1,247 specimens, Dec 1931, Galapagos, R/V Dorado
  2. Expedition 892: 834 specimens, Aug 1965, Gulf of California, Smith
  ...
```

---

## Implementation Order

### Stage 1: Core Algorithm
1. Gap-based splitting with configurable threshold
2. Basic merge scoring (temporal + spatial + collector overlap)
3. Iterative expansion until convergence
4. Fragment cleanup

### Stage 2: Tuning Framework
1. Evaluation metrics calculation
2. Parameter grid search
3. Per-collection configuration storage
4. Tuning report generation

### Stage 3: Signal Enhancements
1. Vessel detection and special handling
2. Density-aware gap detection
3. Collector team detection

### Stage 4: Quality Features
1. Confidence scoring
2. Trajectory analysis (optional)
3. Station/transect pattern detection (optional)

### Stage 5: Validation
1. Manual review of largest/longest expeditions
2. Comparison with any known expedition records
3. Edge case documentation

---

## Success Criteria

1. **No internal gaps > configured threshold** in any expedition
2. **Convergence** achieved in < 10 iterations for all collections
3. **Collector coherence** maintained or improved vs. current clustering
4. **Coverage** maintained (no significant increase in orphaned specimens)
5. **Confidence distribution** skews toward HIGH (> 60% HIGH for most collections)
6. **Manual review** of top 20 largest expeditions per collection passes sanity check

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Over-splitting creates too many fragments | Tune max_gap_days up; use density-aware gaps |
| Over-merging combines unrelated trips | Tune merge_threshold up; enforce gap constraint post-merge |
| Collector coherence drops after expansion | Weight collector signal higher in merge score |
| Long runtime for large collections (botany) | Process in batches; parallelize parameter search |
| Edge cases: very long expeditions (surveys) | Flag expeditions > 30 days for manual review |
| Year boundary edge cases (Dec 31 → Jan 1) | Year is hard boundary; adjacent-year trips are separate |

---

## Timeline Estimate

- Stage 1 (Core): 2-3 days
- Stage 2 (Tuning): 2 days
- Stage 3 (Enhancements): 2-3 days
- Stage 4 (Quality): 1-2 days
- Stage 5 (Validation): 1-2 days
- Full tuning runs: 1 day (compute time)

Total: ~10-12 days for complete implementation and tuning

---

## Appendix: Merge Score Formula

```
merge_score(A, B) =
    w_temporal * temporal_score(A, B) +
    w_spatial * spatial_score(A, B) +
    w_collector * collector_coherence * collector_overlap(A, B) +
    w_vessel * vessel_match(A, B) +
    w_team * team_match(A, B) +
    w_fallback * (1 - collector_coherence) * (temporal_score + spatial_score) / 2

Where:
  temporal_score = max(0, 1 - gap_days / max_merge_gap_days)
  spatial_score = max(0, 1 - distance_km / max_merge_distance_km)
  collector_overlap = |collectors_A ∩ collectors_B| / |collectors_A ∪ collectors_B|
  vessel_match = 1.0 if same vessel, 0.0 otherwise
  team_match = max team co-occurrence score between any collector pair

Default weights:
  w_temporal = 0.25
  w_spatial = 0.25
  w_collector = 0.30
  w_vessel = 0.15
  w_team = 0.10
  w_fallback = 0.20 (only applies when collector_coherence < 0.5)
```

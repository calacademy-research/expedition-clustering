# Alternative Approaches to Expedition Clustering

Research into off-the-shelf tools, neural networks, and other methods that could enhance or replace our current DBSCAN + merge pipeline.

## Tier 1: High impact, low effort (days)

### 1. HDBSCAN instead of DBSCAN

Eliminates the biggest pain point — per-collection epsilon tuning. HDBSCAN finds clusters at varying density levels with just one intuitive parameter (`min_cluster_size`). Already in scikit-learn 1.3+ (`sklearn.cluster.HDBSCAN`). Would replace both `SpatialDBSCAN` and potentially the iterative reconnection stages. Also gives cluster membership probabilities (soft clustering) for flagging borderline specimens.

### 2. Embed collector names with sentence-transformers

Replace normalized string matching with dense vector similarity. `all-MiniLM-L6-v2` (384 dims) would handle "R.H. Beck" / "Rollo Howard Beck" / "Beck, R." as naturally similar without hand-coded rules. Drop-in replacement for the Jaccard collector overlap in merge scoring. `pip install sentence-transformers`.

### 3. Examine ParseGBIF's approach

[ParseGBIF](https://github.com/pablopains/parseGBIF) (Scientific Reports 2024) solves the same problem — parsing GBIF occurrences into unique collection events. Uses a collector dictionary + spatial ranking. Their approach increased useful records by 180% for Myrtaceae.

## Tier 2: Potentially transformative, medium effort (1-2 weeks)

### 4. Splink (probabilistic record linkage)

The expedition clustering problem — "do these specimens belong to the same expedition?" — IS record linkage. [Splink](https://github.com/moj-analytical-services/splink) has built-in `DistanceInKMLevel` (haversine), `DateDiffLevel`, and string similarity comparisons. Crucially, it **learns the optimal weights via EM** instead of hand-tuned `w_temporal=0.25, w_spatial=0.25, w_collector=0.30`. Could replace the entire merge stage, and the per-collection parameter dicts, with learned weights. Handles millions of records on a laptop.

### 5. Graph-based community detection (Louvain/Leiden)

Build a specimen similarity graph, detect communities. The merge stage already computes pairwise scores between expedition summaries — a graph approach would do this globally rather than greedily. NetworkX or igraph provide Louvain/Leiden. Advantage: finds globally optimal groupings instead of greedy merge order dependence.

### 6. Contrastive learning on existing output

Use current DBSCAN+merge labels as pseudo-ground-truth. Train a small encoder with supervised contrastive loss to learn a metric space where same-expedition specimens cluster tightly. Then re-cluster with HDBSCAN on the learned embeddings. This learns the relative importance of spatial vs temporal vs collector features end-to-end, replacing per-collection hand-tuning. Libraries: `pytorch-metric-learning`, `sentence-transformers`.

## Tier 3: Research-grade, high effort (weeks+)

### 7. GNN-based clustering

Build k-NN graph in spatiotemporal space, use Graph Attention Networks to learn expedition-aware embeddings. Theoretically elegant but needs labeled data and significant engineering. PyTorch Geometric / DGL.

### 8. N2D pipeline

Autoencoder on multimodal features -> UMAP -> HDBSCAN. Learns a nonlinear feature representation. `pip install n2d`.

### 9. Constrained clustering with active learning

Must-link/cannot-link constraints from domain rules (same collector + same day + same location = must-link). The [active-semi-supervised-clustering](https://github.com/datamole-ai/active-semi-supervised-clustering) package supports this.

## Not recommended

- **DEC/IDEC/VaDE** — require pre-specifying k (number of clusters), which we don't know
- **Full transformers from scratch** — 200k records with 4 features is too low-dimensional; massively overparameterized
- **ST-DBSCAN package** — our current pipeline is already more sophisticated (haversine support, collector partitioning)
- **MovingPandas/scikit-mobility** — trajectory-focused, doesn't fit discrete specimen events well

## Assessment

For our data (200k records, low-dimensional structured features), **HDBSCAN + better feature engineering will likely outperform any neural approach**. Deep learning shines on high-dimensional unstructured data, which isn't this case. The biggest bang-for-buck is probably **HDBSCAN** (eliminates epsilon tuning) + **Splink** (learns merge weights from data distribution). The contrastive learning path is the most promising neural direction.

## Key references

- [HDBSCAN docs](https://hdbscan.readthedocs.io/en/latest/)
- [Splink](https://github.com/moj-analytical-services/splink) — probabilistic record linkage
- [ParseGBIF](https://github.com/pablopains/parseGBIF) — GBIF occurrence parsing into collection events
- [GBIF clustering](https://github.com/gbif/clustering) — hash-based blocking + assertion comparison
- [Python Record Linkage Toolkit](https://github.com/J535D165/recordlinkage)
- [Dedupe](https://github.com/dedupeio/dedupe) — active learning entity resolution
- [N2D: Not Too Deep Clustering](https://github.com/rymc/n2d)
- [pytorch-metric-learning](https://github.com/KevinMusgrave/pytorch-metric-learning) — contrastive losses
- [sentence-transformers](https://github.com/huggingface/sentence-transformers) — collector name embeddings
- [Bionomia](https://bionomia.net/) — specimen-to-collector attribution via ORCID/Wikidata
- [Collector name disambiguation (Biodiversity Data Journal 2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9836581/)

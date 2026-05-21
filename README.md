# K-Means Clustering — California Housing Segments

> Unsupervised segmentation of California census tracts using K-Means (k=6) on median income, latitude, and longitude — producing geographically coherent clusters that a supervised Decision Tree can reproduce with 100% fidelity, validating cluster consistency.

---

## Problem

Group California housing districts into meaningful segments based on location and income — without any predefined labels. Real estate analysts, urban planners, and lenders use geographic–income segmentation to understand regional housing markets, identify underserved areas, and set risk-adjusted pricing. This is an unsupervised learning problem: there is no "correct" answer, only meaningful vs. meaningless groupings.

## Dataset

- **Source:** California Housing dataset (scikit-learn / GitHub)
- **Size:** 20,640 census tract records
- **Full features available:** MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude, MedHouseVal
- **Features used:** Only 3 — `MedInc`, `Latitude`, `Longitude`

The 3-feature restriction is intentional: location (lat/lon) captures regional housing market dynamics, and income captures socioeconomic tier. Together they define segments that are both geographically interpretable and economically meaningful.

## Pipeline

| Step | Action |
|---|---|
| Feature selection | `MedInc`, `Latitude`, `Longitude` (3 features, per project spec) |
| Train/test split | 80/20 (16,512 train / 4,128 test) |
| K-Means | `KMeans(n_clusters=6, n_init="auto", random_state=42)` |
| Cluster assignment | Labels from training fit applied to both sets |
| Visualisation | 3 scatter plots per set: Lat vs Lon, Lat vs Income, Lon vs Income |
| Validation | Decision Tree trained on cluster labels → 100% accuracy on test set |

## Results

**6 clusters identified** — each corresponding to a geographically and economically distinct California region:

The scatter plots (Latitude vs Longitude coloured by cluster) reveal coherent geographic segments: coastal high-income areas, inland valleys, Southern California metro regions, and rural/agricultural zones — all emerging from the data without any geographic label being provided.

**Validation:** A `DecisionTreeClassifier` trained to predict K-Means cluster labels from the same 3 features achieves **100% accuracy** on the test set. This confirms the clusters have clean, consistent boundaries — they're not noise, they're structure.

## Key Takeaways

- **Unsupervised ≠ no validation:** Without a ground-truth label, cluster quality is assessed differently — do the segments make visual/geographic sense? Can a supervised model reproduce them perfectly? Both checks pass here.
- **Feature choice shapes the clusters entirely:** Using all 9 features would produce clusters driven by housing age, room counts, and population density. Restricting to income + location produces segments that answer the practical question: *where are high-income vs. low-income areas, and how do they cluster geographically?*
- **K=6 is a modelling decision, not a discovery:** The number of clusters is chosen, not found. An elbow plot or silhouette analysis would give a principled way to select k rather than specifying it upfront.

## Tech Stack

`Python` · `scikit-learn` · `pandas` · `Matplotlib` · `Seaborn`

## Run It Locally

```bash
git clone https://github.com/matthewkane-ml/ML_KMeans_MTK.git
cd ML_KMeans_MTK
pip install -r requirements.txt
jupyter notebook src/KMEANS.ipynb
```

Both the K-Means model and the validation Decision Tree are saved to `models/` via `pickle`.

## What I'd Do Next

- Build an **elbow plot** (inertia vs k) and a **silhouette score plot** to find the optimal number of clusters empirically instead of fixing k=6
- Add `HouseAge` or `MedHouseVal` as a 4th feature and compare how the cluster structure changes
- Visualise clusters on an actual map using **Folium** or **Plotly** with California county boundaries for a more interpretable presentation

---

**Author:** Matthew Kane — [LinkedIn](https://www.linkedin.com/in/thomas-k-392094410/) · [GitHub portfolio](https://github.com/matthewkane-ml)

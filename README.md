# GraDis
**GraDis** is an **ArcPy**-based toolkit for automated graben morphometric analysis using DEMs and mapped grabens.  
Developed by **Aditya Ray** & **Nilanjan Dasgupta**, Department of Geology, Presidency University, 86/1 College Street, Kolkata, India.
# GraDis: Graben Morphometric Analysis Pipeline

---

## Requirements

* ArcGIS Pro with `arcpy` and Spatial Analyst extension.
* Python libraries: `pandas`, `numpy`, `matplotlib`, and `scipy`.

---

## Core Features

* **Automated Filtering**: Removes features below minimum length and excludes high-curvature zones or intersections.
* **Transect Generation**: Creates perpendicular sampling lines at defined intervals along graben traces.
* **Signal Processing**: Uses Savitzky-Golay filters to smooth topographic noise for reliable feature detection.
* **Morphometric Analysis**: Identifies floors and shoulders to calculate displacement metrics.
* **Outlier Control**: Implements 2-sigma filtering for final $L/D$ summary statistics.

---

## User Parameters

| Parameter | Description |
| :--- | :--- |
| `transect_spacing` | Distance between profiles along the graben. |
| `perp_line_length` | Total length of the perpendicular transect. |
| `sample_spacing` | Distance between elevation sample points. |
| `angle_threshold` | Limit for excluding sharp bends (degrees). |
| `asymmetry_threshold` | Threshold for validating graben shoulder symmetry. |

---

## Workflow

1. **Initialization**: Configures file geodatabase and output directories.
2. **Preprocessing**: Generates intersection masks and segments polylines.
3. **Sampling**: Extracts DEM values to points and exports to CSV profiles.
4. **Analysis**: Detects local extrema and calculates:
   * **Depth ($D$)**: The minimum vertical distance to the floor from either shoulder.
   * **Width ($L$)**: Determined via iso-elevation crossing from the shallower shoulder.
5. **Visualization**: Saves raw and marked plots for every profile.

---

## Outputs

* **CSV Profiles**: Raw elevation data grouped by individual graben.
* **Profile Plots**: PNG files visualizing floor/shoulder picks.
* **LD Summary**: Master CSV with averaged morphometric metrics per graben ID.

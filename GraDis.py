import os
import math
import arcpy
from arcpy import env
from arcpy.sa import KernelDensity, PointDensity
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema, savgol_filter

# USER PARAMETERS
input_grabens = r"D:/Master/Lunar LD/Data/graben.shp"
dem_path = r"D:/Master/Lunar LD/Data/DEM.tif"
output_dir = r"D:/Master/Lunar LD/Results/DAT"
gdb_name = "graben.gdb"

transect_spacing = 2000  # Distance between transect center-points along graben (m)
sample_spacing = 60      # Distance between sample points along each transect (m)
perp_line_length = 3000  # Length of perpendicular transects (m)
point_vicinity = 1800    # High-curvature exclusion radius (m)
angle_threshold = 120    # Angle threshold for sharp bends (degrees)
min_graben_length = 5000 # Minimum graben length for inclusion (m)
segment_split_interval = None

use_intersection_exclusion = True
intersection_buffer_distance = 5000

savgol_window = 5
savgol_poly = 2
plots_per_profile = True
save_plots = True
asymmetry_threshold = 0.3

# SETUP
def setup_output_folders(output_dir, gdb_name):
    """Initialize output directory structure and geodatabase."""
    os.makedirs(output_dir, exist_ok=True)
    
    gdb_path = os.path.join(output_dir, gdb_name)
    if not arcpy.Exists(gdb_path):
        print(f"Creating file geodatabase at {gdb_path}")
        arcpy.CreateFileGDB_management(output_dir, os.path.splitext(gdb_name)[0])
    
    csv_dir = os.path.join(output_dir, "CSV_Profiles")
    plots_dir = os.path.join(output_dir, "Profile_Plots")
    summary_dir = os.path.join(output_dir, "LD_Summary")
    
    for folder in [csv_dir, plots_dir, summary_dir]:
        os.makedirs(folder, exist_ok=True)
    
    return gdb_path, csv_dir, plots_dir, summary_dir

# SEGMENTATION AND TRANSECTS
def segment_and_make_transects(input_fc, gdb_path, transect_spacing, perp_line_length,
                                point_vicinity, angle_threshold, min_graben_length, segment_split_interval=None,
                                mask_poly_fc=None):
    """
    Segment graben polylines, generate transect center-points and perpendicular transects.
    
    Filtering steps:
    1. Minimum length filter
    2. High-curvature exclusion (angle-based and near-analysis)
    3. Intersection buffer masking
    """
    arcpy.env.workspace = gdb_path
    arcpy.env.overwriteOutput = True
    
    spatial_ref = arcpy.Describe(input_fc).spatialReference
    
    segments_fc = os.path.join(gdb_path, "Graben_Segments")
    points_fc = os.path.join(gdb_path, "Graben_Points")
    transects_fc = os.path.join(gdb_path, "Graben_Transects")
    graben_filtered_fc = os.path.join(gdb_path, "Graben_Filtered")
    
    for fc in [segments_fc, points_fc, transects_fc, graben_filtered_fc]:
        if arcpy.Exists(fc):
            arcpy.Delete_management(fc)
    
    # Filter by minimum length
    print(f"Filtering grabens by minimum length: {min_graben_length} m")
    arcpy.management.CopyFeatures(input_fc, graben_filtered_fc)
    
    try:
        arcpy.AddField_management(graben_filtered_fc, "G_LENGTH", "DOUBLE")
    except Exception:
        pass
    
    with arcpy.da.UpdateCursor(graben_filtered_fc, ["SHAPE@", "G_LENGTH"]) as cur:
        for row in cur:
            row[1] = row[0].length
            cur.updateRow(row)
    
    lyr = "graben_layer"
    if arcpy.Exists(lyr):
        arcpy.Delete_management(lyr)
    
    arcpy.MakeFeatureLayer_management(graben_filtered_fc, lyr)
    arcpy.SelectLayerByAttribute_management(lyr, "NEW_SELECTION", 
                                           f'"G_LENGTH" >= {min_graben_length}')
    
    if segment_split_interval and segment_split_interval > 0:
        seg_src = os.path.join(gdb_path, "graben_split_temp")
        if arcpy.Exists(seg_src):
            arcpy.Delete_management(seg_src)
        print(f"Splitting lines every {segment_split_interval} m")
        arcpy.management.SplitLineAtLength(lyr, seg_src, str(segment_split_interval))
        source_fc = seg_src
    else:
        source_fc = lyr
    
    # Create feature classes with schema
    print("Creating segment, point, and transect feature classes...")
    
    arcpy.CreateFeatureclass_management(gdb_path, "Graben_Segments", "POLYLINE", 
                                       spatial_reference=spatial_ref)
    for field, dtype in [("GrabenOID", "LONG"), ("SegLength", "DOUBLE"), ("Bearing", "DOUBLE")]:
        arcpy.AddField_management(segments_fc, field, dtype)
    
    arcpy.CreateFeatureclass_management(gdb_path, "Graben_Points", "POINT", 
                                       spatial_reference=spatial_ref)
    for field, dtype in [("GrabenOID", "LONG"), ("SegmentOID", "LONG"), 
                         ("Bearing", "DOUBLE"), ("Chainage", "DOUBLE")]:
        arcpy.AddField_management(points_fc, field, dtype)
    
    arcpy.CreateFeatureclass_management(gdb_path, "Graben_Transects", "POLYLINE", 
                                       spatial_reference=spatial_ref)
    for field, dtype in [("GrabenOID", "LONG"), ("SegmentOID", "LONG"), 
                         ("Bearing", "DOUBLE"), ("Chainage", "DOUBLE")]:
        arcpy.AddField_management(transects_fc, field, dtype)
    
    # Generate segments from graben polylines
    print("Generating segments and identifying sharp bends...")
    seg_insert = arcpy.da.InsertCursor(segments_fc, ["SHAPE@", "GrabenOID", "SegLength", "Bearing"])
    seg_counter = 0
    exclusion_vertices = []
    
    with arcpy.da.SearchCursor(lyr, ["OID@", "SHAPE@"]) as sc:
        for oid, shape in sc:
            if shape is None or shape.length < min_graben_length:
                continue
            
            for part in shape:
                pts = [p for p in part if p]
                segment_data = []
                for i in range(len(pts) - 1):
                    p1, p2 = pts[i], pts[i + 1]
                    seg = arcpy.Polyline(arcpy.Array([p1, p2]), spatial_ref)
                    dx, dy = p2.X - p1.X, p2.Y - p1.Y
                    bearing = (math.degrees(math.atan2(dx, dy)) + 360) % 360
                    seg_insert.insertRow([seg, oid, seg.length, bearing])
                    seg_counter += 1
                    segment_data.append((bearing, p2))
                
                # Check smaller interior angle between consecutive segments
                for j in range(len(segment_data) - 1):
                    b1, p_shared = segment_data[j]
                    b2, _ = segment_data[j+1]
                    
                    # Calculate the angular difference
                    diff = abs(b1 - b2)
                    if diff > 180: 
                        diff = 360 - diff
                    
                    # Interior Angle = 180 - the change in bearing
                    interior_angle = 180 - diff
                    
                    # If the bend is sharper than the threshold (e.g., < 120°), mark it
                    if interior_angle < angle_threshold:
                        exclusion_vertices.append((p_shared.X, p_shared.Y))
    
    del seg_insert
    print(f"Created {seg_counter} segments. Found {len(exclusion_vertices)} sharp bend exclusion zones.")
    
    # Generate transect center-points along grabens
    print(f"Generating transect center-points every {transect_spacing} m")
    pt_insert = arcpy.da.InsertCursor(points_fc, ["GrabenOID", "SegmentOID", "Bearing", "Chainage", "SHAPE@"])
    
    with arcpy.da.SearchCursor(lyr, ["OID@", "SHAPE@"]) as sc:
        for gid, geom in sc:
            if geom.length < min_graben_length:
                continue
            
            n = int(geom.length // transect_spacing)
            for i in range(n + 1):
                dist = i * transect_spacing
                pt = geom.positionAlongLine(dist)
                pt_geom = pt.firstPoint
                
                # Check if this point falls within 'point_vicinity' (k) of any bad vertex
                is_excluded = False
                for vx, vy in exclusion_vertices:
                    # Pythagorean distance check
                    if math.hypot(pt_geom.X - vx, pt_geom.Y - vy) < point_vicinity:
                        is_excluded = True
                        break
                
                # Only insert the point if it is safely outside the buffer zone
                if not is_excluded:
                    pt_insert.insertRow([gid, None, None, float(dist), pt])
    
    del pt_insert
    
    # Assign bearings from nearest segment
    print("Assigning bearings...")
    seg_dict = {r[0]: (r[1], r[2], r[3]) for r in arcpy.da.SearchCursor(
        segments_fc, ["OID@", "GrabenOID", "Bearing", "SHAPE@"])}
    
    with arcpy.da.UpdateCursor(points_fc, ["OID@", "GrabenOID", "SegmentOID", "Bearing", "SHAPE@"]) as uc:
        for oid, gid, segid, bear, geom in uc:
            min_d, best_seg, best_bear = float("inf"), None, None
            
            for soid, (sgid, sbear, sgeom) in seg_dict.items():
                if sgid != gid:
                    continue
                d = geom.distanceTo(sgeom)
                if d < min_d:
                    min_d, best_seg, best_bear = d, soid, sbear
            
            uc.updateRow([oid, gid, best_seg, best_bear, geom])
    
    # Remove exact coordinate duplicates if present
    coords_seen = {}
    del_oids = set()
    tol = 1e-3
    
    with arcpy.da.SearchCursor(points_fc, ["OID@", "GrabenOID", "Chainage", "SHAPE@XY"]) as sc:
        for oid, gid, chain, xy in sc:
            key = (round(xy[0] / tol), round(xy[1] / tol))
            if key in coords_seen:
                del_oids.add(oid)
            else:
                coords_seen[key] = oid
    
    del_count = 0
    if del_oids:
        with arcpy.da.UpdateCursor(points_fc, ["OID@"]) as uc:
            for r in uc:
                if r[0] in del_oids:
                    uc.deleteRow()
                    del_count += 1
    print(f"Deleted {del_count} duplicate points")
    
    # High-curvature exclusion (Legacy planar near-analysis check)
    print(f"Excluding points within {point_vicinity} m of each other (high-curvature filter)...")
    
    pt_count = int(arcpy.GetCount_management(points_fc).getOutput(0))
    if pt_count > 1:
        try:
            arcpy.Near_analysis(points_fc, points_fc, search_radius=point_vicinity, method="PLANAR")
        except Exception as e:
            print(f"WARNING: Near_analysis failed. Skipping high-curvature exclusion. Error: {e}")
        
        if arcpy.ListFields(points_fc, "NEAR_DIST"):
            lyr_vicinity = "graben_points_vicinity_layer"
            if arcpy.Exists(lyr_vicinity):
                arcpy.Delete_management(lyr_vicinity)
            
            arcpy.MakeFeatureLayer_management(points_fc, lyr_vicinity)
            exclusion_query = f'"NEAR_DIST" > -1 AND "NEAR_DIST" <= {point_vicinity}'
            arcpy.SelectLayerByAttribute_management(lyr_vicinity, "NEW_SELECTION", exclusion_query)
            
            sel_count = int(arcpy.GetCount_management(lyr_vicinity).getOutput(0))
            if sel_count > 0:
                arcpy.DeleteFeatures_management(lyr_vicinity)
                print(f"Deleted {sel_count} points due to high curvature")
            
            arcpy.management.DeleteField(points_fc, ["NEAR_FID", "NEAR_DIST", "NEAR_ANGLE", "NEAR_X", "NEAR_Y"])
    
    # Intersection buffer masking
    if mask_poly_fc:
        print("Removing transect center-points from intersection buffer zones...")
        
        pts_layer = "graben_points_layer"
        if arcpy.Exists(pts_layer):
            arcpy.Delete_management(pts_layer)
        arcpy.MakeFeatureLayer_management(points_fc, pts_layer)
        
        mask_layer = "density_mask_layer"
        if arcpy.Exists(mask_layer):
            arcpy.Delete_management(mask_layer)
        arcpy.MakeFeatureLayer_management(mask_poly_fc, mask_layer)
        
        arcpy.SelectLayerByLocation_management(pts_layer, "INTERSECT", mask_layer, selection_type="NEW_SELECTION")
        sel_count = int(arcpy.GetCount_management(pts_layer).getOutput(0))
        
        if sel_count > 0:
            arcpy.DeleteFeatures_management(pts_layer)
            print(f"Deleted {sel_count} points intersecting intersection buffer")
    
    pts_count = int(arcpy.GetCount_management(points_fc).getOutput(0))
    print(f"Transect center-points remaining: {pts_count}")
    
    # Create perpendicular transects
    print("Creating perpendicular transects...")
    half = perp_line_length / 2.0
    ins = arcpy.da.InsertCursor(transects_fc, ["GrabenOID", "SegmentOID", "Bearing", "Chainage", "SHAPE@"])
    
    with arcpy.da.SearchCursor(points_fc, ["GrabenOID", "SegmentOID", "Bearing", "Chainage", "SHAPE@"]) as sc:
        for gid, segid, bear, chain, geom in sc:
            if bear is None:
                continue
            
            pt = geom.firstPoint
            perp = (bear + 90) % 360
            dx = half * math.sin(math.radians(perp))
            dy = half * math.cos(math.radians(perp))
            
            p1 = arcpy.Point(pt.X - dx, pt.Y - dy)
            p2 = arcpy.Point(pt.X + dx, pt.Y + dy)
            ins.insertRow([gid, segid, bear, chain, arcpy.Polyline(arcpy.Array([p1, p2]), spatial_ref)])
    
    del ins
    
    tran_count = int(arcpy.GetCount_management(transects_fc).getOutput(0))
    print(f"Transects created: {tran_count}")
    
    return segments_fc, points_fc, transects_fc

# INTERSECTION BUFFER MASK
def create_intersection_buffer_mask(input_fc, gdb_path, buffer_distance):
    """
    Create buffer mask around intersection points.
    
    Returns the dissolved buffer polygon feature class, or None if no intersections found.
    """
    arcpy.env.workspace = gdb_path
    arcpy.env.overwriteOutput = True
    
    intersect_pts = os.path.join(gdb_path, "Intersect_Points")
    buffer_poly = os.path.join(gdb_path, "Intersect_Buffer_Poly")
    buffer_poly_diss = os.path.join(gdb_path, "Intersect_Buffer_Poly_Diss")
    
    for f in [intersect_pts, buffer_poly, buffer_poly_diss]:
        if arcpy.Exists(f):
            arcpy.Delete_management(f)
    
    print("Computing graben intersections...")
    arcpy.analysis.Intersect([input_fc], intersect_pts, output_type="POINT")
    
    count = int(arcpy.GetCount_management(intersect_pts).getOutput(0))
    print(f"Intersection points found: {count}")
    
    if count == 0:
        print("No intersections detected. Skipping intersection masking.")
        return None
    
    print(f"Buffering intersection points by {buffer_distance} m...")
    arcpy.analysis.Buffer(intersect_pts, buffer_poly, f"{buffer_distance} Meters")
    
    print("Dissolving buffers...")
    arcpy.management.Dissolve(buffer_poly, buffer_poly_diss)
    
    return buffer_poly_diss

# DEM SAMPLING AND CSV EXPORT
def sample_dem_and_export_csvs(transect_points_fc, transect_lines_fc, dem_path, 
                               gdb_path, csv_dir, sample_spacing):
    """
    Sample DEM along transects and export profiles as CSV files.
    
    Returns the joined feature class with elevation data.
    """
    arcpy.env.workspace = gdb_path
    arcpy.env.overwriteOutput = True
    
    sample_points_fc = os.path.join(gdb_path, "Transect_Sample_Points")
    sampled_fc = os.path.join(gdb_path, "Sampled_Transect_Points")
    joined_fc = os.path.join(gdb_path, "Joined_Transect_Points")
    
    for f in [sample_points_fc, sampled_fc, joined_fc]:
        if arcpy.Exists(f):
            arcpy.Delete_management(f)
    
    print(f"Generating sample points every {sample_spacing} m along transects...")
    arcpy.management.GeneratePointsAlongLines(
        transect_lines_fc, sample_points_fc, "DISTANCE",
        Distance=f"{sample_spacing} Meters", Include_End_Points="END_POINTS",
        Add_Chainage_Fields="NO_CHAINAGE"
    )
    
    sp_count = int(arcpy.GetCount_management(sample_points_fc).getOutput(0))
    print(f"Sample points generated: {sp_count}")
    
    print("Sampling DEM...")
    try:
        arcpy.CheckOutExtension("Spatial")
        arcpy.sa.ExtractValuesToPoints(sample_points_fc, dem_path, sampled_fc, "NONE", "VALUE_ONLY")
        arcpy.CheckInExtension("Spatial")
    except Exception as e:
        print(f"ERROR: DEM sampling failed. {e}")
        return None
    
    sampled_count = int(arcpy.GetCount_management(sampled_fc).getOutput(0))
    print(f"Points sampled with elevation: {sampled_count}")
    
    print("Joining sample points to transects...")
    arcpy.analysis.SpatialJoin(sampled_fc, transect_lines_fc, joined_fc,
                              "JOIN_ONE_TO_ONE", "KEEP_ALL", match_option="CLOSEST")
    
    joined_count = int(arcpy.GetCount_management(joined_fc).getOutput(0))
    print(f"Joined point count: {joined_count}")
    
    print("Exporting CSV profiles...")
    
    fields = ["SHAPE@XY", "RASTERVALU", "GrabenOID", "SegmentOID", "Bearing", "Chainage"]
    grouped = {}
    
    with arcpy.da.SearchCursor(joined_fc, fields) as cur:
        for xy, elev, gid, segid, bear, chain in cur:
            chain_key = round(chain, 2)
            unique_key = (gid, segid, chain_key)
            
            grouped.setdefault(unique_key, []).append({
                "X": xy[0], "Y": xy[1], "Elevation": elev,
                "GrabenOID": gid, "SegmentOID": segid, "Bearing": bear, "Chainage": chain
            })
    
    for (gid, segid, chain_key), pts in grouped.items():
        folder = os.path.join(csv_dir, f"Graben_{gid}")
        os.makedirs(folder, exist_ok=True)
        
        chain_str = int(round(chain_key, 0))
        out_csv = os.path.join(folder, f"profile_{gid}_{segid}_C{chain_str}.csv")
        
        df = pd.DataFrame(pts)
        
        # Sort points by projection onto profile axis
        if not df.empty and len(df) > 1:
            x0, y0 = df.iloc[0]['X'], df.iloc[0]['Y']
            df['Dist_From_Start'] = df.apply(
                lambda row: math.hypot(row['X'] - x0, row['Y'] - y0), axis=1
            )
            
            farthest_idx = df['Dist_From_Start'].idxmax()
            xf, yf = df.loc[farthest_idx, 'X'], df.loc[farthest_idx, 'Y']
            
            len_sq = (xf - x0)**2 + (yf - y0)**2
            if len_sq > 1e-6:
                df['Sort_Dist'] = df.apply(
                    lambda row: ((row['X'] - x0) * (xf - x0) + (row['Y'] - y0) * (yf - y0)) / len_sq,
                    axis=1
                )
                df = df.sort_values(by='Sort_Dist')
            else:
                df = df.sort_index()
            
            df.to_csv(out_csv, index=False, 
                     columns=[c for c in df.columns if c not in ['Dist_From_Start', 'Sort_Dist']])
        else:
            df.to_csv(out_csv, index=False)
    
    print("CSV export complete")
    return joined_fc

# PROFILE ANALYSIS
def analyze_profiles_and_produce_metrics(csv_dir, plots_dir, summary_dir, sample_interval, 
                                         savgol_window, savgol_poly, asymmetry_threshold):
    """
    Analyze topographic profiles and extract morphometric measurements.
    
    Outputs per-graben metrics and master L/D summary.
    """
    print(f"Analyzing profiles (Asymmetry threshold: {asymmetry_threshold})...")
    
    for graben_folder in os.listdir(csv_dir):
        graben_path = os.path.join(csv_dir, graben_folder)
        
        if not os.path.isdir(graben_path) or not graben_folder.startswith("Graben_"):
            continue
        
        raw_plot_folder = os.path.join(plots_dir, f"{graben_folder}_Plots_Raw")
        marked_plot_folder = os.path.join(plots_dir, f"{graben_folder}_Plots_Marked")
        os.makedirs(raw_plot_folder, exist_ok=True)
        os.makedirs(marked_plot_folder, exist_ok=True)
        
        metrics_csv_path = os.path.join(summary_dir, f"{graben_folder}_metrics.csv")
        
        with open(metrics_csv_path, "w") as metrics_file:
            metrics_file.write("Profile_Name,D1,D2,L\n")
            
            for file in os.listdir(graben_path):
                if not file.startswith("profile_") or not file.endswith(".csv"):
                    continue
                
                csv_path = os.path.join(graben_path, file)
                df = pd.read_csv(csv_path)
                profile_name = os.path.splitext(file)[0]
                
                # Data validation
                if "Elevation" not in df.columns or df["Elevation"].isnull().any() or len(df) < 5:
                    print(f"[SKIPPED] {file}: Insufficient data")
                    continue
                
                y_raw = np.array(df["Elevation"].values)
                x = np.arange(1, len(y_raw) + 1) * sample_interval
                
                # Savitzky-Golay filter with robustness checks
                sw = savgol_window
                if sw % 2 == 0:
                    sw += 1
                if sw < 3:
                    sw = 3
                if sw >= len(y_raw):
                    sw = len(y_raw) - 1
                    if sw % 2 == 0:
                        sw -= 1
                
                if sw < 3:
                    print(f"[SKIPPED] {file}: Insufficient points for S-G filter")
                    continue
                
                try:
                    y_smooth = savgol_filter(y_raw, window_length=sw, polyorder=savgol_poly)
                except Exception:
                    y_smooth = y_raw
                
                # Plot raw vs smoothed
                if save_plots:
                    plt.figure(figsize=(10, 6))
                    plt.plot(x, y_raw, label="Raw Elevation", alpha=0.6)
                    plt.plot(x, y_smooth, label="Smoothed Elevation")
                    plt.title(profile_name)
                    plt.xlabel("Distance (m)")
                    plt.ylabel("Elevation (m)")
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(raw_plot_folder, f"{profile_name}_raw_vs_smooth.png"))
                    plt.close()
                
                # Detect local extrema
                local_min_indices = argrelextrema(y_smooth, np.less)[0]
                local_max = argrelextrema(y_smooth, np.greater)[0]
                
                if len(local_min_indices) == 0 or len(local_max) < 2:
                    print(f"[SKIPPED] {file}: Assymetry criteria not satisfied")
                    continue
                
                # Find deepest minimum with valid shoulders
                sorted_minima = sorted([(y_smooth[idx], idx) for idx in local_min_indices], key=lambda x: x[0])
                
                final_min_index = None
                final_left_shoulder = None
                final_right_shoulder = None
                
                for elev, min_index in sorted_minima:
                    left_shoulders = sorted([i for i in local_max if i < min_index], reverse=True)
                    right_shoulders = sorted([i for i in local_max if i > min_index])
                    
                    if not left_shoulders or not right_shoulders:
                        continue
                    
                    l_shoulder = left_shoulders[0]
                    r_shoulder = right_shoulders[0]
                    
                    D1 = y_smooth[l_shoulder] - y_smooth[min_index]
                    D2 = y_smooth[r_shoulder] - y_smooth[min_index]
                    
                    if D1 <= 0 or D2 <= 0:
                        continue
                    
                    # Asymmetry check
                    asymmetry_met = True
                    D_max = max(D1, D2)
                    
                    if D1 < asymmetry_threshold * D_max:
                        if len(left_shoulders) > 1:
                            l_shoulder_alt = left_shoulders[1]
                            D1_alt = y_smooth[l_shoulder_alt] - y_smooth[min_index]
                            if D1_alt > 0 and D1_alt >= asymmetry_threshold * max(D1_alt, D2):
                                l_shoulder = l_shoulder_alt
                                D1 = D1_alt
                            else:
                                asymmetry_met = False
                        else:
                            asymmetry_met = False
                    
                    if asymmetry_met and D2 < asymmetry_threshold * max(D1, D2):
                        if len(right_shoulders) > 1:
                            r_shoulder_alt = right_shoulders[1]
                            D2_alt = y_smooth[r_shoulder_alt] - y_smooth[min_index]
                            if D2_alt > 0 and D2_alt >= asymmetry_threshold * max(D1, D2_alt):
                                r_shoulder = r_shoulder_alt
                                D2 = D2_alt
                            else:
                                asymmetry_met = False
                        else:
                            asymmetry_met = False
                    
                    if asymmetry_met:
                        final_min_index = min_index
                        final_left_shoulder = l_shoulder
                        final_right_shoulder = r_shoulder
                        break
                
                if final_min_index is None:
                    print(f"[SKIPPED] {file}: No valid shoulders found")
                    continue
                
                min_index = final_min_index
                left_shoulder = final_left_shoulder
                right_shoulder = final_right_shoulder
                
                D1 = y_smooth[left_shoulder] - y_smooth[min_index]
                D2 = y_smooth[right_shoulder] - y_smooth[min_index]
                
                # Calculate width (L) via iso-elevation crossing
                if D1 < D2:
                    shallower_index = left_shoulder
                    search_range = range(min_index + 1, right_shoulder)
                else:
                    shallower_index = right_shoulder
                    search_range = range(min_index - 1, left_shoulder, -1)
                
                target_elev = y_smooth[shallower_index]
                iso_x = None
                
                for i in search_range:
                    if i - 1 < 0 or i >= len(y_smooth):
                        continue
                    
                    y1, y2 = y_smooth[i - 1], y_smooth[i]
                    if (y1 - target_elev) * (y2 - target_elev) < 0:
                        x1, x2 = x[i - 1], x[i]
                        frac = (target_elev - y1) / (y2 - y1) if (y2 - y1) != 0 else 0
                        iso_x = x1 + frac * (x2 - x1)
                        break
                
                L = abs(iso_x - x[shallower_index]) if iso_x else abs(x[right_shoulder] - x[left_shoulder])
                
                # Plot marked profile
                if save_plots:
                    plt.figure(figsize=(10, 6))
                    plt.plot(x, y_smooth, label="Smoothed Elevation")
                    plt.scatter(x[min_index], y_smooth[min_index], color='red', 
                               label="Floor (Min)", zorder=5)
                    plt.scatter(x[left_shoulder], y_smooth[left_shoulder], color='green', 
                               label="Left Shoulder", zorder=5)
                    plt.scatter(x[right_shoulder], y_smooth[right_shoulder], color='blue', 
                               label="Right Shoulder", zorder=5)
                    if iso_x:
                        plt.scatter(iso_x, target_elev, color='orange', 
                                   label="Iso-elevation Point", zorder=6)
                    plt.title(profile_name)
                    plt.xlabel("Distance (m)")
                    plt.ylabel("Elevation (m)")
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(marked_plot_folder, f"{profile_name}_marked.png"))
                    plt.close()
                
                metrics_file.write(f"{profile_name},{D1:.2f},{D2:.2f},{L:.2f}\n")
    
    # Master summary
    print("Generating master L/D summary...")
    summary_csv_path = os.path.join(summary_dir, "Graben_LD_Summary.csv")
    
    with open(summary_csv_path, "w") as summary_file:
        summary_file.write("GrabenOID,D,L,L/D\n")
        
        for graben_folder in os.listdir(csv_dir):
            if not graben_folder.startswith("Graben_"):
                continue
            
            metrics_csv_path = os.path.join(summary_dir, f"{graben_folder}_metrics.csv")
            if not os.path.isfile(metrics_csv_path):
                continue
            
            df = pd.read_csv(metrics_csv_path)
            
            if df.empty or not {'D1', 'D2', 'L'}.issubset(df.columns):
                continue
            
            df["D"] = df[["D1", "D2"]].min(axis=1)
            df["LD"] = df["L"] / df["D"].replace(0, np.nan)
            
            # 2-sigma outlier removal
            mean_ld = df["LD"].mean()
            std_ld = df["LD"].std()
            lower_bound = mean_ld - 2 * std_ld
            upper_bound = mean_ld + 2 * std_ld
            
            df_filtered = df[(df["LD"].notna()) & (df["LD"] >= lower_bound) & (df["LD"] <= upper_bound)]
            
            if df_filtered.empty:
                print(f"WARNING: {graben_folder} filtered entirely by 2-sigma rule")
                continue
            
            avg_D = df_filtered["D"].mean()
            avg_L = df_filtered["L"].mean()
            avg_LD = avg_L / avg_D
            
            graben_id = graben_folder.replace("Graben_", "")
            summary_file.write(f"{graben_id},{avg_D:.2f},{avg_L:.2f},{avg_LD:.2f}\n")
    
    print(f"Profile analysis complete. Summary: {summary_csv_path}")

# MAIN
def main():
    """Execute full graben morphometric analysis pipeline."""
    try:
        print("=" * 50)
        print("GRABEN MORPHOMETRIC ANALYSIS PIPELINE")
        print("=" * 50)
        
        gdb_path, csv_dir, plots_dir, summary_dir = setup_output_folders(output_dir, gdb_name)
        
        # Step 1: Create intersection buffer mask
        intersection_mask_poly = None
        if use_intersection_exclusion:
            intersection_mask_poly = create_intersection_buffer_mask(
                input_grabens, gdb_path, intersection_buffer_distance
            )
        
        # Step 2: Segment and create transects (includes high-curvature and intersection filtering)
        segments_fc, points_fc, transects_fc = segment_and_make_transects(
            input_grabens, gdb_path, transect_spacing, perp_line_length,
            point_vicinity, angle_threshold, min_graben_length, segment_split_interval,
            mask_poly_fc=intersection_mask_poly
        )
        
        # Step 3: Sample DEM and export profiles
        joined_fc = sample_dem_and_export_csvs(
            points_fc, transects_fc, dem_path, gdb_path, csv_dir, sample_spacing
        )
        
        if joined_fc is None:
            print("CSV generation failed. Skipping profile analysis.")
            return
        
        # Step 4: Analyze profiles and compute metrics
        analyze_profiles_and_produce_metrics(
            csv_dir, plots_dir, summary_dir, sample_spacing,
            savgol_window, savgol_poly, asymmetry_threshold
        )
        
        print("=" * 50)
        print(f"EXECUTION COMPLETE. Outputs: {output_dir}")
        print("=" * 50)
        
    except Exception as e:
        print(f"ERROR: {e}")
        raise

if __name__ == "__main__":
    main()

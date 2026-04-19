#!/usr/bin/env python3
"""
visualize.py — 3D scatter plot of K-Means clustering results.

Reads kmeans_results.csv (all feature columns + cluster), lets you pick
any 3 axes, produces a matplotlib PNG and a VTK PolyData .vtp file that
can be opened in ParaView (color by the 'cluster' scalar).

Usage:
  python3 visualize.py [--input FILE] [--x FEAT] [--y FEAT] [--z FEAT]
                       [--output STEM] [--sample N]

Defaults:  --x valence  --y danceability  --z energy
           --input kmeans_results.csv  --output kmeans_3d
"""

import argparse
import csv
import random
import struct
import sys
import os

# Optional: matplotlib for PNG output
try:
    import matplotlib
    matplotlib.use("Agg")           # non-interactive backend
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="3D visualizer for K-Means results")
parser.add_argument("--input",  default="kmeans_results.csv",
                    help="Results CSV from kmeans (default: kmeans_results.csv)")
parser.add_argument("--x",      default="valence",
                    help="Feature for X axis (default: valence)")
parser.add_argument("--y",      default="danceability",
                    help="Feature for Y axis (default: danceability)")
parser.add_argument("--z",      default="energy",
                    help="Feature for Z axis (default: energy)")
parser.add_argument("--output", default="kmeans_3d",
                    help="Output filename stem (default: kmeans_3d) "
                         "→ produces <stem>.png and <stem>.vtp")
parser.add_argument("--sample", type=int, default=0,
                    help="Randomly subsample N points for display (default: all points)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load CSV
# ---------------------------------------------------------------------------
if not os.path.exists(args.input):
    sys.exit(f"[Error] Input file not found: {args.input}")

print(f"[Load] Reading {args.input} ...")
rows = []
with open(args.input, newline="") as f:
    reader = csv.DictReader(f)
    columns = reader.fieldnames
    print(f"[Load] Available features: {', '.join(c for c in columns if c != 'cluster')}")
    for row in reader:
        rows.append(row)

print(f"[Load] {len(rows)} data points loaded.")

# Validate axis choices
for axis_name, axis_val in [("--x", args.x), ("--y", args.y), ("--z", args.z)]:
    if axis_val not in columns or axis_val == "cluster":
        valid = [c for c in columns if c != "cluster"]
        sys.exit(
            f"[Error] {axis_name}='{axis_val}' is not a valid feature.\n"
            f"        Valid options: {', '.join(valid)}"
        )

# Extract coordinate and cluster arrays
xs       = [float(r[args.x])       for r in rows]
ys       = [float(r[args.y])       for r in rows]
zs       = [float(r[args.z])       for r in rows]
clusters = [int(r["cluster"])      for r in rows]
n        = len(rows)
k        = max(clusters) + 1

if args.sample > 0 and args.sample < n:
    indices = random.sample(range(n), args.sample)
    xs       = [xs[i]       for i in indices]
    ys       = [ys[i]       for i in indices]
    zs       = [zs[i]       for i in indices]
    clusters = [clusters[i] for i in indices]
    n        = args.sample
    print(f"[Sample] Subsampled to {n} points for display.")

print(f"[Axes] x={args.x}  y={args.y}  z={args.z}  k={k} clusters")

# ---------------------------------------------------------------------------
# Recolor by nearest 3D centroid (eliminates projection-artifact color mixing)
# ---------------------------------------------------------------------------
# Compute centroid of each cluster in the 3 displayed dimensions
cx = [0.0] * k
cy = [0.0] * k
cz = [0.0] * k
cc = [0]   * k
for i in range(n):
    c = clusters[i]
    cx[c] += xs[i]; cy[c] += ys[i]; cz[c] += zs[i]; cc[c] += 1
for c in range(k):
    if cc[c] > 0:
        cx[c] /= cc[c]; cy[c] /= cc[c]; cz[c] /= cc[c]

# Reassign each point's color to its nearest 3D centroid
def nearest3d(x, y, z):
    best, bestc = float("inf"), 0
    for c in range(k):
        d = (x-cx[c])**2 + (y-cy[c])**2 + (z-cz[c])**2
        if d < best:
            best, bestc = d, c
    return bestc

clusters_3d = [nearest3d(xs[i], ys[i], zs[i]) for i in range(n)]
print("[Color] Recolored by nearest 3D centroid — hard boundaries in plot.")

# ---------------------------------------------------------------------------
# Color palette (up to 20 clusters)
# ---------------------------------------------------------------------------
PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#fffac8", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#ffffff",
]
# Use 3D-reassigned cluster for color, original 14D cluster stored separately
colors = [PALETTE[c % len(PALETTE)] for c in clusters_3d]

# ---------------------------------------------------------------------------
# Matplotlib 3D scatter → PNG
# ---------------------------------------------------------------------------
png_path = f"{args.output}.png"

if HAS_MATPLOTLIB:
    print(f"[Plot] Generating 3D scatter plot → {png_path} ...")
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    for c in range(k):
        idx = [i for i, cl in enumerate(clusters_3d) if cl == c]
        ax.scatter(
            [xs[i] for i in idx],
            [ys[i] for i in idx],
            [zs[i] for i in idx],
            c=PALETTE[c % len(PALETTE)],
            s=2, alpha=0.5, label=f"Cluster {c}"
        )

    ax.set_xlabel(args.x)
    ax.set_ylabel(args.y)
    ax.set_zlabel(args.z)
    ax.set_title(f"K-Means Clusters  (x={args.x}, y={args.y}, z={args.z})")
    ax.legend(markerscale=4, loc="upper left", fontsize="small")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"[Plot] Saved {png_path}")
else:
    print("[Plot] matplotlib not available — skipping PNG. Install with: pip install matplotlib")

# ---------------------------------------------------------------------------
# VTK XML PolyData (.vtp) — opens in ParaView, color by 'cluster' scalar
# ---------------------------------------------------------------------------
vtp_path = f"{args.output}.vtp"
print(f"[VTK]  Writing ParaView file → {vtp_path} ...")

def float32_base64(values):
    """Encode list of floats as VTK inline binary (base64 block with 4-byte length prefix)."""
    import base64
    raw = struct.pack(f"<{len(values)}f", *values)
    # VTK expects a 4-byte header with the byte count before the data
    header = struct.pack("<I", len(raw))
    return base64.b64encode(header + raw).decode("ascii")

# Interleave x,y,z for the Points array
xyz = []
for i in range(n):
    xyz.extend([xs[i], ys[i], zs[i]])

cluster_encoded = float32_base64([float(c) for c in clusters_3d])
xyz_encoded     = float32_base64(xyz)

with open(vtp_path, "w") as f:
    f.write('<?xml version="1.0"?>\n')
    f.write('<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian" '
            'header_type="UInt32" encoding="base64">\n')
    f.write('  <PolyData>\n')
    f.write(f'    <Piece NumberOfPoints="{n}" NumberOfVerts="{n}" '
            'NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="0">\n')

    # Points
    f.write('      <Points>\n')
    f.write('        <DataArray type="Float32" NumberOfComponents="3" '
            'format="binary">\n')
    f.write(f'          {xyz_encoded}\n')
    f.write('        </DataArray>\n')
    f.write('      </Points>\n')

    # Verts (one vertex cell per point so ParaView renders them)
    f.write('      <Verts>\n')
    conn = float32_base64([float(i) for i in range(n)])
    offs = float32_base64([float(i + 1) for i in range(n)])
    f.write('        <DataArray type="Float32" Name="connectivity" format="binary">\n')
    f.write(f'          {conn}\n')
    f.write('        </DataArray>\n')
    f.write('        <DataArray type="Float32" Name="offsets" format="binary">\n')
    f.write(f'          {offs}\n')
    f.write('        </DataArray>\n')
    f.write('      </Verts>\n')

    # Point data: cluster scalar + the 3 chosen features
    f.write('      <PointData Scalars="cluster">\n')

    # cluster
    f.write('        <DataArray type="Float32" Name="cluster" '
            'NumberOfComponents="1" format="binary">\n')
    f.write(f'          {cluster_encoded}\n')
    f.write('        </DataArray>\n')

    # x feature
    f.write(f'        <DataArray type="Float32" Name="{args.x}" '
            'NumberOfComponents="1" format="binary">\n')
    f.write(f'          {float32_base64(xs)}\n')
    f.write('        </DataArray>\n')

    # y feature
    f.write(f'        <DataArray type="Float32" Name="{args.y}" '
            'NumberOfComponents="1" format="binary">\n')
    f.write(f'          {float32_base64(ys)}\n')
    f.write('        </DataArray>\n')

    # z feature
    f.write(f'        <DataArray type="Float32" Name="{args.z}" '
            'NumberOfComponents="1" format="binary">\n')
    f.write(f'          {float32_base64(zs)}\n')
    f.write('        </DataArray>\n')

    f.write('      </PointData>\n')
    f.write('    </Piece>\n')
    f.write('  </PolyData>\n')
    f.write('</VTKFile>\n')

print(f"[VTK]  Saved {vtp_path}")
print()
print("=== Done ===")
print(f"  PNG plot : {png_path}")
print(f"  ParaView : {vtp_path}")
print()
print("To open in ParaView:")
print(f"  1. File → Open → {vtp_path}")
print(f"  2. In the toolbar, set 'Color by' to 'cluster'")
print(f"  3. Apply a discrete colormap (e.g. 'glasbey' or 'jet')")
print()
print("To re-plot with different axes (no re-clustering needed):")
print(f"  python3 visualize.py --x tempo --y acousticness --z loudness")

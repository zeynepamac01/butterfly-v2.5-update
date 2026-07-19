from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D
from collections import defaultdict, deque


# File paths
CSV_PATH = Path("./Data/Table_S1.csv")
OUT_SVG  = Path("./Data/svg_plots/organ_plots.svg")

# Figure dimensions
FIG_W, FIG_H = 40, 10
HEART_X, BASE_Y = 0.0, 0.0

# Colors
COLORS = {"artery": "#d62728", "vein": "#1f77b4"}

LINE_W = 2.6
DOT_SIZE = 68
LABEL_FONT_SIZE = 6
LABEL_ROT_DEG = 12
LABEL_OFFSET = 0.12

# Arrow head sizes
HEAD_A = 22  # Artery arrow head
HEAD_V = 22  # Vein arrow head

FTU_SIZE = 0.42
FTU_LW   = 2.0
CONNECTOR = 0.22  # Distance from FTU edge to connecting line

EXTRA_RIGHT_PAD = 8.0
SHOW_LEFT_AXIS = False

#Edges
E1 = list(range(1, 14))        # 1→2→...→13 (main artery path to acinus)
E2 = [13, 31, 32]              # 13→31→32 (branch to islet)
E3 = [13, 14]                  # 13→14 (branch to duct)
E4 = [33, 16]                  # 33→16 (acinus exit to hub)
E5 = [32, 16]                  # 32→16 (islet exit to hub)
E6 = [14, 15, 16]              # 14→15→16 (duct exit to hub via junction)

E7 = [16, 17, 18, 19, 20, 21, 22, 23, 24]  # 16→...→24 (hub to liver)
E8 = [26, 27, 28, 29, 30]                   # 26→...→30 (liver to heart)

ARTERY_EDGES = [E1, E2, E3, E4, E5, E6]
VEIN_EDGES   = [E7, E8]

FTU_ACINUS = {13, 33}
FTU_DUCT   = {14}
FTU_ISLET  = {32}
FTU_LIVER  = {24, 25, 26}
ALL_FTU_STEPS = FTU_ACINUS | FTU_DUCT | FTU_ISLET | FTU_LIVER


ROW_Y        = BASE_Y + 0.35    # Y position of acinus/duct row
ACINUS_X     = 16.5             # X position of acinus
DUCT_X       = ACINUS_X + 1.5   # X position of duct (moved closer/left)
ISLET_X      = DUCT_X           # X position of islet (aligned with duct)
ISLET_Y      = ROW_Y + 0.95     # Y position of islet (above duct)

STEP31_X     = ACINUS_X + 0.6   # X position of node 31 (between acinus and duct)

# Hub positions (collection points below FTUs)
HUB_X        = ACINUS_X + 0.2   
HUB15_Y      = ROW_Y - 0.65     
HUB16_Y      = HUB15_Y - 0.45   

LIVER_X      = 3.0
LIVER_Y      = BASE_Y - 0.85

PINNED = {15: HUB_X, 16: HUB_X, 31: STEP31_X}

# Artery lane heights
y_E1 = ROW_Y                  
y_E2 = ISLET_Y + 0.15         
y_E3 = ROW_Y + 0.50           
y_E4 = ROW_Y - 0.30           
y_E5_right = ROW_Y - 0.15     
y_E6 = ROW_Y - 0.10           
artery_y = [y_E1, y_E2, y_E3, y_E4, y_E5_right, y_E6]

# Vein lane heights
y_V7 = HUB16_Y - 0.18
y_V8 = LIVER_Y - 0.22
vein_y = [y_V7, y_V8]


#helpers
def topo_order_from_edges(paths, ftu_steps):
    adj = defaultdict(set)
    indeg = defaultdict(int)
    nodes = set()
    
    def add(u, v):
        nodes.add(u)
        nodes.add(v)
        if v in ftu_steps:
            indeg.setdefault(u, 0)
            return
        if v not in adj[u]:
            adj[u].add(v)
            indeg[v] += 1
            indeg.setdefault(u, 0)
    
    for p in paths:
        for i in range(len(p) - 1):
            add(p[i], p[i + 1])
    
    q = deque(sorted([n for n in nodes if indeg.get(n, 0) == 0]))
    order = []
    while q:
        n = q.popleft()
        order.append(n)
        for m in sorted(adj.get(n, [])):
            indeg[m] -= 1
            if indeg[m] == 0:
                q.append(m)
    
    return [n for n in order if n not in ftu_steps]

def make_x_positions(order, left_x, right_x, right_margin=0.99):
    if not order:
        return {}
    span = right_x - left_x
    return {step: left_x + (i / max(1, len(order) - 1)) * (span * right_margin)
            for i, step in enumerate(order)}

def arrow(ax, x1, y1, x2, y2, color, head=16, shrinkA=6, shrinkB=6, z=3):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
        mutation_scale=head, lw=LINE_W, color=color, shrinkA=shrinkA, shrinkB=shrinkB,
        capstyle='round', joinstyle='round', zorder=z))

def line(ax, x1, y1, x2, y2, color, z=4):
    ax.plot([x1, x2], [y1, y2], color=color, lw=LINE_W, solid_capstyle='round', zorder=z)

def label(ax, x, y, text, color, rot, above=True):
    dy = LABEL_OFFSET if above else -LABEL_OFFSET
    ax.text(x, y + dy, text, fontsize=LABEL_FONT_SIZE, rotation=rot,
            va="center", ha="center", color=color, zorder=9,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.78))

def ftu_pos(step):
    if step in FTU_ACINUS:
        return ACINUS_X, ROW_Y
    if step in FTU_DUCT:
        return DUCT_X, ROW_Y
    if step in FTU_ISLET:
        return ISLET_X, ISLET_Y
    if step in FTU_LIVER:
        return LIVER_X, LIVER_Y
    return None

df = pd.read_csv(CSV_PATH)
for c in df.columns:
    if df[c].dtype == object:
        df[c] = df[c].astype(str).str.strip()

#Only works for pancreas
organ = df["Organ"].dropna().unique()[0] if "Organ" in df.columns and df["Organ"].notna().any() else "pancreas"

step_labels = {}
if "PathStep" in df.columns:
    for step, g in df.groupby("PathStep"):
        row = g.sort_values(["PathVessel", "PathVesselID"]).iloc[0]
        name = (row.get("PathVessel", "") or "").strip()
        pid = (row.get("PathVesselID", "") or "").strip()
        if pid.lower() == "nan":
            pid = ""
        step_labels[int(step)] = (f"{name} {pid}".strip() or str(int(step)))

artery_order = topo_order_from_edges(ARTERY_EDGES, ALL_FTU_STEPS | set(PINNED.keys()))
artery_x = make_x_positions(artery_order, HEART_X, ACINUS_X - 0.5, right_margin=0.98)

for s in FTU_ACINUS:
    artery_x[s] = ACINUS_X
for s in FTU_DUCT:
    artery_x[s] = DUCT_X
for s in FTU_ISLET:
    artery_x[s] = ISLET_X
for nid, px in PINNED.items():
    artery_x[nid] = px

artery_x[16] = DUCT_X

vein_x = {}

def leftward(path, start_x):
    L = max(1, len(path) - 1)
    for i, s in enumerate(path):
        t = i / L
        vein_x[s] = start_x - t * (start_x - HEART_X) * 0.96

E7_PAD = 0.55
E7_NODES = E7[:] 

x_start = DUCT_X
x_target_before_box = LIVER_X - (FTU_SIZE / 2) - E7_PAD
span = max(0.01, x_start - x_target_before_box)

L = max(1, len(E7_NODES) - 1)
for i, s in enumerate(E7_NODES):
    t = i / L
    # place nodes evenly from x_start toward the target-before-box
    vein_x[s] = x_start - t * span

vein_x[24] = LIVER_X

for s in FTU_LIVER:
    vein_x[s] = LIVER_X
leftward(E8, start_x=LIVER_X)

for s in FTU_ACINUS:
    vein_x[s] = ACINUS_X
for s in FTU_DUCT:
    vein_x[s] = DUCT_X
for s in FTU_ISLET:
    vein_x[s] = ISLET_X
vein_x[15] = HUB_X
vein_x[16] = DUCT_X  # 16 is at junction (DUCT_X)

plt.figure(figsize=(FIG_W, FIG_H), dpi=150)
ax = plt.gca()
drawn = set()

def exit_ftu(step, toward_y, color, side="top"):
    xftu, yftu = ftu_pos(step)
    y0 = yftu + CONNECTOR if side == "top" else yftu - CONNECTOR
    line(ax, xftu, y0, artery_x.get(step, xftu), toward_y, color)

def drop_to(node, from_y, color):
    if node == 15:
        line(ax, HUB_X, from_y, HUB_X, HUB15_Y, color)
    elif node == 16:
        line(ax, HUB_X, from_y, HUB_X, HUB16_Y, color)


#Arteries draw
for idx, path in enumerate(ARTERY_EDGES):
    y = artery_y[idx]

    # E1: Heart → ... → 13 (main artery)
    if idx == 0:
        arrow(ax, HEART_X, BASE_Y, artery_x.get(path[0], HEART_X), y, COLORS["artery"], head=HEAD_A, shrinkB=0)

    # E2: 13 → 31 → 32 (high lane to islet)
    if path == E2:        
        line(ax, ACINUS_X, ROW_Y + CONNECTOR, ACINUS_X, y, COLORS["artery"])
        arrow(ax, ACINUS_X, y, artery_x.get(31, STEP31_X), y, COLORS["artery"], head=HEAD_A, shrinkB=0)
        arrow(ax, artery_x.get(31, STEP31_X), y, ISLET_X, y, COLORS["artery"], head=HEAD_A, shrinkB=0)
        y32_offset = -0.12 
        y32_dot = ISLET_Y + y32_offset
        line(ax, ISLET_X, y, ISLET_X, y32_dot, COLORS["artery"])
        
        if 31 not in drawn:
            ax.scatter([STEP31_X], [y], s=DOT_SIZE, color=COLORS["artery"], zorder=8)
            label(ax, STEP31_X, y, step_labels.get(31, "31"), COLORS["artery"], LABEL_ROT_DEG, above=True)
            drawn.add(31)
        continue

    # E3: 13 → 14 (acinus right → duct left)
    if path == E3:
        y13_offset = 0.0  # 13 is at center
        y13_dot = ROW_Y + y13_offset
        line(ax, ACINUS_X, y13_dot, ACINUS_X + FTU_SIZE/2, y13_dot, COLORS["artery"])
        line(ax, ACINUS_X + FTU_SIZE/2, y13_dot, ACINUS_X + FTU_SIZE/2, y, COLORS["artery"])
        arrow(ax, ACINUS_X + FTU_SIZE/2, y, DUCT_X - FTU_SIZE/2, y, COLORS["artery"], head=HEAD_A, shrinkB=0)
        y14_offset = 0.0  # 14 is at center
        y14_dot = ROW_Y + y14_offset
        line(ax, DUCT_X - FTU_SIZE/2, y, DUCT_X - FTU_SIZE/2, y14_dot, COLORS["artery"])
        line(ax, DUCT_X - FTU_SIZE/2, y14_dot, DUCT_X, y14_dot, COLORS["artery"])
        continue

    # E4: 33 → 16 (acinus bottom → junction at duct vertical line)
    if path == E4:
        y33_offset = -0.12
        y33_dot = ROW_Y + y33_offset
        
        # Exit from 33 dot at bottom of acinus - go straight down
        line(ax, ACINUS_X, y33_dot, ACINUS_X, HUB16_Y, COLORS["artery"])
        # Then go right to junction at DUCT_X where 16 dot is
        arrow(ax, ACINUS_X, HUB16_Y, DUCT_X, HUB16_Y, COLORS["artery"], head=HEAD_A, shrinkB=0)
        continue

        # connect 13 and 33 inside the acinus
    y13 = ROW_Y + 0.0      
    y33 = ROW_Y - 0.12     
    line(ax, ACINUS_X, y13, ACINUS_X, y33, COLORS["artery"], z=11)

    # E5: 32 → 16 (islet right → join duct vertical line)
    if path == E5:
        y32_offset = -0.12
        y32_dot = ISLET_Y + y32_offset
        
        exit_right_x = ISLET_X + FTU_SIZE/2 + 0.15
        line(ax, ISLET_X, y32_dot, exit_right_x, y32_dot, COLORS["artery"])

        line(ax, exit_right_x, y32_dot, exit_right_x, HUB16_Y, COLORS["artery"])
        arrow(ax, exit_right_x, HUB16_Y, DUCT_X, HUB16_Y, COLORS["artery"], head=HEAD_A, shrinkB=0)
        continue

    # E6: 14 → 15 → 16 (duct bottom → junction → hubs)
    if path == E6:
        y14_dot = ROW_Y  # 14 is at center of duct
        line(ax, DUCT_X, y14_dot, DUCT_X, ROW_Y - CONNECTOR, COLORS["artery"])
        line(ax, DUCT_X, ROW_Y - CONNECTOR, DUCT_X, HUB16_Y, COLORS["artery"])
        if 15 not in drawn:
            ax.scatter([DUCT_X], [HUB15_Y], s=DOT_SIZE, color=COLORS["artery"], zorder=8)
            label(ax, DUCT_X, HUB15_Y, step_labels.get(15, "15"), COLORS["artery"], LABEL_ROT_DEG, above=True)
            drawn.add(15)
        if 16 not in drawn:
            ax.scatter([DUCT_X], [HUB16_Y], s=DOT_SIZE, color=COLORS["artery"], zorder=8)
            label(ax, DUCT_X, HUB16_Y, step_labels.get(16, "16"), COLORS["artery"], LABEL_ROT_DEG, above=True)
            drawn.add(16)
        continue

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        xu, xv = artery_x.get(u, HEART_X), artery_x.get(v, HEART_X)

        if u in ALL_FTU_STEPS:
            if u == 13:
                continue  
            elif u in {33, 14, 32}:
                continue  
            exit_ftu(u, y, COLORS["artery"], side="top")

        arrow(ax, xu, y, xv, y, COLORS["artery"], head=HEAD_A, shrinkB=0)

        # Enter FTUs from lane
        if v in ALL_FTU_STEPS:
            xftu, yftu = ftu_pos(v)
            line(ax, xv, y, xftu, yftu + CONNECTOR, COLORS["artery"])

        if v in (15, 16):
            drop_to(v, y, COLORS["artery"])

    # Draw dots/labels on lanes
    for s in path:
        if s in ALL_FTU_STEPS or s in drawn:
            continue
        x = artery_x.get(s, HEART_X)
        yy = HUB15_Y if s == 15 else HUB16_Y if s == 16 else y
        ax.scatter([x], [yy], s=DOT_SIZE, color=COLORS["artery"], zorder=8)
        label(ax, x, yy, step_labels.get(s, str(s)), COLORS["artery"], LABEL_ROT_DEG, above=True)
        drawn.add(s)

#Veins draw
for idx, path in enumerate(VEIN_EDGES):
    y = vein_y[idx]
    start = path[0]
    
    # E7: 16 → ... → 24 (hub to liver)
    if start == 16:

        line(ax, DUCT_X, HUB16_Y, vein_x.get(start, DUCT_X), y, COLORS["vein"])
    # E8: 26 → ... → 30 (liver to heart)
    elif start in FTU_LIVER:
        line(ax, LIVER_X, LIVER_Y - CONNECTOR, vein_x.get(start, LIVER_X), y, COLORS["vein"])

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]

        if v in FTU_LIVER:
            # land on the exact dot for v (24/25/26) from the correct side
            liver_dot_offset = {24: +0.12, 25: 0.0, 26: -0.12}
            dot_y = LIVER_Y + liver_dot_offset.get(v, 0.0)

            x_u = vein_x.get(u, LIVER_X)
            coming_from_right = x_u > LIVER_X

            pad = 0.10                     # small gap outside box before turning
            if coming_from_right:
                x_after_box = LIVER_X + (FTU_SIZE / 2) + pad
                arrow(ax, x_u, y, x_after_box, y, COLORS["vein"], head=HEAD_V, shrinkA=0, shrinkB=0)
                line(ax, x_after_box, y, x_after_box, dot_y, COLORS["vein"])
                line(ax, x_after_box, dot_y, LIVER_X, dot_y, COLORS["vein"])
            else:
                x_before_box = LIVER_X - (FTU_SIZE / 2) - pad
                arrow(ax, x_u, y, x_before_box, y, COLORS["vein"], head=HEAD_V, shrinkA=0, shrinkB=0)
                line(ax, x_before_box, y, x_before_box, dot_y, COLORS["vein"])
                line(ax, x_before_box, dot_y, LIVER_X, dot_y, COLORS["vein"])
        else:
            arrow(ax, vein_x.get(u, LIVER_X), y, vein_x.get(v, HEART_X), y,
                COLORS["vein"], head=HEAD_V, shrinkA=0, shrinkB=0)


    # Final connection to heart
    if path[-1] == 30:
        arrow(ax, vein_x.get(30, HEART_X), y, HEART_X + 0.06, BASE_Y,
              COLORS["vein"], head=HEAD_V, shrinkA=0, shrinkB=0)

    # Draw vein dots/labels
    for s in path:
        if s in ALL_FTU_STEPS or s in drawn:
            continue
        x = vein_x.get(s, HEART_X)
        ax.scatter([x], [y], s=DOT_SIZE, color=COLORS["vein"], zorder=8)
        label(ax, x, y, step_labels.get(s, str(s)), COLORS["vein"], LABEL_ROT_DEG, above=False)
        drawn.add(s)

def draw_ftu(ax, cx, cy, name, steps, color, offsets=(+0.12, 0.0, -0.12)):
    """Draw an FTU box with vessel dots inside."""
    ax.add_patch(Rectangle((cx - FTU_SIZE/2, cy - FTU_SIZE/2), FTU_SIZE, FTU_SIZE,
                           facecolor="none", edgecolor="black", lw=FTU_LW, zorder=7))
    ax.text(cx + 0.55, cy, name, va="center", ha="left", fontsize=12, color="black", zorder=10)
    
    for s, dy in zip(sorted(steps), offsets[:len(steps)]):
        yb = cy + dy
        ax.scatter([cx], [yb], s=DOT_SIZE, color=color, zorder=11)
        label(ax, cx, yb, step_labels.get(s, str(s)), color, LABEL_ROT_DEG, above=(color == COLORS["artery"]))

draw_ftu(ax, ACINUS_X, ROW_Y, "acinus", FTU_ACINUS, COLORS["artery"], offsets=(0.0, -0.12))
draw_ftu(ax, DUCT_X, ROW_Y, "duct", FTU_DUCT, COLORS["artery"], offsets=(0.0,))
draw_ftu(ax, ISLET_X, ISLET_Y, "islet of Langerhans", FTU_ISLET, COLORS["artery"], offsets=(-0.12,))

liver_steps_sorted = sorted(FTU_LIVER)
draw_ftu(ax, LIVER_X, LIVER_Y, "liver lobule", FTU_LIVER, COLORS["vein"], offsets=(+0.12, 0.0, -0.12))


for i in range(len(liver_steps_sorted) - 1):
    u, v = liver_steps_sorted[i], liver_steps_sorted[i + 1]
    y_u = LIVER_Y + [+0.12, 0.0, -0.12][i]
    y_v = LIVER_Y + [+0.12, 0.0, -0.12][i + 1]
    line(ax, LIVER_X, y_u, LIVER_X, y_v, COLORS["vein"], z=11)  # Use line, not arrow

tri = RegularPolygon((HEART_X, BASE_Y), numVertices=3, radius=0.48, orientation=0.0,
                     facecolor="black", edgecolor="black", zorder=7)
ax.add_patch(tri)
ax.text(HEART_X - 0.65, BASE_Y, "Heart", va="center", ha="right", fontsize=12, zorder=10)

if not SHOW_LEFT_AXIS:
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

ax.set_title(f"{organ}: Corrected routing — 13→14 right-to-left; 14→15,16 T-junction; liver vessels connected")
ax.set_xlabel("Flow: Heart → Arteries → FTUs → Hubs → Liver → Heart")
ax.set_xlim(HEART_X - 1.2, ISLET_X + 3.0 + EXTRA_RIGHT_PAD)
ax.set_ylim(-2.4, 2.5)
ax.set_xticks([])
ax.grid(axis='x', linestyle='--', alpha=0.20, zorder=1)

# Legend
handles = [
    Line2D([0], [0], color=COLORS["artery"], lw=LINE_W, label="Artery"),
    Line2D([0], [0], color=COLORS["vein"], lw=LINE_W, label="Vein"),
    Line2D([0], [0], marker="s", color="black", markerfacecolor="white", lw=FTU_LW, label="FTU (hollow)"),
    Line2D([0], [0], marker="^", color="black", markerfacecolor="black", lw=0, label="Heart"),
]
ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9)

plt.tight_layout()
OUT_SVG.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_SVG, bbox_inches="tight")
print(f"Saved: {OUT_SVG}")
"""
COMP4097 - Problem 1: Explorative Visualisation
Lunar South Pole Interactive Explorer for Artemis III Landing Site Selection

Part A: Multi-scale interactive maps with click-to-sample, contour overlays,
        illumination layers, and region-of-interest zoom.

Requirements: numpy, matplotlib, Pillow, rasterio (or tifffile as fallback)
Run: python3 problem1.py

Dataset expected at: ./dataset/heightmaps/*.tif  and  ./dataset/illumination/*.tif
Each folder contains 3 images at different spatial scales.
"""

import os
import sys
import glob
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# ---------------------------------------------------------------------------
# Optional dependency handling
# ---------------------------------------------------------------------------
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

if not HAS_RASTERIO:
    try:
        import tifffile
        HAS_TIFFFILE = True
    except ImportError:
        HAS_TIFFFILE = False
    from PIL import Image as PILImage
else:
    HAS_TIFFFILE = False

# ---------------------------------------------------------------------------
# LROC elevation constants
# LROC WAC GLD100: elevation in metres relative to a sphere of 1737.4 km
# South pole typical range: approx -9 000 m to +10 000 m
# ---------------------------------------------------------------------------
LUNAR_RADIUS_KM = 1737.4


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

class LunarDataset:
    """Container for one loaded TIFF: pixel data + spatial metadata."""

    def __init__(self, filepath):
        self.filepath = filepath
        self.name = os.path.basename(filepath)
        self.data = None          # 2-D numpy float64 array
        self.width = 0
        self.height = 0
        self.bounds = None        # (left, bottom, right, top) in CRS units
        self.transform = None     # affine transform (rasterio)
        self.pixel_size = None    # metres per pixel (if known)
        self.scale_label = ""
        self._load()

    # ------------------------------------------------------------------
    def _load(self):
        if HAS_RASTERIO:
            self._load_rasterio()
        elif HAS_TIFFFILE:
            self._load_tifffile()
        else:
            self._load_pil()
        self._clean()

    def _load_rasterio(self):
        with rasterio.open(self.filepath) as src:
            self.data = src.read(1).astype(np.float64)
            self.height, self.width = self.data.shape
            self.bounds = src.bounds          # BoundingBox(left, bottom, right, top)
            self.transform = src.transform
            nodata = src.nodata
            if nodata is not None:
                self.data[self.data == nodata] = np.nan
            # compute pixel size from transform
            self.pixel_size = abs(self.transform.a)  # metres per pixel

    def _load_tifffile(self):
        import tifffile
        self.data = tifffile.imread(self.filepath).astype(np.float64)
        if self.data.ndim == 3:
            self.data = self.data[:, :, 0]
        self.height, self.width = self.data.shape

    def _load_pil(self):
        img = PILImage.open(self.filepath)
        self.data = np.array(img, dtype=np.float64)
        if self.data.ndim == 3:
            self.data = self.data[:, :, 0]
        self.height, self.width = self.data.shape

    def _clean(self):
        """Replace extreme sentinel values with NaN."""
        if np.any(np.abs(self.data) > 1e15):
            self.data[np.abs(self.data) > 1e15] = np.nan

    # ------------------------------------------------------------------
    @property
    def extent(self):
        """Return [left, right, bottom, top] for matplotlib imshow."""
        if self.bounds is not None:
            return [self.bounds.left, self.bounds.right,
                    self.bounds.bottom, self.bounds.top]
        return [0, self.width, self.height, 0]

    @property
    def coverage_area(self):
        """Approximate area covered in m^2 (for sorting by scale)."""
        if self.bounds is not None:
            dx = self.bounds.right - self.bounds.left
            dy = self.bounds.top - self.bounds.bottom
            return abs(dx * dy)
        return self.width * self.height  # fallback: pixel area

    def sample(self, x, y):
        """Return scalar value at spatial coordinate (x, y), or NaN."""
        row, col = self._xy_to_rowcol(x, y)
        if 0 <= row < self.height and 0 <= col < self.width:
            return float(self.data[row, col])
        return float("nan")

    def _xy_to_rowcol(self, x, y):
        if self.transform is not None:
            inv = ~self.transform
            col, row = inv * (x, y)
            return int(round(row)), int(round(col))
        return int(round(y)), int(round(x))


class DataStore:
    """Discovers, loads and organises the full dataset."""

    SCALE_LABELS = [
        "Regional overview",
        "Intermediate",
        "Detailed close-up",
    ]

    def __init__(self, root_dir="dataset"):
        self.root = root_dir
        self.heightmaps: list[LunarDataset] = []
        self.illumination: list[LunarDataset] = []
        self._load_folder("heightmaps", self.heightmaps)
        self._load_folder("illumination", self.illumination)
        self._sort_and_label(self.heightmaps)
        self._sort_and_label(self.illumination)
        self._print_summary()

    def _find_tiffs(self, subfolder):
        path = os.path.join(self.root, subfolder)
        files = []
        for ext in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
            files.extend(glob.glob(os.path.join(path, ext)))
        files.sort()
        return files

    def _load_folder(self, subfolder, target_list):
        for fp in self._find_tiffs(subfolder):
            try:
                ds = LunarDataset(fp)
                target_list.append(ds)
            except Exception as exc:
                print(f"[WARN] Could not load {fp}: {exc}")

    def _sort_and_label(self, datasets):
        # Sort largest coverage area first → regional overview first
        datasets.sort(key=lambda d: d.coverage_area, reverse=True)
        for i, ds in enumerate(datasets):
            ds.scale_label = self.SCALE_LABELS[i] if i < len(self.SCALE_LABELS) else f"Scale {i+1}"

    def _print_summary(self):
        print("=" * 60)
        print("LUNAR SOUTH POLE DATASET")
        print("=" * 60)
        for tag, lst in [("Heightmaps", self.heightmaps),
                         ("Illumination", self.illumination)]:
            print(f"\n{tag}:")
            for ds in lst:
                rng = f"{np.nanmin(ds.data):.1f} .. {np.nanmax(ds.data):.1f}"
                print(f"  [{ds.scale_label}] {ds.name}  "
                      f"{ds.width}×{ds.height}  range={rng}")
        print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

class LunarExplorer:
    """Tkinter application: interactive multi-scale lunar south-pole explorer."""

    COLOURMAPS = [
        "terrain", "viridis", "cividis", "inferno",
        "gray", "coolwarm", "RdYlBu_r", "gist_earth",
    ]

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Lunar South Pole Explorer  —  Artemis III")
        self.root.geometry("1450x920")
        self.root.minsize(1000, 650)

        # ── load data ─────────────────────────────────────────────────
        self.store = DataStore()
        if not self.store.heightmaps:
            messagebox.showerror(
                "Dataset not found",
                "No .tif files found in  dataset/heightmaps/\n\n"
                "Please place the LROC TIFF files in:\n"
                "  ./dataset/heightmaps/\n"
                "  ./dataset/illumination/"
            )
            sys.exit(1)

        # ── state ─────────────────────────────────────────────────────
        self.scale_idx = 0
        self.zoom_stack: list[tuple] = []   # previous (xlim, ylim) pairs
        self.cur_xlim = None
        self.cur_ylim = None

        # ── build UI ─────────────────────────────────────────────────
        self._build_gui()
        self._render()

    # ===================================================================
    #  GUI CONSTRUCTION
    # ===================================================================

    def _build_gui(self):
        # ── top-level split: sidebar | map ────────────────────────────
        pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True)

        sidebar = ttk.Frame(pane, width=290)
        pane.add(sidebar, weight=0)

        map_area = ttk.Frame(pane)
        pane.add(map_area, weight=1)

        self._build_sidebar(sidebar)
        self._build_map(map_area)

    # ── sidebar ──────────────────────────────────────────────────────
    def _build_sidebar(self, parent):
        # -- Title --
        ttk.Label(parent, text="Map Controls",
                  font=("Helvetica", 14, "bold")).pack(pady=(12, 4), padx=10)

        # -- Spatial scale --
        frm = ttk.LabelFrame(parent, text="Spatial Scale", padding=6)
        frm.pack(fill=tk.X, padx=10, pady=4)

        self.scale_var = tk.IntVar(value=0)
        for i, ds in enumerate(self.store.heightmaps):
            ttk.Radiobutton(
                frm, text=f"{ds.scale_label}  ({ds.width}×{ds.height})",
                variable=self.scale_var, value=i,
                command=self._on_scale_change,
            ).pack(anchor=tk.W, pady=1)

        # -- Colour map --
        frm = ttk.LabelFrame(parent, text="Elevation Colour Map", padding=6)
        frm.pack(fill=tk.X, padx=10, pady=4)

        self.cmap_var = tk.StringVar(value="terrain")
        cb = ttk.Combobox(frm, textvariable=self.cmap_var,
                          values=self.COLOURMAPS, state="readonly", width=18)
        cb.pack(fill=tk.X)
        cb.bind("<<ComboboxSelected>>", lambda _: self._render())

        # -- Contour isolines --
        frm = ttk.LabelFrame(parent, text="Contour Lines (Isolines)", padding=6)
        frm.pack(fill=tk.X, padx=10, pady=4)

        self.contour_on = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm, text="Show contours",
                        variable=self.contour_on,
                        command=self._render).pack(anchor=tk.W)

        ttk.Label(frm, text="Interval (m):").pack(anchor=tk.W, pady=(6, 0))
        self.interval_var = tk.IntVar(value=500)
        self.interval_scale = ttk.Scale(
            frm, from_=50, to=2000,
            variable=self.interval_var, orient=tk.HORIZONTAL,
            command=self._on_interval_slide,
        )
        self.interval_scale.pack(fill=tk.X)
        self.interval_lbl = ttk.Label(frm, text="500 m")
        self.interval_lbl.pack(anchor=tk.W)

        # -- Illumination overlay --
        frm = ttk.LabelFrame(parent, text="Illumination Overlay", padding=6)
        frm.pack(fill=tk.X, padx=10, pady=4)

        self.illum_on = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm, text="Show illumination layer",
                        variable=self.illum_on,
                        command=self._render).pack(anchor=tk.W)

        ttk.Label(frm, text="Opacity:").pack(anchor=tk.W, pady=(6, 0))
        self.opacity_var = tk.DoubleVar(value=0.45)
        ttk.Scale(frm, from_=0.1, to=0.9,
                  variable=self.opacity_var, orient=tk.HORIZONTAL,
                  command=lambda _: self._render()).pack(fill=tk.X)

        # illumination colourmap selector
        ttk.Label(frm, text="Illumination cmap:").pack(anchor=tk.W, pady=(4, 0))
        self.illum_cmap_var = tk.StringVar(value="hot")
        ttk.Combobox(frm, textvariable=self.illum_cmap_var,
                     values=["hot", "YlOrRd", "magma", "inferno", "gray_r"],
                     state="readonly", width=14).pack(fill=tk.X)

        # -- Navigation --
        frm = ttk.LabelFrame(parent, text="Navigation", padding=6)
        frm.pack(fill=tk.X, padx=10, pady=4)

        ttk.Label(frm,
                  text="Left-click  → sample values\n"
                       "Right-drag → zoom into region",
                  font=("Helvetica", 9), foreground="#555").pack(anchor=tk.W)
        ttk.Button(frm, text="Reset Zoom",
                   command=self._reset_zoom).pack(fill=tk.X, pady=(6, 2))
        ttk.Button(frm, text="← Zoom Back",
                   command=self._zoom_back).pack(fill=tk.X, pady=2)

        # -- Sampled-value readout --
        frm = ttk.LabelFrame(parent, text="Sampled Point", padding=6)
        frm.pack(fill=tk.X, padx=10, pady=4)

        self.sample_text = tk.Text(frm, height=7, width=32,
                                   font=("Courier", 9), state=tk.DISABLED,
                                   bg="#f5f5f0", relief=tk.FLAT)
        self.sample_text.pack(fill=tk.X)

        # -- Dataset statistics --
        frm = ttk.LabelFrame(parent, text="Current Dataset Stats", padding=6)
        frm.pack(fill=tk.X, padx=10, pady=4)

        self.stats_text = tk.Text(frm, height=5, width=32,
                                  font=("Courier", 9), state=tk.DISABLED,
                                  bg="#f5f5f0", relief=tk.FLAT)
        self.stats_text.pack(fill=tk.X)
        self._refresh_stats()

    # ── map canvas ───────────────────────────────────────────────────
    def _build_map(self, parent):
        self.fig = Figure(figsize=(10, 8), dpi=100, facecolor="#f0f0f0")
        self.ax = self.fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Navigation toolbar (pan, zoom, save)
        tb_frame = ttk.Frame(parent)
        tb_frame.pack(fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, tb_frame)

        # Left-click → sample values
        self.canvas.mpl_connect("button_press_event", self._on_click)

        # Right-drag rectangle → zoom into region
        self.selector = RectangleSelector(
            self.ax, self._on_rect_select,
            useblit=True,
            button=[3],           # right mouse button
            minspanx=5, minspany=5,
            spancoords="pixels",
            interactive=True,
            props=dict(facecolor="cyan", edgecolor="white",
                       alpha=0.25, linewidth=1.5),
        )

        self.cbar = None  # will hold colorbar reference

    # ===================================================================
    #  RENDERING
    # ===================================================================

    def _cur_hm(self) -> LunarDataset:
        return self.store.heightmaps[self.scale_idx]

    def _cur_il(self):
        idx = self.scale_idx
        if idx < len(self.store.illumination):
            return self.store.illumination[idx]
        return None

    def _render(self, *_args):
        """Redraw the map with all active layers."""
        ax = self.ax
        ax.clear()

        hm = self._cur_hm()
        data = hm.data
        ext = hm.extent
        vmin, vmax = float(np.nanmin(data)), float(np.nanmax(data))

        # ── 1. Colour-mapped elevation raster ────────────────────────
        self.im = ax.imshow(
            data, cmap=self.cmap_var.get(),
            extent=ext, origin="upper",
            vmin=vmin, vmax=vmax, aspect="equal",
            interpolation="bilinear",
        )

        # ── 2. Contour overlay ───────────────────────────────────────
        if self.contour_on.get():
            interval = max(int(self.interval_var.get()), 10)
            lo = np.floor(vmin / interval) * interval
            hi = np.ceil(vmax / interval) * interval + interval
            levels = np.arange(lo, hi, interval)
            # cap to avoid sluggishness on dense contours
            if len(levels) > 60:
                levels = np.linspace(vmin, vmax, 40)

            h, w = data.shape
            xs = np.linspace(ext[0], ext[1], w)
            ys = np.linspace(ext[3], ext[2], h)   # top→bottom
            X, Y = np.meshgrid(xs, ys)

            # subsample for performance on large grids
            step = max(1, min(h, w) // 800)
            Xs, Ys, Ds = X[::step, ::step], Y[::step, ::step], data[::step, ::step]

            try:
                cs = ax.contour(Xs, Ys, Ds, levels=levels,
                                colors="black", linewidths=0.45, alpha=0.55)
                if len(levels) <= 25:
                    ax.clabel(cs, inline=True, fontsize=6, fmt="%.0f m")
            except Exception:
                pass  # contour can fail on all-NaN slices

        # ── 3. Illumination overlay ──────────────────────────────────
        if self.illum_on.get():
            il = self._cur_il()
            if il is not None:
                il_data = il.data.copy()
                lo_i, hi_i = np.nanmin(il_data), np.nanmax(il_data)
                if hi_i > lo_i:
                    il_norm = (il_data - lo_i) / (hi_i - lo_i)
                else:
                    il_norm = np.zeros_like(il_data)
                ax.imshow(
                    il_norm, cmap=self.illum_cmap_var.get(),
                    extent=il.extent, origin="upper",
                    alpha=float(self.opacity_var.get()), aspect="equal",
                )

        # ── 4. Zoom limits ───────────────────────────────────────────
        if self.cur_xlim is not None:
            ax.set_xlim(self.cur_xlim)
            ax.set_ylim(self.cur_ylim)

        # ── 5. Labels / colour bar ───────────────────────────────────
        ax.set_title(
            f"Lunar South Pole  —  {hm.scale_label}",
            fontsize=13, fontweight="bold", pad=10,
        )
        if hm.bounds is not None:
            ax.set_xlabel("Easting (m)")
            ax.set_ylabel("Northing (m)")
        else:
            ax.set_xlabel("Pixel X")
            ax.set_ylabel("Pixel Y")

        # refresh colour bar
        if self.cbar is not None:
            try:
                self.cbar.remove()
            except Exception:
                pass
        self.cbar = self.fig.colorbar(
            self.im, ax=ax, label="Elevation (m)",
            shrink=0.82, pad=0.02, aspect=30,
        )

        self.fig.tight_layout()
        self.canvas.draw_idle()

    # ===================================================================
    #  INTERACTION HANDLERS
    # ===================================================================

    def _on_click(self, event):
        """Left-click → sample elevation (& illumination) at pointer."""
        if event.inaxes != self.ax:
            return
        if event.button != 1:
            return
        # skip if matplotlib toolbar is in zoom/pan mode
        if self.toolbar.mode:
            return

        x, y = event.xdata, event.ydata
        hm = self._cur_hm()
        elev = hm.sample(x, y)

        lines = []
        if hm.bounds is not None:
            lines.append(f"X (east) : {x:>12.1f} m")
            lines.append(f"Y (north): {y:>12.1f} m")
        else:
            lines.append(f"Pixel X: {x:.0f}")
            lines.append(f"Pixel Y: {y:.0f}")

        lines.append(f"Elevation: {elev:>10.1f} m")
        lines.append(f"           ({elev/1000:.2f} km)")

        il = self._cur_il()
        if il is not None:
            iv = il.sample(x, y)
            lines.append(f"Illumin. : {iv:>10.2f}")
            # interpret as percentage if range suggests it
            il_max = float(np.nanmax(il.data))
            if il_max <= 1.01:
                lines.append(f"           ({iv*100:.1f} %)")
            elif il_max <= 100.1:
                lines.append(f"           ({iv:.1f} %)")

        self._set_textbox(self.sample_text, "\n".join(lines))

        # draw marker
        for art in list(ax_art for ax_art in self.ax.get_children()
                        if getattr(ax_art, "_sample_marker", False)):
            art.remove()
        mk, = self.ax.plot(x, y, "r+", markersize=16, markeredgewidth=2.5)
        mk._sample_marker = True
        self.canvas.draw_idle()

    def _on_rect_select(self, eclick, erelease):
        """Right-drag rectangle → zoom into selected region."""
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        if x1 is None or x2 is None:
            return

        # push current limits onto stack
        self.zoom_stack.append(
            (self.ax.get_xlim(), self.ax.get_ylim())
        )
        self.cur_xlim = (min(x1, x2), max(x1, x2))
        self.cur_ylim = (min(y1, y2), max(y1, y2))
        self._render()

    def _reset_zoom(self):
        self.zoom_stack.clear()
        self.cur_xlim = None
        self.cur_ylim = None
        self._render()

    def _zoom_back(self):
        if self.zoom_stack:
            self.cur_xlim, self.cur_ylim = self.zoom_stack.pop()
        else:
            self.cur_xlim = None
            self.cur_ylim = None
        self._render()

    def _on_scale_change(self):
        self.scale_idx = self.scale_var.get()
        self.zoom_stack.clear()
        self.cur_xlim = None
        self.cur_ylim = None
        self._refresh_stats()
        self._render()

    def _on_interval_slide(self, _val):
        v = int(float(_val))
        self.interval_lbl.config(text=f"{v} m")
        # only re-render if contours are visible
        if self.contour_on.get():
            self._render()

    # ===================================================================
    #  HELPERS
    # ===================================================================

    def _refresh_stats(self):
        hm = self._cur_hm()
        d = hm.data
        lines = [
            f"File : {hm.name}",
            f"Size : {hm.width} × {hm.height} px",
            f"Elev : {np.nanmin(d):.0f} .. {np.nanmax(d):.0f} m",
            f"Mean : {np.nanmean(d):.0f} m",
            f"Std  : {np.nanstd(d):.0f} m",
        ]
        if hm.pixel_size is not None:
            lines.append(f"Res  : {hm.pixel_size:.1f} m/px")
        self._set_textbox(self.stats_text, "\n".join(lines))

    @staticmethod
    def _set_textbox(widget, text):
        widget.config(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, text)
        widget.config(state=tk.DISABLED)


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    root = tk.Tk()

    # Apply a ttk theme for a cleaner look
    style = ttk.Style(root)
    available = style.theme_names()
    for preferred in ("clam", "alt", "default"):
        if preferred in available:
            style.theme_use(preferred)
            break

    app = LunarExplorer(root)
    root.mainloop()


if __name__ == "__main__":
    main()

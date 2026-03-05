"""
COMP4097 - Problem 1: Explorative Visualisation
Lunar South Pole Interactive Explorer for Artemis III Landing Site Selection

Parts A, B & C:
  A – Multi-scale interactive maps, click-to-sample, zoom, illumination overlay
  B – Side-by-side comparison of 3D displacement-map rendering vs isoline/
      isocontour representation, with full GUI parameter control
  C – Creative additions: slope/gradient overlay, permanently shadowed region
      (PSR) detection, composite landing suitability scoring heatmap,
      elevation profile cross-sections

Lecture concepts applied:
  - Visualisation pipeline  (data -> format -> filter -> map -> render)
  - Colour mapping & colourmap design  (perceptual uniformity, diverging scales)
  - Height / displacement plots  (S_displ(x) = x + n(x)*f(x))
  - Contouring / isolines  (marching squares, isovalue control)
  - Colour banding <-> contour duality
  - Inverse mapping  (click-to-sample)
  - Multi-scale overview -> detail
  - Multi-variate overlay  (elevation + illumination)
  - Derived scalar quantities from gradient  (slope = |grad f|)
  - Information layering  (PSR regions, suitability scoring)
  - Filtering / slicing  (1D cross-sections through 2D scalar fields)

Requirements: numpy, matplotlib, scipy, Pillow, rasterio (or tifffile fallback)
Run:  python3 problem1.py
Dataset:  ./dataset/heightmaps/*.tif   ./dataset/illumination/*.tif
"""

import os, sys, glob
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D          # 3-D displacement plots
from scipy import ndimage                         # Part C: slope / gradient

# -- optional I/O backends --
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

LUNAR_RADIUS_KM = 1737.4


# ============================================================================
#  DATA LOADING  (pipeline stage 1: acquisition / import)
# ============================================================================

class LunarDataset:
    """One loaded TIFF: pixel array + spatial metadata."""

    def __init__(self, filepath):
        self.filepath = filepath
        self.name = os.path.basename(filepath)
        self.data = None
        self.width = self.height = 0
        self.bounds = None
        self.transform = None
        self.pixel_size = None
        self.scale_label = ""
        self._load()

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
            self.bounds = src.bounds
            self.transform = src.transform
            nodata = src.nodata
            if nodata is not None:
                self.data[self.data == nodata] = np.nan
            self.pixel_size = abs(self.transform.a)

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
        if np.any(np.abs(self.data) > 1e15):
            self.data[np.abs(self.data) > 1e15] = np.nan

    @property
    def extent(self):
        """[left, right, bottom, top] for imshow."""
        if self.bounds is not None:
            return [self.bounds.left, self.bounds.right,
                    self.bounds.bottom, self.bounds.top]
        return [0, self.width, self.height, 0]

    @property
    def coverage_area(self):
        if self.bounds is not None:
            dx = self.bounds.right - self.bounds.left
            dy = self.bounds.top - self.bounds.bottom
            return abs(dx * dy)
        return self.width * self.height

    def sample(self, x, y):
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
    """Discovers, loads, and sorts the full dataset."""

    SCALE_LABELS = ["Regional overview", "Intermediate", "Detailed close-up"]

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
        # Deduplicate (Windows glob is case-insensitive, so *.tif and *.TIF
        # can return the same file twice)
        seen = set()
        unique = []
        for f in sorted(files):
            normpath = os.path.normcase(os.path.abspath(f))
            if normpath not in seen:
                seen.add(normpath)
                unique.append(f)
        return unique

    def _load_folder(self, subfolder, target):
        for fp in self._find_tiffs(subfolder):
            try:
                target.append(LunarDataset(fp))
            except Exception as exc:
                print(f"[WARN] Could not load {fp}: {exc}")

    def _sort_and_label(self, datasets):
        datasets.sort(key=lambda d: d.coverage_area, reverse=True)
        for i, ds in enumerate(datasets):
            ds.scale_label = (self.SCALE_LABELS[i]
                              if i < len(self.SCALE_LABELS)
                              else f"Scale {i+1}")

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
                      f"{ds.width}x{ds.height}  range={rng}")
        print("=" * 60)


# ============================================================================
#  MAIN APPLICATION
# ============================================================================

class LunarExplorer:
    """Tkinter application: Parts A & B combined."""

    COLOURMAPS = [
        # Elevation is a sequential quantity: prefer sequential maps over
        # diverging schemes such as coolwarm.
        "terrain", "viridis", "cividis", "inferno",
        "magma", "plasma", "gray", "gist_earth",
    ]
    VIEW_MODES = [
        "2D Contour Map",
        "3D Displacement Map",
        "Side-by-Side Comparison",
    ]

    # 3-D surface sub-sampling cap (max grid points per axis)
    SURF_MAX = 400

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Lunar South Pole Explorer  -  Artemis III")
        self.root.geometry("1500x950")
        self.root.minsize(1060, 700)

        # -- load data --
        self.store = DataStore()
        if not self.store.heightmaps:
            messagebox.showerror(
                "Dataset not found",
                "No .tif files found in  dataset/heightmaps/\n\n"
                "Place the LROC TIFFs in:\n"
                "  ./dataset/heightmaps/\n  ./dataset/illumination/")
            sys.exit(1)

        # -- state --
        self.scale_idx = 0
        self.zoom_stack: list[tuple] = []
        self.cur_xlim = self.cur_ylim = None
        self._press_xy = None

        # Part C: profile cross-section state
        self.profile_pts: list[tuple] = []   # [(x1,y1), (x2,y2)]

        # Part C: cached derived layers (invalidated on scale change)
        self._cached_slope = None
        self._cached_slope_idx = -1
        self._cached_suit = None
        self._cached_suit_idx = -1
        self._pending_render_job = None

        # -- build UI & first render --
        self._build_gui()
        self._render()

    # ==================================================================
    #  GUI CONSTRUCTION
    # ==================================================================

    def _build_gui(self):
        pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True)

        # scrollable sidebar
        sidebar_outer = ttk.Frame(pane, width=300)
        pane.add(sidebar_outer, weight=0)

        canvas_sb = tk.Canvas(sidebar_outer, width=285, highlightthickness=0)
        scrollbar = ttk.Scrollbar(sidebar_outer, orient=tk.VERTICAL,
                                  command=canvas_sb.yview)
        self.sidebar = ttk.Frame(canvas_sb)

        self.sidebar.bind(
            "<Configure>",
            lambda e: canvas_sb.configure(scrollregion=canvas_sb.bbox("all")))
        canvas_sb.create_window((0, 0), window=self.sidebar, anchor="nw")
        canvas_sb.configure(yscrollcommand=scrollbar.set)

        canvas_sb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        map_area = ttk.Frame(pane)
        pane.add(map_area, weight=1)

        self._build_sidebar(self.sidebar)
        self._build_map(map_area)

    # -- SIDEBAR -----------------------------------------------------------
    def _build_sidebar(self, parent):
        PAD = dict(padx=8, pady=3)

        ttk.Label(parent, text="Artemis III - Map Controls",
                  font=("Helvetica", 13, "bold")).pack(pady=(10, 2), padx=8)

        # -- View Mode (Part B: comparison toggle) -------------------------
        frm = ttk.LabelFrame(parent, text="View Mode", padding=5)
        frm.pack(fill=tk.X, **PAD)

        self.view_var = tk.StringVar(value=self.VIEW_MODES[0])
        for mode in self.VIEW_MODES:
            ttk.Radiobutton(frm, text=mode, variable=self.view_var,
                            value=mode, command=self._render
                            ).pack(anchor=tk.W, pady=1)

        # -- Spatial Scale -------------------------------------------------
        frm = ttk.LabelFrame(parent, text="Spatial Scale", padding=5)
        frm.pack(fill=tk.X, **PAD)

        self.scale_var = tk.IntVar(value=0)
        for i, ds in enumerate(self.store.heightmaps):
            ttk.Radiobutton(
                frm, text=f"{ds.scale_label}  ({ds.width}x{ds.height})",
                variable=self.scale_var, value=i,
                command=self._on_scale_change,
            ).pack(anchor=tk.W, pady=1)

        # -- Colour Map ----------------------------------------------------
        frm = ttk.LabelFrame(parent, text="Elevation Colour Map", padding=5)
        frm.pack(fill=tk.X, **PAD)

        self.cmap_var = tk.StringVar(value="terrain")
        cb = ttk.Combobox(frm, textvariable=self.cmap_var,
                          values=self.COLOURMAPS, state="readonly", width=16)
        cb.pack(fill=tk.X)
        cb.bind("<<ComboboxSelected>>", lambda _: self._render())

        # -- Contour / Isoline controls ------------------------------------
        frm = ttk.LabelFrame(parent, text="Contour Lines (Isolines)", padding=5)
        frm.pack(fill=tk.X, **PAD)

        self.contour_on = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm, text="Show contours",
                        variable=self.contour_on,
                        command=self._render).pack(anchor=tk.W)

        ttk.Label(frm, text="Contour interval (m):").pack(anchor=tk.W, pady=(4, 0))
        self.interval_var = tk.IntVar(value=500)
        self.interval_scale = ttk.Scale(
            frm, from_=50, to=2000,
            variable=self.interval_var, orient=tk.HORIZONTAL,
            command=self._on_interval_slide)
        self.interval_scale.pack(fill=tk.X)
        self.interval_lbl = ttk.Label(frm, text="500 m")
        self.interval_lbl.pack(anchor=tk.W)

        ttk.Label(frm, text="Line width:").pack(anchor=tk.W, pady=(4, 0))
        self.contour_lw_var = tk.DoubleVar(value=0.5)
        ttk.Scale(frm, from_=0.2, to=3.0,
                  variable=self.contour_lw_var, orient=tk.HORIZONTAL,
                  command=lambda _: self._render()).pack(fill=tk.X)

        ttk.Label(frm, text="Line colour:").pack(anchor=tk.W, pady=(4, 0))
        self.contour_col_var = tk.StringVar(value="black")
        ttk.Combobox(frm, textvariable=self.contour_col_var,
                     values=["black", "white", "red", "blue",
                             "yellow", "cyan", "magenta"],
                     state="readonly", width=10).pack(fill=tk.X)

        self.contour_labels_on = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm, text="Label contour values",
                        variable=self.contour_labels_on,
                        command=self._render).pack(anchor=tk.W, pady=(3, 0))

        # -- 3-D Displacement-Map controls (Part B) ------------------------
        frm = ttk.LabelFrame(parent, text="3D Displacement Map", padding=5)
        frm.pack(fill=tk.X, **PAD)

        ttk.Label(frm, text="Azimuth (deg):").pack(anchor=tk.W)
        self.azi_var = tk.IntVar(value=-135)
        self.azi_scale = ttk.Scale(frm, from_=-180, to=180,
                                   variable=self.azi_var, orient=tk.HORIZONTAL,
                                   command=lambda _: self._on_3d_param("azi"))
        self.azi_scale.pack(fill=tk.X)
        self.azi_lbl = ttk.Label(frm, text="-135 deg")
        self.azi_lbl.pack(anchor=tk.W)

        ttk.Label(frm, text="Elevation angle (deg):").pack(anchor=tk.W, pady=(4, 0))
        self.elev_var = tk.IntVar(value=45)
        self.elev_scale = ttk.Scale(frm, from_=5, to=90,
                                    variable=self.elev_var, orient=tk.HORIZONTAL,
                                    command=lambda _: self._on_3d_param("elev"))
        self.elev_scale.pack(fill=tk.X)
        self.elev_lbl = ttk.Label(frm, text="45 deg")
        self.elev_lbl.pack(anchor=tk.W)

        ttk.Label(frm, text="Vertical exaggeration:").pack(anchor=tk.W, pady=(4, 0))
        self.vexag_var = tk.DoubleVar(value=3.0)
        self.vexag_scale = ttk.Scale(frm, from_=0.5, to=15.0,
                                     variable=self.vexag_var, orient=tk.HORIZONTAL,
                                     command=lambda _: self._on_3d_param("vexag"))
        self.vexag_scale.pack(fill=tk.X)
        self.vexag_lbl = ttk.Label(frm, text="3.0x")
        self.vexag_lbl.pack(anchor=tk.W)

        self.shade_3d = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm, text="Surface shading (light source)",
                        variable=self.shade_3d,
                        command=self._render).pack(anchor=tk.W, pady=(3, 0))

        self.contour_3d_on = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm, text="Project contours onto 3D surface",
                        variable=self.contour_3d_on,
                        command=self._render).pack(anchor=tk.W, pady=(1, 0))

        # -- Illumination overlay ------------------------------------------
        frm = ttk.LabelFrame(parent, text="Illumination Overlay", padding=5)
        frm.pack(fill=tk.X, **PAD)

        self.illum_on = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm, text="Show illumination layer",
                        variable=self.illum_on,
                        command=self._render).pack(anchor=tk.W)

        ttk.Label(frm, text="Opacity:").pack(anchor=tk.W, pady=(4, 0))
        self.opacity_var = tk.DoubleVar(value=0.45)
        ttk.Scale(frm, from_=0.1, to=0.9, variable=self.opacity_var,
                  orient=tk.HORIZONTAL,
                  command=lambda _: self._render()).pack(fill=tk.X)

        ttk.Label(frm, text="Illumination cmap:").pack(anchor=tk.W, pady=(4, 0))
        self.illum_cmap_var = tk.StringVar(value="hot")
        ttk.Combobox(frm, textvariable=self.illum_cmap_var,
                     values=["hot", "YlOrRd", "magma", "inferno", "gray_r"],
                     state="readonly", width=14).pack(fill=tk.X)

        # ==============================================================
        #  PART C — Creative Analysis Overlays
        # ==============================================================

        # -- C1: Slope / Gradient overlay ----------------------------------
        frm = ttk.LabelFrame(parent,
                             text="Slope / Gradient  (Part C)", padding=5)
        frm.pack(fill=tk.X, **PAD)

        self.slope_on = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm, text="Show slope overlay",
                        variable=self.slope_on,
                        command=self._render).pack(anchor=tk.W)

        ttk.Label(frm, text="Danger threshold (deg):").pack(
            anchor=tk.W, pady=(4, 0))
        self.slope_thresh_var = tk.DoubleVar(value=15.0)
        ttk.Scale(frm, from_=2, to=45,
                  variable=self.slope_thresh_var, orient=tk.HORIZONTAL,
                  command=lambda _: self._render()).pack(fill=tk.X)

        ttk.Label(frm, text="Slope opacity:").pack(anchor=tk.W, pady=(4, 0))
        self.slope_alpha_var = tk.DoubleVar(value=0.5)
        ttk.Scale(frm, from_=0.1, to=0.9,
                  variable=self.slope_alpha_var, orient=tk.HORIZONTAL,
                  command=lambda _: self._render()).pack(fill=tk.X)

        # -- C2: Permanently Shadowed Regions (PSR) indicators -------------
        frm = ttk.LabelFrame(parent,
                             text="Shadowed Regions / PSR  (Part C)", padding=5)
        frm.pack(fill=tk.X, **PAD)

        self.psr_on = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm, text="Highlight permanently shadowed regions",
                        variable=self.psr_on,
                        command=self._render).pack(anchor=tk.W)

        ttk.Label(frm, text="Shadow threshold (percentile):").pack(
            anchor=tk.W, pady=(4, 0))
        self.psr_thresh_var = tk.DoubleVar(value=10.0)
        ttk.Scale(frm, from_=1, to=40,
                  variable=self.psr_thresh_var, orient=tk.HORIZONTAL,
                  command=lambda _: self._render()).pack(fill=tk.X)

        # -- C3: Landing Suitability Score ---------------------------------
        frm = ttk.LabelFrame(parent,
                             text="Landing Suitability Score  (Part C)",
                             padding=5)
        frm.pack(fill=tk.X, **PAD)

        self.suit_on = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm, text="Show suitability heatmap",
                        variable=self.suit_on,
                        command=self._render).pack(anchor=tk.W)

        ttk.Label(frm, text="Weight — flat terrain:").pack(
            anchor=tk.W, pady=(4, 0))
        self.w_flat_var = tk.DoubleVar(value=0.4)
        ttk.Scale(frm, from_=0, to=1,
                  variable=self.w_flat_var, orient=tk.HORIZONTAL,
                  command=lambda _: self._render()).pack(fill=tk.X)

        ttk.Label(frm, text="Weight — illumination:").pack(
            anchor=tk.W, pady=(4, 0))
        self.w_illum_var = tk.DoubleVar(value=0.35)
        ttk.Scale(frm, from_=0, to=1,
                  variable=self.w_illum_var, orient=tk.HORIZONTAL,
                  command=lambda _: self._render()).pack(fill=tk.X)

        ttk.Label(frm, text="Weight — PSR proximity:").pack(
            anchor=tk.W, pady=(4, 0))
        self.w_psr_var = tk.DoubleVar(value=0.25)
        ttk.Scale(frm, from_=0, to=1,
                  variable=self.w_psr_var, orient=tk.HORIZONTAL,
                  command=lambda _: self._render()).pack(fill=tk.X)

        ttk.Label(frm, text="Suitability opacity:").pack(
            anchor=tk.W, pady=(4, 0))
        self.suit_alpha_var = tk.DoubleVar(value=0.55)
        ttk.Scale(frm, from_=0.1, to=0.9,
                  variable=self.suit_alpha_var, orient=tk.HORIZONTAL,
                  command=lambda _: self._render()).pack(fill=tk.X)

        # -- C4: Elevation Profile Cross-Section ---------------------------
        frm = ttk.LabelFrame(parent,
                             text="Elevation Profile  (Part C)", padding=5)
        frm.pack(fill=tk.X, **PAD)

        self.profile_mode = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm, text="Profile mode (click 2 pts on map)",
                        variable=self.profile_mode).pack(anchor=tk.W)
        ttk.Button(frm, text="Clear Profile",
                   command=self._clear_profile).pack(fill=tk.X, pady=(4, 0))

        # -- Navigation ----------------------------------------------------
        frm = ttk.LabelFrame(parent, text="Navigation", padding=5)
        frm.pack(fill=tk.X, **PAD)

        ttk.Label(frm,
                  text="Left-click  -> sample values\n"
                       "  (or set profile pts)\n"
                       "Left-drag  -> zoom into region\n"
                       "Right-drag -> zoom into region\n"
                       "(2D view only)",
                  font=("Helvetica", 9), foreground="#555").pack(anchor=tk.W)
        ttk.Button(frm, text="Reset Zoom",
                   command=self._reset_zoom).pack(fill=tk.X, pady=(5, 2))
        ttk.Button(frm, text="<- Zoom Back",
                   command=self._zoom_back).pack(fill=tk.X, pady=2)

        # -- Sampled-value readout -----------------------------------------
        frm = ttk.LabelFrame(parent, text="Sampled Point", padding=5)
        frm.pack(fill=tk.X, **PAD)

        self.sample_text = tk.Text(frm, height=10, width=32,
                                   font=("Courier", 9), state=tk.DISABLED,
                                   bg="#f5f5f0", relief=tk.FLAT)
        self.sample_text.pack(fill=tk.X)

        # -- Dataset statistics --------------------------------------------
        frm = ttk.LabelFrame(parent, text="Current Dataset Stats", padding=5)
        frm.pack(fill=tk.X, **PAD)

        self.stats_text = tk.Text(frm, height=5, width=32,
                                  font=("Courier", 9), state=tk.DISABLED,
                                  bg="#f5f5f0", relief=tk.FLAT)
        self.stats_text.pack(fill=tk.X)
        self._refresh_stats()

    # -- MAP CANVAS --------------------------------------------------------
    def _build_map(self, parent):
        self.fig = Figure(figsize=(10, 8), dpi=100, facecolor="#f0f0f0")
        self.ax = None
        self.ax3d = None

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        tb_frame = ttk.Frame(parent)
        tb_frame.pack(fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, tb_frame)

        self.canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self.canvas.mpl_connect("button_release_event", self._on_click)

        self.selector = None
        self.cbar = None

    # ==================================================================
    #  RENDERING DISPATCH
    # ==================================================================

    def _cur_hm(self) -> LunarDataset:
        return self.store.heightmaps[self.scale_idx]

    def _cur_il(self):
        idx = self.scale_idx
        if idx < len(self.store.illumination):
            return self.store.illumination[idx]
        return None

    def _render(self, *_):
        # Clear any queued render to avoid duplicate redraws.
        if self._pending_render_job is not None:
            try:
                self.root.after_cancel(self._pending_render_job)
            except Exception:
                pass
            self._pending_render_job = None
        mode = self.view_var.get()
        if mode == self.VIEW_MODES[0]:
            self._render_2d()
        elif mode == self.VIEW_MODES[1]:
            self._render_3d()
        else:
            self._render_sidebyside()

    def _schedule_render(self, delay_ms=80):
        """Debounce expensive full redraws while sliders are dragged."""
        if self._pending_render_job is not None:
            self.root.after_cancel(self._pending_render_job)
        self._pending_render_job = self.root.after(delay_ms, self._render)

    def _update_3d_view_only(self):
        """
        Cheap interaction path for azimuth/elevation changes:
        update camera angle without rebuilding all artists.
        """
        if self.ax3d is None:
            return False

        elev = int(float(self.elev_var.get()))
        azim = int(float(self.azi_var.get()))
        self.ax3d.view_init(elev=elev, azim=azim)

        if self.view_var.get() == self.VIEW_MODES[1]:
            hm = self._cur_hm()
            vexag = float(self.vexag_var.get())
            self.ax3d.set_title(
                f"Lunar South Pole  -  {hm.scale_label}  (3D Displacement Map)\n"
                f"Vert. exagg. {vexag:.1f}x   "
                f"Azimuth {azim} deg   "
                f"Elev {elev} deg",
                fontsize=11, fontweight="bold", pad=6)

        self.canvas.draw_idle()
        return True

    # ==================================================================
    #  2-D CONTOUR / COLOUR-MAP VIEW  (Part A)
    # ==================================================================

    def _render_2d(self):
        self.fig.clf()
        self.ax3d = None
        self.ax = self.fig.add_subplot(111)

        hm = self._cur_hm()
        data, ext = hm.data, hm.extent
        vmin, vmax = float(np.nanmin(data)), float(np.nanmax(data))

        # 1 -- colour-mapped elevation raster
        self.im = self.ax.imshow(
            data, cmap=self.cmap_var.get(), extent=ext,
            origin="upper", vmin=vmin, vmax=vmax,
            aspect="equal", interpolation="bilinear")

        # 2 -- contour overlay
        self._overlay_contours_2d(self.ax, data, ext, vmin, vmax)

        # 3 -- illumination overlay
        self._overlay_illumination(self.ax)

        # 4 -- Part C overlays
        self._overlay_slope(self.ax)
        self._overlay_psr(self.ax)
        self._overlay_suitability(self.ax)
        self._draw_profile_on_map(self.ax)

        # 5 -- zoom limits
        if self.cur_xlim is not None:
            self.ax.set_xlim(self.cur_xlim)
            self.ax.set_ylim(self.cur_ylim)

        # 6 -- labels & colour bar
        self.ax.set_title(
            f"Lunar South Pole  -  {hm.scale_label}  (2D Contour Map)",
            fontsize=12, fontweight="bold", pad=8)
        self._set_axis_labels(self.ax, hm)
        self.cbar = self.fig.colorbar(
            self.im, ax=self.ax, label="Elevation (m)",
            shrink=0.82, pad=0.02, aspect=30)

        # 7 -- rectangle selector for zoom (left/right drag)
        self.selector = RectangleSelector(
            self.ax, self._on_rect_select, useblit=True, button=[1, 3],
            minspanx=5, minspany=5, spancoords="pixels",
            interactive=True,
            props=dict(facecolor="cyan", edgecolor="white",
                       alpha=0.25, linewidth=1.5))

        self.fig.tight_layout()
        self.canvas.draw_idle()

    # ==================================================================
    #  3-D DISPLACEMENT-MAP VIEW  (Part B)
    #
    #  Lecture 4 - Height / displacement plots:
    #    S_displ(x) = x + n(x) * f(x)
    #    For terrain: S = xy-plane, displacement along z.
    #    Shading provides additional perceptual cue for fine-grained
    #    (small-scale) data variations invisible in a flat colour map.
    # ==================================================================

    def _render_3d(self):
        self.fig.clf()
        self.ax = None
        self.selector = None
        self.cbar = None

        self.ax3d = self.fig.add_subplot(111, projection="3d")
        ax3 = self.ax3d

        hm = self._cur_hm()
        data, ext = hm.data, hm.extent

        # -- sub-sample for performance --
        X, Y, Z = self._prepare_3d_grid(data, ext)

        vmin, vmax = float(np.nanmin(Z)), float(np.nanmax(Z))
        vexag = float(self.vexag_var.get())

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = matplotlib.colormaps.get_cmap(self.cmap_var.get())

        # -- surface with optional hill-shading --
        if self.shade_3d.get():
            shade = self._compute_hillshade(Z)
            facecolors = cmap(norm(Z))
            facecolors[..., :3] *= shade[..., np.newaxis]
            facecolors = np.clip(facecolors, 0, 1)
            ax3.plot_surface(
                X, Y, Z * vexag,
                facecolors=facecolors,
                rstride=1, cstride=1,
                linewidth=0, antialiased=False,
                shade=False)
        else:
            ax3.plot_surface(
                X, Y, Z * vexag,
                cmap=self.cmap_var.get(),
                vmin=vmin * vexag, vmax=vmax * vexag,
                rstride=1, cstride=1,
                linewidth=0, antialiased=False)

        # -- contours projected onto 3-D surface --
        if self.contour_3d_on.get() and self.contour_on.get():
            self._draw_contours_3d(ax3, X, Y, Z, vmin, vmax, vexag)

        # -- viewpoint --
        ax3.view_init(elev=int(self.elev_var.get()),
                      azim=int(self.azi_var.get()))

        ax3.set_title(
            f"Lunar South Pole  -  {hm.scale_label}  (3D Displacement Map)\n"
            f"Vert. exagg. {vexag:.1f}x   "
            f"Azimuth {int(self.azi_var.get())} deg   "
            f"Elev {int(self.elev_var.get())} deg",
            fontsize=11, fontweight="bold", pad=6)

        self._set_axis_labels_3d(ax3, hm, vexag)

        sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        self.cbar = self.fig.colorbar(
            sm, ax=ax3, label="Elevation (m)",
            shrink=0.65, pad=0.08, aspect=25)

        self.fig.tight_layout()
        self.canvas.draw_idle()

    # ==================================================================
    #  SIDE-BY-SIDE COMPARISON  (Part B core requirement)
    #
    #  Directly compares 3D perspective displacement map rendering
    #  against isoline / isocontour representation so the user can
    #  evaluate trade-offs: shading cues vs precision, occlusion vs
    #  quantitative readability.
    # ==================================================================

    def _render_sidebyside(self):
        self.fig.clf()
        self.selector = None
        self.cbar = None

        ax2d = self.fig.add_subplot(121)
        ax3d = self.fig.add_subplot(122, projection="3d")
        self.ax = ax2d
        self.ax3d = ax3d

        hm = self._cur_hm()
        data, ext = hm.data, hm.extent
        vmin, vmax = float(np.nanmin(data)), float(np.nanmax(data))

        # -- LEFT: 2-D contour map --
        self.im = ax2d.imshow(
            data, cmap=self.cmap_var.get(), extent=ext,
            origin="upper", vmin=vmin, vmax=vmax,
            aspect="equal", interpolation="bilinear")
        self._overlay_contours_2d(ax2d, data, ext, vmin, vmax)
        self._overlay_illumination(ax2d)
        self._overlay_slope(ax2d)
        self._overlay_psr(ax2d)
        self._overlay_suitability(ax2d)
        self._draw_profile_on_map(ax2d)
        if self.cur_xlim is not None:
            ax2d.set_xlim(self.cur_xlim)
            ax2d.set_ylim(self.cur_ylim)
        ax2d.set_title("Isoline / Contour View", fontsize=10, fontweight="bold")
        self._set_axis_labels(ax2d, hm)

        # -- RIGHT: 3-D displacement map --
        X, Y, Z = self._prepare_3d_grid(data, ext)
        vexag = float(self.vexag_var.get())

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = matplotlib.colormaps.get_cmap(self.cmap_var.get())

        if self.shade_3d.get():
            shade = self._compute_hillshade(Z)
            fc = cmap(norm(Z))
            fc[..., :3] *= shade[..., np.newaxis]
            fc = np.clip(fc, 0, 1)
            ax3d.plot_surface(X, Y, Z * vexag,
                              facecolors=fc, rstride=1, cstride=1,
                              linewidth=0, antialiased=False, shade=False)
        else:
            ax3d.plot_surface(X, Y, Z * vexag,
                              cmap=self.cmap_var.get(),
                              vmin=vmin * vexag, vmax=vmax * vexag,
                              rstride=1, cstride=1,
                              linewidth=0, antialiased=False)

        if self.contour_3d_on.get() and self.contour_on.get():
            self._draw_contours_3d(ax3d, X, Y, Z, vmin, vmax, vexag)

        ax3d.view_init(elev=int(self.elev_var.get()),
                       azim=int(self.azi_var.get()))
        ax3d.set_title(
            f"3D Displacement Map  (x{vexag:.1f})",
            fontsize=10, fontweight="bold")
        self._set_axis_labels_3d(ax3d, hm, vexag)

        # shared colour bar
        sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        self.cbar = self.fig.colorbar(
            sm, ax=[ax2d, ax3d], label="Elevation (m)",
            shrink=0.7, pad=0.03, aspect=28)

        # rectangle selector on the 2-D panel
        self.selector = RectangleSelector(
            ax2d, self._on_rect_select, useblit=True, button=[1, 3],
            minspanx=5, minspany=5, spancoords="pixels",
            interactive=True,
            props=dict(facecolor="cyan", edgecolor="white",
                       alpha=0.25, linewidth=1.5))

        self.fig.tight_layout()
        self.canvas.draw_idle()

    # ==================================================================
    #  SHARED RENDERING HELPERS
    # ==================================================================

    def _overlay_contours_2d(self, ax, data, ext, vmin, vmax):
        """
        Lecture 4 - Contouring / isolines:
        I(f0) = {x in D | f(x) = f0}
        Contours are always closed curves and never self-intersect.
        matplotlib.contour implements marching squares internally.
        """
        if not self.contour_on.get():
            return
        interval = max(int(self.interval_var.get()), 10)
        lo = np.floor(vmin / interval) * interval
        hi = np.ceil(vmax / interval) * interval + interval
        levels = np.arange(lo, hi, interval)
        if len(levels) > 60:
            levels = np.linspace(vmin, vmax, 40)

        h, w = data.shape
        xs = np.linspace(ext[0], ext[1], w)
        ys = np.linspace(ext[3], ext[2], h)
        X, Y = np.meshgrid(xs, ys)
        step = max(1, min(h, w) // 800)
        Xs, Ys, Ds = X[::step, ::step], Y[::step, ::step], data[::step, ::step]
        try:
            cs = ax.contour(
                Xs, Ys, Ds, levels=levels,
                colors=self.contour_col_var.get(),
                linewidths=float(self.contour_lw_var.get()),
                alpha=0.6)
            if self.contour_labels_on.get() and len(levels) <= 25:
                ax.clabel(cs, inline=True, fontsize=6, fmt="%.0f m")
        except Exception as exc:
            print(f"[WARN] 2D contour overlay failed: {exc}", file=sys.stderr)

    def _draw_contours_3d(self, ax3, X, Y, Z, vmin, vmax, vexag):
        """Project contour lines onto the 3-D displaced surface."""
        interval = max(int(self.interval_var.get()), 10)
        lo = np.floor(vmin / interval) * interval
        hi = np.ceil(vmax / interval) * interval + interval
        levels = np.arange(lo, hi, interval)
        if len(levels) > 40:
            levels = np.linspace(vmin, vmax, 30)
        try:
            ax3.contour(
                X, Y, Z * vexag,
                levels=levels * vexag,
                colors=self.contour_col_var.get(),
                linewidths=float(self.contour_lw_var.get()),
                alpha=0.7)
        except Exception as exc:
            print(f"[WARN] 3D contour projection failed: {exc}",
                  file=sys.stderr)

    def _overlay_illumination(self, ax):
        """Semi-transparent illumination layer on 2-D axes."""
        if not self.illum_on.get():
            return
        il = self._cur_il()
        if il is None:
            return
        il_data = il.data.copy()
        lo_i, hi_i = np.nanmin(il_data), np.nanmax(il_data)
        if hi_i > lo_i:
            il_norm = (il_data - lo_i) / (hi_i - lo_i)
        else:
            il_norm = np.zeros_like(il_data)
        ax.imshow(il_norm, cmap=self.illum_cmap_var.get(),
                  extent=il.extent, origin="upper",
                  alpha=float(self.opacity_var.get()), aspect="equal")

    # ==================================================================
    #  PART C — CREATIVE ANALYSIS OVERLAYS
    # ==================================================================

    # -- C1: Slope / Gradient overlay ----------------------------------
    def _compute_slope_grid(self):
        """
        Lecture 4 — derived scalar quantities from gradient.
        Compute slope magnitude |grad f| from elevation, convert to degrees.
        Slope = arctan(|grad f|) is critical for landing safety.
        """
        hm = self._cur_hm()
        if self._cached_slope is not None and self._cached_slope_idx == self.scale_idx:
            return self._cached_slope
        data = hm.data
        # Compute gradient using Sobel for better noise handling
        dx = ndimage.sobel(data, axis=1).astype(np.float64)
        dy = ndimage.sobel(data, axis=0).astype(np.float64)
        # Account for pixel spacing if known
        if hm.pixel_size is not None and hm.pixel_size > 0:
            dx /= (8.0 * hm.pixel_size)   # Sobel divides by 8*spacing
            dy /= (8.0 * hm.pixel_size)
        else:
            dx /= 8.0
            dy /= 8.0
        grad_mag = np.sqrt(dx**2 + dy**2)
        slope_deg = np.degrees(np.arctan(grad_mag))
        self._cached_slope = slope_deg
        self._cached_slope_idx = self.scale_idx
        return slope_deg

    def _overlay_slope(self, ax):
        """
        Overlay slope as a diverging colourmap highlighting dangerous areas.
        Green = flat/safe, yellow = moderate, red = dangerously steep.
        """
        if not self.slope_on.get():
            return
        hm = self._cur_hm()
        slope_deg = self._compute_slope_grid()
        thresh = float(self.slope_thresh_var.get())
        # Create a colourmap: green (safe) -> yellow (caution) -> red (danger)
        from matplotlib.colors import LinearSegmentedColormap
        slope_cmap = LinearSegmentedColormap.from_list(
            "slope_safety",
            [(0, "green"), (0.5, "yellow"), (1.0, "red")])
        ax.imshow(slope_deg, cmap=slope_cmap,
                  extent=hm.extent, origin="upper",
                  vmin=0, vmax=thresh * 2,
                  alpha=float(self.slope_alpha_var.get()),
                  aspect="equal")
        # Add a contour at the danger threshold
        h, w = slope_deg.shape
        ext = hm.extent
        xs = np.linspace(ext[0], ext[1], w)
        ys = np.linspace(ext[3], ext[2], h)
        X, Y = np.meshgrid(xs, ys)
        step = max(1, min(h, w) // 600)
        try:
            ax.contour(X[::step, ::step], Y[::step, ::step],
                       slope_deg[::step, ::step],
                       levels=[thresh], colors=["red"],
                       linewidths=1.2, linestyles="dashed", alpha=0.8)
        except Exception as exc:
            print(f"[WARN] Slope threshold contour failed: {exc}",
                  file=sys.stderr)

    # -- C2: Permanently Shadowed Regions (PSR) indicators -------------
    def _overlay_psr(self, ax):
        """
        Identify permanently shadowed regions from illumination data.
        Low illumination => likely PSR => potential water ice deposits.
        Overlaid as hatched semi-transparent regions.
        """
        if not self.psr_on.get():
            return
        il = self._cur_il()
        if il is None:
            return
        il_data = il.data.copy()
        lo_i, hi_i = np.nanmin(il_data), np.nanmax(il_data)
        if hi_i <= lo_i:
            return
        il_norm = (il_data - lo_i) / (hi_i - lo_i)

        thresh_pct = float(self.psr_thresh_var.get()) / 100.0
        psr_mask = il_norm <= thresh_pct

        # Draw as a blue-tinted semi-transparent overlay
        psr_rgba = np.zeros((*psr_mask.shape, 4), dtype=np.float64)
        psr_rgba[psr_mask, 0] = 0.1    # R
        psr_rgba[psr_mask, 1] = 0.15   # G
        psr_rgba[psr_mask, 2] = 0.9    # B
        psr_rgba[psr_mask, 3] = 0.4    # A
        ax.imshow(psr_rgba, extent=il.extent, origin="upper", aspect="equal")

        # Add contour boundary around PSR regions
        h, w = il_norm.shape
        ext = il.extent
        xs = np.linspace(ext[0], ext[1], w)
        ys = np.linspace(ext[3], ext[2], h)
        X, Y = np.meshgrid(xs, ys)
        step = max(1, min(h, w) // 600)
        try:
            ax.contour(X[::step, ::step], Y[::step, ::step],
                       il_norm[::step, ::step],
                       levels=[thresh_pct], colors=["dodgerblue"],
                       linewidths=1.0, linestyles="solid", alpha=0.9)
        except Exception as exc:
            print(f"[WARN] PSR boundary contour failed: {exc}",
                  file=sys.stderr)

    # -- C3: Landing Suitability Score ---------------------------------
    def _compute_suitability(self):
        """
        Multi-variate derived quantity (Lectures 3-4):
        Combine flatness, illumination, and PSR proximity into a single
        composite score. Uses normalised weighting of three criteria:
          - Flatness: 1 - slope/max_slope  (flat = good)
          - Illumination: normalised brightness  (well-lit = good)
          - PSR proximity: distance to nearest shadow  (close = good
            for science, not too close for safety)
        """
        hm = self._cur_hm()
        if (self._cached_suit is not None
                and self._cached_suit_idx == self.scale_idx):
            return self._cached_suit

        # Flatness component (from slope)
        slope_deg = self._compute_slope_grid()
        max_slope = max(np.nanmax(slope_deg), 1.0)
        flatness = 1.0 - np.clip(slope_deg / max_slope, 0, 1)

        # Illumination component
        il = self._cur_il()
        if il is not None:
            il_data = il.data.copy()
            lo_i, hi_i = np.nanmin(il_data), np.nanmax(il_data)
            if hi_i > lo_i:
                illum_score = (il_data - lo_i) / (hi_i - lo_i)
            else:
                illum_score = np.ones_like(il_data) * 0.5
            # Resample to match heightmap size if different
            if illum_score.shape != hm.data.shape:
                from PIL import Image as _PILImg
                illum_pil = _PILImg.fromarray(illum_score)
                illum_pil = illum_pil.resize(
                    (hm.width, hm.height), _PILImg.BILINEAR)
                illum_score = np.array(illum_pil, dtype=np.float64)
        else:
            illum_score = np.ones_like(hm.data) * 0.5

        # PSR proximity component
        # Compute distance transform from PSR boundary: close to PSR
        # is scientifically valuable (water ice) but must not be *in* PSR
        psr_thresh = float(self.psr_thresh_var.get()) / 100.0
        psr_mask = illum_score <= psr_thresh
        if np.any(psr_mask) and not np.all(psr_mask):
            dist_to_psr = ndimage.distance_transform_edt(~psr_mask)
            max_dist = max(np.nanmax(dist_to_psr), 1.0)
            # Gaussian-like proximity: peaks near PSR, decays with distance
            # Penalise being *inside* PSR (too dark), reward being nearby
            psr_prox = np.exp(-0.5 * (dist_to_psr / (max_dist * 0.15))**2)
            psr_prox[psr_mask] = 0.1   # inside PSR = low score (too dark)
        else:
            psr_prox = np.ones_like(hm.data) * 0.5

        # Weighted combination
        w_f = float(self.w_flat_var.get())
        w_i = float(self.w_illum_var.get())
        w_p = float(self.w_psr_var.get())
        total_w = w_f + w_i + w_p
        if total_w < 1e-6:
            total_w = 1.0
        score = (w_f * flatness + w_i * illum_score + w_p * psr_prox) / total_w
        score = np.clip(score, 0, 1)

        self._cached_suit = score
        self._cached_suit_idx = self.scale_idx
        return score

    def _overlay_suitability(self, ax):
        """
        Visualise composite landing suitability as a heatmap overlay.
        Hot colours = good candidate sites; cool colours = poor sites.
        """
        if not self.suit_on.get():
            return
        # Invalidate cache — weights or PSR threshold may have changed
        self._cached_suit = None
        self._cached_suit_idx = -1
        hm = self._cur_hm()
        score = self._compute_suitability()
        ax.imshow(score, cmap="RdYlGn", vmin=0, vmax=1,
                  extent=hm.extent, origin="upper",
                  alpha=float(self.suit_alpha_var.get()),
                  aspect="equal")

    # -- C4: Elevation Profile Cross-Section ---------------------------
    def _draw_profile_on_map(self, ax):
        """Draw profile line and endpoint markers on the 2D map."""
        if len(self.profile_pts) == 0:
            return
        # Draw placed points
        for px, py in self.profile_pts:
            mk, = ax.plot(px, py, "mo", markersize=10, markeredgewidth=2,
                          markerfacecolor="none")
            mk._profile_marker = True
        # Draw line if we have both points
        if len(self.profile_pts) == 2:
            (x1, y1), (x2, y2) = self.profile_pts
            line, = ax.plot([x1, x2], [y1, y2], "m-",
                            linewidth=2, alpha=0.8)
            line._profile_marker = True

    def _render_profile_subplot(self):
        """
        Lecture concept: filtering / slicing through 2D scalar data.
        Extract a 1D cross-section along the user-drawn line and
        display it as an elevation profile in a popup window.
        """
        if len(self.profile_pts) != 2:
            return
        hm = self._cur_hm()
        (x1, y1), (x2, y2) = self.profile_pts

        # Sample N points along the line
        N = 500
        xs = np.linspace(x1, x2, N)
        ys = np.linspace(y1, y2, N)
        elevs = np.array([hm.sample(x, y) for x, y in zip(xs, ys)])

        # Compute distance along profile in metres
        if hm.pixel_size is not None:
            dists = np.sqrt((xs - x1)**2 + (ys - y1)**2)
        else:
            dists = np.linspace(0, np.sqrt((x2-x1)**2 + (y2-y1)**2), N)

        # Also sample illumination if available
        il = self._cur_il()
        illum_vals = None
        if il is not None:
            illum_vals = np.array([il.sample(x, y) for x, y in zip(xs, ys)])

        # Create popup window
        if hasattr(self, '_profile_win') and self._profile_win is not None:
            try:
                self._profile_win.destroy()
            except Exception:
                pass

        self._profile_win = tk.Toplevel(self.root)
        self._profile_win.title("Elevation Profile  -  Cross Section")
        self._profile_win.geometry("800x500")

        n_plots = 1 + (1 if illum_vals is not None else 0)
        pfig = Figure(figsize=(9, 5), dpi=100, facecolor="#f8f8f4")

        # Elevation profile
        ax_el = pfig.add_subplot(n_plots, 1, 1)
        ax_el.fill_between(dists, elevs, alpha=0.3, color="sienna")
        ax_el.plot(dists, elevs, "k-", linewidth=1.2)
        ax_el.set_ylabel("Elevation (m)")
        ax_el.set_title("Elevation Profile Cross-Section", fontweight="bold")
        ax_el.grid(True, alpha=0.3)

        # Mark slope danger zones on the profile
        slope_deg = self._compute_slope_grid()
        slope_along = np.array([
            slope_deg[min(int(round(r)), slope_deg.shape[0]-1),
                      min(int(round(c)), slope_deg.shape[1]-1)]
            for r, c in (hm._xy_to_rowcol(x, y)
                         for x, y in zip(xs, ys))
        ])
        thresh = float(self.slope_thresh_var.get())
        danger = slope_along > thresh
        if np.any(danger):
            ax_el.fill_between(dists, elevs, where=danger,
                               alpha=0.3, color="red",
                               label=f"Slope > {thresh:.0f} deg")
            ax_el.legend(fontsize=8, loc="upper right")

        # Illumination profile
        if illum_vals is not None:
            ax_il = pfig.add_subplot(n_plots, 1, 2, sharex=ax_el)
            ax_il.fill_between(dists, illum_vals, alpha=0.3, color="orange")
            ax_il.plot(dists, illum_vals, color="darkorange", linewidth=1.2)
            ax_il.set_ylabel("Illumination")
            ax_il.set_xlabel("Distance along profile (m)")
            ax_il.grid(True, alpha=0.3)
        else:
            ax_el.set_xlabel("Distance along profile (m)")

        pfig.tight_layout()
        pcanvas = FigureCanvasTkAgg(pfig, master=self._profile_win)
        pcanvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        pcanvas.draw()

    def _clear_profile(self):
        self.profile_pts.clear()
        if hasattr(self, '_profile_win') and self._profile_win is not None:
            try:
                self._profile_win.destroy()
            except Exception:
                pass
            self._profile_win = None
        self._render()

    def _prepare_3d_grid(self, data, ext):
        """Build X, Y, Z meshgrids for 3-D surface, sub-sampled for speed."""
        h, w = data.shape
        step = max(1, max(h, w) // self.SURF_MAX)
        Z = data[::step, ::step].copy()
        nan_mask = np.isnan(Z)
        if nan_mask.any():
            Z[nan_mask] = np.nanmin(data)
        rows, cols = Z.shape
        xs = np.linspace(ext[0], ext[1], cols)
        ys = np.linspace(ext[3], ext[2], rows)
        X, Y = np.meshgrid(xs, ys)
        return X, Y, Z

    @staticmethod
    def _compute_hillshade(Z, azimuth_deg=315, altitude_deg=45):
        """
        Lambertian hill-shading (Lecture 4):
        Shading acts as an additional perceptual cue, emphasising
        fine-grained (small-scale) data variations that would be
        invisible in a flat colour map alone.
        """
        az = np.radians(azimuth_deg)
        alt = np.radians(altitude_deg)
        dy, dx = np.gradient(Z)
        slope = np.sqrt(dx**2 + dy**2)
        aspect = np.arctan2(-dy, dx)
        shade = (np.sin(alt) * np.cos(np.arctan(slope)) +
                 np.cos(alt) * np.sin(np.arctan(slope)) *
                 np.cos(az - aspect))
        shade = np.clip(shade, 0.15, 1.0)
        lo, hi = shade.min(), shade.max()
        shade = 0.3 + 0.7 * (shade - lo) / (hi - lo + 1e-12)
        return shade

    def _set_axis_labels(self, ax, hm):
        if hm.bounds is not None:
            ax.set_xlabel("Easting (m)")
            ax.set_ylabel("Northing (m)")
        else:
            ax.set_xlabel("Pixel X")
            ax.set_ylabel("Pixel Y")

    def _set_axis_labels_3d(self, ax3, hm, vexag):
        if hm.bounds is not None:
            ax3.set_xlabel("Easting (m)", fontsize=8)
            ax3.set_ylabel("Northing (m)", fontsize=8)
        else:
            ax3.set_xlabel("X (px)", fontsize=8)
            ax3.set_ylabel("Y (px)", fontsize=8)
        ax3.set_zlabel(f"Elevation x{vexag:.1f} (m)", fontsize=8)

    # ==================================================================
    #  INTERACTION HANDLERS
    # ==================================================================

    def _on_mouse_press(self, event):
        """Track left-press location to distinguish click from drag."""
        if event.button != 1:
            self._press_xy = None
            return
        target_ax = self.ax
        if target_ax is None or event.inaxes != target_ax:
            self._press_xy = None
            return
        if self.toolbar.mode:
            self._press_xy = None
            return
        if event.x is None or event.y is None:
            self._press_xy = None
            return
        self._press_xy = (event.x, event.y)

    def _on_click(self, event):
        """Left-release -> sample values or set profile points.
        Implements the inverse mapping concept from Lecture 2.
        Part C adds slope, suitability readout and profile cross-sections."""
        if event.button != 1:
            self._press_xy = None
            return
        press_xy = self._press_xy
        self._press_xy = None

        target_ax = self.ax
        if target_ax is None or event.inaxes != target_ax:
            return
        if self.toolbar.mode:
            return
        if event.x is None or event.y is None:
            return
        if press_xy is not None:
            dx = event.x - press_xy[0]
            dy = event.y - press_xy[1]
            # Left-drag draws a zoom rectangle; avoid sampling on drag releases.
            if (dx * dx + dy * dy) > 25:
                return
        if event.xdata is None or event.ydata is None:
            return

        x, y = event.xdata, event.ydata

        # -- Profile mode: collect two points, then show cross-section --
        if self.profile_mode.get():
            self.profile_pts.append((x, y))
            if len(self.profile_pts) > 2:
                self.profile_pts = self.profile_pts[-2:]
            self._render()
            if len(self.profile_pts) == 2:
                self._render_profile_subplot()
            return

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

        # Illumination readout
        il = self._cur_il()
        if il is not None:
            iv = il.sample(x, y)
            lines.append(f"Illumin. : {iv:>10.2f}")
            il_max = float(np.nanmax(il.data))
            if il_max <= 1.01:
                lines.append(f"           ({iv*100:.1f} %)")
            elif il_max <= 100.1:
                lines.append(f"           ({iv:.1f} %)")

        # Part C: slope readout
        try:
            slope_grid = self._compute_slope_grid()
            row, col = hm._xy_to_rowcol(x, y)
            if 0 <= row < slope_grid.shape[0] and 0 <= col < slope_grid.shape[1]:
                sv = slope_grid[row, col]
                thresh = float(self.slope_thresh_var.get())
                flag = " STEEP!" if sv > thresh else ""
                lines.append(f"Slope  :  {sv:>8.1f} deg{flag}")
        except Exception:
            pass

        # Part C: suitability readout
        try:
            if self.suit_on.get():
                suit = self._compute_suitability()
                row, col = hm._xy_to_rowcol(x, y)
                if 0 <= row < suit.shape[0] and 0 <= col < suit.shape[1]:
                    sc = suit[row, col]
                    lines.append(f"Suitab.:  {sc:>8.2f} / 1.0")
        except Exception:
            pass

        self._set_textbox(self.sample_text, "\n".join(lines))

        # draw cross-hair marker
        for art in list(a for a in target_ax.get_children()
                        if getattr(a, "_sample_marker", False)):
            art.remove()
        mk, = target_ax.plot(x, y, "r+", markersize=16, markeredgewidth=2.5)
        mk._sample_marker = True
        self.canvas.draw_idle()

    def _on_rect_select(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        if x1 is None or x2 is None:
            return
        if self.ax is not None:
            self.zoom_stack.append(
                (self.ax.get_xlim(), self.ax.get_ylim()))
        self.cur_xlim = (min(x1, x2), max(x1, x2))
        self.cur_ylim = (min(y1, y2), max(y1, y2))
        self._render()

    def _reset_zoom(self):
        self.zoom_stack.clear()
        self.cur_xlim = self.cur_ylim = None
        self._render()

    def _zoom_back(self):
        if self.zoom_stack:
            self.cur_xlim, self.cur_ylim = self.zoom_stack.pop()
        else:
            self.cur_xlim = self.cur_ylim = None
        self._render()

    def _on_scale_change(self):
        self.scale_idx = self.scale_var.get()
        self.zoom_stack.clear()
        self.cur_xlim = self.cur_ylim = None
        # Invalidate Part C caches
        self._cached_slope = None
        self._cached_slope_idx = -1
        self._cached_suit = None
        self._cached_suit_idx = -1
        self.profile_pts.clear()
        self._refresh_stats()
        self._render()

    def _on_interval_slide(self, _val):
        v = int(float(_val))
        self.interval_lbl.config(text=f"{v} m")
        if self.contour_on.get():
            self._render()

    def _on_3d_param(self, changed="view"):
        self.azi_lbl.config(text=f"{int(float(self.azi_var.get()))} deg")
        self.elev_lbl.config(text=f"{int(float(self.elev_var.get()))} deg")
        self.vexag_lbl.config(text=f"{float(self.vexag_var.get()):.1f}x")
        mode = self.view_var.get()
        if mode in (self.VIEW_MODES[1], self.VIEW_MODES[2]):
            # Azimuth/elevation can be updated interactively without
            # rebuilding the full figure. Vertical exaggeration changes
            # surface geometry, so keep it as a full render (debounced).
            if changed in ("azi", "elev") and self._update_3d_view_only():
                return
            self._schedule_render()

    # ==================================================================
    #  HELPERS
    # ==================================================================

    def _refresh_stats(self):
        hm = self._cur_hm()
        d = hm.data
        lines = [
            f"File : {hm.name}",
            f"Size : {hm.width} x {hm.height} px",
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


# ============================================================================
#  ENTRY POINT
# ============================================================================

def main():
    root = tk.Tk()
    style = ttk.Style(root)
    for preferred in ("clam", "alt", "default"):
        if preferred in style.theme_names():
            style.theme_use(preferred)
            break
    app = LunarExplorer(root)
    root.mainloop()


if __name__ == "__main__":
    main()

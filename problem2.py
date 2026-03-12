import os, sys, glob, textwrap
import tkinter as tk
from tkinter import ttk
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import FancyBboxPatch, Wedge, FancyArrowPatch
from matplotlib.colors import Normalize, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patheffects as pe
import matplotlib.cm as mcm
from scipy import ndimage

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

# ── calibration constants (from LROC metadata; see problem1.py) ─────────────
LDEM_UINT_SCALE_M  = 0.5
LDEM_UINT_OFFSET_M = -10000.0
LUNAR_RADIUS_KM    = 1737.4

# ── colour palette — space-themed dark scheme ────────────────────────────────
BG_DARK   = "#0b0b1e"
BG_PANEL  = "#10102a"
BG_CARD   = "#181840"
BG_PLOT   = "#0a0a18"
FG_WHITE  = "#eeeeff"
FG_DIM    = "#7777aa"
FG_MUTED  = "#444470"
ACCENT    = "#4fc3f7"
GOLD      = "#ffd54f"
SAFE_CLR  = "#00e676"
DANGER    = "#ff5252"
PSR_CLR   = "#1565c0"
WARM_ORG  = "#ff9800"
IS_MACOS  = (sys.platform == "darwin")

# Chapter panel layout: keep chart/stat columns compact to maximise map area.
CHAPTER_WIDTH_RATIOS = [4.6, 1.25, 0.62, 0.62]
CHAPTER_LAYOUT = {
    "hspace": 0.32,
    "wspace": 0.16,
    "left": 0.03,
    "right": 0.985,
    "top": 0.92,
    "bottom": 0.07,
}

# ── chapter definitions ─────────────────────────────────────────────────────
CHAPTERS = [
    {
        "key":   "landscape",
        "title": "THE LANDSCAPE",
        "sub":   "What does the lunar south pole look like?",
        "icon":  "1",
        "narrative": (
            "The lunar south pole is one of the most dramatic terrains "
            "in the Solar System.  Billions of years of meteorite impacts "
            "have carved enormous craters — some wider than Wales.\n\n"
            "The terrain map (top-left) shows elevation above the lunar "
            "reference sphere, with Lambertian hillshading adding a 3D "
            "depth cue.  The histogram (top-right) reveals how elevation "
            "is distributed.  The comparison panel (bottom-right) shows "
            "just how extreme this terrain is compared to Earth.\n\n"
            "The perceptually-uniform 'terrain' colourmap avoids the "
            "non-linear luminance jumps of rainbow scales (Lec 7)."
        ),
    },
    {
        "key":   "shadow",
        "title": "LIGHT & SHADOW",
        "sub":   "Where the Sun never shines",
        "icon":  "2",
        "narrative": (
            "Because the Moon's axis is almost perfectly upright, "
            "the Sun only skims the horizon at the poles.  Deep craters "
            "never receive direct sunlight — creating Permanently "
            "Shadowed Regions (PSRs) that have been dark for billions "
            "of years.\n\n"
            "The illumination map (top-left) uses a perceptually "
            "uniform colourmap ('plasma', Lec 7) to show % time each "
            "pixel is sunlit.  The donut chart (top-right) instantly "
            "communicates how much of the region is in permanent "
            "shadow.  The bar chart (bottom-right) compares "
            "conditions inside PSRs with sunlit ridges.\n\n"
            "Temperatures in PSRs drop to −230°C — cold enough to "
            "trap water ice from comets.  This ice is the reason "
            "NASA chose the south pole."
        ),
    },
    {
        "key":   "sunlight",
        "title": "SUNLIGHT & POWER",
        "sub":   "The Peaks of Eternal Light",
        "icon":  "3",
        "narrative": (
            "Just kilometres from permanent darkness, elevated ridges "
            "and crater rims receive near-continuous sunlight.  These "
            "'Peaks of Eternal Light' are ideal sites for solar panels.\n\n"
            "The illumination contour map (top-left) uses isolines "
            "(Lec 4) at 20/40/60/80% to show how sunlight varies.  "
            "The bar chart (top-right) ranks how many pixels fall into "
            "each illumination band.  The cross-section profile "
            "(bottom-right) traces an elevation slice through the "
            "brightest ridge and the deepest shadow, showing the "
            "dramatic terrain that creates these extreme lighting "
            "differences within a few kilometres.\n\n"
            "A base camp on a sunlit ridge close to a PSR gets "
            "reliable solar power AND access to water ice — the best "
            "of both worlds."
        ),
    },
    {
        "key":   "suitability",
        "title": "MISSION READY",
        "sub":   "Where is it safe to land?",
        "icon":  "4",
        "narrative": (
            "Choosing a landing site balances multiple competing "
            "factors: the terrain must be flat enough to land safely, "
            "but close to both sunlight (power) and shadow (water "
            "ice).\n\n"
            "The suitability map (top-left) uses a diverging "
            "colourmap (RdYlGn, Lec 7) — green = ideal, red = avoid "
            "— so the audience can instantly identify the best zones.  "
            "The radar chart (top-right) shows how the top candidate "
            "site scores on each criterion.  The stacked bar chart "
            "(bottom-right) breaks down the contribution of each "
            "factor across different zones.\n\n"
            "The first crewed Moon landing since Apollo 17 (1972) "
            "will explore the south pole — and for the first time, "
            "a woman will walk on the Moon."
        ),
    },
]


# ═════════════════════════════════════════════════════════════════════════════
#  DATA LOADING  (pipeline stage 1 — reused from problem1.py)
# ═════════════════════════════════════════════════════════════════════════════

class LunarDataset:
    """Single loaded TIFF with calibrated elevation data."""

    def __init__(self, filepath, role="unknown"):
        self.filepath = filepath
        self.name = os.path.basename(filepath)
        self.role = role
        self.data = None
        self.width = self.height = 0
        self.bounds = None
        self.transform = None
        self.pixel_size = None
        self.scale_label = ""
        self.raw_dtype = "unknown"
        self.raw_min = float("nan")
        self.raw_max = float("nan")
        self.crs = None
        self.is_georeferenced = False
        self.calibration_note = "No calibration applied"
        self._load()

    @property
    def coverage_area(self):
        if self.is_georeferenced and self.bounds:
            w = abs(self.bounds.right - self.bounds.left)
            h = abs(self.bounds.top  - self.bounds.bottom)
            return w * h
        return self.width * self.height

    # ── loading backends ─────────────────────────────────────────────────────

    def _load(self):
        if HAS_RASTERIO:
            self._load_rasterio()
        elif HAS_TIFFFILE:
            self._load_tifffile()
        else:
            self._load_pil()
        self._capture_raw_stats()
        self._apply_height_calibration()
        self._clean()

    def _load_rasterio(self):
        with rasterio.open(self.filepath) as src:
            raw = src.read(1)
            self.raw_dtype = str(raw.dtype)
            self.data = raw.astype(np.float64)
            self.height, self.width = self.data.shape
            self.transform = src.transform
            nodata = src.nodata
            if nodata is not None:
                self.data[self.data == nodata] = np.nan
            self.crs = str(src.crs) if src.crs else None
            identity = rasterio.Affine.identity()
            has_non_identity = (
                self.transform is not None and
                any(abs(a - b) > 1e-12
                    for a, b in zip(self.transform, identity))
            )
            self.is_georeferenced = bool(self.crs) or has_non_identity
            if self.is_georeferenced:
                self.bounds = src.bounds
                self.pixel_size = abs(self.transform.a)
            else:
                self.bounds = self.transform = self.pixel_size = None

    def _load_tifffile(self):
        import tifffile
        with tifffile.TiffFile(self.filepath) as tf:
            page = tf.pages[0]
            try:
                raw = page.asarray()
            except Exception:
                raw = np.array(PILImage.open(self.filepath))
        self.raw_dtype = str(raw.dtype)
        self.data = raw.astype(np.float64)
        if self.data.ndim == 3:
            self.data = self.data[:, :, 0]
        self.height, self.width = self.data.shape

    def _load_pil(self):
        img = PILImage.open(self.filepath)
        raw = np.array(img)
        self.raw_dtype = str(raw.dtype)
        self.data = raw.astype(np.float64)
        if self.data.ndim == 3:
            self.data = self.data[:, :, 0]
        self.height, self.width = self.data.shape

    def _capture_raw_stats(self):
        if self.data is not None and self.data.size:
            valid = self.data[np.isfinite(self.data)]
            if valid.size:
                self.raw_min = float(valid.min())
                self.raw_max = float(valid.max())

    def _apply_height_calibration(self):
        if self.role != "heightmaps":
            self.calibration_note = "N/A (illumination layer)"
            return
        try:
            dtype = np.dtype(self.raw_dtype)
        except Exception:
            dtype = self.data.dtype
        rmin, rmax = self.raw_min, self.raw_max
        if np.issubdtype(dtype, np.unsignedinteger) and np.isfinite(rmax) and rmax > 1000:
            self.data = self.data * LDEM_UINT_SCALE_M + LDEM_UINT_OFFSET_M
            self.calibration_note = "DN->m: elev = DN*0.5 - 10000"
            return
        if (np.issubdtype(dtype, np.floating) and
                np.isfinite(rmin) and np.isfinite(rmax) and
                -50.0 <= rmin <= 50.0 and -50.0 <= rmax <= 50.0):
            self.data = self.data * 1000.0
            self.calibration_note = "km->m: elev = value*1000"
            return
        self.calibration_note = "Assumed metres (no conversion)"

    def _clean(self):
        if self.data is None:
            return
        if np.any(np.abs(self.data) > 1e15):
            self.data[np.abs(self.data) > 1e15] = np.nan

    def sample(self, x, y):
        r, c = self._xy_to_rowcol(x, y)
        if 0 <= r < self.height and 0 <= c < self.width:
            return float(self.data[r, c])
        return float("nan")

    def _xy_to_rowcol(self, x, y):
        if self.transform is not None:
            inv = ~self.transform
            c, r = inv * (x, y)
            return int(round(r)), int(round(c))
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
        print(f"Dataset loaded: {len(self.heightmaps)} heightmap(s), "
              f"{len(self.illumination)} illumination file(s)")

    def _find_tiffs(self, subfolder):
        path = os.path.join(self.root, subfolder)
        files = []
        for ext in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
            files.extend(glob.glob(os.path.join(path, ext)))
        seen, unique = set(), []
        for f in sorted(files):
            n = os.path.normcase(os.path.abspath(f))
            if n not in seen:
                seen.add(n)
                unique.append(f)
        return unique

    def _load_folder(self, subfolder, target):
        for fp in self._find_tiffs(subfolder):
            try:
                target.append(LunarDataset(fp, role=subfolder))
            except Exception as exc:
                print(f"[WARN] Could not load {fp}: {exc}")

    def _sort_and_label(self, datasets):
        datasets.sort(key=lambda d: d.coverage_area, reverse=True)
        for i, ds in enumerate(datasets):
            ds.scale_label = (self.SCALE_LABELS[i]
                              if i < len(self.SCALE_LABELS) else f"Scale {i+1}")


# ═════════════════════════════════════════════════════════════════════════════
#  DERIVED LAYER HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def compute_hillshade(elev, azimuth=315.0, altitude=45.0):
    """
    Lambertian hillshade (Lec 5: reflectance / shading models).
    Azimuth 315° (NW) and altitude 45° chosen to maximise ridge
    visibility for a general audience rather than simulate true solar
    position.
    """
    az  = np.radians(360.0 - azimuth + 90.0)
    alt = np.radians(altitude)
    dy, dx = np.gradient(elev.astype(np.float32))
    slope  = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dy, dx)
    shade  = (np.sin(alt) * np.cos(slope) +
              np.cos(alt) * np.sin(slope) * np.cos(az - aspect))
    return np.clip(shade, 0.0, 1.0)


def compute_slope_deg(elev):
    """Slope in degrees from Sobel gradient — Lec 7: |nabla f|."""
    sx = ndimage.sobel(elev.astype(np.float32), axis=1)
    sy = ndimage.sobel(elev.astype(np.float32), axis=0)
    return np.degrees(np.arctan(np.hypot(sx, sy) / 8.0))


def compute_psr_mask(illum, threshold=0.05):
    """
    Boolean PSR mask: pixels with normalised illumination below threshold.
    Uses 2nd–98th percentile normalisation so border zeros are excluded.
    """
    lo = np.nanpercentile(illum, 2)
    hi = np.nanpercentile(illum, 98)
    norm = np.clip((illum - lo) / (hi - lo + 1e-9), 0, 1)
    return norm < threshold


def compute_suitability(elev, illum=None, psr_mask=None):
    """
    Composite score in [0,1]:  0.5*flat + 0.3*illumination + 0.2*psr_proximity.
    Diverging colourmap (RdYlGn, Lec 7) encodes deviation from neutral.
    """
    h, w = elev.shape
    slope = compute_slope_deg(elev)
    flat  = np.clip(1.0 - slope / 20.0, 0, 1)

    if illum is not None:
        if illum.shape != elev.shape:
            from scipy.ndimage import zoom
            illum = zoom(illum, (h / illum.shape[0], w / illum.shape[1]), order=1)
        lo = np.nanpercentile(illum, 2)
        hi = np.nanpercentile(illum, 98)
        illum_score = np.clip((illum - lo) / (hi - lo + 1e-9), 0, 1)
    else:
        illum_score = np.ones((h, w), dtype=np.float32)

    if psr_mask is not None:
        if psr_mask.shape != elev.shape:
            from scipy.ndimage import zoom
            psr_mask = zoom(psr_mask.astype(float),
                            (h / psr_mask.shape[0], w / psr_mask.shape[1]),
                            order=0) > 0.5
        dist = ndimage.distance_transform_edt(~psr_mask)
        psr_prox = np.exp(-((dist - 5.0)**2) / (2 * 8.0**2))
        psr_prox = (psr_prox - psr_prox.min()) / (psr_prox.max() - psr_prox.min() + 1e-9)
    else:
        psr_prox = np.ones((h, w), dtype=np.float32)

    score = 0.5 * flat + 0.3 * illum_score + 0.2 * psr_prox
    return np.nan_to_num(score, nan=0.0).astype(np.float32)


def resample_to(arr, target_shape, order=1):
    """Resample arr to target_shape using scipy zoom."""
    if arr.shape == target_shape:
        return arr
    from scipy.ndimage import zoom as _zoom
    zy = target_shape[0] / arr.shape[0]
    zx = target_shape[1] / arr.shape[1]
    return _zoom(arr.astype(np.float64), (zy, zx), order=order)


# ═════════════════════════════════════════════════════════════════════════════
#  PLOT STYLING HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def style_axis_dark(ax, title="", xlabel="", ylabel="", title_size=9):
    """Apply consistent dark-theme styling to any axes."""
    ax.set_facecolor(BG_PLOT)
    if title:
        ax.set_title(title, color=FG_WHITE, fontsize=title_size,
                      fontweight="bold", pad=6)
    if xlabel:
        ax.set_xlabel(xlabel, color=FG_DIM, fontsize=7)
    if ylabel:
        ax.set_ylabel(ylabel, color=FG_DIM, fontsize=7)
    ax.tick_params(colors=FG_DIM, labelsize=6.5)
    for spine in ax.spines.values():
        spine.set_edgecolor(FG_MUTED)
        spine.set_linewidth(0.4)


def style_colorbar(cbar, label="", horizontal=False):
    """Dark-themed colorbar."""
    cbar.set_label(label, color=FG_WHITE, fontsize=7.5)
    if horizontal:
        cbar.ax.xaxis.set_tick_params(color=FG_DIM, labelsize=6.5)
        for lbl in cbar.ax.get_xticklabels():
            lbl.set_color(FG_DIM)
    else:
        cbar.ax.yaxis.set_tick_params(color=FG_DIM, labelsize=6.5)
        for lbl in cbar.ax.get_yticklabels():
            lbl.set_color(FG_DIM)


def big_stat(ax, value, unit, description, color=ACCENT):
    """Draw a large key-statistic number — infographic style (Lec 8)."""
    ax.set_facecolor(BG_CARD)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(FG_MUTED)
        spine.set_linewidth(0.3)
    ax.text(0.5, 0.62, value, transform=ax.transAxes,
            ha="center", va="center", fontsize=18, fontweight="bold",
            color=color)
    ax.text(0.5, 0.38, unit, transform=ax.transAxes,
            ha="center", va="center", fontsize=8, color=FG_DIM)
    ax.text(0.5, 0.15, description, transform=ax.transAxes,
            ha="center", va="center", fontsize=6.5, color=FG_WHITE,
            style="italic")


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ═════════════════════════════════════════════════════════════════════════════

class ArtemisInfoVis:
    """
    Multi-panel infographic dashboard for general audiences.

    Each chapter renders a multi-panel matplotlib figure that combines:
      • A scientific visualisation panel (map/terrain from LROC data)
      • Information visualisation panels (charts, histograms, profiles)
      • Infographic elements (key statistics, comparisons, annotations)

    Layout per chapter (GridSpec):
      ┌──────────────────┬──────────┬──────────┐
      │                  │ InfoVis  │          │
      │   SciVis MAP     │  panel   │  Stats   │
      │   (main panel)   ├──────────┤  strip   │
      │                  │ InfoVis  │          │
      │                  │  panel 2 │          │
      └──────────────────┴──────────┴──────────┘

    Lecture 8 principles:
      • "Control reading order" — chapters guide the audience through
        a curated narrative from overview to decision.
      • "Provide context" — charts alongside maps contextualise raw
        data that non-experts would otherwise find hard to interpret.
      • "Highlight to focus attention" — annotations direct the eye
        to the most important features.
    """

    def __init__(self, root: tk.Tk, store: DataStore):
        self.root  = root
        self.store = store
        self._chapter_idx = 0
        self._scale_idx   = 0
        self._show_annotations = tk.BooleanVar(value=True)
        self._show_psr = tk.BooleanVar(value=True)
        self._psr_threshold = tk.DoubleVar(value=0.05)
        self._cache: dict = {}

        # zoom state
        self._zoom_stack: list = []
        self._cur_xlim = None
        self._cur_ylim = None
        # click state
        self._click_col = None
        self._click_row = None
        self._press_xy  = None
        self._selector  = None
        self._click_marker_artists = []
        # main map axes reference (for zoom/click)
        self._map_ax    = None
        self._base_xlim = None
        self._base_ylim = None

        self._build_ui()
        self._refresh_chapter()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        self.root.title("Artemis III — Lunar South Pole Infographic")
        self.root.configure(bg=BG_DARK)
        self.root.geometry("1440x860")
        self.root.minsize(1100, 680)

        self._build_title_bar()
        self._build_main_area()
        self._build_bottom_bar()

    @staticmethod
    def _control_style_kwargs(bg_color):
        """Ensure control background colours are respected on macOS Tk."""
        if not IS_MACOS:
            return {}
        return {
            "highlightbackground": bg_color,
            "highlightcolor": bg_color,
            "highlightthickness": 0,
        }

    def _clear_click_marker_artists(self):
        for artist in self._click_marker_artists:
            try:
                artist.remove()
            except ValueError:
                pass
        self._click_marker_artists = []

    def _draw_click_marker(self):
        self._clear_click_marker_artists()
        if self._map_ax is None or self._click_col is None or self._click_row is None:
            return
        m1 = self._map_ax.plot(self._click_col, self._click_row,
                               "+", color="#ff3355", markersize=18,
                               markeredgewidth=2.5, zorder=10)[0]
        m2 = self._map_ax.plot(self._click_col, self._click_row,
                               "o", color="none", markersize=14,
                               markeredgecolor="#ff3355", markeredgewidth=1.5,
                               zorder=10)[0]
        self._click_marker_artists = [m1, m2]

    def _apply_zoom_to_map(self):
        if self._map_ax is None:
            return
        if self._cur_xlim is not None and self._cur_ylim is not None:
            self._map_ax.set_xlim(self._cur_xlim)
            self._map_ax.set_ylim(self._cur_ylim)
            return
        if self._base_xlim is not None and self._base_ylim is not None:
            self._map_ax.set_xlim(self._base_xlim)
            self._map_ax.set_ylim(self._base_ylim)

    def _chapter_gridspec(self):
        return GridSpec(
            2, 4, figure=self._fig,
            width_ratios=CHAPTER_WIDTH_RATIOS,
            **CHAPTER_LAYOUT
        )

    def _add_horizontal_map_colorbar(self, ax_map, mappable, label,
                                     ticks=None, ticklabels=None):
        """
        Add a horizontal scale bar directly under the map, spanning map width.
        """
        divider = make_axes_locatable(ax_map)
        cax = divider.append_axes("bottom", size="4.5%", pad=0.36)
        cbar = self._fig.colorbar(mappable, cax=cax, orientation="horizontal")
        if ticks is not None:
            cbar.set_ticks(ticks)
        if ticklabels is not None:
            cbar.set_ticklabels(ticklabels)
        style_colorbar(cbar, label, horizontal=True)
        cax.set_facecolor(BG_PLOT)
        for spine in cax.spines.values():
            spine.set_edgecolor(FG_MUTED)
            spine.set_linewidth(0.4)
        return cbar

    @staticmethod
    def _keep_ylabels_inside(ax, fontsize=5.8):
        """
        Keep y tick labels within narrow info charts so they don't spill onto map.
        """
        ax.tick_params(axis="y", pad=-2, labelsize=fontsize)
        for lbl in ax.get_yticklabels():
            lbl.set_horizontalalignment("left")
            lbl.set_clip_on(True)

    def _build_title_bar(self):
        """Title banner — first visual element seen (Lec 8: reading order)."""
        bar = tk.Frame(self.root, bg="#060618", height=50)
        bar.pack(fill=tk.X)
        bar.pack_propagate(False)

        tk.Label(bar, text="\u263E", bg="#060618", fg=ACCENT,
                 font=("Segoe UI", 22, "bold")).pack(side=tk.LEFT, padx=14)
        tk.Label(bar,
                 text="ARTEMIS III  \u00b7  EXPLORING THE LUNAR SOUTH POLE",
                 bg="#060618", fg=ACCENT,
                 font=("Segoe UI", 14, "bold")).pack(side=tk.LEFT)
        tk.Label(bar,
                 text="  \u2014  An interactive data story for the public",
                 bg="#060618", fg=FG_DIM,
                 font=("Segoe UI", 10)).pack(side=tk.LEFT)
        tk.Label(bar, text="Data: NASA LROC  |  COMP4097",
                 bg="#060618", fg=FG_DIM,
                 font=("Segoe UI", 8)).pack(side=tk.RIGHT, padx=14)

    def _build_main_area(self):
        """Central area: matplotlib canvas (left) + narrative panel (right)."""
        pane = tk.Frame(self.root, bg=BG_DARK)
        pane.pack(fill=tk.BOTH, expand=True, padx=6, pady=(2, 0))

        # ── matplotlib canvas ─────────────────────────────────────────────
        left = tk.Frame(pane, bg=BG_DARK)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._fig = Figure(figsize=(11, 7), facecolor=BG_DARK)
        self._canvas = FigureCanvasTkAgg(self._fig, master=left)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._canvas.mpl_connect("button_press_event", self._on_press)
        self._canvas.mpl_connect("button_release_event", self._on_map_click)

        # ── narrative panel ───────────────────────────────────────────────
        right = tk.Frame(pane, bg=BG_PANEL, width=280)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(6, 0))
        right.pack_propagate(False)

        self._lbl_chnum = tk.Label(right, text="1", bg=BG_PANEL, fg=ACCENT,
                                    font=("Segoe UI", 32, "bold"))
        self._lbl_chnum.pack(pady=(14, 0))

        self._lbl_title = tk.Label(right, text="", bg=BG_PANEL, fg=GOLD,
                                    font=("Segoe UI", 11, "bold"),
                                    wraplength=260, justify=tk.CENTER)
        self._lbl_title.pack(pady=(2, 0))

        self._lbl_sub = tk.Label(right, text="", bg=BG_PANEL, fg=FG_DIM,
                                  font=("Segoe UI", 9, "italic"),
                                  wraplength=260, justify=tk.CENTER)
        self._lbl_sub.pack(pady=(2, 6))

        tk.Frame(right, bg=ACCENT, height=1).pack(fill=tk.X, padx=16, pady=2)

        # Scrollable narrative text
        frame_text = tk.Frame(right, bg=BG_PANEL)
        frame_text.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        scroll = tk.Scrollbar(frame_text, bg=BG_PANEL, troughcolor=BG_CARD)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._txt_body = tk.Text(
            frame_text, wrap=tk.WORD, bg=BG_PANEL, fg=FG_WHITE,
            font=("Segoe UI", 8, "italic"), relief=tk.FLAT, bd=0,
            spacing1=2, spacing2=1, insertbackground=FG_WHITE,
            yscrollcommand=scroll.set, state=tk.DISABLED)
        self._txt_body.pack(fill=tk.BOTH, expand=True)
        scroll.config(command=self._txt_body.yview)

        # Click readout at bottom of narrative
        tk.Frame(right, bg=FG_MUTED, height=1).pack(fill=tk.X, padx=16, pady=4)
        self._lbl_click = tk.Label(
            right, text="Click the map to explore values",
            bg=BG_PANEL, fg=FG_DIM, font=("Segoe UI", 7, "italic"),
            wraplength=260, justify=tk.LEFT, anchor=tk.W)
        self._lbl_click.pack(padx=10, pady=(0, 8), anchor=tk.W)

    def _build_bottom_bar(self):
        """Chapter selector + controls along bottom."""
        bar = tk.Frame(self.root, bg=BG_CARD, height=60)
        bar.pack(fill=tk.X, side=tk.BOTTOM)
        bar.pack_propagate(False)

        # Chapter buttons
        chap_frame = tk.Frame(bar, bg=BG_CARD)
        chap_frame.pack(side=tk.LEFT, padx=10, pady=6)
        tk.Label(chap_frame, text="CHAPTER:", bg=BG_CARD, fg=FG_DIM,
                 font=("Segoe UI", 7, "bold")).pack(anchor=tk.W)
        btn_row = tk.Frame(chap_frame, bg=BG_CARD)
        btn_row.pack()
        self._chap_btns: list[tk.Button] = []
        for i, ch in enumerate(CHAPTERS):
            b = tk.Button(
                btn_row,
                text=f"  {ch['icon']}  {ch['title']}  ",
                command=lambda idx=i: self._select_chapter(idx),
                bg=BG_CARD, fg=FG_WHITE, activebackground=ACCENT,
                activeforeground=BG_DARK,
                font=("Segoe UI", 8, "bold"), relief=tk.FLAT, bd=0,
                padx=8, pady=4, cursor="hand2",
                **self._control_style_kwargs(BG_CARD))
            b.pack(side=tk.LEFT, padx=2)
            self._chap_btns.append(b)

        # Separator
        tk.Frame(bar, bg=FG_DIM, width=1).pack(side=tk.LEFT, fill=tk.Y,
                                                padx=6, pady=8)

        # Controls
        ctrl = tk.Frame(bar, bg=BG_CARD)
        ctrl.pack(side=tk.LEFT, padx=6, pady=4)

        tk.Checkbutton(ctrl, text="Annotations",
                       variable=self._show_annotations,
                       command=self._refresh_chapter,
                       bg=BG_CARD, fg=FG_WHITE, selectcolor=BG_DARK,
                       activebackground=BG_CARD,
                       font=("Segoe UI", 8),
                       **self._control_style_kwargs(BG_CARD)).grid(
                           row=0, column=0, padx=4)

        tk.Checkbutton(ctrl, text="PSR overlay",
                       variable=self._show_psr,
                       command=self._refresh_chapter,
                       bg=BG_CARD, fg=FG_WHITE, selectcolor=BG_DARK,
                       activebackground=BG_CARD,
                       font=("Segoe UI", 8),
                       **self._control_style_kwargs(BG_CARD)).grid(
                           row=0, column=1, padx=4)

        tk.Label(ctrl, text="Scale:", bg=BG_CARD, fg=FG_WHITE,
                 font=("Segoe UI", 8)).grid(row=0, column=2, padx=(10, 2))
        scales = [ds.scale_label for ds in self.store.heightmaps] or ["(none)"]
        self._scale_var = tk.StringVar(value=scales[0])
        cb = ttk.Combobox(ctrl, textvariable=self._scale_var,
                          values=scales, state="readonly", width=16,
                          font=("Segoe UI", 8))
        cb.grid(row=0, column=3, padx=4)
        cb.bind("<<ComboboxSelected>>", self._on_scale_change)

        # Zoom controls
        tk.Frame(bar, bg=FG_DIM, width=1).pack(side=tk.LEFT, fill=tk.Y,
                                                padx=6, pady=8)
        zoom_frame = tk.Frame(bar, bg=BG_CARD)
        zoom_frame.pack(side=tk.LEFT, padx=4)
        tk.Button(zoom_frame, text="Reset Zoom", command=self._reset_zoom,
                  bg=BG_PANEL, fg=FG_WHITE, activebackground=ACCENT,
                  font=("Segoe UI", 7, "bold"), relief=tk.FLAT,
                  padx=6, pady=2, cursor="hand2",
                  **self._control_style_kwargs(BG_PANEL)).pack(
                      side=tk.LEFT, padx=2)
        tk.Button(zoom_frame, text="Zoom Back", command=self._zoom_back,
                  bg=BG_PANEL, fg=FG_WHITE, activebackground=ACCENT,
                  font=("Segoe UI", 7, "bold"), relief=tk.FLAT,
                  padx=6, pady=2, cursor="hand2",
                  **self._control_style_kwargs(BG_PANEL)).pack(
                      side=tk.LEFT, padx=2)
        tk.Button(zoom_frame, text="Clear Marker", command=self._clear_marker,
                  bg=BG_PANEL, fg=FG_WHITE, activebackground=ACCENT,
                  font=("Segoe UI", 7, "bold"), relief=tk.FLAT,
                  padx=6, pady=2, cursor="hand2",
                  **self._control_style_kwargs(BG_PANEL)).pack(
                      side=tk.LEFT, padx=2)

        # Status
        self._lbl_status = tk.Label(
            bar, text="", bg=BG_CARD, fg=FG_DIM,
            font=("Segoe UI", 7, "italic"), anchor=tk.E)
        self._lbl_status.pack(side=tk.RIGHT, padx=10)

    # ── chapter switching ────────────────────────────────────────────────────

    def _select_chapter(self, idx):
        self._chapter_idx = idx
        self._zoom_stack.clear()
        self._cur_xlim = self._cur_ylim = None
        self._base_xlim = self._base_ylim = None
        self._click_col = self._click_row = None
        self._refresh_chapter()

    def _refresh_chapter(self, *_):
        ch = CHAPTERS[self._chapter_idx]

        # Update button highlights
        for i, b in enumerate(self._chap_btns):
            if i == self._chapter_idx:
                cfg = {
                    "bg": GOLD,
                    "fg": BG_DARK,
                    "font": ("Segoe UI", 8, "bold"),
                }
                cfg.update(self._control_style_kwargs(GOLD))
                b.configure(**cfg)
            else:
                cfg = {
                    "bg": BG_CARD,
                    "fg": FG_WHITE,
                    "font": ("Segoe UI", 8),
                }
                cfg.update(self._control_style_kwargs(BG_CARD))
                b.configure(**cfg)

        # Update narrative
        self._lbl_chnum.configure(text=ch["icon"])
        self._lbl_title.configure(text=ch["title"])
        self._lbl_sub.configure(text=ch["sub"])
        self._txt_body.configure(state=tk.NORMAL)
        self._txt_body.delete("1.0", tk.END)
        self._txt_body.insert(tk.END, ch["narrative"])
        self._txt_body.configure(state=tk.DISABLED)

        # Rebuild figure
        self._fig.clf()
        self._selector = None
        self._map_ax = None
        self._click_marker_artists = []
        key = ch["key"]
        if key == "landscape":
            self._draw_landscape()
        elif key == "shadow":
            self._draw_shadow()
        elif key == "sunlight":
            self._draw_sunlight()
        elif key == "suitability":
            self._draw_suitability()

        # Save default map limits and then restore user zoom if active.
        if self._map_ax is not None:
            self._base_xlim = self._map_ax.get_xlim()
            self._base_ylim = self._map_ax.get_ylim()
        self._apply_zoom_to_map()

        # Click marker
        self._draw_click_marker()

        # Rectangle selector for zoom
        if self._map_ax is not None:
            self._selector = RectangleSelector(
                self._map_ax, self._on_rect_select,
                useblit=True, button=[1],
                minspanx=5, minspany=5, spancoords="pixels",
                interactive=True,
                props=dict(facecolor="cyan", edgecolor="white",
                           alpha=0.15, linewidth=1))

        # Status
        hm = self._current_heightmap()
        self._lbl_status.configure(
            text=f"Dataset: {hm.name if hm else '-'}  |  "
                 f"Chapter {self._chapter_idx + 1}/4")
        self._canvas.draw_idle()

    # ── data helpers ─────────────────────────────────────────────────────────

    def _current_heightmap(self):
        if not self.store.heightmaps:
            return None
        return self.store.heightmaps[min(self._scale_idx,
                                         len(self.store.heightmaps) - 1)]

    def _current_illumination(self):
        if not self.store.illumination:
            return None
        return self.store.illumination[min(self._scale_idx,
                                            len(self.store.illumination) - 1)]

    def _get_cached(self, key):
        return self._cache.get((self._scale_idx, key))

    def _set_cached(self, key, val):
        self._cache[(self._scale_idx, key)] = val

    def _get_elev(self):
        v = self._get_cached("elev")
        if v is None:
            hm = self._current_heightmap()
            v = hm.data if hm else np.zeros((100, 100))
            self._set_cached("elev", v)
        return v

    def _get_illum(self):
        v = self._get_cached("illum")
        if v is None:
            il = self._current_illumination()
            v = il.data if il else None
            self._set_cached("illum", v)
        return v

    def _get_hillshade(self):
        v = self._get_cached("hs")
        if v is None:
            v = compute_hillshade(self._get_elev())
            self._set_cached("hs", v)
        return v

    def _get_illum_aligned(self):
        v = self._get_cached("illum_a")
        if v is None:
            raw = self._get_illum()
            v = resample_to(raw, self._get_elev().shape, 1) if raw is not None else None
            self._set_cached("illum_a", v)
        return v

    def _get_psr_mask(self):
        k = f"psr_{self._psr_threshold.get():.3f}"
        v = self._get_cached(k)
        if v is None:
            raw_illum = self._get_illum()
            if raw_illum is not None:
                raw_mask = compute_psr_mask(raw_illum, self._psr_threshold.get())
                v = resample_to(raw_mask.astype(np.float32),
                                self._get_elev().shape, 0) > 0.5
            else:
                v = np.zeros(self._get_elev().shape, dtype=bool)
            self._set_cached(k, v)
        return v

    def _get_suitability(self):
        v = self._get_cached("suit")
        if v is None:
            v = compute_suitability(self._get_elev(),
                                    self._get_illum_aligned(),
                                    self._get_psr_mask())
            self._set_cached("suit", v)
        return v

    def _get_slope(self):
        v = self._get_cached("slope")
        if v is None:
            v = compute_slope_deg(self._get_elev())
            self._set_cached("slope", v)
        return v

    def _norm_illum(self, illum=None):
        """Return illumination normalised to [0,1] (2nd–98th percentile)."""
        if illum is None:
            illum = self._get_illum_aligned()
        if illum is None:
            return None
        lo = np.nanpercentile(illum, 2)
        hi = np.nanpercentile(illum, 98)
        return np.clip((illum - lo) / (hi - lo + 1e-9), 0, 1)

    def _get_norm_illum(self):
        v = self._get_cached("norm_illum")
        if v is None:
            illum = self._get_illum_aligned()
            v = self._norm_illum(illum) if illum is not None else None
            self._set_cached("norm_illum", v)
        return v

    # =====================================================================
    #  CHAPTER 1: THE LANDSCAPE
    #  SciVis: hillshaded terrain map
    #  InfoVis: elevation histogram, Earth-comparison bar chart
    #  Infographic: key statistics strip
    # =====================================================================

    def _draw_landscape(self):
        gs = self._chapter_gridspec()

        # ── main map: hillshaded terrain ──────────────────────────────────
        ax_map = self._fig.add_subplot(gs[:, 0])
        self._map_ax = ax_map

        elev = self._get_elev()
        hs   = self._get_hillshade()
        H, W = elev.shape
        vmin = float(np.nanpercentile(elev, 2))
        vmax = float(np.nanpercentile(elev, 98))

        norm_e = np.clip((elev - vmin) / (vmax - vmin + 1e-9), 0, 1)
        rgba = matplotlib.colormaps["terrain"](norm_e)
        rgba[..., :3] *= hs[..., np.newaxis] * 0.6 + 0.4
        rgba[..., 3] = 1.0
        ax_map.imshow(rgba, origin="upper", aspect="equal",
                      interpolation="bilinear")

        # Colourbar for elevation
        sm = mcm.ScalarMappable(cmap="terrain",
                                norm=Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        self._add_horizontal_map_colorbar(ax_map, sm, "Elevation (m)")
        style_axis_dark(ax_map, "Terrain Map (Hillshaded)",
                         "West \u2190 East (px)", "South \u2190 North (px)",
                         title_size=10)

        if self._show_annotations.get():
            self._annotate_landscape(ax_map, elev)

        # ── histogram: elevation distribution (InfoVis) ───────────────────
        ax_hist = self._fig.add_subplot(gs[0, 1])
        valid = elev[np.isfinite(elev)].ravel()
        band_edges = np.linspace(vmin, vmax, 7)  # wider bins for readable labels
        counts, edges = np.histogram(valid, bins=band_edges)
        centers = (edges[:-1] + edges[1:]) / 2
        widths = np.diff(edges) * 0.86
        bars = ax_hist.bar(
            centers, counts, width=widths, align="center",
            color=ACCENT, alpha=0.85, edgecolor=BG_PLOT, linewidth=0.35)

        # Put bin labels directly on bars to avoid axis-label overflow.
        total = max(valid.size, 1)
        for bar, lo, hi, c in zip(bars, edges[:-1], edges[1:], counts):
            if c <= 0:
                continue
            label = (
                f"{lo/1000:+.1f}→{hi/1000:+.1f} km\n"
                f"{(c/total)*100:.1f}%"
            )
            y = max(c * 0.55, 1)
            t = ax_hist.text(
                bar.get_x() + bar.get_width() / 2, y, label,
                ha="center", va="center", fontsize=5.2, color=FG_WHITE,
                zorder=6, clip_on=True)
            t.set_path_effects([
                pe.withStroke(linewidth=1.6, foreground="#000000bb")
            ])

        ax_hist.axvline(0, color=GOLD, ls="--", lw=0.8, label="Sea level equiv.")
        ax_hist.legend(fontsize=5.5, facecolor=BG_CARD, edgecolor=FG_DIM,
                       labelcolor=FG_WHITE, loc="upper right")
        ax_hist.set_xticks([])
        style_axis_dark(ax_hist, "Elevation Distribution",
                         "", "Pixel count")

        # ── comparison bar: Moon vs Earth (Infographic) ───────────────────
        ax_comp = self._fig.add_subplot(gs[1, 1])
        relief = vmax - vmin
        comparisons = {
            "Moon South\nPole relief": relief,
            "Mt Everest\n(8,849 m)": 8849,
            "Grand Canyon\n(1,857 m)": 1857,
            "Mariana Trench\n(10,994 m)": 10994,
        }
        names = list(comparisons.keys())
        vals  = list(comparisons.values())
        colors = [ACCENT, "#66bb6a", "#ffa726", "#ef5350"]
        bars = ax_comp.barh(names, vals, color=colors, height=0.6,
                            edgecolor=BG_PLOT)
        for bar, v in zip(bars, vals):
            ax_comp.text(bar.get_width() + 150, bar.get_y() + bar.get_height()/2,
                         f"{v:,.0f} m", va="center", fontsize=6, color=FG_WHITE)
        style_axis_dark(ax_comp, "How Extreme Is It?\n(vs Earth)",
                         "Metres", "")
        self._keep_ylabels_inside(ax_comp, fontsize=5.4)
        ax_comp.invert_yaxis()

        # ── key statistics strip (Infographic) ────────────────────────────
        ax_s1 = self._fig.add_subplot(gs[0, 2])
        big_stat(ax_s1, f"{relief/1000:.0f}", "km",
                 "Total relief", ACCENT)

        ax_s2 = self._fig.add_subplot(gs[0, 3])
        big_stat(ax_s2, "1.62", "m/s\u00b2",
                 "Surface gravity", GOLD)

        ax_s3 = self._fig.add_subplot(gs[1, 2])
        big_stat(ax_s3, f"{vmin/1000:+.1f}", "km",
                 "Deepest point", "#42a5f5")

        ax_s4 = self._fig.add_subplot(gs[1, 3])
        big_stat(ax_s4, f"{vmax/1000:+.1f}", "km",
                 "Highest ridge", WARM_ORG)

    def _annotate_landscape(self, ax, elev):
        """Data-driven annotations pointing to real minima/maxima."""
        H, W = elev.shape
        mr, mc = max(1, H // 10), max(1, W // 10)
        inner = elev[mr:H-mr, mc:W-mc].astype(np.float64)
        inner = np.where(np.isfinite(inner), inner, np.nanmedian(inner))
        smooth = ndimage.gaussian_filter(inner, sigma=max(H, W) / 60.0)

        min_ri, min_ci = np.unravel_index(np.argmin(smooth), smooth.shape)
        max_ri, max_ci = np.unravel_index(np.argmax(smooth), smooth.shape)
        min_r, min_c = min_ri + mr, min_ci + mc
        max_r, max_c = max_ri + mr, max_ci + mc

        min_e = float(elev[min_r, min_c])
        max_e = float(elev[max_r, max_c])

        def _push(r, c, d=0.22):
            cx, cy = W / 2, H / 2
            dx, dy = c - cx, r - cy
            ln = max((dx**2 + dy**2)**0.5, 1)
            return (float(np.clip(c + d * W * dx / ln, W * .06, W * .92)),
                    float(np.clip(r + d * H * dy / ln, H * .06, H * .92)))

        tx, ty = _push(min_r, min_c)
        ax.annotate(f"Deepest crater\n{min_e/1000:+.2f} km",
                    xy=(min_c, min_r), xytext=(tx, ty),
                    arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1),
                    color=ACCENT, fontsize=6.5, ha="center",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#00000088",
                              ec=ACCENT, lw=0.7))

        tx2, ty2 = _push(max_r, max_c)
        ax.annotate(f"Highest ridge\n{max_e/1000:+.2f} km",
                    xy=(max_c, max_r), xytext=(tx2, ty2),
                    arrowprops=dict(arrowstyle="->", color=GOLD, lw=1),
                    color=GOLD, fontsize=6.5, ha="center",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#00000088",
                              ec=GOLD, lw=0.7))

    # =====================================================================
    #  CHAPTER 2: LIGHT & SHADOW
    #  SciVis: illumination map with PSR overlay
    #  InfoVis: donut chart (% area), bar chart (PSR vs sunlit conditions)
    #  Infographic: key statistics
    # =====================================================================

    def _draw_shadow(self):
        gs = self._chapter_gridspec()

        illum = self._get_illum_aligned()
        norm_ill = self._get_norm_illum()
        elev  = self._get_elev()
        H, W  = elev.shape

        # ── main map: illumination ────────────────────────────────────────
        ax_map = self._fig.add_subplot(gs[:, 0])
        self._map_ax = ax_map

        if norm_ill is not None:
            im = ax_map.imshow(norm_ill * 100, cmap="plasma",
                               vmin=0, vmax=100, origin="upper",
                               aspect="equal", interpolation="bilinear")
            self._add_horizontal_map_colorbar(
                ax_map, im, "% Time Illuminated")

            if self._show_psr.get():
                psr = self._get_psr_mask()
                psr_rgba = np.zeros((H, W, 4), dtype=np.float32)
                psr_rgba[psr] = [0.08, 0.39, 0.75, 0.50]
                ax_map.imshow(psr_rgba, origin="upper", aspect="equal",
                              interpolation="none")
        else:
            ax_map.text(0.5, 0.5, "No illumination data",
                        transform=ax_map.transAxes, ha="center", va="center",
                        color=FG_WHITE, fontsize=11)

        style_axis_dark(ax_map, "Illumination Map with PSR Overlay",
                         "West \u2190 East (px)", "South \u2190 North (px)",
                         title_size=10)

        if self._show_annotations.get() and norm_ill is not None:
            self._annotate_shadow(ax_map, norm_ill)

        # ── donut chart: % area in shadow (InfoVis) ───────────────────────
        ax_donut = self._fig.add_subplot(gs[0, 1])
        ax_donut.set_facecolor(BG_PLOT)
        psr = self._get_psr_mask()
        psr_frac = float(psr.sum()) / max(psr.size, 1)
        lit_frac = 1.0 - psr_frac
        wedge_colors = [PSR_CLR, GOLD]
        sizes = [psr_frac, lit_frac]
        wedges, texts = ax_donut.pie(
            sizes, colors=wedge_colors, startangle=90,
            wedgeprops=dict(width=0.35, edgecolor=BG_PLOT, linewidth=1.5))
        ax_donut.text(0, 0, f"{psr_frac*100:.1f}%\nPSR",
                      ha="center", va="center", fontsize=9,
                      fontweight="bold", color=PSR_CLR)
        ax_donut.set_title("Area in Permanent Shadow",
                            color=FG_WHITE, fontsize=8.5, fontweight="bold",
                            pad=6)
        # legend
        import matplotlib.patches as mpatches
        ax_donut.legend(
            [mpatches.Patch(color=PSR_CLR), mpatches.Patch(color=GOLD)],
            [f"Shadow ({psr_frac*100:.1f}%)", f"Sunlit ({lit_frac*100:.1f}%)"],
            fontsize=5.5, facecolor=BG_CARD, edgecolor=FG_DIM,
            labelcolor=FG_WHITE, loc="lower center",
            bbox_to_anchor=(0.5, -0.12))

        # ── bar chart: PSR vs Sunlit conditions (Infographic) ─────────────
        ax_bar = self._fig.add_subplot(gs[1, 1])
        categories = ["Temperature", "Water ice\npotential", "Solar power\npotential"]
        psr_vals   = [40, 95, 5]     # relative scale for visual comparison
        lit_vals   = [250, 10, 90]
        x = np.arange(len(categories))
        w = 0.35
        ax_bar.barh(x - w/2, psr_vals, w, color=PSR_CLR, label="PSR (shadow)")
        ax_bar.barh(x + w/2, lit_vals, w, color=GOLD, label="Sunlit ridge")
        ax_bar.set_yticks(x)
        ax_bar.set_yticklabels(categories, fontsize=6)
        ax_bar.legend(fontsize=5.5, facecolor=BG_CARD, edgecolor=FG_DIM,
                      labelcolor=FG_WHITE, loc="lower right")
        style_axis_dark(ax_bar, "Shadow vs Sunlight\nConditions",
                         "Relative score", "")
        self._keep_ylabels_inside(ax_bar, fontsize=5.4)
        ax_bar.invert_yaxis()

        # ── key stats ─────────────────────────────────────────────────────
        ax_s1 = self._fig.add_subplot(gs[0, 2])
        big_stat(ax_s1, "\u2212230", "\u00b0C",
                 "PSR temperature", PSR_CLR)

        ax_s2 = self._fig.add_subplot(gs[0, 3])
        big_stat(ax_s2, "~1.6", "degrees",
                 "Max sun elevation", GOLD)

        ax_s3 = self._fig.add_subplot(gs[1, 2])
        big_stat(ax_s3, "2009", "",
                 "LCROSS ice discovery", ACCENT)

        ax_s4 = self._fig.add_subplot(gs[1, 3])
        psr_area_est = psr_frac * H * W * 0.001  # rough pixel-based estimate
        big_stat(ax_s4, f"{psr_frac*100:.0f}%", "of area",
                 "Permanently shadowed", DANGER)

    def _annotate_shadow(self, ax, norm_illum):
        H, W = norm_illum.shape
        blurred = ndimage.gaussian_filter(norm_illum, sigma=max(H, W) / 40)
        mr, mc = max(1, H // 10), max(1, W // 10)
        inner = blurred[mr:H-mr, mc:W-mc]

        dr, dc = np.unravel_index(np.argmin(inner), inner.shape)
        br, bc = np.unravel_index(np.argmax(inner), inner.shape)
        dr += mr; dc += mc
        br += mr; bc += mc

        ax.annotate("Permanently\nShadowed Region",
                    xy=(dc, dr),
                    xytext=(float(np.clip(dc + W * .18, W * .05, W * .88)),
                            float(np.clip(dr + H * .12, H * .05, H * .88))),
                    arrowprops=dict(arrowstyle="->", color="#80c8ff", lw=1),
                    color="#80c8ff", fontsize=6.5, ha="center",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#00000099",
                              ec="#80c8ff", lw=0.7))
        ax.annotate("Peak of\nEternal Light",
                    xy=(bc, br),
                    xytext=(float(np.clip(bc - W * .16, W * .05, W * .88)),
                            float(np.clip(br + H * .12, H * .05, H * .88))),
                    arrowprops=dict(arrowstyle="->", color=GOLD, lw=1),
                    color=GOLD, fontsize=6.5, ha="center",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#00000099",
                              ec=GOLD, lw=0.7))

    # =====================================================================
    #  CHAPTER 3: SUNLIGHT & POWER
    #  SciVis: illumination contour map with hillshade base
    #  InfoVis: illumination band bar chart, elevation cross-section
    #  Infographic: key statistics
    # =====================================================================

    def _draw_sunlight(self):
        gs = self._chapter_gridspec()

        elev  = self._get_elev()
        hs    = self._get_hillshade()
        norm_ill = self._get_norm_illum()
        H, W  = elev.shape

        # ── main map: hillshade + illumination contours ───────────────────
        ax_map = self._fig.add_subplot(gs[:, 0])
        self._map_ax = ax_map

        ax_map.imshow(hs, cmap="gray", vmin=0, vmax=1, origin="upper",
                      aspect="equal", interpolation="bilinear", alpha=0.75)

        if norm_ill is not None:
            ax_map.imshow(norm_ill * 100, cmap="YlOrRd", vmin=0, vmax=100,
                          alpha=0.50, origin="upper", aspect="equal",
                          interpolation="bilinear")

            # Smooth before contouring (avoid scan-line artefacts; Lec 4)
            smooth = ndimage.gaussian_filter(norm_ill * 100,
                                              sigma=max(H, W) / 40)
            lvls = [20, 40, 60, 80]
            ct = ax_map.contour(smooth, levels=lvls,
                                colors=[ACCENT, ACCENT, GOLD, GOLD],
                                linewidths=[0.6, 0.6, 0.8, 0.8], alpha=0.8)
            ax_map.clabel(ct, inline=True, fontsize=5.5,
                          fmt={20: "20%", 40: "40%", 60: "60%", 80: "80%"},
                          colors=FG_WHITE)

            sm = mcm.ScalarMappable(cmap="YlOrRd",
                                    norm=Normalize(vmin=0, vmax=100))
            sm.set_array([])
            self._add_horizontal_map_colorbar(
                ax_map, sm, "% Time Illuminated")

            if self._show_psr.get():
                psr = self._get_psr_mask()
                psr_rgba = np.zeros((H, W, 4), dtype=np.float32)
                psr_rgba[psr] = [0.08, 0.39, 0.75, 0.40]
                ax_map.imshow(psr_rgba, origin="upper", aspect="equal",
                              interpolation="none")

        style_axis_dark(ax_map, "Sunlight Contour Map (Isolines, Lec 4)",
                         "West \u2190 East (px)", "South \u2190 North (px)",
                         title_size=10)

        if self._show_annotations.get() and norm_ill is not None:
            blurred = ndimage.gaussian_filter(norm_ill, sigma=max(H, W) / 40)
            mr, mc = max(1, H // 10), max(1, W // 10)
            inner = blurred[mr:H-mr, mc:W-mc]
            br, bc = np.unravel_index(np.argmax(inner), inner.shape)
            br += mr; bc += mc
            ax_map.annotate("Best solar\npanel site",
                            xy=(bc, br),
                            xytext=(float(np.clip(bc - W * .15, W * .05, W * .88)),
                                    float(np.clip(br + H * .12, H * .05, H * .88))),
                            arrowprops=dict(arrowstyle="->", color=GOLD, lw=1),
                            color=GOLD, fontsize=6.5, ha="center",
                            bbox=dict(boxstyle="round,pad=0.25",
                                      fc="#00000099", ec=GOLD, lw=0.7))

        # ── bar chart: illumination bands (InfoVis) ───────────────────────
        ax_bands = self._fig.add_subplot(gs[0, 1])
        if norm_ill is not None:
            illum_pct = norm_ill * 100
            valid = illum_pct[np.isfinite(illum_pct)].ravel()
            band_edges = [0, 20, 40, 60, 80, 100]
            band_labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
            counts = []
            for lo, hi in zip(band_edges[:-1], band_edges[1:]):
                counts.append(np.sum((valid >= lo) & (valid < hi)))
            band_colors = ["#311b92", "#7b1fa2", "#f57f17", "#ff8f00", "#ffd54f"]
            ax_bands.barh(band_labels, counts, color=band_colors, height=0.6,
                          edgecolor=BG_PLOT)
            for i, (c, lbl) in enumerate(zip(counts, band_labels)):
                pct = c / max(valid.size, 1) * 100
                ax_bands.text(c + valid.size * 0.02, i,
                              f"{pct:.1f}%", va="center",
                              fontsize=5.5, color=FG_WHITE)
        style_axis_dark(ax_bands, "Illumination Band\nDistribution",
                         "Pixel count", "")
        self._keep_ylabels_inside(ax_bands, fontsize=5.5)
        ax_bands.invert_yaxis()

        # ── cross-section profile (InfoVis: filtering/slicing, Lec 2) ─────
        ax_prof = self._fig.add_subplot(gs[1, 1])
        if norm_ill is not None:
            norm_ill_full = norm_ill
            blurred = ndimage.gaussian_filter(norm_ill_full, sigma=max(H, W)/40)
            mr, mc = max(1, H // 10), max(1, W // 10)
            inner_b = blurred[mr:H-mr, mc:W-mc]
            br, bc = np.unravel_index(np.argmax(inner_b), inner_b.shape)
            br += mr; bc += mc
            # Horizontal cross-section through brightest point
            row = br
            elev_slice = elev[row, :]
            illum_slice = norm_ill_full[row, :] * 100
            x_px = np.arange(W)

            ax_prof.fill_between(x_px, elev_slice, alpha=0.3, color=ACCENT,
                                 label="Elevation (m)")
            ax_prof.plot(x_px, elev_slice, color=ACCENT, lw=0.8)
            ax2 = ax_prof.twinx()
            ax2.plot(x_px, illum_slice, color=GOLD, lw=0.8,
                     label="Illumination %")
            ax2.set_ylabel("Illumination %", color=GOLD, fontsize=6)
            ax2.tick_params(colors=GOLD, labelsize=5.5)
            ax2.set_ylim(0, 105)

            # Draw line on map showing cross-section location
            ax_map.axhline(y=row, color=SAFE_CLR, ls="--", lw=0.7, alpha=0.7)
            ax_map.text(5, row - H * 0.02, f"cross-section (row {row})",
                        color=SAFE_CLR, fontsize=5, alpha=0.8)

        style_axis_dark(ax_prof, "Elevation + Illumination Cross-Section",
                         "Pixel column", "Elevation (m)")

        # ── key stats ─────────────────────────────────────────────────────
        ax_s1 = self._fig.add_subplot(gs[0, 2])
        big_stat(ax_s1, ">80%", "sunlit",
                 "Best ridges", GOLD)

        ax_s2 = self._fig.add_subplot(gs[0, 3])
        big_stat(ax_s2, "1,360", "W/m\u00b2",
                 "Solar irradiance", WARM_ORG)

        ax_s3 = self._fig.add_subplot(gs[1, 2])
        big_stat(ax_s3, "<1", "km",
                 "PEL to PSR distance", SAFE_CLR)

        ax_s4 = self._fig.add_subplot(gs[1, 3])
        big_stat(ax_s4, "H\u2082O", "+ O\u2082",
                 "Ice splits to fuel", "#42a5f5")

    # =====================================================================
    #  CHAPTER 4: MISSION READY
    #  SciVis: suitability heatmap (diverging colourmap, Lec 7)
    #  InfoVis: radar chart (multi-criteria), stacked-bar breakdown
    #  Infographic: key statistics
    # =====================================================================

    def _draw_suitability(self):
        gs = self._chapter_gridspec()

        suit = self._get_suitability()
        hs   = self._get_hillshade()
        psr  = self._get_psr_mask()
        elev = self._get_elev()
        H, W = suit.shape

        # ── main map: suitability heatmap ─────────────────────────────────
        ax_map = self._fig.add_subplot(gs[:, 0])
        self._map_ax = ax_map

        ax_map.imshow(hs, cmap="gray", vmin=0, vmax=1, origin="upper",
                      aspect="equal", interpolation="bilinear", alpha=0.3)
        im = ax_map.imshow(suit, cmap="RdYlGn", vmin=0, vmax=1,
                           alpha=0.8, origin="upper", aspect="equal",
                           interpolation="bilinear")
        self._add_horizontal_map_colorbar(
            ax_map, im, "Landing Suitability",
            ticks=[0, 0.5, 1], ticklabels=["Avoid", "Caution", "Ideal"])

        # Highlight top 5%
        threshold = float(np.nanpercentile(suit, 95))
        best_mask = suit >= threshold
        best_rgba = np.zeros((H, W, 4), dtype=np.float32)
        best_rgba[best_mask] = [0.0, 0.9, 0.46, 0.55]
        ax_map.imshow(best_rgba, origin="upper", aspect="equal",
                      interpolation="none")

        if self._show_psr.get():
            ax_map.contour(psr.astype(float), levels=[0.5],
                           colors=[PSR_CLR], linewidths=[1], alpha=0.8)

        style_axis_dark(ax_map, "Landing Suitability (Diverging, Lec 7)",
                         "West \u2190 East (px)", "South \u2190 North (px)",
                         title_size=10)

        if self._show_annotations.get():
            rows, cols = np.where(best_mask)
            if rows.size:
                cy, cx = float(rows.mean()), float(cols.mean())
                ax_map.annotate("Best candidate\nlanding zone",
                                xy=(cx, cy),
                                xytext=(cx - W * .2, cy - H * .16),
                                arrowprops=dict(arrowstyle="->",
                                                color=SAFE_CLR, lw=1.2),
                                color=SAFE_CLR, fontsize=7, ha="center",
                                fontweight="bold",
                                bbox=dict(boxstyle="round,pad=0.3",
                                          fc="#00000099", ec=SAFE_CLR, lw=0.8))

        # ── radar chart: multi-criteria score (InfoVis) ───────────────────
        ax_radar = self._fig.add_subplot(gs[0, 1], polar=True)
        ax_radar.set_facecolor(BG_PLOT)

        # Compute average scores in best zone
        slope = self._get_slope()
        norm_ill = self._get_norm_illum()
        flat_score = float(np.clip(1 - np.nanmean(slope[best_mask]) / 20, 0, 1))
        if norm_ill is not None:
            light_score = float(np.nanmean(norm_ill[best_mask]))
        else:
            light_score = 0.5
        dist_to_psr = ndimage.distance_transform_edt(~psr)
        psr_prox_raw = np.exp(-((dist_to_psr - 5) ** 2) / (2 * 8 ** 2))
        psr_prox_raw = (psr_prox_raw - psr_prox_raw.min()) / (
            psr_prox_raw.max() - psr_prox_raw.min() + 1e-9)
        psr_score = float(np.nanmean(psr_prox_raw[best_mask]))
        elev_smooth = float(np.clip(
            1 - np.nanstd(elev[best_mask]) / 3000, 0, 1))

        labels = ["Flat terrain", "Sunlight", "Near PSR\n(water ice)",
                  "Smooth\nsurface"]
        scores = [flat_score, light_score, psr_score, elev_smooth]
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        scores_closed = scores + [scores[0]]
        angles_closed = angles + [angles[0]]

        ax_radar.plot(angles_closed, scores_closed, color=SAFE_CLR, lw=1.5)
        ax_radar.fill(angles_closed, scores_closed, color=SAFE_CLR, alpha=0.2)
        ax_radar.set_xticks(angles)
        ax_radar.set_xticklabels(labels, fontsize=5.5, color=FG_WHITE)
        ax_radar.set_ylim(0, 1)
        ax_radar.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax_radar.set_yticklabels(["0.25", "0.50", "0.75", "1.00"],
                                  fontsize=5, color=FG_DIM)
        ax_radar.set_title("Best Zone — Criteria Scores",
                            color=FG_WHITE, fontsize=8.5, fontweight="bold",
                            pad=14)
        ax_radar.spines["polar"].set_color(FG_MUTED)
        ax_radar.tick_params(colors=FG_DIM)
        ax_radar.set_facecolor(BG_PLOT)

        # ── stacked bar: factor contributions by zone (InfoVis) ───────────
        ax_stack = self._fig.add_subplot(gs[1, 1])
        # Sample 4 representative zones
        zone_names = ["Best zone\n(top 5%)", "Ridges", "Crater\nfloor", "Slopes"]
        zone_masks = [
            best_mask,
            norm_ill > 0.7 if norm_ill is not None else np.ones_like(best_mask),
            psr,
            slope > 15,
        ]
        flat_scores, illum_scores, psr_scores = [], [], []
        for mask in zone_masks:
            if mask.sum() == 0:
                flat_scores.append(0)
                illum_scores.append(0)
                psr_scores.append(0)
            else:
                flat_scores.append(float(np.clip(
                    1 - np.nanmean(slope[mask]) / 20, 0, 1)))
                illum_scores.append(float(
                    np.nanmean(norm_ill[mask]) if norm_ill is not None else 0.5))
                psr_scores.append(float(np.nanmean(psr_prox_raw[mask])))

        x = np.arange(len(zone_names))
        w_bar = 0.55
        b1 = ax_stack.bar(x, flat_scores, w_bar, label="Flat terrain (50%)",
                          color=SAFE_CLR, alpha=0.8)
        b2 = ax_stack.bar(x, illum_scores, w_bar, bottom=flat_scores,
                          label="Sunlight (30%)", color=GOLD, alpha=0.8)
        bottoms2 = [f + i for f, i in zip(flat_scores, illum_scores)]
        b3 = ax_stack.bar(x, psr_scores, w_bar, bottom=bottoms2,
                          label="PSR proximity (20%)", color=PSR_CLR, alpha=0.8)
        ax_stack.set_xticks(x)
        ax_stack.set_xticklabels(zone_names, fontsize=5.5)
        ax_stack.legend(fontsize=5, facecolor=BG_CARD, edgecolor=FG_DIM,
                        labelcolor=FG_WHITE, loc="upper right")
        style_axis_dark(ax_stack, "Factor Breakdown by Zone",
                         "", "Score")

        # ── key stats ─────────────────────────────────────────────────────
        overall = float(np.nanmean(suit[best_mask])) if best_mask.sum() else 0
        ax_s1 = self._fig.add_subplot(gs[0, 2])
        big_stat(ax_s1, f"{overall:.2f}", "/ 1.0",
                 "Best zone score", SAFE_CLR)

        ax_s2 = self._fig.add_subplot(gs[0, 3])
        big_stat(ax_s2, "<10\u00b0", "slope",
                 "Max safe gradient", GOLD)

        ax_s3 = self._fig.add_subplot(gs[1, 2])
        big_stat(ax_s3, "2026", "",
                 "Artemis III target", ACCENT)

        ax_s4 = self._fig.add_subplot(gs[1, 3])
        big_stat(ax_s4, "1st", "woman",
                 "On the Moon", "#f48fb1")

    # ── interactivity ────────────────────────────────────────────────────

    def _on_press(self, event):
        if event.inaxes == self._map_ax:
            self._press_xy = (event.x, event.y)
        else:
            self._press_xy = None

    def _on_map_click(self, event):
        """Inverse mapping (Lec 2): click pixel → data values."""
        if event.button != 1:
            self._press_xy = None
            return
        if self._press_xy is not None:
            dx = event.x - self._press_xy[0]
            dy = event.y - self._press_xy[1]
            self._press_xy = None
            if dx * dx + dy * dy > 25:
                return  # drag, not click
        else:
            return
        if event.inaxes != self._map_ax or event.xdata is None:
            return

        col = int(round(event.xdata))
        row = int(round(event.ydata))
        self._click_col = col
        self._click_row = row

        lines = [f"Pixel  col={col}, row={row}"]

        hm = self._current_heightmap()
        if hm is not None and 0 <= row < hm.height and 0 <= col < hm.width:
            ev = float(hm.data[row, col])
            lines.append(f"  Elevation: {ev:+.0f} m  ({ev/1000:+.2f} km)")

        illum = self._get_illum_aligned()
        if illum is not None and 0 <= row < illum.shape[0] and 0 <= col < illum.shape[1]:
            norm_illum = self._get_norm_illum()
            norm_val = float(norm_illum[row, col]) if norm_illum is not None else 0.0
            lines.append(f"  Illuminated: {norm_val*100:.1f}%")
            psr = self._get_psr_mask()
            if 0 <= row < psr.shape[0] and 0 <= col < psr.shape[1]:
                lines.append(f"  In shadow: {'Yes' if psr[row, col] else 'No'}")

        ch = CHAPTERS[self._chapter_idx]
        if ch["key"] == "suitability":
            suit = self._get_suitability()
            if 0 <= row < suit.shape[0] and 0 <= col < suit.shape[1]:
                sv = float(suit[row, col])
                label = "Ideal" if sv > 0.7 else "Caution" if sv > 0.4 else "Avoid"
                lines.append(f"  Suitability: {sv:.2f} ({label})")

        self._lbl_click.configure(text="\n".join(lines), fg=ACCENT)
        if ch["key"] == "sunlight":
            self._refresh_chapter()
        else:
            self._draw_click_marker()
            self._canvas.draw_idle()

    def _on_rect_select(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        if None in (x1, x2, y1, y2):
            return
        self._zoom_stack.append((self._cur_xlim, self._cur_ylim))
        self._cur_xlim = (min(x1, x2), max(x1, x2))
        self._cur_ylim = (max(y1, y2), min(y1, y2))
        self._apply_zoom_to_map()
        self._canvas.draw_idle()

    def _zoom_back(self):
        if self._zoom_stack:
            self._cur_xlim, self._cur_ylim = self._zoom_stack.pop()
        else:
            self._cur_xlim = self._cur_ylim = None
        self._apply_zoom_to_map()
        self._canvas.draw_idle()

    def _reset_zoom(self):
        self._zoom_stack.clear()
        self._cur_xlim = self._cur_ylim = None
        self._apply_zoom_to_map()
        self._canvas.draw_idle()

    def _clear_marker(self):
        self._click_col = self._click_row = None
        self._lbl_click.configure(
            text="Click the map to explore values", fg=FG_DIM)
        if CHAPTERS[self._chapter_idx]["key"] == "sunlight":
            self._refresh_chapter()
        else:
            self._clear_click_marker_artists()
            self._canvas.draw_idle()

    def _on_scale_change(self, event=None):
        if not self.store.heightmaps:
            return
        labels = [ds.scale_label for ds in self.store.heightmaps]
        val = self._scale_var.get()
        if val in labels:
            self._scale_idx = labels.index(val)
        self._cache.clear()
        self._refresh_chapter()


# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main():
    store = DataStore(root_dir="dataset")
    if not store.heightmaps and not store.illumination:
        print("[ERROR] No TIFF files found.")
        print("  Expected:  ./dataset/heightmaps/*.tif")
        print("             ./dataset/illumination/*.tif")
        sys.exit(1)

    root = tk.Tk()
    try:
        root.iconbitmap("")
    except Exception:
        pass

    app = ArtemisInfoVis(root, store)
    root.mainloop()


if __name__ == "__main__":
    main()

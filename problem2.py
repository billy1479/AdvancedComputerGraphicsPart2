"""
COMP4097 – Problem 2: Infographic Visualisation for a General Audience
Artemis III: Exploring the Lunar South Pole

Design philosophy (Lectures 7 & 8):
────────────────────────────────────
  INFOGRAPHICS (Lec 8): "Visual representations of information, data or
    knowledge mainly for communication" — complex information explained
    quickly and clearly (Wikipedia, cited in lecture slides).

  NARRATIVE STRUCTURE (Lec 8): The visualisation is structured as a
    four-chapter story, implementing the principle that "visualisation
    tells stories about the data" and that "control of reading order"
    guides the audience's understanding.

  COLOURMAP SELECTION (Lec 7): Rainbow colourmaps are deliberately
    avoided because their luminance varies non-monotonically and warm
    colours attract disproportionate attention (Lec 7 slides).  Instead:
      • 'terrain' / 'gist_earth' for elevation (sequential, intuitive)
      • 'plasma' for illumination percentage (perceptually uniform, Lec 7)
      • Custom green→red diverging scheme for suitability (middle-neutral,
        deviation from average is emphasised — Lec 7 diverging colour scale)

  PERCEPTUAL MAPPING (Lec 8): Colour, shape, and size are used as
    preattentive features to segment regions (PSRs, landing zones) so
    the audience can "scan, recognise, and remember images" quickly
    (Lec 8 – Eye and the Mind slide).

  MULTIPLE LEVELS OF DETAIL (Lec 8): The overview map (regional scale)
    is shown first; the user can drill into specific chapters for finer
    detail — implementing the "overview → detail" hierarchy.

  DATA-INK / PROPORTIONAL INK (Lec 8): Chart decorations are kept
    minimal.  Every coloured area encodes real data; annotations are
    concise and anchored to data features.

  VISUALISATION AMPLIFIES COGNITION (Lec 8): The panels are designed to
    "highlight to focus attention", "control reading order", and
    "provide context" for the numbers shown.

Runs with:
    python3 problem2.py
Dataset: ./dataset/heightmaps/*.tif   ./dataset/illumination/*.tif
"""

import os, sys, glob, textwrap
import tkinter as tk
from tkinter import ttk, font as tkfont
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe
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

# ── calibration constants (derived from metadata inspection; see problem1.py) ──
LDEM_UINT_SCALE_M  = 0.5
LDEM_UINT_OFFSET_M = -10000.0

# ── colour palette for UI (space-themed dark scheme) ──────────────────────────
BG_DARK   = "#0d0d1a"
BG_PANEL  = "#12122a"
BG_CARD   = "#1a1a3a"
FG_WHITE  = "#f0f0ff"
FG_DIM    = "#8888bb"
ACCENT    = "#4fc3f7"    # NASA blue
GOLD      = "#ffd54f"
PSR_CLR   = "#1565c0"    # PSR overlay blue
SAFE_CLR  = "#00e676"    # landing zone green

# ── chapter definitions ────────────────────────────────────────────────────────
CHAPTERS = [
    {
        "key":   "landscape",
        "title": "1 · THE LANDSCAPE",
        "sub":   "What does the lunar south pole look like?",
        "icon":  "🌑",
        "headline": "Mountains, craters, and extreme relief",
        "body": (
            "The lunar south pole is one of the most dramatic terrains in "
            "the Solar System.  Billions of years of meteorite impacts have "
            "carved enormous craters — some wider than Wales.\n\n"
            "The map to the left shows elevation above the lunar reference "
            "sphere (radius 1,737 km).  Blues and greens mark deep crater "
            "floors; yellows and whites mark elevated ridges and peaks.\n\n"
            "KEY FACTS\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "• Lowest point:   ~−8 km  (deep craters)\n"
            "• Highest point:  ~+10 km (polar mountain ridges)\n"
            "• Total relief:   ~18 km  across the region\n"
            "• Surface gravity: 1.62 m/s²  (⅙ of Earth's)\n\n"
            "HOW WE MADE THIS\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "The terrain is derived from the Lunar Reconnaissance Orbiter "
            "Camera (LROC) Digital Elevation Model.  Hillshading is computed "
            "using Lambertian reflectance — the same technique used in "
            "problem 1.  The perceptually-uniform 'terrain' colourmap avoids "
            "the non-linear luminance jumps of rainbow scales (Lec 7)."
        ),
    },
    {
        "key":   "shadow",
        "title": "2 · LIGHT & SHADOW",
        "sub":   "Where the Sun never shines",
        "icon":  "🔦",
        "headline": "Permanent darkness hides a precious resource",
        "body": (
            "Because the Moon's axis is almost perfectly upright, the Sun "
            "skims along the horizon at the poles — never rising high enough "
            "to illuminate the floors of deep craters.\n\n"
            "These Permanently Shadowed Regions (PSRs, shown in blue) have "
            "been in darkness for BILLIONS of years.  Temperatures can drop "
            "to −230 °C — colder than Pluto.  This cold-trap preserves "
            "water ice that likely arrived via comets and asteroids.\n\n"
            "KEY FACTS\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "• PSR area at south pole: ~13,000 km²\n"
            "• Ice confirmed by LCROSS mission (2009)\n"
            "• Temperature in PSR: as low as −238 °C\n"
            "• Sun elevation at pole: max ~1.6°\n\n"
            "WHY THIS MATTERS\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "Water ice is mission-critical: it can be split into hydrogen "
            "fuel and oxygen for breathing.  A site near a PSR gives "
            "astronauts access to this resource without having to travel far."
        ),
    },
    {
        "key":   "sunlight",
        "title": "3 · SUNLIGHT & POWER",
        "sub":   "Where the Sun always shines",
        "icon":  "☀️",
        "headline": "Peaks of Eternal Light power the mission",
        "body": (
            "Just a few kilometres from the PSRs, ridge crests and "
            "elevated peaks receive near-continuous sunlight.  These 'Peaks "
            "of Eternal Light' (PEL) are ideal locations for solar panels.\n\n"
            "The map shows the percentage of time each area is illuminated "
            "(from the LROC illumination dataset).  Yellow regions are "
            "nearly always in sunlight; deep purple areas are the PSRs.\n\n"
            "KEY FACTS\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "• Best-lit ridges: >80 % sunlit time\n"
            "• Haworth crater rim: ~70 % sunlit\n"
            "• Solar power potential: ~1,360 W/m²\n"
            "• Nearest landing zone: <1 km from PEL\n\n"
            "THE PERFECT LOCATION\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "A base camp on a sunlit ridge close to a PSR gets the best of "
            "both worlds: reliable solar power AND access to water ice.  No "
            "other location in the Solar System offers this combination so "
            "conveniently."
        ),
    },
    {
        "key":   "suitability",
        "title": "4 · MISSION READY",
        "sub":   "Where is it safe to land?",
        "icon":  "🚀",
        "headline": "Finding the safest and most valuable landing zone",
        "body": (
            "Choosing a landing site requires balancing many factors:\n\n"
            "  ✔  Flat terrain  (slope < 10°)\n"
            "  ✔  Close to sunlight  (solar power)\n"
            "  ✔  Close to PSR  (water ice access)\n"
            "  ✔  Avoid deep craters and boulders\n\n"
            "The composite 'Suitability Score' (green = ideal, red = avoid) "
            "combines elevation gradient (slope), illumination fraction, and "
            "distance from PSR edges.  The diverging green→red colourmap "
            "uses a neutral midpoint to highlight the best and worst areas "
            "at a glance — a diverging scheme is appropriate here because "
            "the score has a meaningful neutral centre (Lec 7).\n\n"
            "KEY FACTS\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "• Candidate sites: Shackleton crater rim, Haworth, de Gerlache\n"
            "• Landing ellipse size: ~50 m × 100 m\n"
            "• Max safe slope for lander: ~10°\n"
            "• Artemis III target year: 2026–2027\n\n"
            "ARTEMIS III MISSION\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "The first crewed Moon landing since Apollo 17 (1972).  For the "
            "first time, astronauts will explore the lunar south pole — and "
            "a woman will walk on the Moon."
        ),
    },
]


# ═════════════════════════════════════════════════════════════════════════════
#  DATA LOADING  (pipeline stage 1 — reused from problem 1)
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
            import rasterio as _ra
            identity = _ra.Affine.identity()
            has_non_identity = (
                self.transform is not None and
                any(abs(a - b) > 1e-12 for a, b in zip(self.transform, identity))
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
            self.data = self.data[0]
        self.height, self.width = self.data.shape

    def _load_pil(self):
        img = PILImage.open(self.filepath)
        raw = np.array(img)
        self.raw_dtype = str(raw.dtype)
        self.data = raw.astype(np.float64)
        if self.data.ndim == 3:
            self.data = self.data[0]
        self.height, self.width = self.data.shape

    def _capture_raw_stats(self):
        if self.data is not None and self.data.size:
            valid = self.data[np.isfinite(self.data)]
            if valid.size:
                self.raw_min = float(valid.min())
                self.raw_max = float(valid.max())

    def _apply_height_calibration(self):
        """Convert uint16 DN → metres if needed."""
        if self.role != "heightmaps":
            self.calibration_note = "N/A (illumination layer)"
            return
        if self.raw_dtype.startswith(("uint", "int")) or (
            np.isfinite(self.raw_min) and self.raw_min > 100
        ):
            self.data = self.data * LDEM_UINT_SCALE_M + LDEM_UINT_OFFSET_M
            self.calibration_note = (
                f"uint16 DN → m  (×{LDEM_UINT_SCALE_M} + {LDEM_UINT_OFFSET_M})"
            )
        else:
            # float32 already in km; convert to metres for consistency
            if np.isfinite(self.raw_max) and abs(self.raw_max) < 100:
                self.data = self.data * 1000.0
                self.calibration_note = "float32 km × 1000 → m"
            else:
                self.calibration_note = "assumed metres (float)"

    def _clean(self):
        if self.data is None:
            return
        # Clip extreme artefact spikes to 3-sigma
        valid = self.data[np.isfinite(self.data)]
        if valid.size:
            mu, sg = float(valid.mean()), float(valid.std())
            if sg > 0:
                self.data = np.clip(self.data, mu - 6*sg, mu + 6*sg)

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
                seen.add(n); unique.append(f)
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

def compute_hillshade(elev: np.ndarray, azimuth=315.0, altitude=45.0) -> np.ndarray:
    """
    Lambertian hillshade.  The Sun direction is fixed at azimuth 315°
    (north-west), altitude 45° for the landscape chapter — chosen to
    maximise ridge visibility rather than simulate actual solar position.
    (In chapters 2/3 we display actual illumination data instead.)
    """
    az_rad  = np.radians(360.0 - azimuth + 90.0)
    alt_rad = np.radians(altitude)
    dy, dx = np.gradient(elev.astype(np.float32))
    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dy, dx)
    shade = (np.sin(alt_rad) * np.cos(slope)
             + np.cos(alt_rad) * np.sin(slope) * np.cos(az_rad - aspect))
    shade = np.clip(shade, 0.0, 1.0)
    return shade


def compute_slope_deg(elev: np.ndarray) -> np.ndarray:
    """
    Slope in degrees from the Sobel gradient of the elevation field.
    Lecture 7 — derived scalar quantities from gradient: |∇f|.
    """
    sx = ndimage.sobel(elev.astype(np.float32), axis=1)
    sy = ndimage.sobel(elev.astype(np.float32), axis=0)
    grad_mag = np.hypot(sx, sy)
    # approximate scale: pixel spacing ~1 (normalised), so slope in arb units
    return np.degrees(np.arctan(grad_mag / 8.0))


def compute_psr_mask(illum: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """
    Boolean PSR mask: pixels whose time-averaged illumination fraction
    falls below `threshold` are classified as permanently shadowed.

    Uses 2nd–98th percentile normalisation instead of nanmin/nanmax so that
    zero-valued nodata rows at LROC polar image borders are NOT classified as
    PSR.  With nanmin, the border zeros anchor the minimum, making entire
    rows at the image edge pass the threshold and appear as horizontal bars.
    """
    lo = np.nanpercentile(illum, 2)
    hi = np.nanpercentile(illum, 98)
    norm = np.clip((illum - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    return norm < threshold


def compute_suitability(elev: np.ndarray,
                        illum: np.ndarray | None = None,
                        psr_mask: np.ndarray | None = None) -> np.ndarray:
    """
    Composite landing suitability score in [0, 1]:
        score = w_flat * flat_score
              + w_illum * illum_score
              + w_psr   * psr_proximity_score

    Lecture 8 — multi-variate overlay; combining scientific data layers
    into a single human-readable score for a general audience.
    Diverging colourmap (RdYlGn) maps 0→red (avoid), 1→green (ideal).
    """
    h, w = elev.shape

    # 1. Flatness (inverse slope): steep = bad
    slope   = compute_slope_deg(elev)
    flat    = np.clip(1.0 - slope / 20.0, 0.0, 1.0)

    # 2. Illumination score
    if illum is not None:
        if illum.shape != elev.shape:
            from scipy.ndimage import zoom
            zy = elev.shape[0] / illum.shape[0]
            zx = elev.shape[1] / illum.shape[1]
            illum = zoom(illum, (zy, zx), order=1)
        lo = np.nanpercentile(illum, 2)
        hi = np.nanpercentile(illum, 98)
        norm_illum = np.clip((illum - lo) / (hi - lo + 1e-9), 0.0, 1.0)
        illum_score = norm_illum
    else:
        illum_score = np.ones((h, w), dtype=np.float32)

    # 3. PSR proximity (within a few km is good; deep inside is bad)
    if psr_mask is not None:
        if psr_mask.shape != elev.shape:
            from scipy.ndimage import zoom
            zy = elev.shape[0] / psr_mask.shape[0]
            zx = elev.shape[1] / psr_mask.shape[1]
            psr_mask = zoom(psr_mask.astype(float), (zy, zx), order=0) > 0.5
        dist  = ndimage.distance_transform_edt(~psr_mask)
        norm_dist = np.clip(dist / (0.05 * max(h, w)), 0.0, 1.0)
        # ideal distance: 2–15 px from PSR edge → score peaks ~5 px
        psr_prox  = np.exp(-((dist - 5.0) ** 2) / (2 * 8.0**2))
        psr_prox  = (psr_prox - psr_prox.min()) / (psr_prox.max() - psr_prox.min() + 1e-9)
    else:
        psr_prox = np.ones((h, w), dtype=np.float32)

    # weighted sum
    score = 0.5 * flat + 0.3 * illum_score + 0.2 * psr_prox
    score = np.nan_to_num(score, nan=0.0)
    return score.astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ═════════════════════════════════════════════════════════════════════════════

class ArtemisInfoVis:
    """
    Tkinter root + matplotlib canvas: infographic-style public visualisation.

    GUI architecture:
      ┌────────────────────────────────────────────────────────┐
      │  TITLE BAR                                             │
      ├───────────────────────────────┬────────────────────────┤
      │                               │  NARRATIVE PANEL       │
      │   MATPLOTLIB MAP CANVAS       │  (chapter headline +   │
      │                               │   annotated facts)     │
      │                               │                        │
      ├───────────────────────────────┴────────────────────────┤
      │  CHAPTER SELECTOR  │  LAYER CONTROLS  │  STATUS BAR    │
      └────────────────────────────────────────────────────────┘

    Lecture 8 principles applied in layout:
      • "Control reading order" — the title and chapter buttons guide
        the audience from overview to detail.
      • "Provide context" — the narrative panel adds verbal explanation
        alongside the visual, forming a complete infographic.
      • "Highlight to focus attention" — the active chapter button is
        highlighted in gold; the map is annotated with arrows and labels.
    """

    _LAYER_CHOICES = ["Elevation + Hillshade", "Illumination %",
                      "Slope (°)", "Landing Suitability"]

    def __init__(self, root: tk.Tk, store: DataStore):
        self.root  = root
        self.store = store
        self._chapter_idx = 0
        self._scale_idx   = 0
        self._show_annotations = tk.BooleanVar(value=True)
        self._show_psr    = tk.BooleanVar(value=True)
        self._illum_threshold = tk.DoubleVar(value=0.05)
        self._cache: dict = {}
        self._click_info = ""
        # zoom state
        self._zoom_stack: list = []
        self._cur_xlim = None
        self._cur_ylim = None
        # click-marker state
        self._click_col = None
        self._click_row = None
        # drag detection
        self._press_xy = None
        self._selector = None

        self._build_ui()
        self._refresh_chapter()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        self.root.title("Artemis III: Exploring the Lunar South Pole")
        self.root.configure(bg=BG_DARK)
        self.root.geometry("1380x820")
        self.root.minsize(1100, 680)

        self._build_title()
        self._build_main_area()
        self._build_bottom_bar()

    def _build_title(self):
        """
        Title banner — Lec 8: 'Control reading order'; the title is the
        first visual element the audience sees.
        """
        bar = tk.Frame(self.root, bg="#07071a", height=56)
        bar.pack(fill=tk.X)
        bar.pack_propagate(False)

        # Mission logo / icon
        tk.Label(bar, text="🌙", bg="#07071a",
                 font=("Segoe UI Emoji", 26)).pack(side=tk.LEFT, padx=14)

        # Mission title
        tk.Label(bar,
                 text="ARTEMIS III  ·  EXPLORING THE LUNAR SOUTH POLE",
                 bg="#07071a", fg=ACCENT,
                 font=("Segoe UI", 15, "bold")).pack(side=tk.LEFT)

        # Subtitle
        tk.Label(bar,
                 text="  —  An interactive data story for the public",
                 bg="#07071a", fg=FG_DIM,
                 font=("Segoe UI", 11)).pack(side=tk.LEFT)

        # NASA attribution
        tk.Label(bar, text="Data: NASA LROC  |  COMP4097  Visualisation",
                 bg="#07071a", fg=FG_DIM,
                 font=("Segoe UI", 9)).pack(side=tk.RIGHT, padx=14)

    def _build_main_area(self):
        pane = tk.Frame(self.root, bg=BG_DARK)
        pane.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4, 0))

        # ── left: matplotlib canvas ──────────────────────────────────────────
        left = tk.Frame(pane, bg=BG_DARK)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._fig = Figure(figsize=(9, 6.5), facecolor=BG_DARK)
        self._ax  = self._fig.add_subplot(111)
        self._fig.subplots_adjust(left=0.04, right=0.93, top=0.93, bottom=0.07)

        self._canvas = FigureCanvasTkAgg(self._fig, master=left)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._canvas.mpl_connect("button_press_event", self._on_press)
        self._canvas.mpl_connect("button_release_event", self._on_map_click)

        # ── right: narrative panel ───────────────────────────────────────────
        right = tk.Frame(pane, bg=BG_PANEL, width=320)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))
        right.pack_propagate(False)

        # Chapter icon + headline
        self._lbl_icon = tk.Label(right, text="🌑", bg=BG_PANEL, fg=FG_WHITE,
                                  font=("Segoe UI Emoji", 36))
        self._lbl_icon.pack(pady=(18, 2))

        self._lbl_chapter = tk.Label(right, text="", bg=BG_PANEL, fg=ACCENT,
                                     font=("Segoe UI", 11, "bold"),
                                     wraplength=295, justify=tk.CENTER)
        self._lbl_chapter.pack()

        self._lbl_headline = tk.Label(right, text="", bg=BG_PANEL, fg=GOLD,
                                      font=("Segoe UI", 10, "italic"),
                                      wraplength=295, justify=tk.CENTER)
        self._lbl_headline.pack(pady=(4, 8))

        # Separator
        tk.Frame(right, bg=ACCENT, height=1).pack(fill=tk.X, padx=16, pady=2)

        # Scrollable body text
        frame_text = tk.Frame(right, bg=BG_PANEL)
        frame_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        scroll = tk.Scrollbar(frame_text, bg=BG_PANEL, troughcolor=BG_CARD,
                               activebackground=ACCENT)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self._txt_body = tk.Text(
            frame_text, wrap=tk.WORD,
            bg=BG_PANEL, fg=FG_WHITE,
            font=("Segoe UI", 9),
            relief=tk.FLAT, bd=0,
            spacing1=2, spacing2=1,
            insertbackground=FG_WHITE,
            yscrollcommand=scroll.set,
            state=tk.DISABLED
        )
        self._txt_body.pack(fill=tk.BOTH, expand=True)
        scroll.config(command=self._txt_body.yview)

        # Bottom-right click readout is created in _build_bottom_bar().

    def _build_bottom_bar(self):
        bar = tk.Frame(self.root, bg=BG_CARD, height=98)
        bar.pack(fill=tk.X, side=tk.BOTTOM, padx=0, pady=0)
        bar.pack_propagate(False)

        # Chapter selector buttons
        chap_frame = tk.Frame(bar, bg=BG_CARD)
        chap_frame.pack(side=tk.LEFT, padx=12, pady=8)
        tk.Label(chap_frame, text="CHAPTER:", bg=BG_CARD, fg=FG_DIM,
                 font=("Segoe UI", 8, "bold")).pack(anchor=tk.W)
        btn_row = tk.Frame(chap_frame, bg=BG_CARD)
        btn_row.pack()
        self._chap_btns: list[tk.Button] = []
        for i, ch in enumerate(CHAPTERS):
            b = tk.Button(
                btn_row,
                text=f"{ch['icon']}  {ch['title']}",
                command=lambda idx=i: self._select_chapter(idx),
                bg=BG_CARD, fg=FG_WHITE, activebackground=ACCENT,
                activeforeground=BG_DARK,
                font=("Segoe UI", 9), relief=tk.FLAT, bd=0,
                padx=10, pady=5, cursor="hand2"
            )
            b.pack(side=tk.LEFT, padx=3)
            self._chap_btns.append(b)

        # Divider
        tk.Frame(bar, bg=FG_DIM, width=1).pack(side=tk.LEFT, fill=tk.Y,
                                                padx=8, pady=8)

        # Layer / option controls
        ctrl = tk.Frame(bar, bg=BG_CARD)
        ctrl.pack(side=tk.LEFT, padx=8, pady=4)
        tk.Label(ctrl, text="OPTIONS:", bg=BG_CARD, fg=FG_DIM,
                 font=("Segoe UI", 8, "bold")).grid(row=0, column=0,
                                                    columnspan=4, sticky=tk.W)

        tk.Checkbutton(ctrl, text="Show annotations",
                       variable=self._show_annotations,
                       command=self._refresh_chapter,
                       bg=BG_CARD, fg=FG_WHITE, selectcolor=BG_DARK,
                       activebackground=BG_CARD,
                       font=("Segoe UI", 9)).grid(row=1, column=0,
                                                   padx=6, sticky=tk.W)

        tk.Checkbutton(ctrl, text="Highlight PSR",
                       variable=self._show_psr,
                       command=self._refresh_chapter,
                       bg=BG_CARD, fg=FG_WHITE, selectcolor=BG_DARK,
                       activebackground=BG_CARD,
                       font=("Segoe UI", 9)).grid(row=1, column=1,
                                                   padx=6, sticky=tk.W)

        tk.Label(ctrl, text="Scale:", bg=BG_CARD, fg=FG_WHITE,
                 font=("Segoe UI", 9)).grid(row=1, column=2, padx=(12, 2))
        scales = [ds.scale_label for ds in self.store.heightmaps]
        if not scales:
            scales = ["(no data)"]
        self._scale_var = tk.StringVar(value=scales[0])
        scale_cb = ttk.Combobox(ctrl, textvariable=self._scale_var,
                                 values=scales, state="readonly", width=18,
                                 font=("Segoe UI", 9))
        scale_cb.grid(row=1, column=3, padx=4)
        scale_cb.bind("<<ComboboxSelected>>", self._on_scale_change)

        tk.Label(ctrl, text="PSR threshold:",
                 bg=BG_CARD, fg=FG_WHITE,
                 font=("Segoe UI", 9)).grid(row=1, column=4, padx=(12, 2))
        tk.Scale(ctrl, from_=0.01, to=0.20, resolution=0.01,
                 variable=self._illum_threshold,
                 orient=tk.HORIZONTAL, length=110,
                 bg=BG_CARD, fg=FG_WHITE, troughcolor=BG_DARK,
                 activebackground=ACCENT, highlightthickness=0,
                 command=lambda _: self._refresh_chapter()
                 ).grid(row=1, column=5, padx=4)

        # Right: status + interaction controls + click info (bottom-right)
        status = tk.Frame(bar, bg=BG_CARD)
        status.pack(side=tk.RIGHT, padx=12, pady=4, anchor=tk.E)

        self._lbl_status = tk.Label(
            status, text="Loading...",
            bg=BG_CARD, fg=FG_DIM,
            font=("Segoe UI", 8, "italic"),
            justify=tk.RIGHT, anchor=tk.E
        )
        self._lbl_status.pack(anchor=tk.E)

        action_row = tk.Frame(status, bg=BG_CARD)
        action_row.pack(anchor=tk.E, pady=(4, 2))

        tk.Button(
            action_row, text="Reset Zoom",
            command=self._reset_zoom,
            bg=BG_PANEL, fg=FG_WHITE,
            activebackground=ACCENT, activeforeground=BG_DARK,
            font=("Segoe UI", 8, "bold"),
            relief=tk.FLAT, bd=0, padx=8, pady=3, cursor="hand2"
        ).pack(side=tk.LEFT, padx=(0, 6))

        tk.Button(
            action_row, text="Clear Marker",
            command=self._clear_marker,
            bg=BG_PANEL, fg=FG_WHITE,
            activebackground=ACCENT, activeforeground=BG_DARK,
            font=("Segoe UI", 8, "bold"),
            relief=tk.FLAT, bd=0, padx=8, pady=3, cursor="hand2"
        ).pack(side=tk.LEFT)

        self._lbl_click = tk.Label(
            status,
            text="Click map to explore data values",
            bg=BG_CARD, fg=FG_DIM,
            font=("Segoe UI", 8, "italic"),
            wraplength=420, justify=tk.RIGHT, anchor=tk.E
        )
        self._lbl_click.pack(anchor=tk.E, fill=tk.X)

    # ── chapter switching ─────────────────────────────────────────────────────

    def _select_chapter(self, idx: int):
        """
        Switch to chapter `idx` and update the entire view.
        Implements "control reading order" (Lec 8): the user advances
        through a curated narrative sequence.
        """
        self._chapter_idx = idx
        self._refresh_chapter()

    def _refresh_chapter(self, *_):
        """Rebuild the matplotlib canvas and narrative text for the current chapter."""
        ch = CHAPTERS[self._chapter_idx]

        # ── highlight active chapter button ──
        for i, b in enumerate(self._chap_btns):
            if i == self._chapter_idx:
                b.configure(bg=GOLD, fg=BG_DARK, font=("Segoe UI", 9, "bold"))
            else:
                b.configure(bg=BG_CARD, fg=FG_WHITE, font=("Segoe UI", 9))

        # ── update narrative panel ──
        self._lbl_icon.configure(text=ch["icon"])
        self._lbl_chapter.configure(text=ch["title"] + "\n" + ch["sub"])
        self._lbl_headline.configure(text=ch["headline"])
        self._txt_body.configure(state=tk.NORMAL)
        self._txt_body.delete("1.0", tk.END)
        self._txt_body.insert(tk.END, ch["body"])
        self._txt_body.configure(state=tk.DISABLED)

        # ── rebuild axes cleanly ──────────────────────────────────────────────
        # ax.clear() leaves colourbar axes behind as orphaned figure axes.
        # Delete every axes on the figure and create a fresh one each time.
        for _a in self._fig.axes[:]:
            self._fig.delaxes(_a)
        self._ax = self._fig.add_subplot(111)
        self._fig.subplots_adjust(left=0.04, right=0.93, top=0.93, bottom=0.07)
        self._ax.set_facecolor("#05050f")

        # ── draw the map ──
        key = ch["key"]
        if key == "landscape":
            self._draw_landscape()
        elif key == "shadow":
            self._draw_shadow()
        elif key == "sunlight":
            self._draw_sunlight()
        elif key == "suitability":
            self._draw_suitability()

        # ── restore zoom window ───────────────────────────────────────────────
        if self._cur_xlim is not None:
            self._ax.set_xlim(self._cur_xlim)
            self._ax.set_ylim(self._cur_ylim)

        # ── draw click marker ─────────────────────────────────────────────────
        # Red crosshair + hollow circle; redrawn after every refresh so it
        # persists across chapter switches and zoom changes.
        if self._click_col is not None and self._click_row is not None:
            kw = dict(linestyle="none", zorder=10, transform=self._ax.transData)
            self._ax.plot(self._click_col, self._click_row,
                          "+", color="#ff3355", markersize=22,
                          markeredgewidth=2.5, **kw)
            self._ax.plot(self._click_col, self._click_row,
                          "o", color="none", markersize=16,
                          markeredgecolor="#ff3355", markeredgewidth=1.8, **kw)

        # ── attach drag-to-zoom rectangle selector ────────────────────────────
        # Must be re-attached after every axes rebuild.
        self._selector = RectangleSelector(
            self._ax, self._on_rect_select,
            useblit=True, button=[1],
            minspanx=5, minspany=5, spancoords="pixels",
            interactive=True,
            props=dict(facecolor="cyan", edgecolor="white",
                       alpha=0.15, linewidth=1.2)
        )

        # ── status bar ──
        hm = self._current_heightmap()
        n  = hm.name if hm else "—"
        self._lbl_status.configure(
            text=f"Dataset: {n}  |  Chapter {self._chapter_idx+1}/4")
        self._canvas.draw_idle()

    # ── data helpers ──────────────────────────────────────────────────────────

    def _current_heightmap(self) -> LunarDataset | None:
        if not self.store.heightmaps:
            return None
        return self.store.heightmaps[min(self._scale_idx,
                                         len(self.store.heightmaps) - 1)]

    def _current_illumination(self) -> LunarDataset | None:
        if not self.store.illumination:
            return None
        return self.store.illumination[min(self._scale_idx,
                                            len(self.store.illumination) - 1)]

    def _get_cached(self, key: str):
        """Simple cache keyed on (scale_idx, key)."""
        return self._cache.get((self._scale_idx, key))

    def _set_cached(self, key: str, val):
        self._cache[(self._scale_idx, key)] = val

    def _get_elev(self):
        k = "elev"
        v = self._get_cached(k)
        if v is None:
            hm = self._current_heightmap()
            v  = hm.data if hm is not None else np.zeros((100, 100))
            self._set_cached(k, v)
        return v

    def _get_illum(self):
        k = "illum"
        v = self._get_cached(k)
        if v is None:
            il = self._current_illumination()
            v  = il.data if il is not None else None
            self._set_cached(k, v)
        return v

    def _get_hillshade(self):
        k = "hs"
        v = self._get_cached(k)
        if v is None:
            v = compute_hillshade(self._get_elev())
            self._set_cached(k, v)
        return v

    def _resample_to_elev(self, arr: np.ndarray, order: int = 1) -> np.ndarray:
        """
        Resample `arr` to match the current heightmap shape using scipy zoom.
        All layers (illumination, PSR, suitability) are normalised to this
        single canonical shape so every imshow call covers identical pixel
        extents — avoiding the misalignment that occurs when matplotlib
        stretches differently-sized arrays to the same axes.
        """
        elev = self._get_elev()
        if arr.shape == elev.shape:
            return arr
        from scipy.ndimage import zoom as _zoom
        zy = elev.shape[0] / arr.shape[0]
        zx = elev.shape[1] / arr.shape[1]
        return _zoom(arr.astype(np.float64), (zy, zx), order=order)

    def _get_illum_aligned(self) -> np.ndarray | None:
        """Return illumination resampled to heightmap shape, cached."""
        k = "illum_aligned"
        v = self._get_cached(k)
        if v is None:
            raw = self._get_illum()
            v = self._resample_to_elev(raw, order=1) if raw is not None else None
            self._set_cached(k, v)
        return v

    def _get_psr_mask(self):
        k = f"psr_{self._illum_threshold.get():.3f}"
        v = self._get_cached(k)
        if v is None:
            # Compute PSR on the raw illumination array, then resample to elev shape
            illum_raw = self._get_illum()
            if illum_raw is not None:
                raw_mask = compute_psr_mask(illum_raw, self._illum_threshold.get())
                # resample boolean mask with nearest-neighbour (order=0)
                v = self._resample_to_elev(raw_mask.astype(np.float32), order=0) > 0.5
            else:
                v = np.zeros(self._get_elev().shape, dtype=bool)
            self._set_cached(k, v)
        return v

    def _get_suitability(self):
        k = "suit"
        v = self._get_cached(k)
        if v is None:
            # Use aligned illumination so compute_suitability receives
            # arrays of identical shape — no internal zoom needed there
            v = compute_suitability(self._get_elev(),
                                    self._get_illum_aligned(),
                                    self._get_psr_mask())
            self._set_cached(k, v)
        return v

    def _get_slope(self):
        k = "slope"
        v = self._get_cached(k)
        if v is None:
            v = compute_slope_deg(self._get_elev())
            self._set_cached(k, v)
        return v

    # ── chapter renderers ─────────────────────────────────────────────────────

    def _common_axis_style(self, title: str):
        """Apply dark background and minimal axis chrome (Lec 8: data-ink ratio)."""
        ax = self._ax
        ax.set_facecolor("#05050f")
        ax.set_title(title, color=FG_WHITE, fontsize=11, pad=8, fontweight="bold")
        ax.tick_params(colors=FG_DIM, labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor(FG_DIM)
            spine.set_linewidth(0.5)
        ax.set_xlabel("West  ←  East  (pixel columns)", color=FG_DIM, fontsize=7)
        ax.set_ylabel("South  ←  North  (pixel rows)",  color=FG_DIM, fontsize=7)

    def _draw_landscape(self):
        """
        Chapter 1: Elevation with Lambertian hillshade blended in.

        Colourmap choice: 'terrain' provides an intuitive mapping
        (blue=low, green=plains, yellow/white=high peaks) that a general
        audience will find natural — avoiding rainbow artefacts (Lec 7).
        Hillshading adds a 3-D depth cue via Lambertian reflectance (Lec 5).
        """
        elev = self._get_elev()
        hs   = self._get_hillshade()
        vmin = np.nanpercentile(elev, 2)
        vmax = np.nanpercentile(elev, 98)

        # Blend hillshade into the colourmap image (Lec 5: intensity modulation)
        import matplotlib.cm as cm
        norm_e = (elev - vmin) / (vmax - vmin + 1e-9)
        norm_e = np.clip(norm_e, 0, 1)
        rgba   = matplotlib.colormaps["terrain"](norm_e)       # (H,W,4)
        # multiply RGB channels by hillshade for shading
        rgba[..., :3] *= hs[..., np.newaxis] * 0.6 + 0.4
        rgba[..., 3]   = 1.0
        self._ax.imshow(rgba, origin="upper", aspect="auto",
                        interpolation="bilinear")

        # Colourbar — elevation in metres (Lec 7: colourmap must be invertible)
        import matplotlib.colors as mcolors
        sm = matplotlib.cm.ScalarMappable(
            cmap="terrain",
            norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = self._fig.colorbar(sm, ax=self._ax, fraction=0.028, pad=0.02)
        cbar.set_label("Elevation (m)", color=FG_WHITE, fontsize=8)
        cbar.ax.yaxis.set_tick_params(color=FG_DIM, labelsize=7)
        plt_label_color(cbar.ax, FG_DIM)

        self._common_axis_style("The Lunar South Pole — Terrain Map")

        if self._show_annotations.get():
            self._annotate_landscape(elev)

    def _annotate_landscape(self, elev):
        """
        Add callout arrows to key geographic features.
        Lecture 8: "Highlight to focus attention" — annotations direct
        the audience to the most important data features.

        Data-driven detection: restrict argmin/argmax search to the inner
        80% of the image (10% margin each side) so that nodata-zero border
        pixels — which calibrate to extreme values after DN×0.5−10000 —
        do not mis-locate the labels.  A light Gaussian smooth (σ = W/60)
        finds basin/ridge centres rather than single-pixel spikes.
        The actual calibrated elevation (km) is shown in each label.
        """
        ax, H, W = self._ax, elev.shape[0], elev.shape[1]

        # Search only the inner 80 % to skip nodata borders
        mr, mc = max(1, H // 10), max(1, W // 10)
        inner = elev[mr:H - mr, mc:W - mc].astype(np.float64)
        inner = np.where(np.isfinite(inner), inner, np.nanmedian(inner))
        smooth = ndimage.gaussian_filter(inner, sigma=max(H, W) / 60.0)

        min_ri, min_ci = np.unravel_index(np.argmin(smooth), smooth.shape)
        max_ri, max_ci = np.unravel_index(np.argmax(smooth), smooth.shape)

        # Translate inner-array indices back to full-image coordinates
        min_r, min_c = min_ri + mr, min_ci + mc
        max_r, max_c = max_ri + mr, max_ci + mc

        # True calibrated elevation at the detected pixel
        min_e_m = float(elev[min_r, min_c])
        max_e_m = float(elev[max_r, max_c])

        def label_pos(r, c, dist=0.24):
            """Push label away from image centre to avoid obscuring arrowhead."""
            cx2, cy2 = W / 2.0, H / 2.0
            dx, dy = c - cx2, r - cy2
            length = max((dx**2 + dy**2)**0.5, 1.0)
            tx = c + dist * W * dx / length
            ty = r + dist * H * dy / length
            return (float(np.clip(tx, W * 0.06, W * 0.92)),
                    float(np.clip(ty, H * 0.06, H * 0.92)))

        tx, ty = label_pos(min_r, min_c)
        ax.annotate(
            f"Deep crater floor\n(cold trap for ice)\n{min_e_m / 1000:+.2f} km",
            xy=(min_c, min_r), xytext=(tx, ty),
            arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.2),
            color=ACCENT, fontsize=7.5, ha="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="#00000088", ec=ACCENT, lw=0.8)
        )

        tx2, ty2 = label_pos(max_r, max_c)
        ax.annotate(
            f"Highest ridge\n(solar power peak)\n{max_e_m / 1000:+.2f} km",
            xy=(max_c, max_r), xytext=(tx2, ty2),
            arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.2),
            color=GOLD, fontsize=7.5, ha="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="#00000088", ec=GOLD, lw=0.8)
        )
        ax.text(0.02, 0.02,
                "Elevation from NASA LROC DEM  |  Hillshading: Lambertian (Lec 5)",
                transform=ax.transAxes, color=FG_DIM, fontsize=6.5, va="bottom")

    def _draw_shadow(self):
        """
        Chapter 2: Illumination map with PSR overlay.

        Colourmap: 'plasma' (perceptually uniform sequential, Lec 7) maps
        illumination percentage from low (dark purple) to high (yellow).
        This ensures the audience can accurately judge relative brightness
        without hue confusion from a rainbow scale.

        PSR regions are overlaid as a semi-transparent blue mask —
        a preattentive feature (Lec 8) that instantly signals danger / cold.

        All arrays are resampled to the heightmap shape via
        _get_illum_aligned() / _get_psr_mask() so that every imshow call
        covers identical pixel extents and overlays align correctly.
        """
        # Use the shape-aligned illumination throughout so all imshow calls
        # render over the same pixel grid (fixes misalignment bug).
        illum = self._get_illum_aligned()
        if illum is None:
            self._ax.text(0.5, 0.5, "No illumination data found.\n"
                          "Ensure dataset/illumination/*.tif is present.",
                          transform=self._ax.transAxes,
                          ha="center", va="center", color=FG_WHITE, fontsize=11)
            self._common_axis_style("Light & Shadow — Illumination Map")
            return

        H, W = illum.shape
        lo = np.nanpercentile(illum, 2)
        hi = np.nanpercentile(illum, 98)
        norm_illum = np.clip((illum - lo) / (hi - lo + 1e-9), 0.0, 1.0)

        im = self._ax.imshow(norm_illum * 100.0, cmap="plasma",
                              vmin=0, vmax=100,
                              origin="upper", aspect="auto",
                              interpolation="bilinear")
        cbar = self._fig.colorbar(im, ax=self._ax, fraction=0.028, pad=0.02)
        cbar.set_label("% Time Illuminated", color=FG_WHITE, fontsize=8)
        cbar.ax.yaxis.set_tick_params(color=FG_DIM, labelsize=7)
        plt_label_color(cbar.ax, FG_DIM)

        # PSR overlay — already resampled to elev shape by _get_psr_mask()
        if self._show_psr.get():
            psr = self._get_psr_mask()   # guaranteed same shape as illum (aligned)
            psr_rgba = np.zeros((H, W, 4), dtype=np.float32)
            psr_rgba[psr] = [0.08, 0.39, 0.75, 0.55]
            self._ax.imshow(psr_rgba, origin="upper", aspect="auto",
                             interpolation="none")

        self._common_axis_style("Light & Shadow — Where the Sun Never Shines")

        if self._show_annotations.get():
            # Find darkest (PSR centre) and brightest (peak of eternal light)
            # using inner-80% search to skip nodata borders
            blurred = ndimage.gaussian_filter(norm_illum, sigma=max(H, W) / 40)
            mr, mc = max(1, H // 10), max(1, W // 10)
            inner = blurred[mr:H - mr, mc:W - mc]
            dr, dc = np.unravel_index(np.argmin(inner), inner.shape)
            br, bc = np.unravel_index(np.argmax(inner), inner.shape)
            dr += mr; dc += mc; br += mr; bc += mc

            self._ax.annotate(
                "Permanently\nShadowed Region\n(water ice!)",
                xy=(dc, dr),
                xytext=(float(np.clip(dc + W * 0.20, W * 0.05, W * 0.88)),
                        float(np.clip(dr + H * 0.14, H * 0.05, H * 0.88))),
                arrowprops=dict(arrowstyle="->", color="#80c8ff", lw=1.2),
                color="#80c8ff", fontsize=7.5, ha="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="#00000099",
                          ec="#80c8ff", lw=0.8)
            )
            self._ax.annotate(
                "Peak of\nEternal Light",
                xy=(bc, br),
                xytext=(float(np.clip(bc - W * 0.18, W * 0.05, W * 0.88)),
                        float(np.clip(br + H * 0.14, H * 0.05, H * 0.88))),
                arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.2),
                color=GOLD, fontsize=7.5, ha="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="#00000099",
                          ec=GOLD, lw=0.8)
            )
            import matplotlib.patches as mpatches
            psr_patch = mpatches.Patch(color="#1565c0", alpha=0.6,
                                        label="Permanently Shadowed Region (PSR)")
            self._ax.legend(handles=[psr_patch], loc="lower right",
                             facecolor=BG_CARD, edgecolor=FG_DIM,
                             labelcolor=FG_WHITE, fontsize=7.5)
            self._ax.text(0.02, 0.02,
                          f"Shadow threshold: illumination < {self._illum_threshold.get()*100:.0f}%"
                          "  |  Colourmap: plasma (perceptually uniform, Lec 7)",
                          transform=self._ax.transAxes,
                          color=FG_DIM, fontsize=6.5, va="bottom")

    def _draw_sunlight(self):
        """
        Chapter 3: Elevation hillshade + illumination contours.

        Combines the scientific elevation data (SciVis — displacement map)
        with the illumination dataset (InfoVis — contour overlay) in a
        single multi-variate view (Lec 7: multi-variate overlay).

        Contour lines at 20 %, 40 %, 60 %, 80 % illumination levels act
        as isolines (Lec 4: contouring / isovalue control) and help the
        audience count "how sunlit" each zone is at a glance.

        All arrays use _get_illum_aligned() / _get_psr_mask() so every
        imshow/contour call is over the same pixel grid.
        """
        elev  = self._get_elev()
        illum = self._get_illum_aligned()   # resampled to elev shape
        hs    = self._get_hillshade()
        H, W  = elev.shape

        # Base: greyscale hillshade for 3-D depth perception
        self._ax.imshow(hs, cmap="gray", vmin=0, vmax=1,
                         origin="upper", aspect="auto",
                         interpolation="bilinear", alpha=0.8)

        if illum is not None:
            lo = np.nanpercentile(illum, 2)
            hi = np.nanpercentile(illum, 98)
            norm_illum = np.clip((illum - lo) / (hi - lo + 1e-9), 0.0, 1.0)
            # illum is already (H, W) — same as hs, so overlay aligns correctly
            self._ax.imshow(norm_illum * 100.0, cmap="YlOrRd",
                             vmin=0, vmax=100, alpha=0.55,
                             origin="upper", aspect="auto",
                             interpolation="bilinear")

            # Smooth before contouring to avoid tracing LROC scan-line artefacts
            # as horizontal bars (Lec 4: isolines trace field features, not noise)
            smooth_illum = ndimage.gaussian_filter(norm_illum * 100.0,
                                                   sigma=max(H, W) / 40.0)
            lvls = [20, 40, 60, 80]
            ct = self._ax.contour(smooth_illum, levels=lvls,
                                   colors=[ACCENT, ACCENT, GOLD, GOLD],
                                   linewidths=[0.7, 0.7, 0.9, 0.9], alpha=0.85)
            self._ax.clabel(ct, inline=True, fontsize=6,
                            fmt={20: "20%", 40: "40%", 60: "60%", 80: "80%"},
                            colors=FG_WHITE)

            import matplotlib.colors as mcolors
            sm = matplotlib.cm.ScalarMappable(
                cmap="YlOrRd",
                norm=mcolors.Normalize(vmin=0, vmax=100))
            sm.set_array([])
            cbar = self._fig.colorbar(sm, ax=self._ax, fraction=0.028, pad=0.02)
            cbar.set_label("% Time Illuminated", color=FG_WHITE, fontsize=8)
            cbar.ax.yaxis.set_tick_params(color=FG_DIM, labelsize=7)
            plt_label_color(cbar.ax, FG_DIM)

        if self._show_psr.get():
            # _get_psr_mask() already resampled to elev shape — safe to overlay
            psr = self._get_psr_mask()
            psr_rgba = np.zeros((H, W, 4), dtype=np.float32)
            psr_rgba[psr] = [0.08, 0.39, 0.75, 0.45]
            self._ax.imshow(psr_rgba, origin="upper", aspect="auto",
                             interpolation="none")

        self._common_axis_style("Sunlight & Power — Peaks of Eternal Light")

        if self._show_annotations.get():
            if illum is not None:
                blurred = ndimage.gaussian_filter(norm_illum, sigma=max(H, W) / 40)
                mr, mc = max(1, H // 10), max(1, W // 10)
                inner = blurred[mr:H - mr, mc:W - mc]
                br, bc = np.unravel_index(np.argmax(inner), inner.shape)
                br += mr; bc += mc
                self._ax.annotate(
                    "Best solar panel\nsite (>60% sunlit)",
                    xy=(bc, br),
                    xytext=(float(np.clip(bc - W * 0.18, W * 0.05, W * 0.88)),
                            float(np.clip(br + H * 0.14, H * 0.05, H * 0.88))),
                    arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.2),
                    color=GOLD, fontsize=7.5, ha="center",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#00000099",
                              ec=GOLD, lw=0.8)
                )
            self._ax.text(0.02, 0.02,
                          "Isolines: 20/40/60/80% illumination (Lec 4: contouring)  "
                          "|  Overlay: YlOrRd (intuitive warm=bright)",
                          transform=self._ax.transAxes,
                          color=FG_DIM, fontsize=6.5, va="bottom")

    def _draw_suitability(self):
        """
        Chapter 4: Composite landing suitability heatmap.

        Diverging colourmap (RdYlGn) — Lecture 7 (diverging / double-ended
        scales): "middle considered neutral; emphasises far ends (deviation
        from the average)."  Here:
          • Green  = high suitability  (flat, sunlit, near PSR)
          • Red    = low suitability   (steep, dark, or deep crater)
          • Yellow = neutral / caution

        All arrays (hs, suit, psr) are guaranteed the same shape because
        _get_suitability() / _get_psr_mask() / _get_hillshade() all derive
        from _get_elev() via _resample_to_elev().
        """
        suit = self._get_suitability()   # shape == elev shape
        hs   = self._get_hillshade()     # shape == elev shape
        psr  = self._get_psr_mask()      # shape == elev shape
        H, W = suit.shape

        # Hillshade base for context
        self._ax.imshow(hs, cmap="gray", vmin=0, vmax=1,
                         origin="upper", aspect="auto",
                         interpolation="bilinear", alpha=0.35)

        # Suitability overlay (diverging — Lec 7)
        im = self._ax.imshow(suit, cmap="RdYlGn",
                              vmin=0, vmax=1, alpha=0.82,
                              origin="upper", aspect="auto",
                              interpolation="bilinear")
        cbar = self._fig.colorbar(im, ax=self._ax, fraction=0.028, pad=0.02)
        cbar.set_label("Landing Suitability", color=FG_WHITE, fontsize=8)
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(["Avoid", "Caution", "Ideal"])
        cbar.ax.yaxis.set_tick_params(color=FG_DIM, labelsize=7.5)
        plt_label_color(cbar.ax, FG_DIM)

        # Highlight top 5 % of suitable pixels
        threshold = np.nanpercentile(suit, 95)
        best_mask = suit >= threshold
        best_rgba = np.zeros((H, W, 4), dtype=np.float32)
        best_rgba[best_mask] = [0.0, 0.9, 0.46, 0.6]
        self._ax.imshow(best_rgba, origin="upper", aspect="auto",
                         interpolation="none")

        # PSR boundary contour — same shape as suit, no resampling needed
        if self._show_psr.get():
            self._ax.contour(psr.astype(float), levels=[0.5],
                              colors=[PSR_CLR], linewidths=[1.2], alpha=0.9)

        self._common_axis_style("Mission Ready — Landing Suitability Score")

        if self._show_annotations.get():
            rows, cols = np.where(best_mask)
            if rows.size:
                cy, cx = rows.mean(), cols.mean()
                self._ax.annotate(
                    "Best candidate\nlanding zone",
                    xy=(cx, cy), xytext=(cx - W*0.22, cy - H*0.18),
                    arrowprops=dict(arrowstyle="->", color=SAFE_CLR, lw=1.4),
                    color=SAFE_CLR, fontsize=8, ha="center", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#00000099",
                              ec=SAFE_CLR, lw=1.0)
                )
            self._ax.text(0.02, 0.02,
                          "Score = 0.5×(flat) + 0.3×(sunlit) + 0.2×(near PSR)  "
                          "|  Colourmap: RdYlGn diverging (Lec 7)",
                          transform=self._ax.transAxes,
                          color=FG_DIM, fontsize=6.5, va="bottom")

            import matplotlib.patches as mpatches
            psr_patch  = mpatches.Patch(facecolor="none",
                                         edgecolor=PSR_CLR, lw=1.2,
                                         label="PSR boundary")
            best_patch = mpatches.Patch(color=SAFE_CLR, alpha=0.55,
                                         label="Top 5% candidate zones")
            self._ax.legend(handles=[psr_patch, best_patch],
                             loc="lower right",
                             facecolor=BG_CARD, edgecolor=FG_DIM,
                             labelcolor=FG_WHITE, fontsize=7.5)

    # ── interactivity ─────────────────────────────────────────────────────────

    # ── zoom & marker helpers ─────────────────────────────────────────────────

    def _on_press(self, event):
        """Record mouse-down pixel so we can distinguish a click from a drag."""
        if event.inaxes == self._ax:
            self._press_xy = (event.x, event.y)
        else:
            self._press_xy = None

    def _on_rect_select(self, eclick, erelease):
        """
        RectangleSelector callback — drag-to-zoom.
        Pushes the current view limits onto the undo stack, then applies
        the dragged rectangle as the new view window.
        (Lec 2: interactive view navigation / inverse mapping)
        """
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        if None in (x1, x2, y1, y2):
            return
        self._zoom_stack.append((self._cur_xlim, self._cur_ylim))
        self._cur_xlim = (min(x1, x2), max(x1, x2))
        # imshow y-axis is inverted (row 0 = top); preserve that orientation
        self._cur_ylim = (max(y1, y2), min(y1, y2))
        self._refresh_chapter()

    def _zoom_back(self):
        """Step back one zoom level."""
        if self._zoom_stack:
            self._cur_xlim, self._cur_ylim = self._zoom_stack.pop()
        else:
            self._cur_xlim = self._cur_ylim = None
        self._refresh_chapter()

    def _reset_zoom(self):
        """Clear zoom history and return to the full-image view."""
        self._zoom_stack.clear()
        self._cur_xlim = self._cur_ylim = None
        self._refresh_chapter()

    def _clear_marker(self):
        """Remove the click marker and reset the info readout."""
        self._click_col = None
        self._click_row = None
        self._lbl_click.configure(
            text="Click the map to explore data values", fg=FG_DIM)
        self._refresh_chapter()

    def _on_map_click(self, event):
        """
        Click-to-explore: shows data values at the clicked pixel and places
        a persistent marker on the map.
        Implements "inverse mapping" (Lec 2) — click pixel → data value.
        Drag-vs-click detection: ignore release events where the mouse moved
        more than 5 px since button_press (those belong to the zoom selector).
        """
        if event.button != 1:
            self._press_xy = None
            return
        # Ignore drag releases
        if self._press_xy is not None:
            dx = event.x - self._press_xy[0]
            dy = event.y - self._press_xy[1]
            self._press_xy = None
            if dx * dx + dy * dy > 25:
                return
        else:
            return
        if event.inaxes != self._ax or event.xdata is None:
            return

        col = int(round(event.xdata))
        row = int(round(event.ydata))
        # Store position — marker is drawn in _refresh_chapter so it
        # survives chapter switches and zoom changes
        self._click_col = col
        self._click_row = row
        ch = CHAPTERS[self._chapter_idx]

        lines = [f"Pixel  col={col}, row={row}"]

        hm = self._current_heightmap()
        if hm is not None and 0 <= row < hm.height and 0 <= col < hm.width:
            elev_val = float(hm.data[row, col])
            lines.append(f"  Elevation: {elev_val:+.0f} m  ({elev_val/1000:+.2f} km)")

        illum = self._get_illum_aligned()
        if illum is not None and 0 <= row < illum.shape[0] and 0 <= col < illum.shape[1]:
            raw_val = float(illum[row, col])
            lo = float(np.nanpercentile(illum, 2))
            hi = float(np.nanpercentile(illum, 98))
            norm_val = float(np.clip((raw_val - lo) / (hi - lo + 1e-9), 0, 1))
            lines.append(f"  Illuminated: {norm_val*100:.1f}%")
            psr_mask = self._get_psr_mask()
            if psr_mask is not None and 0 <= row < psr_mask.shape[0] and 0 <= col < psr_mask.shape[1]:
                in_psr = bool(psr_mask[row, col])
                lines.append(f"  In shadowed region: {'Yes' if in_psr else 'No'}")

        if ch["key"] == "suitability":
            suit = self._get_suitability()
            if 0 <= row < suit.shape[0] and 0 <= col < suit.shape[1]:
                sv = float(suit[row, col])
                lines.append(
                    f"  Suitability: {sv:.2f}  "
                    f"({'Ideal' if sv > 0.7 else 'Caution' if sv > 0.4 else 'Avoid'})"
                )

        self._lbl_click.configure(text="\n".join(lines), fg=ACCENT)
        self._refresh_chapter()  # redraws marker at stored position

    def _on_scale_change(self, event):
        if not self.store.heightmaps:
            return
        labels = [ds.scale_label for ds in self.store.heightmaps]
        val    = self._scale_var.get()
        if val in labels:
            self._scale_idx = labels.index(val)
        self._refresh_chapter()


# ── small colour-styling helper ───────────────────────────────────────────────

def plt_label_color(cbar_ax, color):
    """Set colourbar tick label colours (matplotlib API is inconsistent)."""
    for lbl in cbar_ax.get_yticklabels():
        lbl.set_color(color)
    for lbl in cbar_ax.get_xticklabels():
        lbl.set_color(color)


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

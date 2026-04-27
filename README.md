# Thorncroft Rossby-Wave-Breaking Life-Cycle Simulations (LC1 / LC2)

Idealised baroclinic-wave life-cycle experiments following **Thorncroft,
Hoskins & McIntyre (1993, QJRMS 119, 17–55)**. Two dry, frictionless,
flat-topography runs with 15 σ-levels and ∇⁶ hyperdiffusion:

| Run | Basic state Z                                     | Perturbation               |
| --- | ------------------------------------------------- | -------------------------- |
| LC1 | u₀=47 m/s mid-latitude jet (J&W 2006 thermal wind) | 1 mb wave-6 surface-pressure envelope |
| LC2 | LC1 + barotropic +15/−15 m/s at 20°N/50°N         | same as LC1               |

The LC1 basic state is the analytical **Jablonowski & Williamson (2006)**
thermally-balanced zonal-wind field rescaled to peak at 47 m/s — chosen
because Thorncroft 1993 does not publish closed-form equations (they defer
to Hoskins & Simmons 1975). J&W 2006 produces an NH jet with a sloping
tropopause and near-zero surface wind, matching the Z1 specification.

Dynamics are integrated with [`SpeedyWeather.jl`](https://speedyweather.github.io/)
in `PrimitiveDryModel` with `dynamics_only=true` (all physics off) and
`drag=nothing` (frictionless), so only the resolved dynamics and ∇⁶
hyperdiffusion act on the flow.

## Governing equations

Fully three-dimensional, **baroclinic, hydrostatic, dry primitive
equations** on the rotating sphere, integrated in spectral /
finite-difference hybrid form (Simmons & Hoskins 1975 formulation as
implemented in `SpeedyWeather.jl/src/dynamics/`). Prognostic variables:

- absolute vorticity ζ + f  (spectral, one per σ-level)
- horizontal divergence D  (spectral, one per σ-level)
- virtual temperature Tᵥ  (spectral, one per σ-level)
- ln(surface pressure) ln pₛ  (spectral, single 2-D field)
- specific humidity q is carried but **disabled** (`dynamics_only=true`,
  PrimitiveDryModel), so the thermodynamic equation is dry.

Symbolically (σ-coordinates, after Simmons & Hoskins 1975):

$$
\frac{\partial \zeta}{\partial t} = -\nabla\!\cdot\!(\mathbf{v}(\zeta+f))
    - \hat{\mathbf{k}}\!\cdot\!\nabla\!\times\!(\dot\sigma \partial_\sigma \mathbf{v}
    + R T_v \nabla \ln p_s) - \nu\,(-\nabla^2)^{n}\zeta
$$

$$
\frac{\partial D}{\partial t} = \hat{\mathbf{k}}\!\cdot\!\nabla\!\times\!(\dots)
    - \nabla^2\!\bigl[\Phi + R T_v \ln p_s + \tfrac{1}{2}|\mathbf{v}|^2\bigr]
    - \nu\,(-\nabla^2)^{n}D
$$

$$
\frac{\partial T_v}{\partial t} = -\mathbf{v}\!\cdot\!\nabla T_v
    - \dot\sigma\,\partial_\sigma T_v + \tfrac{RT_v}{c_p}\omega/p
    - \nu\,(-\nabla^2)^{n}T_v
$$

$$
\frac{\partial \ln p_s}{\partial t} = -\tfrac{1}{p_s}\nabla\!\cdot\!\!\int_0^1\! p_s \mathbf{v}\,d\sigma
$$

Hydrostatic balance closes Φ = Φₛ + ∫ RTᵥ d(ln p).  Hyperdiffusion is
∇²ⁿ with n = 3 (i.e. ∇⁶) and a **1-hour damping time** on the
smallest retained spectral scale (paper spec: "decay rate of 1 h⁻¹ for
the shortest retained scale, n = 95", Thorncroft 1993 §3).

This is a **baroclinic** model (vertically stratified Tᵥ and u(σ,φ),
baroclinic instability is the growth mechanism). The **barotropic**
Rossby-wave equation (one vertical level, 2-D) is not used here; it
appears in the sibling project at
`../../vallis04_barotropic/speedy_weather/`.

### Vertical discretisation

- **15 σ-levels** (σ = p/pₛ) with full-level values from Hoskins &
  Simmons 1975: `σ = 0.967, 0.887, 0.784, 0.674, 0.569, 0.477, 0.400,
  0.338, 0.287, 0.241, 0.197, 0.152, 0.106, 0.060, 0.018`.
- Half-levels are derived by midpoint averaging with σ_half[1]=1 and
  σ_half[end]=0.
- Terrain-following coordinate over flat topography (Φₛ = 0
  everywhere), so σ ≡ p/pₛ.

### Horizontal discretisation

- Spectral truncation **T95** (spherical-harmonic basis on the full
  sphere) → collocated on a **128 × 256 FullGaussianGrid** (~1.4°).
- All physics off (dry, frictionless), so there is no PBL, no
  radiation, no moisture, no gravity-wave drag.

### Boundary conditions

- **Horizontal**: fully **spherical / global** domain — no channel,
  no zonal periodic strip, no polar cap imposed. The spherical-harmonic
  basis handles polar regularity automatically.
- **Top** (σ → 0): rigid lid, σ̇ = 0.
- **Bottom** (σ = 1): rigid lid, σ̇ = 0, flat topography Φₛ = 0, and
  **free-slip** (`drag=nothing`) — no surface friction.
- Zonal direction: periodic (inherent in spherical harmonics).

### Initial condition ("fixed mode")

The basic state is zonally symmetric, thermal-wind-balanced, and
held across both experiments:

- Jablonowski & Williamson (2006) analytic jet: u₀ = 47 m/s,
  η₀ = 0.252 (250 hPa jet core), balanced Tᵥ(σ,φ).
- For LC2 only, a depth-independent **barotropic** addition of
  ±10 m/s at 20°N / 50°N with ±10° Gaussian half-widths is
  superposed on u₀, leaving Tᵥ unchanged (no thermal-wind shift).

This zonally symmetric state is then perturbed by a **zonal
wave-number 6 surface-pressure anomaly** of 1 mb with a
45°N-centred, 15° Gaussian meridional envelope — the seed for the
normal-mode baroclinic instability.

## Tunable knobs → different RWB life-cycles

Everything below is editable in `config/lc1.toml` / `config/lc2.toml`:

| Knob (TOML path)                | Effect on lifecycle |
| --- | --- |
| `basic_state.u0`                | Jet strength (m/s). Stronger jet → faster growth, more pronounced breaking. |
| `basic_state.eta0`              | Height of jet core (smaller η → higher / narrower jet). |
| `basic_state.barotropic_add`    | `false` → LC1 (equivalent-barotropic), `true` → LC2 basic state. Most important single switch for anticyclonic-vs-cyclonic breaking. |
| `basic_state.bt_amp`, `bt_lat_*`, `bt_width` | Tune the barotropic addition. Increasing `bt_amp` or narrowing the meridional footprint makes LC2 more cyclonic (tighter roll-up of PV on 330 K). |
| `perturbation.zonal_wavenumber` | k = 6 matches Thorncroft 1993. Try k = 5 or 7 to excite different unstable modes; longer waves give slower, broader breaking. |
| `perturbation.amplitude_mb`     | Sets when the wave reaches finite amplitude. 0.1 mb delays onset by ~4–5 d; 1.0 mb matches paper. |
| `perturbation.envelope_lat`, `envelope_width` | Where the seed is placed (latitudinal centre and half-width). |
| `diffusion.power`               | n in ∇²ⁿ. SpeedyWeather uses `power=3` for ∇⁶. Increasing to 4 (∇⁸) gives less large-scale damping and sharper PV filaments (but may destabilise). |
| `diffusion.time_scale_hours`    | Damping time on the shortest retained scale. **1 h matches the paper**; larger → weaker diffusion → sharper filaments but risk of blow-up at wave-breaking time. |
| `simulation.dt_at_T31_minutes`  | Leapfrog time step at T31, linearly scaled to trunc at runtime. Default 30 min; we use **15 min** (→5 min at T95) so LC2 remains CFL-stable through day 15 when PV filaments are sharpest. |
| `simulation.n_days`             | Run length. 16 d captures the full lifecycle; extend to 20–25 d to see the post-breaking zonal flow adjustment. |
| `grid.trunc` / `nlayers`        | Horizontal / vertical resolution. T42 / 7 σ is the classic 1993 setup; T95 / 15 σ is the higher-resolution default here. |

Quick recipes:

- **LC1** (anticyclonic, equatorward wrap) — `barotropic_add=false`.
- **LC2** (cyclonic, poleward roll-up) — `barotropic_add=true`,
  `bt_amp=15`.
- **LC1 with earlier breaking** — keep `barotropic_add=false`,
  set `perturbation.amplitude_mb=5.0` (still linear but wave reaches
  saturation ~2 days sooner).
- **Quasi-linear baseline** — `amplitude_mb=0.01`, `n_days=25`;
  gives ~8-d exponential-growth phase visible before any wrap-up.
- **Ultra-clean filaments** — `diffusion.time_scale_hours=6.0`,
  `simulation.n_days=12` (blow-up is then the main risk).

## Workflow (v2.10.0 — repo relocation + idealized 3-row layout redesign)

**v2.10.0 changes** on top of v2.9.0:

1. **Repository relocation**. The project moved from
   `…/barotropic_vorticity_model/thorncroft_rwb` to
   `…/literature_review/rwb/thorncroft93_baroclinic/thorncroft_rwb`.
   The sibling SpeedyWeather barotropic project moved to
   `…/literature_review/rwb/vallis04_barotropic/speedy_weather` (its
   SpeedyWeather.jl source clone now lives at
   `/net/flood/data/users/x_yan/SpeedyWeather.jl` and is symlinked
   in). All hard-coded `ROOT` paths in the python scripts and
   `scripts/run_tempest.sh` were updated. Old absolute paths inside
   `outputs/**/{run.log,parameters.txt,*.log}` were left as-is
   (immutable historical artefacts).
2. **Idealized 3-row plot layout** (`scripts/idealized_plot.py`):
   - All three rows now span the **same horizontal extent**
     (`left=0.04, right=0.96`).
   - **Every panel** has its own attached colour bar via
     `mpl_toolkits.axes_grid1.make_axes_locatable`. Row 0 panels
     share the `∂q/∂t` range `r1`; row 1 panels share the basis
     range `rb`; row 2 panels each have an independent range.
   - Removed `set_aspect("equal")` so panels fill their gridspec
     slot (the slight 4:3 → row-dependent stretch is acceptable for
     these schematic figures and the alternative leaves large gaps).
   - Row 2 titles broken into two lines so the `α` annotation no
     longer collides with the cbar exponent label.

## Workflow (v2.9.0 — tighter backward walk + dual-polarity polar-cap MP4)

From v2.4.0 onwards the pipeline uses **zeta250 as the sole canonical
tracking method**. The `pv330` and `theta_pv2` pipelines remain in the
codebase (and `scripts/_config.py::METHOD`) for optional use, but all
default scripts, CLI flags, and outputs target `zeta250` only.  Their
previously-committed outputs have been moved to `outputs/archive/`
(on disk, not tracked by git).

**v2.9.0 changes** on top of v2.8.0:

1. **Tighter `extend_backward.py` walk gates** — prevents jumping to a
   spurious neighbouring extremum once the original feature dissolves
   into the meridional background:
   - `R_SEARCH_DEG`: 5° → 3.5°
   - `JUMP_MAX_DEG_PER_H`: 7° → 3.5°
   - `BACK_THRESH_FACTOR`: 0.25 → 0.4
   - **New magnitude-continuity gate**: stop if
     `|val_new| < 0.5·|val_prev|` AND `|val_new| < 0.5·mask_thresh`.
   Backward extension shrinks LC1 +30/+31 h (was +43/+44 h);
   LC2 +60 h (was +71 h). LC2 tilt time series no longer has the
   sudden spikes the v2.8 walk produced when it overshot into noise.
2. **Polar-cap MP4 always shows BOTH C and AC tracks**. Running
   `--polarity C` alone (or `AC` alone) now also reads the existing
   extended tracks of the *other* polarity from disk so the MP4
   contains blue/cyan (C) **and** orange/gold (AC) trails together.
3. **Tilt animations & static PNGs** use **hours since simulation
   start** (day 5 → hour 120). Std-onset markers in legends now show
   both hour and day.
4. **Idealized 3-row plot for the backward composite**.
   `idealized_plot.process(..., method="zeta250_back", t_k=12)` is
   called per polarity in `extend_backward.py::run_idealized_back`.
   `t_k = 12` (12 h after the earliest backward composite frame ≈
   ½ day after the earliest backward-tracked timestamp). Output:
   `outputs/<lc>/projections/zeta250_back/idealized_plot/<lc>_idealized_3row_{C,AC}.png`.

### Earlier history

**v2.8.0** added `extend_backward.py` (looser gates, single-polarity
MP4, simulation-day x-axis). Superseded by v2.9.0 above.

**v2.7.0 changes** on top of v2.6.0:

1. **Asymmetric Lagrangian patch** — `PATCH_HALF_LON=40°`, `PATCH_HALF_LAT=30°`.
   Composites are now **61 rows × 81 cols** (lat × lon) instead of
   the previous 81×81 square.  The guard-circle radius is bounded by
   the shorter (latitude) half-width: `PATCH_HALF = PATCH_HALF_LAT = 30°`.
   - `build_composites.py` uses `x_rel = ±PATCH_HALF_LON`, `y_rel = ±PATCH_HALF_LAT`.
   - All plot axes use `xlim(±40°)`, `ylim(±30°)` accordingly.
   - Guard circle: `PATCH_HALF_LAT − GUARD_PAD_DEG = 25°` (unchanged physics).
2. `outputs/archive/` contents cleared (old 81×81 results removed).

**Per-LC composite window (`scripts/_config.py:WINDOW_BY_LC`)**:
- `lc1` → day **6 → 10** (hours 144..240)
- `lc2` → day **8 → 12** (hours 192..288)

Each window covers 96 h → **97 hourly composite frames**.

```mermaid
graph TD
    A[config/lc1.toml, config/lc2.toml] --> B[scripts/run_lc.jl]
    B -->|SpeedyWeather.jl, T95, 17 days| C[outputs/&lt;lc&gt;/raw/run_*/output.nc]
    C --> D[scripts/postprocess.py]
    D --> E[outputs/&lt;lc&gt;/processed.nc]

    CFG[scripts/_config.py<br/>CANONICAL_METHOD=zeta250,<br/>METHOD dict, WINDOW_BY_LC] -.-> H
    GS[scripts/_grad_safe.py<br/>safe_gradient + mask_vmax] -.-> P & R & ID
    CFG -.-> J & L & EXP & N & P & R & W & FIG & ID

    E --> H[scripts/prep_track_inputs.py<br/>build zeta250_anom.nc]
    H --> I2[zeta250_anom.nc]
    I2 --> J[scripts/run_tempest.sh<br/>per LC &times; C/AC]
    J --> K[tracks/zeta250/tracks_&#123;C,AC&#125;.txt]
    K --> L[scripts/select_top6.py]
    L --> M[tracks/zeta250/tracks_&#123;C,AC&#125;_top6.txt]
    M --> EXP[scripts/export_track_csv.py]
    EXP --> CSV[tracks/zeta250/track_centers_&#123;C,AC&#125;.csv]
    CSV --> N[scripts/build_composites.py<br/>track-centred 81&times;81 patch &plusmn;40&deg;,<br/>total + anom]
    N --> O[composites/zeta250/&#123;C,AC&#125;_composite.nc]
    O --> P[scripts/project_composite.py<br/>5-basis decomp anchored at first<br/>valid hour, mask-based cbar]
    P --> Q[projections/zeta250/plots/decomp_*.png]
    O --> R[scripts/tilt_evolution.py<br/>center-blob mask + |q'|-weighted ellipse,<br/>safe_gradient + mask_vmax cbar]
    R --> S[projections/zeta250/plots/<br/>theta_tilt_*.png, tilt_animation_*.mp4]
    R --> SD[projections/zeta250/data/tilt_&#123;C,AC&#125;.npz]
    O --> ID[scripts/idealized_plot.py<br/>3-row idealised figure<br/>row1 dq/dt+recon+resid,<br/>row2 &Phi;&#8321;..&Phi;&#8325;, row3 scaled bases<br/>+ &alpha; stretching arrows]
    ID --> IDP[projections/zeta250/idealized_plot/<br/>&lt;lc&gt;_idealized_3row_&#123;C,AC&#125;.png]
    O --> W[plotting/make_tracked_anim.py<br/>zeta250 default]
    W --> WPMP[outputs/&lt;lc&gt;/plots/zeta250_tracked.mp4]
    E --> FIG[plotting/make_figures.py<br/>cf.remove fix, no ax.clear]
    FIG --> MP4[outputs/&lt;lc&gt;/mp4/&#123;pv_330K,zeta_250hPa,<br/>theta_on_pv2,T_surface&#125;.mp4]
```

### Tracking method (canonical)

| Tag | Field | Sign convention (C / AC) | `mask_thresh` |
| --- | ----- | ------------------------ | ------------- |
| `zeta250`   | $\zeta'_{250\,\text{hPa}}$ (relative-vorticity anomaly) | C: searchbymax(`>+8e-6 s⁻¹`), AC: searchbymin(`<-8e-6 s⁻¹`) | 8e-6 s⁻¹ |

`pv330` and `theta_pv2` are defined in `scripts/_config.py::METHOD` and can
be run optionally with `--method pv330` / `--method theta_pv2`; their
outputs are not tracked by git (archived to `outputs/archive/`).

### Anchor & ellipse fit

- **Basis anchor** = first hour where ≥ 90 % of the 81×81 patch is
  finite (interpreted as the most-circular onset). `SYMMETRIZE=False`.
- **Ellipse fit** uses signed-anomaly mask (C: $q' > +\text{thr}$,
  AC: $q' < -\text{thr}$) AND a guard circle of radius
  $\text{PATCH\_HALF} - \text{GUARD\_PAD\_DEG} = 35^{\circ}$, then
  **restricted to the connected component containing (or closest to)
  the patch origin**. Weighted-covariance second moments with
  **latitude weight** $w_{ij} \propto 1/\cos(\phi_{ij})$ (normalised,
  clipped at ±85°) give $(\theta, a, b, x_c, y_c)$ with
  $\theta \in [-90, 90)^{\circ}$. The latitude weight emphasises
  higher-latitude pixels, consistent with the physical importance of
  near-polar flow in RWB dynamics.
  $(\theta, a, b, x_c, y_c)$, with $\theta \in [-90, 90)^{\circ}$.
  Major-axis line of length $2a$ at angle $\theta$ through
  $(x_c, y_c)$ is drawn on every animation frame (green = obs,
  cyan dashed = 1-h prediction).
- **Method comparator** scores each method by mean wrap-aware
  $|\Delta\theta_{\text{obs}} - \Delta\theta_{\text{pred}}|$ (C+AC) and
  writes `projections/winner.json`. All 3 methods' tracked mp4s are
  rendered to `outputs/<lc>/plots/<method>_tracked.mp4` so you can
  compare them side-by-side; no symlink is created.

## Reproducing Thorncroft 1993 RWB tracking

Dual-polarity cyclonic (**C**, $q' > +0.1$ PVU) and anticyclonic
(**AC**, $q' < -0.1$ PVU) wave-breaking life cycles are diagnosed on
the $\theta = 330$ K isentrope from the SpeedyWeather runs:

- **[SpeedyWeather.jl](https://speedyweather.github.io/)** — dry,
  frictionless `PrimitiveDryModel` at T95 &times; 15 &sigma;-levels integrates
  the baroclinic primitive equations with $\nabla^6$ hyperdiffusion
  (1 h damping on the smallest retained scale). Initial condition is
  the Jablonowski &amp; Williamson (2006) analytic jet rescaled to
  $\eta_0 = 0.252$ plus a wave-6, 1 mb surface-pressure
  perturbation; LC2 adds the $\pm 15$ m s$^{-1}$ barotropic addition
  at 20&deg;/50&deg;N.
- **[TempestExtremes](https://climate.ucdavis.edu/tempestextremes.php)**
  `DetectNodes` is run twice per LC: once with
  `--searchbymax pv_anom_330 --thresholdcmd pv_anom_330,>,+0.1,0`
  (cyclonic branch) and once with
  `--searchbymin pv_anom_330 --thresholdcmd pv_anom_330,<,-0.1,0`
  (anticyclonic branch). Both are restricted to
  $35^\circ\text{N} \le \varphi \le 75^\circ\text{N}$ and merged
  within 8&deg;. `StitchNodes` connects hourly nodes with
  `--range 7.0 --mintime 90h --maxgap 6h`. `DetectBlobs` masks the
  anomaly to areas &ge; $5\times 10^{5}$ km$^{2}$ for visualisation.
  All constants live in [`scripts/_config.py`](scripts/_config.py).
- **Top-6 selection** (`scripts/select_top6.py`) ranks tracks by
  $\text{peak}\,|q'| \times \text{span}$ and trims the six survivors
  to the common $[\max t_{\text{start}}, \min t_{\text{end}}]$ window
  so every member contributes simultaneously.
- **Track-centred Lagrangian patch** (`scripts/build_composites.py`).
  For each (track, hour) the 61&times;61 patch is sampled at
  $(\lambda_0(t) + \Delta\lambda_j,\; \varphi_0(t) + \Delta\varphi_i)$
  with $\Delta\lambda,\Delta\varphi \in [-30^\circ, +30^\circ]$ at
  1&deg; spacing, where both $\lambda_0(t)$ and $\varphi_0(t)$ are the
  $|q'|$-weighted mass centroid of the containing DetectBlobs component
  (see [`filter_cand_by_blob.py`](scripts/filter_cand_by_blob.py)).
  The 145-hour composite is
  $\bar q_{ijk} = \text{nanmean}_{m}\, q(\lambda_0^{(m)}(t_k) +
  \Delta\lambda_j,\, 55 + \Delta\varphi_i,\, t_k)$.
- **5-basis decomposition** (`scripts/project_composite.py`).
  `pvtend.compute_orthogonal_basis` builds the orthonormal basis
  $\{F_{\text{INT}}, F_{\text{DEF}}, F_{\text{PROP}}, L, I\}$ from
  $\bar q, \partial_x \bar q, \partial_y \bar q$ at the peak hour
  (peak defined per polarity: $\operatorname*{argmax}_k \sum_{ij}
  \max(\text{sgn}\cdot q'_a, 0)$). `project_field` returns the five
  coefficients plus residual. `SYMMETRIZE=False` in `_config.py`
  means the raw composite is used directly; set `True` to rotationally
  average over 36 angles before projection.
- **Tilt evolution &amp; 1 h ellipse forecast** (`scripts/tilt_evolution.py`).
  At each hour we fit a covariance ellipse to $\max(\text{sgn}\cdot q'_a, 0)$
  and define the tilt
  $$
  \theta(t) = \tfrac12 \operatorname{atan2}\!\bigl(2 C_{xy},\,
  C_{xx} - C_{yy}\bigr),\qquad C = \frac{\sum w(\mathbf x-\bar{\mathbf x})
  (\mathbf x-\bar{\mathbf x})^\top}{\sum w}.
  $$
  The predicted ellipse is obtained by advancing $\bar q \to
  \bar q + \Delta t\,(F_{\text{INT}} + F_{\text{DEF}} +
  F_{\text{PROP}} + L + I)$ over $\Delta t = 1$ h and recomputing
  $\theta_{\text{pred}}$. The accumulated-angle figure
  (`theta_tilt_accum_{C,AC}.png`) plots the cumulative unwrapped
  $\sum \Delta\theta_{\text{obs}}$ (green) and cumulative
  $\sum (\theta_{\text{pred}} - \theta_{\text{obs}})$ (cyan) versus
  hour. Animations are written as MP4 (`FFMpegWriter`,
  `yuv420p`) with lower-row PV-tendency and $F_{\text{DEF}}$
  colourbars symmetrically clipped at the 95th percentile of
  $|\cdot|$ for visual clarity.

Large generated files (raw SpeedyWeather output, `processed.nc`,
`pv330_anom.nc`, composites, blob NetCDFs, MP4s) are excluded via
`.gitignore`; regenerate with `scripts/run_all.sh` plus the
post-processing chain.

### RWB recipe (from `processed.nc`)

```bash
cd /net/flood/data2/users/x_yan/literature_review/rwb/thorncroft93_baroclinic/thorncroft_rwb
ENV=blocking          # has pvtend, xarray, TempestExtremes wrappers

# 1. Build per-LC anomaly NetCDFs for the 3 tracking methods
#    (LC1 → day 6..10, LC2 → day 8..12, see WINDOW_BY_LC)
micromamba run -n $ENV python scripts/prep_track_inputs.py lc1 lc2

# 2. Run TempestExtremes for each (LC, method, polarity)
for lc in lc1 lc2; do
  for m in pv330 zeta250 theta_pv2; do
    bash scripts/run_tempest.sh $lc $m
  done
done

# 3. Track curation, CSV export, Lagrangian composite
micromamba run -n $ENV python scripts/select_top6.py        lc1 lc2
micromamba run -n $ENV python scripts/export_track_csv.py   lc1 lc2
micromamba run -n $ENV python scripts/build_composites.py   lc1 lc2

# 4. 5-basis decomposition + ellipse-tilt evolution
#    decomp_*.png has the strict ellipse mask as a black dashed contour
#    on the ∂q/∂t panel.
#    tilt_animation_*.mp4 LR panel shows -γ₁φ₄ - γ₂φ₅ (def tendency);
#    LL panel overlays the same dashed mask on ∂q/∂t.
micromamba run -n $ENV python scripts/project_composite.py  lc1 lc2
micromamba run -n $ENV python scripts/tilt_evolution.py     lc1 lc2

# 5. Method comparator + winning mp4
#    Auto-pick by min mean |Δθ_obs - Δθ_pred|, OR force a method:
micromamba run -n $ENV python scripts/compare_methods.py    lc1 lc2 --force zeta250
micromamba run -n speedy_weather python plotting/make_tracked_anim.py lc1 lc2
```

Tunable constants live in [`scripts/_config.py`](scripts/_config.py):
`PATCH_HALF`, `DX`, `CENTER_LAT`, `LAT_MIN/MAX`, `POS_THRESH`,
`NEG_THRESH`, `AREA_MIN_KM2`, `MERGE_DIST_DEG`, `STITCH_RANGE_DEG`,
`STITCH_MAXGAP`, `SPAN_REQ_H`, `JUMP_MAX_DEG_PER_H`, `TOP_N`,
`N_COMPOSITE_HOURS`, `THETA_LEVEL`, `SYMMETRIZE`, `N_ROT`,
`SMOOTHING_DEG`, `INCLUDE_LAP`, `ANIM_FPS`, `PCTL_CBAR`,
`DT_PRED_HOURS`.

## Layout

```
thorncroft_rwb/
├── Project.toml                 # Julia env (SpeedyWeather + TOML)
├── SpeedyWeather.jl -> ../speedy_weather/SpeedyWeather.jl  (dev source)
├── config/
│   ├── lc1.toml
│   └── lc2.toml
├── scripts/
│   ├── env.sh                   # module load julia + micromamba activate
│   ├── setup_julia.jl           # Pkg.instantiate
│   ├── run_lc.jl                # main Julia driver
│   ├── run_all.sh               # background launcher (24 threads × 2)
│   └── postprocess.py           # derive θ, PV, fluxes → processed.nc
├── plotting/
│   ├── thorncroft_figs.py       # Figs 5-10 analogues (PNG)
│   └── make_figures.py          # hourly PV & ζ₂₅₀ animations (MP4)
└── outputs/
    ├── lc1/ {raw, plots, mp4, logs, processed.nc}
    └── lc2/ {raw, plots, mp4, logs, processed.nc}
```

## Running

```bash
cd /net/flood/data2/users/x_yan/literature_review/rwb/thorncroft93_baroclinic/thorncroft_rwb
source scripts/env.sh                 # julia 1.10.8 + speedy_weather env
julia --project=. scripts/setup_julia.jl   # first time only

# quick smoke test (T31, 1 day, ~seconds)
julia --project=. --threads=4 scripts/run_lc.jl --config config/lc1.toml --smoke

# full T95 × 16-day runs in background (24 threads each, 48 cores total)
bash scripts/run_all.sh
tail -f outputs/lc1/logs/run.log outputs/lc2/logs/run.log

# once both runs finish:
micromamba run -n speedy_weather python scripts/postprocess.py lc1
micromamba run -n speedy_weather python scripts/postprocess.py lc2
micromamba run -n speedy_weather python plotting/thorncroft_figs.py
micromamba run -n speedy_weather python plotting/make_figures.py lc1
micromamba run -n speedy_weather python plotting/make_figures.py lc2
```

## Sanity-check figures (Thorncroft 1993)

`plotting/thorncroft_figs.py` writes the following paper-number
analogues to `outputs/<lc>/plots/`:

- **Fig 3** — lat-σ zonal-mean [u] (shading) and [θ] (contours) at
  days 0 and 10 for LC1 and LC2. At day 0 LC1 and LC2 basic states
  differ only by the ±10 m/s barotropic addition in LC2.
- **Fig 4** — column-integrated EKE vs time [J m⁻²]. Paper: LC1 peaks
  near 1.5 × 10⁶ J m⁻² around day 9–11, LC2 grows faster with a
  broader late-time plateau.
- **Fig 5** — LC1 T at σ=0.967 days 4–9, NH polar stereo, 4 K
  contours, 0°C dotted, negative dashed. Classic anticyclonically
  wrapped cold front / warm sector.
- **Fig 6** — LC1 surface pressure days 4–9, 4 mb contours,
  1000 mb dotted, <1000 mb dashed.
- **Fig 7** — LC1 PV on 315 K θ, NH polar stereo, days 4–9. The
  paper's classic equatorward/westward anticyclonic wave-break.
- **Fig 8** — LC2 T at σ=0.967 days 4–9 (same style as Fig 5).
  Stronger warm-frontal contrast due to the cyclonic shear of Z2.
- **Fig 9** — LC2 surface pressure days 4–9 (same style as Fig 6).
- **Fig 10** — LC2 PV on 330 K θ, NH polar stereo, days 4–9. The
  cyclonic poleward roll-up of PV.
- **Fig 15** — LC1/LC2 eddy momentum flux [u'v'] lat-σ cross sections.

`plotting/make_figures.py` writes MP4 animations of PV on θ,
250 hPa relative vorticity, and surface-layer T to
`outputs/<lc>/mp4/`.

## Notes

- **Paper setup match** (comparing to Thorncroft 1993 §3):

  | item | paper | this repo | match? |
  | --- | --- | --- | --- |
  | horizontal truncation | T95 | T95 | ✔ |
  | vertical layers | 15 σ (H&S 1975) | 15 σ (same values) | ✔ |
  | hyperdiffusion | ∇⁶, 1 h at n=95 | ∇⁶, 1 h at n=95 | ✔ |
  | dry, frictionless | yes | yes | ✔ |
  | initial perturbation | 1 mb wave-6 most-unstable mode | 1 mb wave-6 Gaussian envelope at 45°N ± 15° | **approx** (we seed with a broadband wave-6 bump rather than the exact normal-mode structure; onset is therefore ~1–2 d later) |
  | basic state Z1 | H&S 1975 analytic | J&W 2006 analytic, rescaled to u₀=47 m/s | **approx** (J&W 2006 is the modern analogue; peak wind and jet latitude match) |
  | Z2 barotropic addition | ±10 m/s at 20°N/50°N | ±15 m/s at 20°N/50°N | **stronger** (sharper cyclonic roll-up of PV on 330 K) |
  | Δt | not published | 15 min at T31 (→5 min at T95) | extra precaution for LC2 CFL |

- ∇⁶ hyperdiffusion: SpeedyWeather's `HyperDiffusion.power` is the
  exponent n of the Laplacian (∇²ⁿ); ∇⁶ ⇒ `power=3`. Damping time 1 h
  on the smallest retained spectral scale (`config/*.toml → [diffusion]`).
- σ-levels from the user spec (concentrated near the tropopause):
  `0.967 0.887 0.784 0.674 0.569 0.477 0.400 0.338 0.287 0.241 0.197
  0.152 0.106 0.060 0.018`.
- NetCDF output on FullGaussianGrid with `nlat_half=64` (≈ 256×128).
- Hourly output cadence; 16-day integration = 385 frames per run.
- The SpeedyWeather NetCDF writes **`temp` in °C** (not K); `postprocess.py`
  auto-detects and converts before computing θ/PV.
- Surface pressure is written as `mslp` in **hPa** (not `pres` in Pa).
- EKE is column-integrated via EKE = (pₛ/g) · Σₖ ½(u'²+v'²)ₖ Δσₖ, reported
  in **J m⁻²** (matches paper Fig 4).

## References

- Thorncroft, C. D., Hoskins, B. J., & McIntyre, M. E. (1993).
  "Two paradigms of baroclinic-wave life-cycle behaviour."
  *Quarterly Journal of the Royal Meteorological Society*, 119, 17–55.
  <https://doi.org/10.1002/qj.49711950903>
- Jablonowski, C., & Williamson, D. L. (2006). "A baroclinic instability
  test case for atmospheric model dynamical cores."
  *QJRMS*, 132, 2943–2975. <https://doi.org/10.1256/qj.06.12>
- Hoskins, B. J., & Simmons, A. J. (1975). "A multi-layer spectral
  model and the semi-implicit method." *QJRMS*, 101, 637–655.
  <https://doi.org/10.1002/qj.49710142918>

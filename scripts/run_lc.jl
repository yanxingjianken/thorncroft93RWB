#!/usr/bin/env julia
# Thorncroft, Hoskins & McIntyre (1993) LC1 / LC2 life-cycle run.
#
# Dry, frictionless PrimitiveDryModel on T95 with 15 σ-levels from
# Hoskins & Simmons (1975). ∇⁶ hyperdiffusion with 1 h decay at the
# shortest retained scale. Z1 (LC1) = Jablonowski-Williamson 2006
# analytical thermal-wind-balanced jet with u₀=47 m/s at ~45°N on a
# sloping tropopause, zero surface wind by construction to within a few
# percent. Z2 (LC2) = Z1 + barotropic zonal wind of +10 m/s at 20°N and
# −10 m/s at 50°N. Initial perturbation: 1 mb wave-6 surface-pressure
# with a latitude envelope centred on the jet — this broadband seed
# projects onto the most unstable normal mode at zonal wavenumber 6 (see
# Thorncroft et al. 1993 §4: "10 mb vs 1 mb modal vs non-modal
# perturbations produce very similar life cycles — what matters is the
# initial mean flow Z1 vs Z2").
#
# Usage (after `source scripts/env.sh`):
#     julia --project=. --threads=N scripts/run_lc.jl --config config/lc1.toml
#     julia --project=. --threads=N scripts/run_lc.jl --config config/lc1.toml --smoke

using Pkg
Pkg.activate(dirname(@__DIR__))

using Dates
using TOML
using SpeedyWeather

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
function parse_args(args)
    cfg_path = nothing
    smoke = false
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--config"
            cfg_path = args[i + 1]; i += 2
        elseif a == "--smoke"
            smoke = true; i += 1
        else
            error("Unknown argument $a")
        end
    end
    cfg_path === nothing && error("--config required")
    return cfg_path, smoke
end

# ---------------------------------------------------------------------
# σ half-levels from user-specified full levels
# ---------------------------------------------------------------------
function build_sigma_half(sigma_full::Vector{<:Real})
    # SpeedyWeather wants σ_half ordered top → surface, i.e. starts at 0
    # (model top) and ends at 1 (surface).  Accept the user spec in
    # either order and always return the ascending top-down form.
    σf = sort(Vector{Float64}(sigma_full))            # ascending: top → surface
    n = length(σf)
    σh = Vector{Float64}(undef, n + 1)
    σh[1] = 0.0                                       # model top
    @inbounds for k in 1:(n - 1)
        σh[k + 1] = 0.5 * (σf[k] + σf[k + 1])
    end
    σh[n + 1] = 1.0                                   # surface
    return σh
end

# ---------------------------------------------------------------------
# Barotropic u(φ) addition (Z2): +A at bt_lat_west, −A at bt_lat_east.
# Returned as a pair of closures: u_bt(λ,φ,σ) [m/s] and the matching
# relative vorticity ζ_bt(λ,φ,σ) [1/s] (v ≡ 0 so ζ = -∂u/∂y - u tan φ/a
# on the sphere, then ζ = -(1/a) dU/dφ_rad + (u/a) tan φ).
# ---------------------------------------------------------------------
const EARTH_RADIUS = 6.371e6   # m

function make_barotropic_u(amp_ms::Real, lat_w::Real, lat_e::Real, width_deg::Real)
    function u_bt(λ, φ, σ)
        gw = exp(-((φ - lat_w) / width_deg)^2)
        ge = exp(-((φ - lat_e) / width_deg)^2)
        return amp_ms * (gw - ge)
    end
    return u_bt
end

function make_barotropic_vor(amp_ms::Real, lat_w::Real, lat_e::Real,
                             width_deg::Real; a::Real = EARTH_RADIUS)
    # du/dφ in m/s per radian
    function du_dphi_rad(φ_deg)
        gw  = exp(-((φ_deg - lat_w) / width_deg)^2)
        ge  = exp(-((φ_deg - lat_e) / width_deg)^2)
        dgw = -2 * (φ_deg - lat_w) / width_deg^2 * gw
        dge = -2 * (φ_deg - lat_e) / width_deg^2 * ge
        return amp_ms * (dgw - dge) * (180 / π)       # per rad
    end
    function ζ_bt(λ, φ, σ)
        gw = exp(-((φ - lat_w) / width_deg)^2)
        ge = exp(-((φ - lat_e) / width_deg)^2)
        u  = amp_ms * (gw - ge)
        return (-du_dphi_rad(φ) + u * tand(φ)) / a
    end
    return ζ_bt
end

# ---------------------------------------------------------------------
# Wave-6 log-surface-pressure perturbation with latitude envelope.
# ln(ps + δps) - ln(ps) ≈ δps/ps. 1 mb / 1000 hPa = 1e-3 in ln(ps).
# ---------------------------------------------------------------------
function make_wave6_pressure(amp_mb::Real, m::Int, lat0::Real, width_deg::Real;
                             ps_ref_hPa::Real = 1000.0)
    δlnp = (amp_mb * 100.0) / (ps_ref_hPa * 100.0)    # amp_mb[Pa] / ps_ref[Pa]
    function lnp_pert(λ, φ)                           # λ, φ in degrees
        env = exp(-((φ - lat0) / width_deg)^2)
        return δlnp * env * cosd(m * λ)
    end
    return lnp_pert
end

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
function main(args)
    cfg_path, smoke = parse_args(args)
    cfg = TOML.parsefile(cfg_path)

    trunc    = smoke ? Int(cfg["simulation"]["smoke_trunc"]) : Int(cfg["grid"]["trunc"])
    nlayers  = Int(cfg["grid"]["nlayers"])
    σ_full   = Vector{Float64}(cfg["grid"]["sigma_full"])
    n_days   = smoke ? Int(cfg["simulation"]["smoke_days"]) : Int(cfg["simulation"]["n_days"])

    u0            = Float64(cfg["basic_state"]["u0"])
    η0            = Float64(cfg["basic_state"]["eta0"])
    bt_add        = Bool(cfg["basic_state"]["barotropic_add"])
    bt_amp        = Float64(cfg["basic_state"]["bt_amp"])
    bt_lat_w      = Float64(cfg["basic_state"]["bt_lat_west"])
    bt_lat_e      = Float64(cfg["basic_state"]["bt_lat_east"])
    bt_width      = Float64(cfg["basic_state"]["bt_width"])

    m_wave        = Int(cfg["perturbation"]["zonal_wavenumber"])
    amp_mb        = Float64(cfg["perturbation"]["amplitude_mb"])
    env_lat       = Float64(cfg["perturbation"]["envelope_lat"])
    env_width     = Float64(cfg["perturbation"]["envelope_width"])

    diff_power    = Int(cfg["diffusion"]["power"])
    diff_hours    = Float64(cfg["diffusion"]["time_scale_hours"])

    # Optional explicit leapfrog Δt at T31 (default 30 min). Smaller
    # values improve stability for LC2 late-time cyclonic filamentation.
    dt_T31_min    = haskey(cfg, "simulation") && haskey(cfg["simulation"], "dt_at_T31_minutes") ?
                    Float64(cfg["simulation"]["dt_at_T31_minutes"]) : 30.0

    out_id        = smoke ? "smoke_$(cfg["output"]["id"])" : String(cfg["output"]["id"])
    out_dt_h      = Int(cfg["output"]["dt_hours"])
    nlat_half_out = Int(cfg["output"]["nlat_half_out"])

    project_root = dirname(@__DIR__)
    outputs_dir  = joinpath(project_root, "outputs", String(cfg["output"]["id"]), "raw")
    mkpath(outputs_dir)

    @info "Thorncroft 1993 LC run" cfg_path smoke trunc nlayers n_days u0 η0 bt_add bt_amp diff_power diff_hours out_id

    # --- vertical coordinate ---------------------------------------------
    σ_half = build_sigma_half(σ_full)
    vcoord = SigmaCoordinates(nlayers, σ_half)
    @info "Sigma half-levels" σ_half

    # --- spectral grid ---------------------------------------------------
    spectral_grid = SpectralGrid(trunc = trunc, nlayers = nlayers)
    geometry      = Geometry(spectral_grid; vertical_coordinates = vcoord)

    # --- ∇⁶ hyperdiffusion, 1 h at shortest retained scale ---------------
    tmin = max(1, Int(round(diff_hours * 60)))
    horizontal_diffusion = HyperDiffusion(spectral_grid;
        time_scale = Minute(tmin),
        power      = diff_power,
    )

    # --- leapfrog time stepping (override Δt if configured) ---------------
    time_stepping = Leapfrog(spectral_grid;
        Δt_at_T31 = Minute(Int(round(dt_T31_min))),
    )

    # --- initial conditions: J&W 2006 zonal flow + J&W temperature -------
    # u₀ rescaled to 47 m/s (Thorncroft Z1 peak wind). Localised J&W
    # perturbation turned off (perturb_uₚ = 0); we impose our own wave-6
    # surface-pressure perturbation and (for LC2) the barotropic addition
    # in the steps below.
    initial_conditions = InitialConditions(
        vordiv = ZonalWind(spectral_grid; u₀ = u0, η₀ = η0, perturb_uₚ = 0.0),
        temp   = JablonowskiTemperature(spectral_grid; u₀ = u0, η₀ = η0),
        pres   = PressureOnOrography(spectral_grid),
    )

    # --- NetCDF output ---------------------------------------------------
    output = NetCDFOutput(spectral_grid, PrimitiveDry;
        output_dt   = Hour(out_dt_h),
        output_grid = FullGaussianGrid(nlat_half_out),
        path        = outputs_dir,
        id          = out_id,
        overwrite   = true,
    )

    # --- model: dry, frictionless, flat ---------------------------------
    model = PrimitiveDryModel(spectral_grid;
        geometry,
        initial_conditions,
        orography            = NoOrography(spectral_grid),
        drag                 = nothing,                 # frictionless
        horizontal_diffusion,
        time_stepping,
        output,
        physics              = false,                   # all parameterisations off
    )

    simulation = initialize!(model)

    # --- Z2 addition: barotropic ±10 m/s at 20°N/50°N --------------------
    # v ≡ 0, so we add the analytic ζ_bt directly (avoids an upstream
    # bug in set!(u=…, v=…, add=true) on 3D prognostic vor).
    if bt_add
        ζ_bt = make_barotropic_vor(bt_amp, bt_lat_w, bt_lat_e, bt_width)
        @info "Adding barotropic vorticity (±$(bt_amp) m/s at $(bt_lat_w)°/$(bt_lat_e)°)"
        set!(simulation; vor = ζ_bt, add = true)
    end

    # --- 1 mb wave-6 surface-pressure perturbation -----------------------
    lnp_pert = make_wave6_pressure(amp_mb, m_wave, env_lat, env_width)
    @info "Adding wave-$m_wave surface-pressure perturbation" amp_mb env_lat env_width
    set!(simulation; pres = lnp_pert, add = true)

    # --- run -------------------------------------------------------------
    @info "Running $(n_days) days"
    run!(simulation; period = Day(n_days), output = true)

    run_folder = String(model.output.run_folder)
    if !isabspath(run_folder)
        run_folder = joinpath(outputs_dir, run_folder)
    end
    nc_path = joinpath(run_folder, "output.nc")
    @info "Done" outputs_dir nc_path isfile(nc_path)
    return nothing
end

main(ARGS)

#!/usr/bin/env julia
# One-off: instantiate the Julia project (resolves deps with the
# shared Manifest from ../speedy_weather if desired, otherwise fresh).
using Pkg
Pkg.activate(dirname(@__DIR__))
Pkg.instantiate()
Pkg.precompile()

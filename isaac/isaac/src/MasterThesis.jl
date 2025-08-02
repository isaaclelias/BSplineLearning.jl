module MasterThesis

using Plots
using Dates
using Statistics
using Optimization
using Zygote
using ComponentArrays
using Random
using Logging
using Printf
using Serialization
using StaticArrays
using LibGit2
using Distributions
using preBurner
using LaTeXStrings
using NaNMath; const nm = NaNMath
using UnPack
using DataFrames
using Optimisers
using ForwardDiff
using Measures

# remove Arrhenius, Enzyme, ForwardDiff and ReverseDiff from deps. When I tried it went wrong

import Base.size

const pkg_root_dir::String = pathof(MasterThesis) |>
                             x->splitpath(x) |>
                             x->x[begin:end-2] |>
                             x->joinpath(x)

include("constants.jl")

include("BSpline.jl")
export BSpline, BSpline2
export control_point_coordinates, control_points_coordinates, b_spline_definition, b_spline_coxdeboor

include("solve_bspline.jl")
export solve_bspline_curve

include("solve_adaptive_bspline.jl")
export solve_adaptive_bspline_curve

include("Curve.jl")
export Curve, curve_sample, curves_sample, min_curve_points_inside_u_step, min_curves_points_inside_u_step, x_labels, y_labels

include("MBSpline.jl")
export MBSpline, calc_mbspline

include("solve_mbspline.jl")
export initialize_mbs0, solve_mbspline_curves

include("solve_mbspline2.jl")
export solve_mbspline_curves2, parameters_solve_mbspline_curves2

include("mbspline_error_metrics.jl")
export maximum_percentage_error_and_errors

include("NMBSpline.jl")
export calc_nmbs_naive, calc_nmbs_cox, NMBSpline

include("plotting.jl")
export savefig_with_date, plot_bspline!, plot_bspline, plot_mbspline_curves, plot_mbspline_curves!, plot_mbspline, plot_iterations_labels_log, plot_heatmap_valuess, plot_iterations_labels_colors_linestyles_log

include("load_data.jl")
export df_hoq_training_data_1d, df_hoq_training_data_2d, curves_hoq_training_data_1d, curves_hoq_training_data_2d

end #module

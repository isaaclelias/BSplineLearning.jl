using MasterThesis
using Logging, LoggingExtras
using Dates
using Plots
using LaTeXStrings
using DelimitedFiles
using Printf
using Serialization
using Measures
using Debugger

# Hyperparameters
const curves::Vector{Curve} = curves_hoq_training_data_1d()
df = df_hoq_training_data_1d()
const n_u_steps_aut::Int = 2
const k::Int = 4
const maxiters::Int = 999
const n_additional_iterations = 2001
const r_refinement::Bool = solve_u_steps::Bool = true
const alpha_d::Float64 = alpha = 0.2
const atol::Float64 = 0
const n_points_per_u_step_max::Int = 100
const clamp_n_points_u_steps::Bool = false
const h_refinement::Bool = true
const complete_check_after::Int = 10
const n_iter_refinement::Int = 200
const alpha_h::Float64 = 0.3
const alpha_r::Float64 = 10^-8
const alpha_opt::Float64 = 0.025

# Construct the cases
## Complete case
const _methods_which = [:error, :quadratic_error]
const _methods_where = [:first_order_moment, :second_order_moment]
const _methods_how_high = [:spline_prediction, :curve_middle]
## Small case
#const _methods_which = [:error, :quadratic_error]
#const _methods_where = [:first_order_moment, :second_order_moment]
#const _methods_how_high = [:spline_prediction, :curve_middle]
## Construct vectors
methods_which    = Symbol[]
methods_where    = Symbol[]
methods_how_high = Symbol[]
for method_which    in _methods_which,
    method_where    in _methods_where,
    method_how_high in _methods_how_high
    
    push!(methods_which, method_which)
    push!(methods_where, method_where)
    push!(methods_how_high, method_how_high)
end

# logging setup
log_dir = "../logs/"; mkpath(log_dir)
log_title = basename(@__FILE__) |> x -> replace(x, ".jl" => "")
log_format_date = Dates.format(now(), "yyyy-mm-dd_HH-MM")
log_path = log_dir * log_format_date * " " * log_title * ".log"
log_info_path  = log_path * ".info" ; touch(log_info_path)
log_debug_path = log_path * ".debug"; touch(log_debug_path)
const logger = TeeLogger(
    ConsoleLogger(stdout                    , Logging.Info ),
    SimpleLogger( open(log_info_path , "w+"), Logging.Info ),
    SimpleLogger( open(log_debug_path, "w+"), Logging.Debug),
); global_logger(logger)
@info "Logging" log_info_path log_debug_path

# stuff to save the study results
_title = "trying with r and h maxiters-$(maxiters) npusm-$(n_points_per_u_step_max) a_opt-$(alpha_opt) nir-$(n_iter_refinement) nppusm-$(n_points_per_u_step_max)"
filename = basename(@__FILE__) |> x -> replace(x, ".jl" => "")
title = MasterThesis.format_date_githash_title_now(_title)
path_save = "../results/" * filename * "/"*title*"/"; mkpath(path_save)

# data loading
curves = curves_hoq_training_data_1d()

# saying hi to the user with useful information
@info "$(filename)" n_points_per_u_step_max

# Heavy lifting

## Initializing vectors for quantities to be plotted
n_cases = length(methods_which)
lossess = Vector{Vector{Float64}}(undef, n_cases)
lossess_comp = Vector{Vector{Float64}}(undef, n_cases)
maess_comp = Vector{Vector{Float64}}(undef, n_cases)
maess_norm_comp = Vector{Vector{Float64}}(undef, n_cases)
maess_norm = Vector{Vector{Float64}}(undef, n_cases)
maess = Vector{Vector{Float64}}(undef, n_cases)
mpes_i_best_comp = Vector{Vector{Float64}}(undef, n_cases)
maes_i_best = Vector{Vector{Float64}}(undef, n_cases)
mpess_i_comp = Vector{Vector{Vector{Float64}}}(undef, n_cases)
mpess_comp = Vector{Vector{Float64}}(undef, n_cases)
reps = Vector{NamedTuple}(undef, n_cases)

## Start the computation batches
Threads.@threads for (i, (method_which, method_where, method_how_high)) in collect(enumerate(zip(methods_which, methods_where, methods_how_high))) # tell me sonic, tell me how to split this line into multiple ones, sonic
  iter_title = "$(filename) mwhich-$(method_which) mwhere-$(method_where) mhh-$(method_how_high) mi-$(maxiters) k-$(k) nusa0-$(n_u_steps_aut) rr-$(r_refinement) hr-$(h_refinement) $(_title)"
    mbs, rep = solve_mbspline_curves2(
        n_u_steps_aut, k, curves;
        maxiters,
        title = iter_title,
        solve_u_steps = r_refinement,
        h_refinement,
        atol, # should not exit because of it
        clamp_n_points_u_steps,
        n_points_per_u_step_max,
        alpha_opt,
        alpha_u_steps = alpha_r,
        complete_check_after = complete_check_after,
        method_which,
        method_where,
        method_how_high,
        n_iter_refinement,
        n_additional_iterations,
    )
    reps[i] = rep
    lossess[i] = rep.losses
    lossess_comp[i] = rep.losses_comp
    maess_norm[i] = rep.maes_norm
    maess_norm_comp[i] = rep.maes_norm_comp
    maess[i] = rep.maes
    maess_comp[i] = rep.maes_comp
    mpes_i_best_comp[i] = rep.mpe_i_best_comp
    mpess_i_comp[i] = rep.mpes_i_comp
    mpess_comp[i] = rep.mpes_comp
    maes_i_best[i] = rep.mae_i_best
end

# Plotting

## Plotting constants
legend_column = 1
linestyles = repeat([:dash], 16)
colors =[palette(:default)[i] for i in 1:16]

iterations = collect(1:maximum(length.(lossess)))
labels = [
    "$(method_which), $(methods_where), $(method_how_high)"
    for (method_which, methods_where, method_how_high) in zip(methods_which, methods_where, methods_how_high)
]

## Losses plot
plt_lossess = plot_iterations_labels_colors_linestyles_log(
    lossess, labels, L"\mathcal{L} \ [-]", colors, linestyles, size = (480, 380),
    legend_column = legend_column
)
file_plot_lossess_png = path_save * "losses.png"
file_plot_lossess_pdf = path_save * "losses.pdf"
touch(file_plot_lossess_png)
savefig(plt_lossess, file_plot_lossess_png)
savefig(plt_lossess, file_plot_lossess_pdf)

## Maximum Absolute Error plot
plt_mpess = plot_iterations_labels_colors_linestyles_log(
    mpess_comp, labels, L"Maximum Relative Error $[\%]$", colors, linestyles,
    n = complete_check_after, size = (480, 380), legend_column = legend_column,
    bottom_margin = 1mm, left_margin = 0mm, right_margin = 8mm, top_margin = 10mm,
)
file_plot_mpess_png = path_save * "normalized_maximum_absolute_error.png"
file_plot_mpess_pdf = path_save * "normalized_maximum_absolute_error.pdf"
touch(file_plot_mpess_png)
savefig(plt_mpess, file_plot_mpess_png)
savefig(plt_mpess, file_plot_mpess_pdf)

## Maximum Error heatmap
labels_nl = [
    "- $(method_which), $(method_how_high),\n$(method_where)."
    for (method_which, method_where, method_how_high) in zip(methods_which, methods_where, methods_how_high)
]
scale_plt1 = 1.5
#labels_nl[1] = L"k = %$(ks[1]) \ , \ n_{reg} = %$(ns_u_steps_aut[1]-1)"
plt_mpe_i = plot_heatmap_valuess(
    mpes_i_best_comp, mpes_i_best_comp, labels_nl, y_labels(curves),
    label_plt = L"\mathrm{MRE} \ [\%]", size = (480*scale_plt1, 280*scale_plt1),
    bottom_margin = 12mm, left_margin = 8mm, right_margin = 10mm, top_margin = 3mm,
    yrotation = 45, xtickfontsize = 12, ytickfontsize = 6, annotation_size = 10
)
file_plot_mpe_i_png = path_save * "maximum-percentual-error.png"
file_plot_mpe_i_pdf = path_save * "maximum-percentual-error.pdf"
touch(file_plot_mpe_i_png)
touch(file_plot_mpe_i_pdf)
savefig(plt_mpe_i, file_plot_mpe_i_png)
savefig(plt_mpe_i, file_plot_mpe_i_pdf)

# export to PGFPlots
labels_dat = [
    "mwhich-$(method_which)__mwhere-$(method_where)__mhh-$(method_how_high)"
    for (method_which, method_where, method_how_high) in zip(methods_which, methods_where, methods_how_high)
]
iters_dat = vcat(vcat(["iter", iterations]...))

file_lossess_dat = path_save * "losses.dat"
touch(file_lossess_dat)
io_lossess_dat = open(file_lossess_dat, "w")
lossess_dat = hcat(vcat(iters_dat...), vcat(hcat(labels_dat...), hcat(lossess...)))
writedlm(io_lossess_dat, lossess_dat)
close(io_lossess_dat)

#maes_pgfplots_dat = hcat(vcat(iters_pgfplots...), vcat(hcat(labels_pgfplots...), hcat(maes_norm...)))

# Report

## Small one, I want to deprecate it in favor of always saving the complete info
file_rep_jl = path_save*"report.jl"
rep = """
n_points_per_u_step_max = $(n_points_per_u_step_max)
iterations = $(iterations)
labels = $(labels)
labels_nl = $(labels_nl)
lossess = $(lossess)
lossess_comp = $(lossess_comp)
maess_norm = $(maess_norm)
maess_comp = $(maess_comp)
maess_norm_comp = $(maess_norm_comp)
mpes_i_best_comp = $(mpes_i_best_comp)
mpess_i_comp = $(mpess_i_comp)
mpess_comp = $(mpess_comp)
maess = $(maess)
"""
file_reps_jl  = path_save * "complete-report.jl" ; touch(file_reps_jl )
file_reps_jls = path_save * "complete-report.jls"; touch(file_reps_jls)
io_reps_jl = open(file_reps_jl, "w")
print(io_reps_jl, reps)
close(io_reps_jl)
serialize(file_reps_jls, reps)

@info "$(filename) exited ;)"

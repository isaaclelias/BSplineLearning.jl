using MasterThesis
using Logging, LoggingExtras
using Dates
using Plots#; gaston()
using LaTeXStrings
using DelimitedFiles
using Printf
using Serialization

const ks             = [1 , 1 , 1 , 2 , 2 , 2 , 3 , 3 , 3 , 4 , 4 , 4 , 5 , 5 , 5 ]
const ns_u_steps_aut = [11, 21, 41, 11, 21, 41, 11, 21, 41, 11, 21, 41, 11, 21, 41]
# small case
#const ks             = [1 , 3 ]#, 1 , 2 , 2 , 2 , 3 , 3 , 3 , 4 , 4 , 4 , 5 , 5 , 5 ]
#const ns_u_steps_aut = [11, 21]#, 40, 10, 20, 40, 10, 20, 40, 10, 20, 40, 10, 20, 40]
@assert length(ks) == length(ns_u_steps_aut)
const maxiters = 1000
const alpha_opt = 0.01
const n_points_u_steps = 10
const h_refinement = false
const alpha_rr = alpha_u_steps = 10^-6
const r_refinement = false
const complete_check_after = 10

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
_title = "maxiters-$(maxiters) npus-$(n_points_u_steps) a_opt-$(alpha_opt) a_rr-$(alpha_rr)"
filename = basename(@__FILE__) |> x -> replace(x, ".jl" => "")
title = MasterThesis.format_date_githash_title_now(_title)
path_save = "../results/" * filename * "/"*title*"/"; mkpath(path_save)

curves = curves_hoq_training_data_1d()

@info "$(filename)" n_points_u_steps

n_cases = length(ns_u_steps_aut)
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
Threads.@threads for (i, (k, n_u_steps_aut)) in collect(enumerate(zip(ks, ns_u_steps_aut)))
    iter_title = "$(filename) mi-$(maxiters) k-$(k) nusa0-$(n_u_steps_aut) rr-$(r_refinement) hr-$(h_refinement) $(_title)"
    mbs, rep = solve_mbspline_curves2(
        n_u_steps_aut, k, curves;
        maxiters,
        title = iter_title,
        solve_u_steps = r_refinement,
        atol = 0, # should not exit because of it
        clamp_n_points_u_steps = false,
        n_points_per_u_step_max = n_points_u_steps,
        alpha_opt = alpha_opt,
        alpha_u_steps = alpha_u_steps,
        complete_check_after = complete_check_after
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


linestyles = repeat([:dashdotdot, :dashdot, :dash], 5)
colors = vcat([fill(palette(:default)[i], 3) for i in 1:5]...)    

iterations = collect(1:maximum(length.(lossess)))
labels = [
    L"$k = %$(k)$, $n_{regions} = %$(n_u_steps_aut-1)$" # the -1 is a gambiarra
    for (k, n_u_steps_aut) in zip(ks, ns_u_steps_aut)
]

## Losses

plt_lossess = plot_iterations_labels_colors_linestyles_log(lossess, labels, L"\mathcal{L} \ [-]", colors, linestyles)
file_plot_lossess_png = path_save * "losses.png"
file_plot_lossess_pdf = path_save * "losses.pdf"
touch(file_plot_lossess_png)
savefig(plt_lossess, file_plot_lossess_png)
savefig(plt_lossess, file_plot_lossess_pdf)

## Maximum Absolute Error

plt_mpess = plot_iterations_labels_colors_linestyles_log(mpess_comp, labels, L"Maximum Relative Error $[\%]$", colors, linestyles, n = complete_check_after)
file_plot_mpess_png = path_save * "normalized_maximum_absolute_error.png"
file_plot_mpess_pdf = path_save * "normalized_maximum_absolute_error.pdf"
touch(file_plot_mpess_png)
savefig(plt_mpess, file_plot_mpess_png)
savefig(plt_mpess, file_plot_mpess_pdf)

## Maximum Error heatmap

labels_nl = [
    L"k = %$(k) \ , \ n_{reg} = %$(n_u_steps_aut-1)" # the -1 is a gambiarra
    for (k, n_u_steps_aut) in zip(ks, ns_u_steps_aut)
]
#labels_nl[1] = L"k = %$(ks[1]) \ , \ n_{reg} = %$(ns_u_steps_aut[1]-1)"
plt_mpe_i = plot_heatmap_valuess(mpes_i_best_comp, maes_i_best, labels_nl, y_labels(curves), label_plt = L"\mathrm{MRE} \ [\%]", size = (450, 280))
file_plot_mpe_i_png = path_save * "maximum-percentual-error.png"
file_plot_mpe_i_pdf = path_save * "maximum-percentual-error.pdf"
touch(file_plot_mpe_i_png)
touch(file_plot_mpe_i_pdf)
savefig(plt_mpe_i, file_plot_mpe_i_png)
savefig(plt_mpe_i, file_plot_mpe_i_pdf)

# PGFPlots

labels_dat = [
    "k-$(k)n-$(n_u_steps_aut)"
    for (k, n_u_steps_aut) in zip(ks, ns_u_steps_aut)
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
ns_points_u_steps = $(n_points_u_steps)
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

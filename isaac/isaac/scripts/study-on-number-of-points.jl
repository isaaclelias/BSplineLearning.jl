using MasterThesis
using Logging, LoggingExtras
using Dates
using Plots#; gaston()
using LaTeXStrings
using ProgressMeter

const maxiters = 300
const n_u_steps_aut = 10
const k = 7
const solve_u_steps = false
const alpha_opt = 0.01

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

_title = "k-$(k) nusa-$(n_u_steps_aut) sus-$(solve_u_steps) maxiters-$(maxiters)"
filename = basename(@__FILE__) |> x -> replace(x, ".jl" => "")
title = MasterThesis.format_date_githash_title_now(_title)
path_save = "../results/" * filename * "/"*title*"/"; mkpath(path_save)

curves = curves_hoq_training_data_1d()

mbs0 = MasterThesis.initialize_mbs02(curves, n_u_steps_aut, k, 1)
n_points_u_steps_min = min_curves_points_inside_u_step(mbs0, curves)
@assert n_points_u_steps_min > 0

ns_points_u_steps_below = 1:n_points_u_steps_min
ns_points_u_steps_above = [Int(round(n_points_u_steps_min * n * 1.25, RoundUp)) for n in 1:5]
ns_points_u_steps = [ns_points_u_steps_below; ns_points_u_steps_above]
#ns_points_u_steps = [1, 2] # small case for debugging

@info "study-on-number-of-points" n_points_u_steps_min

n_ns_points_u_steps = length(ns_points_u_steps)
lossess = Vector{Vector{Float64}}(undef, n_ns_points_u_steps)
maess_norm = Vector{Vector{Float64}}(undef, n_ns_points_u_steps)
@showprogress desc = "Training..." Threads.@threads for (i, n_points_u_steps) in collect(enumerate(ns_points_u_steps))
    mbs, rep = solve_mbspline_curves2(
        n_u_steps_aut, k, curves;
        maxiters,
        title = "n_points_u_steps big sweep npus-$(n_points_u_steps) nusa-$(n_u_steps_aut) k-$(k) sus-$(solve_u_steps) a_opt-$(alpha_opt)",
        solve_u_steps = solve_u_steps,
        atol = 0, # should not exit because of it
        clamp_n_points_u_steps = false,
        n_points_per_u_step_max = n_points_u_steps,
        alpha_opt = alpha_opt
    )
    lossess[i] = rep.losses
    maess_norm[i] = rep.maes_norm
end

formats_ns_points_u_steps = string.(ns_points_u_steps) .* " points per region"
plt_lossess = plot_iterations_labels_log(lossess, formats_ns_points_u_steps, L"\mathcal{L} \ [-]")
file_plot_lossess_png = path_save * "losses.png"
touch(file_plot_lossess_png)
savefig(plt_lossess, file_plot_lossess_png)

plt_maess = plot_iterations_labels_log(maess_norm, formats_ns_points_u_steps, L"Normalized Maximum Absolute Error $[-]$")
file_plot_maess_png = path_save * "normalized_maximum_absolute_error.png"
touch(file_plot_maess_png)
savefig(plt_maess, file_plot_maess_png)

file_rep_txt = path_save*"report.txt"
rep = """
ns_points_u_steps = $(ns_points_u_steps)
lossess = $(lossess)
maess_norm = $(maess_norm)
"""
touch(file_rep_txt)
io_rep = open(file_rep_txt, "w")
write(io_rep, rep)
close(io_rep)

@info "study-on-number-of-points reached the end ;)"

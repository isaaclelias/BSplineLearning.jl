using MasterThesis
using Logging, LoggingExtras
using Dates
using Plots#; gaston()
using LaTeXStrings

r_refinements = [false, true]
const maxiters = 100
const n_u_steps_aut = 10
const k = 5
const alpha_opt = 0.01
const n_points_u_steps = 5
const h_refinement = false
const alpha_rr = alpha_u_steps = 10^-6
const r_refinement = true

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
_title = "k-$(k) nusa0-$(n_u_steps_aut) a_rr-$(alpha_rr) maxiters-$(maxiters) npus-$(n_points_u_steps) a_opt-$(alpha_opt)"
filename = basename(@__FILE__) |> x -> replace(x, ".jl" => "")
title = MasterThesis.format_date_githash_title_now(_title)
path_save = "../results/" * filename * "/"*title*"/"; mkpath(path_save)

curves = curves_hoq_training_data_1d()

@info "$(filename)" n_points_u_steps

n_r_refinements = length(r_refinements)
lossess = Vector{Vector{Float64}}(undef, n_r_refinements)
maess_norm = Vector{Vector{Float64}}(undef, n_r_refinements)
Threads.@threads for (i, r_refinement) in collect(enumerate(r_refinements))
    iter_title = "$(filename) rr-$(r_refinement) hr-$(h_refinement)  $(_title)"

    mbs, rep = solve_mbspline_curves2(
        n_u_steps_aut, k, curves;
        maxiters,
        title = iter_title,
        solve_u_steps = r_refinement,
        atol = 0, # should not exit because of it
        clamp_n_points_u_steps = false,
        n_points_per_u_step_max = n_points_u_steps,
        alpha_opt = alpha_opt,
        alpha_u_steps = alpha_u_steps
    )
    lossess[i] = rep.losses
    maess_norm[i] = rep.maes_norm
end

labels = L"$r$-refinement - " .* string.(r_refinements)
plt_lossess = plot_iterations_labels_log(lossess, labels, L"\mathcal{L} \ [-]")
file_plot_lossess_png = path_save * "losses.png"
touch(file_plot_lossess_png)
savefig(plt_lossess, file_plot_lossess_png)

plt_maess = plot_iterations_labels_log(maess_norm, labels, L"Normalized Maximum Absolute Error $[-]$")
file_plot_maess_png = path_save * "normalized_maximum_absolute_error.png"
touch(file_plot_maess_png)
savefig(plt_maess, file_plot_maess_png)

file_rep_txt = path_save*"report.txt"
rep = """
ns_points_u_steps = $(n_points_u_steps)
lossess = $(lossess)
maess_norm = $(maess_norm)
"""
touch(file_rep_txt)
io_rep = open(file_rep_txt, "w")
write(io_rep, rep)
close(io_rep)

@info "$(filename) reached the end ;)"

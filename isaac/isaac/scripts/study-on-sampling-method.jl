using MasterThesis
using Logging, LoggingExtras
using Dates
using Plots#; gaston()
using LaTeXStrings
using Serialization

const maxiters = 300
const n_u_steps_aut = 10
const k = 5
const solve_u_steps = false
const alpha_opt = 0.01
const n_points_u_steps = 32
const h_refinement = false
const r_refinement = false

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
_title = "mi-$(maxiters) k-$(k) nusa0-$(n_u_steps_aut) rr-$(solve_u_steps) hr-$(h_refinement) maxiters-$(maxiters) npus-$(n_points_u_steps)"
filename = basename(@__FILE__) |> x -> replace(x, ".jl" => "")
title = MasterThesis.format_date_githash_title_now(_title)
path_save = "../results/" * filename * "/"*title*"/"; mkpath(path_save)

curves = curves_hoq_training_data_1d()

sampling_methods = [:random, :region]

@info "$(filename)" n_points_u_steps

n_sampling_methods = length(sampling_methods)
lossess = Vector{Vector{Float64}}(undef, n_sampling_methods)
maess_norm = Vector{Vector{Float64}}(undef, n_sampling_methods)
reps = Vector{NamedTuple}(undef, n_sampling_methods)
Threads.@threads for (i, sampling_method) in collect(enumerate(sampling_methods))
    iter_title = "$(filename) $(string(sampling_method)) npus-$(n_points_u_steps) nusa-$(n_u_steps_aut) k-$(k) sus-$(solve_u_steps) a_opt-$(alpha_opt)"

    mbs, rep = solve_mbspline_curves2(
        n_u_steps_aut, k, curves;
        maxiters,
        title = iter_title,
        solve_u_steps = r_refinement,
        atol = 0, # should not exit because of it
        clamp_n_points_u_steps = false,
        n_points_per_u_step_max = n_points_u_steps,
        alpha_opt = alpha_opt,
        sampling = sampling_method,
    )
    lossess[i] = rep.losses
    maess_norm[i] = rep.maes_norm
    reps[i] = rep
end

labels = "\"" .* string.(sampling_methods) .* "\"" .* " sampling"
plt_lossess = plot_iterations_labels_log(lossess, labels, L"\mathcal{L} \ [-]")
file_plot_lossess_png = path_save * "losses.png"
touch(file_plot_lossess_png)
savefig(plt_lossess, file_plot_lossess_png)

plt_maess = plot_iterations_labels_log(maess_norm, labels, L"NMAE $[-]$")
file_plot_maess_png = path_save * "normalized_maximum_absolute_error.png"
touch(file_plot_maess_png)
savefig(plt_maess, file_plot_maess_png)

file_rep_txt = path_save*"report.txt"
rep = """
ns_points_u_steps = $(n_points_u_steps)
sampling_methods = $(sampling_methods)
lossess = $(lossess)
maess_norm = $(maess_norm)
"""
touch(file_rep_txt)
io_rep = open(file_rep_txt, "w")
write(io_rep, rep)
close(io_rep)

file_rep_jls = path_save*"report.jls"
serialize(file_rep_jls, reps)

@info "$(filename) reached the end ;)"

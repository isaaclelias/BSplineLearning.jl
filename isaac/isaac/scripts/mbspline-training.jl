using MasterThesis
using Debugger
using Optimization
using DataFrames
using Logging
using Logging, LoggingExtras
using Dates
using Pkg # issue #56216
using Revise

# Choose an AD alg
#using Enzyme     ; const adtype = AutoEnzyme()
#using Zygote     ; const adtype = AutoZygote()
#using ReverseDiff; const adtype = AutoReverseDiff()
#using ForwardDiff; const adtype = AutoForwardDiff()
#using FiniteDiff ; const adtype = AutoFiniteDiff()

# logging setup
log_dir = "../logs/"; mkpath(log_dir)
log_title = basename(@__FILE__) |> x -> replace(x, ".jl" => "")
log_format_date = Dates.format(now(), "yyyy-mm-dd_HH-MM")
log_path = log_dir * log_format_date * " " * log_title * ".log"
log_info_path  = log_path * ".info" ; touch(log_info_path)
log_debug_path = log_path * ".debug"; touch(log_debug_path)
const logger = TeeLogger(
    ConsoleLogger(stdout                    , Logging.Info ), # Console
    SimpleLogger( open(log_info_path , "w+"), Logging.Info ), # File for info
#    SimpleLogger( open(log_debug_path, "w+"), Logging.Debug), # File for debug
); global_logger(logger)
@info "Logging" log_info_path log_debug_path

# stuff to save the study results
_title = "trying to get the long training plot"
filename = basename(@__FILE__) |> x -> replace(x, ".jl" => "")
title = MasterThesis.format_date_githash_title_now(_title)
path_save = "../results/" * filename * "/"*title*"/"; mkpath(path_save)

# debugger setup
break_on(:error)

const curves::Vector{Curve} = curves_hoq_training_data_1d()
df = df_hoq_training_data_1d()
const n_u_steps_aut::Int = 2
const k::Int = 3
const maxiters::Int = 100
const n_additional_iterations::Int = 50
const solve_u_steps::Bool = false
const alpha_d::Float64 = alpha = 0.3
const atol::Float64 = 10^-6
const n_points_per_u_step_max::Int = 100
const clamp_n_points_u_steps::Bool = false
const h_refinement::Bool = true
const complete_check_after::Int = 10
const n_iter_refinement::Int = 200
const alpha_h::Float64 = 0.3
const alpha_opt::Float64 = 0.025

title_solve = "trying h refinement for real k-$(k) nusa-$(n_u_steps_aut) sus-$(solve_u_steps) alpha_d-$(alpha_d) nppusm-$(n_points_per_u_step_max) cnpus-$(clamp_n_points_u_steps) hr-$(h_refinement) nir-$(n_iter_refinement) cca-$(complete_check_after) ah-$(alpha_h) ao-$(alpha_opt)"

run_training() = solve_mbspline_curves2(
    n_u_steps_aut, k, curves;
    atol,
    alpha_d,
    maxiters,
    title = title_solve,
    solve_u_steps,
    n_points_per_u_step_max,
    clamp_n_points_u_steps,
    h_refinement,
    complete_check_after,
    n_iter_refinement,
    alpha_h,
    alpha_opt,
    n_additional_iterations,
)

mbs, rep = run_training() 

#=
_p2 = parameters_solve_mbspline_curves2(
    n_u_steps_aut, k, _curves;
    alpha_d,
    maxiters,
    title = title_solve,
    solve_u_steps 
)

min_curve_points_inside_u_step
=#


#= # Using solve_mbspline_curves

const alpha_n::Float64 = 0.05

p1 = MasterThesis.parameters_solve_mbspline_curves(
    curves, n_u_steps_aut, k;
    alpha, alpha_n,
    maxiters, solve_u_steps
)

mbs1 = solve_mbspline_curves(
    curves, n_u_steps_aut, k;
    alpha, alpha_n,
    maxiters,
    title = title_solve * " alpha_n-$(alpha_n)",
    adtype,
    solve_u_steps
)
=#


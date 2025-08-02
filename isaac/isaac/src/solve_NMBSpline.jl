###########################################################################################
# 
# the final boss
# let's do it
#
###########################################################################################

"""

"""
function solve_nmbs(
    degree  ::Int,
    surfaces::ASurfaces;
    atol                ::Union{Real, AbstractVector{Real}},
    maxiters            ::Int,
    n_samples_per_reg   ::Int,
    n_min_points_per_reg::Int,
    n_reg_start         ::Int,
    n_reg_max           ::Int,
    alpha_d             ::Real,
    check_iter          ::Int,
    opt_alg,
    alpha_opt,
    r_ref               ::Bool,
    alpha_r             ::Real,
    #h_ref               ::Bool,
    #h_ref_iter          ::Int,
    #h_ref_sensor        ::Symbol,
    #h_ref_loc           ::Symbol,
    #h_ref_interp        ::Symbol,
    save_result         ::Bool = true,
    save_path           ::AbstractString,
    save_title          ::AbstractVector,
)
    # Check inputs sanity
    @assert n_auth_reg_start >= 1 "There must be at least one "
    @assert length(atol) == 1 || length(atol) == length(surfaces)

    # Initialize
    nmbs0 = init_nmbs(degree, n_reg_start, alpha_d)
    parameters = init_parameters_solve_nmbs()
    history = init_history_solve_nmbs()
    @unpack nmbs0, nmbs0_norm, surfaces_norm, xs_min, xs_min, ys_min, ys_max, fmt_start_time = parameters
    @unpack tralala = history
    u = trainable_parameters()
    opt_state = Optimisers.setup(opt_alg(alpha_opt), u)
    nmbs = nmbs0
    nmbs_norm = nmbs_norm
    iter = 0
    good_enough = false

    # Inform the user
    @info "solve_nmbs" save_title fmt_start_time
    (n_reg_max - n_reg_start) && @warn "Maximum number of regions will never be achieved"

    # Train the spline
    while !good_enough && iter < maxiters
        iter = iter + 1

        inter = intermediates_solve_nmbs()
        @unpack

        # Check for early exit criteria and exit if so
        if n_points_least_populated_region < n_min_points_per_reg
            @warn ""
        end
    end
end

function init_parameters_solve_nmbs(arguments)
  
end

function init_nmbs(arguments)
  
end

function init_history_solve_nmbs(arguments)
    error()
end

function loss_solve_nmbs(arguments)
  
end

function grad_loss_solve_nmbs(arguments)
  
end

function update_solve_nmbs(arguments)
  
end

function callback_solve_nmbs(arguments)
    error()
end

function report_solve_nmbs(arguments)
    error()
end

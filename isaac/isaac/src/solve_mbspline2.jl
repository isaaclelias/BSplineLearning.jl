###########################################################################################
#
# solve_mbspline_curves2, o inimigo agora é outro
#
# - curves_sample_u_steps instead of curves_sample_random
# - 
#
###########################################################################################

"""
    solve_mbspline_curves2(
        n_u_steps_aut0::Int, # number of "authorized nodes", minumum of two, because we need one at each end of the domain
        k::Int, # order of the resulting spline
        curves::ACurves; # data to train agains
    )

Use Optimisers.jl instead of Optimization.jl

Accepts any optimiser that Optimisers.setup also accepts

Tho check how the hyper parameters closure is assembled, see also parameters_solve_mbspline_curves.

For this solver, it's assumed that the provided curves have all the same domain in principle. For my problem, all curves are functions of the same progress variable. So the difference between each y_min and y_max can be arbitrary, the dependent variable can be whatsoever, and inside here it will be normalized properly, so no need to normalize it before. But this is **not** the case for the independent variable x, as there is **only one** `x_min` and one `x_max`.

Keyword arguments:

- `k::Int`: order of the resulting spline
- curves::ACurves; # data to train against
- atol                   ::Union{Real, AbstractVector{Real}}, # absolute tolerance, 
- solve_u_steps          ::Bool           = false, # TASK rename to r_refinement
- h_refinement           ::Bool           = false, # use h-refinement
- alpha_d                ::Real           = 0.25, # how far apart the last (unauthorized) nodes should be from the end of the data domain # TASK
- alpha_h                ::Real           = 0.25, # learning rate for the h-refinement # TASK
- maxiters               ::Int            = 50, # stop after this much iterations, if it didn't stop beacause of something else
- title                  ::AbstractString = "", # title for saving the results
- save_result            ::Bool           = true, # should the results be saved? ~please do save them~
- n_points_per_u_step_max::Int            = 20, # maximum number of points batched **per region**
- clamp_n_points_u_steps ::Bool           = false, # if the region is "overpopulated" (have more points than the least populated region), only use the number of points equal to the least populated instead of using `n_points_per_u_step_max`
- alpha_opt              ::Real           = 0.025, # learning rate for the optimiser
- sampling               ::Symbol         = :region, # method for batching
- opt_alg                                 = Optimisers.Adam, # optimization algorithm
- alpha_u_steps          ::Real           = 10^-7, # relaxation for the r-refinement
- n_iter_refinement      ::Int            = 20, # refine after this much iterations
- complete_check_after   ::Int            = 10, # make a complete (expensive) check after this much iterations. It's here that it determines if the current spline is the best one.
- method_which           ::Symbol         = :error_center_of_mass_region, # method to decide which regions should be refined
- method_where           ::Symbol         = :first_order_moment, # method to decide where the regions marked for refinement should be refined
- method_how_high        ::Symbol         = :curve_middle, # method to decide the value of the weight that is going to be inserted
- min_points_per_region  ::Int            = 4 # don't refine regions with less than this much data points inside. Refining regions with few points can lead to regions with only one point inside, which impossibilidates the error calculation.
"""
function solve_mbspline_curves2(
    n_u_steps_aut0::Int, # number of "authorized nodes", minumum of two, because we need one at each end of the domain
    k::Int, # order of the resulting spline
    curves::ACurves; # data to train against
    atol                   ::Union{Real, AbstractVector{Real}}, # absolute tolerance, 
    solve_u_steps          ::Bool           = false, # TASK rename to r_refinement
    h_refinement           ::Bool           = false,
    alpha_d                ::Real           = 0.25, # TASK
    alpha_h                ::Real           = 0.25, # TASK
    maxiters               ::Int            = 50,
    title                  ::AbstractString = "",
    save_result            ::Bool           = true,
    n_points_per_u_step_max::Int            = 20,
    clamp_n_points_u_steps ::Bool           = false,
    alpha_opt              ::Real           = 0.025,
    sampling               ::Symbol         = :region,
    opt_alg                                 = Optimisers.Adam,
    alpha_u_steps          ::Real           = 10^-7,
    n_iter_refinement      ::Int            = 20,
    complete_check_after   ::Int            = 10,
    method_which           ::Symbol         = :error_center_of_mass_region,
    method_where           ::Symbol         = :first_order_moment,
    method_how_high        ::Symbol         = :curve_middle,
    min_points_per_region  ::Int            = 4,
    n_additional_iterations::Int            = 0,
)
    @assert length(atol) == 1 || length(atol) == length(curves)    
    @assert maxiters >= complete_check_after >= 1

    mbs0 = initialize_mbs02(curves, n_u_steps_aut0, k, alpha_d) 

    p = parameters_solve_mbspline_curves2(
        mbs0, curves; solve_u_steps,
        alpha_d, maxiters, title, save_result, alpha_opt, sampling, complete_check_after
    ); @unpack mbs0_norm, curves_norm, n_out, x_min, x_max, ys_min, ys_max, format_start_time = p

    @info "solve_mbspline_curve2" title format_start_time

    # Initialization
    u = trainable_parameters(mbs0_norm, solve_u_steps)
    opt_state = Optimisers.setup(opt_alg(alpha_opt), u)
    mbs = mbs0
    mbs_norm = mbs0_norm
    n_iter = 0
    is_good_enough = false
    while !is_good_enough && n_iter < maxiters + n_additional_iterations
        n_iter = n_iter + 1

        inter = intermediates_solve_mbspline_curves2(
            mbs, mbs_norm, curves, curves_norm, n_out, u, n_points_per_u_step_max,
            clamp_n_points_u_steps, sampling, p, alpha_u_steps, solve_u_steps
        )
        @unpack curves_sample_idxss, curves_sample, curves_norm_sample, loss, loss_i, grad, mae_i, mae_i_norm, mae, mae_norm, grad_alpha = inter

        # Update the training parameters
        opt_state, u = update_solve_mbspline_curves2(opt_state, u, grad_alpha)

        # Contruct (!!!!!) a new MBSpline and extract training parameters from it
        mbs_norm = mbs_from_u_mbs(u, mbs_norm)
        mbs = unnormalize11_xy_mbspline(mbs_norm, x_min, x_max, ys_min, ys_max)
        u = trainable_parameters(mbs_norm, solve_u_steps)

        # Check if the optimiser is triying to screw me up
        if !(min_curves_points_inside_u_step(mbs, curves) > 0)
            @error "solve_mbspline_curve2\nMBSpline has become uncontrollable by the optimiser. Breaking the training run and returning..." title n_iter
            break
        end

        is_refinement_iter = n_iter % n_iter_refinement == 0 && maxiters >= n_iter 
        #@show h_refinement is_refinement_iter n_iter n_iter_refinement
        if is_refinement_iter && h_refinement
            #@show "got inside refinement area"
            mbs = refine_mbspline(
                mbs, curves, alpha_h; method_which, method_where, method_how_high,
                min_points_per_region
            )
            mbs_norm = normalize11_xy_mbspline(mbs, x_min, x_max, ys_min, ys_max)
            u = trainable_parameters(mbs_norm, solve_u_steps)
            opt_state = Optimisers.setup(opt_alg(alpha_opt), u)
        end

        # Print the current state and save, both to memory and disk, important stuff
        callback_solve_mbspline_curves2!(
            p;
            opt_state, u, loss, loss_i, grad, save_result, mbs_norm, mbs,
            curves_sample_idxss, mae_i, mae_i_norm, mae, mae_norm, n_iter, grad_alpha,
            complete_check_after, is_refinement_iter,
        ) # Appends lots of stuff into `p` containers, save stuff to the disk

        # Check if it's already good enough
        is_good_enough = sieve_solve_mbspline_curves2(mbs, curves, atol)
    end

    # We always want the best to be given to the caller
    idx_best = find_best_idx(p.losses)
    mbs_best = p.mbss[idx_best]
    rep = report_solve_mbspline_curves2(p)

    return mbs_best, rep
end

function refine_mbspline(
    mbs, curves, alpha_h;
    method_which::Symbol = :pe_mean_squared,
    method_where::Symbol = :mass,
    method_how_high::Symbol = :prediction,
    min_points_per_region::Int
)::MBSpline
    weightss_old = mbs.weightss
    u_steps_old = mbs.u_steps
    k = mbs.k
    
    idxs_reg_to_refine = which_regions_to_refine(mbs, curves, alpha_h, method_which, min_points_per_region)
    #@show idxs_reg_to_refine
    u_steps_new = insert_u_steps(u_steps_old, mbs, curves, idxs_reg_to_refine; method_where)
    weightss_new = insert_weightss(
        weightss_old, mbs, curves, idxs_reg_to_refine; method_how_high, method_where
    )
    #@show length(u_steps_new) length(u_steps_old) length.(weightss_new) length.(weightss_old)

    @info "refine_mbspline" idxs_reg_to_refine

    return MBSpline(weightss_new, u_steps_new, k)
end

"""
    first_authorized_node_idx(k)

find the index of the first authorized node. the one that sits right on the edge of the domain, but in this case, left on the edge. it's the smallest one.
"""
first_authorized_node_idx(k::Int)::Int = k + 1

"""
    last_authorized_node_idx(k, n_u_steps)

find the index of the first authorized node. the one that sits right on the edge of the domain, it's the biggest one.
"""
last_authorized_node_idx(k::Int, n_u_steps::Int)::Int = n_u_steps - k

"""
    idxs_authorized_regions(k, n_u_steps)

Get the indices of the nodes vector `u_steps` that lie within the fully controlled regions.
"""
function idxs_authorized_regions(k, n_u_steps)

    return first_authorized_node_idx(k):last_authorized_node_idx(k, n_u_steps)-1
end

"""
    which_regions_to_refine(
        mbs::MBSpline, curves::ACurves, alpha_h, method_which::Symbol
    )::Vector{Int}

Determine the indexes of the regions that should be refined based on a given criteria.

- `error_center_of_mass_region`: x coordinate of the center of mass
"""
function which_regions_to_refine(
    mbs::MBSpline, curves::ACurves, alpha_h, method_which::Symbol, min_points_per_region
)::Vector{Int}
    weightss = mbs.weightss
    u_steps = mbs.u_steps
    k = mbs.k
    n_u_steps = length(u_steps)
    n_aut_reg = length(idxs_authorized_regions(k, n_u_steps))
    n_refinements = Int(round(n_aut_reg*alpha_h, RoundUp))

    # calculate various ranking criteria
    rankings = NamedTuple[]
    for idx_reg in eachindex(u_steps)[begin:end-1] #idxs_authorized_regions(k, n_u_steps)
        idxs = minimum([length(findall_idxs_x_inside_idx_reg(u_steps, idx_reg, curve)[begin:end]) for curve in curves])#-1]) for curve in curves])
        # if it's not authorized, it's -Inf, so it gets to the end of the list and - I assumed - never get refined
        # pe_max: maximum percentage error for all curves
        # pe_i_max: vector with max percentage error for each curve
        # pe_max_abs: absolute value for the max percentage error (originally the pe_max is signed)
        if !is_authorized_idx(idx_reg, k, n_u_steps) || idxs <= min_points_per_region
            len = length(maximum_percentage_error_and_errors_reg2(mbs, curves, idx_reg)[2])
            pe_max, pe_i_max = (-Inf, fill(-Inf, len))
            pe_max = -Inf; pe_i_max = fill(-Inf, len)
            pe_max_abs, pe_i_max_abs = (-Inf, fill(-Inf, len))
            pe_mean = -Inf
            pe_mean_squared = -Inf
        else
            pe_max, pe_i_max = maximum_percentage_error_and_errors_reg2(
                mbs, curves, idx_reg
            )
            pe_max = abs(pe_max); pe_i_max = abs.(pe_i_max)
            pe_max_abs, pe_i_max_abs = (abs(pe_max), abs.(pe_i_max))
            pe_mean = mean(pe_i_max_abs)
            pe_mean_squared = sum(pe_i_max_abs.^2) / length(pe_i_max_abs)
        end
        push!(rankings, (; pe_max_abs, pe_mean, pe_mean_squared))
    end

    # sort by ranking
    if method_which == :error
        perm = sortperm([ranking.pe_mean for ranking in rankings], rev = true)
    elseif method_which == :quadratic_error
        perm = sortperm([ranking.pe_mean_squared for ranking in rankings], rev = true)
    else
        error("Invalid `method_which`")
    end

    # return only the top ranked ones
    #ret = perm[1:n_refinements]
    ret = [perm[1]] # ignore h refinement learning rate
    @assert !any(isnan.(ret)) # isn't better to be safe than sorry? ~take on meeeee~
    #@show ret rankings perm
    return ret
end

"""
    is_authorized_idx(idx, k, n_u_steps)

return true if an index maps to a authorized node.
"""
is_authorized_idx(idx, k, n_u_steps) = idx in idxs_authorized_regions(k, n_u_steps)

"""
    error_center_of_mass_region(mbs::MBSpline, curves, idx_reg)

Calculate the first order momentum of the error within a given region.

Assumes that the `curves` are normalized.

Regions are always to the right of the index that identifies them.
"""
function error_center_of_mass_region(mbs::MBSpline, curves, idx_reg)
    xs_cm_curve = Float64[]
    areas_curve = Float64[]
    u_steps = mbs.u_steps

    # iterate over the curves
    for (i, curve) in enumerate(curves)
        idxs = findall_idxs_x_inside_idx_reg(u_steps, idx_reg, curve)[begin:end-1]
        #@show idxs
        #isempty(idxs) && @show i isempty(idxs)
        isempty(idxs) && continue
        As    = Float64[]
        xs_cm = Float64[]

        # iterate over the indexes of the curve that lie within the region `idx_reg`
        for idx in idxs[begin:end-1]
            A = abs(curve.y[idx] - mbs(curve.x[idx])[i]) * (curve.x[idx+1] - curve.x[idx])
            #A == 0 && @show idx A curve.x[idx+1] curve.x[idx] curve.y[idx+1] curve.y[idx]
            @assert !isnan(A)
            x_cm = (curve.x[idx] + curve.x[idx+1])/2
            @assert !isnan(x_cm)
            push!(As, A)
            push!(xs_cm, x_cm)
        end
        isempty(As) && continue
        area_curve = sum(As)
        @assert !isnan(area_curve)
        x_cm_curve = sum(As .* xs_cm) / area_curve
        #@show i sum(As .* xs_cm) area_curve sum(As) sum(xs_cm) As
        @assert !isnan(x_cm_curve)

        push!(areas_curve, area_curve)
        push!(xs_cm_curve, x_cm_curve)
    end

    ret = sum(xs_cm_curve .* areas_curve) / sum(areas_curve)
    @assert !isnan(ret)
    return ret
end

"""
    error_quadratic_center_of_mass_region(mbs::MBSpline, curves, idx_reg)

calculate the x position for the box error, but only inside the region corresponding to idx_reg.
"""
function error_quadratic_center_of_mass_region(mbs::MBSpline, curves, idx_reg)
    xs_cm_curve = Float64[]
    areas_curve = Float64[]
    u_steps = mbs.u_steps

    # iterate over the curves
    for (i, curve) in enumerate(curves)
        idxs = findall_idxs_x_inside_idx_reg(u_steps, idx_reg, curve)[begin:end-1]
        #@show idxs
        #isempty(idxs) && @show i isempty(idxs)
        isempty(idxs) && continue
        As    = Float64[]
        xs_cm = Float64[]

        # iterate over the indexes of the curve that lie within the region `idx_reg`
        for idx in idxs[begin:end-1]
            A = abs(curve.y[idx] - mbs(curve.x[idx])[i])^2 * (curve.x[idx+1] - curve.x[idx])
            #A == 0 && @show idx A curve.x[idx+1] curve.x[idx] curve.y[idx+1] curve.y[idx]
            @assert !isnan(A)
            x_cm = (curve.x[idx] + curve.x[idx+1])/2
            @assert !isnan(x_cm)
            push!(As, A)
            push!(xs_cm, x_cm)
        end
        isempty(As) && continue
        area_curve = sum(As)
        @assert !isnan(area_curve)
        x_cm_curve = sum(As .* xs_cm) / area_curve
        #@show i sum(As .* xs_cm) area_curve sum(As) sum(xs_cm) As
        @assert !isnan(x_cm_curve)

        push!(areas_curve, area_curve)
        push!(xs_cm_curve, x_cm_curve)
    end

    ret = sum(xs_cm_curve .* areas_curve) / sum(areas_curve)
    @assert !isnan(ret)
    return ret
end

"""
    insert_u_steps(
        u_steps, mbs, curves, which_regions; method_where = :first_order_moment
    )

insert new nodes inside a vector of nodes, using the given methods against the data provided by the curves.

the indexes received in `which_regions` are assumed to be all controlled.
"""
function insert_u_steps(
    u_steps, mbs, curves, which_regions; method_where = :first_order_moment
)
    u_steps_new = copy(u_steps)
    for idx_reg in which_regions
        u_step_new = begin
            if method_where == :first_order_moment
                #@show idx_reg
                error_center_of_mass_region(mbs, curves, idx_reg)
            elseif method_where == :second_order_moment
                error_quadratic_center_of_mass_region(mbs, curves, idx_reg)
                #error("second order moment not implemented yet")
            end
        end
        @assert !any(isnan.(u_steps_new))
        insert!(u_steps_new, idx_reg, u_step_new)
    end

    return u_steps_new
end

function insert_weightss(
    weightss, mbs::MBSpline, curves, which_regions::Vector{Int};
    method_how_high::Symbol, method_where::Symbol
)

    weightss_new = Vector{Float64}[]
    x_min, x_max, ys_min, ys_max = extrema_curves(curves)

    for (i, (weights, y_min, y_max)) in enumerate(zip(weightss, ys_min, ys_max))
        weights_new = copy(weights)
        for idx_reg in which_regions
            # first we need to know where to calculate the value for the weight
            where_to_calc = 0.0
            if method_where == :first_order_moment
                where_to_calc = error_center_of_mass_region(mbs, curves, idx_reg)
            elseif method_where == :second_order_moment
                where_to_calc = error_quadratic_center_of_mass_region(mbs, curves, idx_reg)
            else
                error("Invalid method_where")
            end

            # now we can finally calculate how high the weight should be inserted
            if method_how_high == :spline_prediction # not the best
                #@warn "insert_weightss\nspline_prediction is probably not working" # but it should be right now
                weight_new = mbs(where_to_calc)[i]
            elseif method_how_high == :curve_middle # the easiest to implement
                weight_new = (y_min+y_max)/2
            else
                error("Invalid method_how_high")
            end
            insert!(weights_new, idx_reg, weight_new)
        end
        push!(weightss_new, weights_new)
    end

    return weightss_new
end

#=
# Convenience
function solve_mbspline_curves2(
     curves::ACurves;
    alpha_d = 0.25,
    maxiters = 50,
    title = "",
    save_result = true,
    solve_u_steps = false,
    atol = 10^-8,
    clamp_n_points_u_steps = true,
    n_points_per_u_step_max = 20,
    alpha_opt = 0.01
)

    return solve_mbspline_curves2(
        mbs0, curves;
        alpha_d, maxiters, title, solve_u_steps, save_result, atol, n_points_per_u_step_max,
        clamp_n_points_u_steps, alpha_opt
    )
end
=#

function initialize_mbs02(curves, n_u_steps_aut, k, alpha_d)
    x_min, x_max, ys_min, ys_max = extrema_curves(curves)
    n_curves = length(curves)

    return initialize_mbs02(x_min, x_max, ys_min, ys_max, n_u_steps_aut, k, alpha_d, n_curves)
end

function initialize_mbs02(
    x_min ::Real, x_max ::Real, ys_min, ys_max,
    n_u_steps_aut, k, alpha_d, n_curves
)
    @assert 0 < alpha_d
    n_u_steps_out = 2*k
    x_span = x_max - x_min

    u_step_out_distance = x_span*alpha_d/n_u_steps_out
    u_steps = [
        [x_min - i*u_step_out_distance for i in k:-1:1]; # unauthorized points on the left
        LinRange(x_min, x_max, n_u_steps_aut)          ; # authorized points
        [x_max + i*u_step_out_distance for i in 1: 1:k]; # unauthorized points on the right 
    ]
    n_u_steps = length(u_steps)
    n_weights = n_u_steps - k - 1
    weightss = [fill((y_min+y_max)/2, n_weights) for (y_min, y_max) in zip(ys_min, ys_max)]

    return MBSpline(weightss, u_steps, k)

    # Here I choose to initialize the weights with a normal distribution, because the profesor of Physics-Aware Machine Learning told to. There's a name for this method. 
end

function parameters_solve_mbspline_curves2(
    mbs0::MBSpline, curves::ACurves;

    maxiters::Int,

    title::AbstractString = "", 
    savefig::Bool = true,
    save_result::Bool = true,
    solve_u_steps::Bool = false,
    alpha_d,
    alpha_opt,
    sampling,
    complete_check_after,
)
    # Where, when and why
    start_time = now(); format_start_time = Dates.format(start_time, "YYYY-mm-dd_HH-MM")
    path_save = "../results/solve_mbspline_curve2/"*format_date_githash_title_now(title)*"/"
    
    # About mbs0
    n_weightss = length(mbs0.weightss)
    n_weights0 = length(mbs0.weightss[begin])
    n_u_steps0 = length(mbs0.u_steps)
    k = mbs0.k
    n_weights0_total = n_weights0 * n_weightss

    x_min, x_max, ys_min, ys_max = extrema_curves(curves)
    
    # normalize bspline
    n_out = Int(number_u_steps_outside_domain(mbs0)/2)
    n_u_steps_aut0 = number_u_steps_inside_domain(mbs0)
    #mbss_norm = [mbs0_norm]
    
    # About the curves
    ns_points = length.(curves)
    #ns_points_sample = length.(curves_norm_samples[begin])
    n_curves = length(curves)

    ## Normalizations
    #mbs0_norm = normalize11_y_mbspline(mbs0, ys_min, ys_max)
    #curves_norm = normalize11_y_curves(curves, ys_min, ys_max)
    mbs0_norm = normalize11_xy_mbspline(mbs0, x_min, x_max, ys_min, ys_max)
    curves_norm = normalize11_xy_curves(curves, x_min, x_max, ys_min, ys_max)

    @info "parameters_solve_mbspline_curves2" extrema_curves(curves) extrema_curves(curves_norm) #mbs0.u_steps

    # Containers, now i think this should be separate from parameters
    mbss                 = MBSpline[]
    mbss_norm            = MBSpline[]
    losses               = Float64[]
    losses_i             = Vector{Float64}[]
    grads                = ComponentVector[] # put a type in this please TASK
    grads_alpha          = ComponentVector[] # put a type in this please TASK
    us                   = ComponentVector[] # same here
    curves_samples_idxs  = Vector{Curve}[]
    #mbss_best            = MBSpline[]
    #mbss_norm_best       = MBSpline[]
    #losses_best          = Float64[]
    #losses_i_best        = Vector{Float64}[]
    maes                 = Float64[]
    maes_norm            = Float64[]
    maes_i               = Vector{Float64}[]
    maes_i_norm          = Vector{Float64}[]
    ## complete checks
    losses_comp          = Float64[] 
    losses_i_comp        = Vector{Float64}[]
    maes_i_comp          = Vector{Float64}[]
    maes_comp            = Float64[]
    maes_i_norm_comp     = Vector{Float64}[]
    maes_norm_comp       = Float64[]
    mpes_i_comp          = Vector{Float64}[]
    mpes_comp       = Float64[]
    #curves_norm_samples = Vector{Curve}[]

    p = (;
        mbs0, mbs0_norm, curves, start_time, losses, title, solve_u_steps,
        format_start_time, path_save, x_min, x_max, ys_min, ys_max, n_weights0, ns_points,
        k, n_weightss, savefig, n_out, n_u_steps0, n_weights0_total, n_curves,
        curves_samples_idxs, mbss, losses_i, grads, us, curves_norm, mbss_norm,
        n_u_steps_aut0, maes, maes_norm, maes_i, maes_i_norm, alpha_opt, alpha_d, sampling,
        grads_alpha, losses_comp, losses_i_comp, maes_i_comp, maes_comp, maes_i_norm_comp,
        maes_norm_comp, mpes_i_comp, mpes_comp, complete_check_after
        # mbss_best, losses_best, losses_i_best, mbss_norm_best,
    )
    
    return p
end

function normalize11_xy_curves(curves::ACurves, x_min, x_max, ys_min, ys_max)::ACurves
    curves_norm = similar(curves)

    for (i, (curve, y_min, y_max)) in enumerate(zip(curves, ys_min, ys_max))
        curves_norm[i] = Curve(
            normalize_11.(curve.x, x_min, x_max),
            normalize_11.(curve.y, y_min, y_max),
            curve.xlabel,
            curve.ylabel,
        )
    end

    return curves_norm
end

function normalize11_xy_mbspline(mbs, x_min, x_max, ys_min, ys_max)
    weightss_norm = normalize11.(mbs.weightss, ys_min, ys_max)
    u_steps_norm = normalize11.(mbs.u_steps, x_min, x_max)
    k = mbs.k
    mbs_norm = MBSpline(weightss_norm, u_steps_norm, k)

    return mbs_norm
end

function unnormalize11_xy_mbspline(mbs_norm, x_min, x_max, ys_min, ys_max)::MBSpline
    weightss_norm = mbs_norm.weightss
    u_steps_norm = mbs_norm.u_steps
    weightss = [
        unnormalize11.(weights_norm, y_min, y_max)
        for (weights_norm, y_min, y_max) in zip(weightss_norm, ys_min, ys_max)
    ]
    k = mbs_norm.k
    u_steps = unnormalize11.(u_steps_norm, x_min, x_max)
    mbs = MBSpline(weightss, u_steps, k)

    return mbs
end

function intermediates_solve_mbspline_curves2(
    mbs, mbs_norm, curves, curves_norm, n_out::Int, u, n_points_per_u_step,
    clamp_n_points_u_steps, sampling, p, alpha_r, solve_u_steps
)
    weightss = mbs.weightss
    weightss_norm = mbs_norm.weightss
    u_steps = mbs.u_steps
    u_steps_norm = mbs_norm.u_steps
    k = mbs.k

    # curves::ACurves, u_steps::Vector, n_out::Int, n_points_per_u_step_max,
    # clamp_n_points_u_steps
    curves_sample_idxss = sampling_solve_mbspline_curves2_idxss(p, mbs, n_points_per_u_step)
    curves_sample = curves_sample_by_idxss(curves_norm, curves_sample_idxss)
    curves_norm_sample = curves_sample_by_idxss(curves_norm, curves_sample_idxss)

    loss, loss_i = loss_and_loss_i_mbspline_curves2(u, mbs_norm, curves_norm_sample)
    grad = gradient_loss_mbspline_curves2_forwarddiff(u, mbs_norm, curves_norm_sample)
    grad_alpha = gradient_loss_mbspline_curves2_alpha_forwarddiff(u, mbs_norm, curves_norm_sample, alpha_r, solve_u_steps)
    mae, mae_i = maximum_absolute_error_and_errors2(k, weightss, u_steps, curves_sample)
    mae_norm, mae_i_norm = maximum_absolute_error_and_errors2(k, weightss_norm, u_steps_norm, curves_norm_sample)

    return (;
        curves_sample_idxss, curves_sample, curves_norm_sample, loss, loss_i, grad,
        mae_i, mae_i_norm, mae, mae_norm, grad_alpha
    )
end

function sampling_solve_mbspline_curves2_idxss(p, mbs, n_points_per_u_step)
    @unpack n_curves, curves, sampling, n_out = p
    u_steps = mbs.u_steps
    n_u_steps_aut = number_u_steps_inside_domain(mbs)
    n_regions = n_u_steps_aut + 1
    n_regions_total = n_regions * n_curves
    #@show u_steps n_u_steps_aut n_regions n_regions_total

    if sampling == :region
        ret = curves_sample_u_steps_min_idxss(
            curves, u_steps, n_out, n_points_per_u_step, false
        )
        return ret
    elseif sampling == :random
        return curves_sample_random_n_points_idxss(curves, n_points_per_u_step * n_regions)
    end

end

function gradient_loss_mbspline_curves2_forwarddiff(u, mbs, curves)
    f(u) = loss_mbspline_curves2(u, mbs, curves)
    grad = ForwardDiff.gradient(f, u)

    return grad
end

function gradient_loss_mbspline_curves2_alpha_forwarddiff(
    u, mbs, curves, alpha_r, solve_u_steps
)
    n_out = Int(number_u_steps_outside_domain(mbs)/2)
    f(u) = loss_mbspline_curves2(u, mbs, curves)
    grad = ForwardDiff.gradient(f, u)
    grad_alpha = begin 
        if solve_u_steps
            u_steps_grad = grad.u_steps
            u_steps_grad[begin:n_out] .= zero(eltype(u_steps_grad))
            u_steps_grad[end-n_out+1:end] .= zero(eltype(u_steps_grad))
            ComponentVector{Float64}((
                weightss = collect(grad.weightss), u_steps = alpha_r * u_steps_grad
            ))
        else
            grad
        end
    end

    return grad_alpha
end

function jacobian_loss_i_mbspline_curves2_forwarddiff(u, mbs, curves)
    f(u) = loss_i_mbspline_curves2(u, mbs, curves)
    jaco = ForwardDiff.gradient(f, u)

    return jaco
end

function loss_and_loss_i_mbspline_curves2(u, mbs, curves)
    n_curves = length(curves)
    ns_points = length.(curves)
    k = mbs.k
    weightss = haskey(u, :weightss) ? u.weightss : mbs.weightss
    u_steps  = haskey(u, :u_steps ) ? u.u_steps  : mbs.u_steps

    return mean_squared_error_and_errors2(
        k, weightss, u_steps, curves, ns_points, n_curves
    )
end

function loss_mbspline_curves2(u, mbs, curves)
    return loss_and_loss_i_mbspline_curves2(u, mbs, curves)[1]
end

function loss_i_mbspline_curves2(u, mbs, curves)
    return loss_and_loss_i_mbspline_curves2(u, mbs, curves)[2]
end

# # Training parameters update routines

function update_solve_mbspline_curves2(opt_state, u, grad)
    opt_state, u = Optimisers.update!(opt_state, u, grad)

    return opt_state, u

    # Just realized that there is a naming ambiguity involving `u`, which can be both the training parameters current state, of the place that a *BSpline is evaluated. I'll switch to `x`on the notation for *BSplines.
end

# # Callback routines

"""

LPoK: This from now on is a _**L**ost **P**iece of **K**nowledge_. I'll use that to denote random stuff that I'd like to have written down, but too busy to write them somewhere else.

LPoK: By default, saving stuff to the disc and appending new information to the parameters closure `p` is done solely inside the callback.
"""
function callback_solve_mbspline_curves2!(
    p;
    opt_state, u, loss, loss_i, grad, save_result, mbs_norm, mbs, curves_sample_idxss,
    mae_i, mae_i_norm, mae, mae_norm, n_iter, grad_alpha, complete_check_after,
    is_refinement_iter,
)
    @unpack curves, title, losses, path_save, x_min, x_max, ys_min, ys_max, ns_points, n_out, curves_norm, mbss_norm, mbss, grads, mbs0, us, losses_i, curves_samples_idxs, n_u_steps_aut0, maes_i, maes, maes_norm, maes_i_norm, grads_alpha, losses_comp, losses_i_comp, maes_i_comp, maes_comp, maes_i_norm_comp, maes_norm_comp, mpes_i_comp, mpes_comp = p

    curves_sample = curves_sample_by_idxss(curves, curves_sample_idxss)
    curves_norm_sample = curves_sample_by_idxss(curves_norm, curves_sample_idxss)
    k = mbs.k
    u_steps = mbs.u_steps
    n_points_u_step_min = min_curves_points_inside_u_step(curves, u_steps, n_out)
    n_u_steps = length(u_steps)
    n_points_sample = sum(length.(curves_sample))
    n_u_steps_aut = number_u_steps_inside_domain(mbs)
    n_regions = n_u_steps_aut + 1
    weightss = mbs.weightss
    weightss_norm = mbs_norm.weightss

    # Update stuff
    ## update the curves_sample
    push!(curves_samples_idxs)
    ## update the history of mbs
    push!(mbss, mbs)
    push!(mbss_norm, mbs_norm)
    ## Gradient
    push!(grads, grad)
    push!(grads_alpha, grad_alpha)
    ## current training parameters
    push!(us, u)
    push!(losses, loss)
    push!(losses_i, loss_i)
    push!(maes_i, mae_i)
    push!(maes, mae)
    push!(maes_i_norm, mae_i_norm)
    push!(maes_norm, mae_norm)
    ## best results
    idx_best = find_best_idx(losses)
    loss_best = losses[idx_best]
    isbest = loss <= loss_best

    # let's do a complete and expensive check
    if complete_check_after % n_iter == 0
        @info "callback_solve_mbspline_curves2\nPerforming a complete check" n_iter
        loss_comp, loss_i_comp = loss_and_loss_i_mbspline_curves2(u, mbs_norm, curves_norm)
        mae_comp, mae_i_comp = maximum_absolute_error_and_errors2(mbs, curves)
        mae_norm_comp, mae_i_norm_comp = maximum_absolute_error_and_errors2(
            mbs_norm, curves_norm
        )
        mpe_comp, mpe_i_comp = maximum_percentage_error_and_errors2(mbs, curves)

        push!(losses_comp, loss_comp)
        push!(losses_i_comp, loss_i_comp)
        push!(maes_i_comp, mae_i_comp)
        push!(maes_comp, mae_comp)
        push!(maes_i_norm_comp, mae_i_norm_comp)
        push!(maes_norm_comp, mae_norm_comp)
        push!(mpes_i_comp, mpe_i_comp)
        push!(mpes_comp, mpe_comp)
    end

    @info "callback_solve_mbspline_curves2" title n_iter loss findmax(loss_i) findmin(loss_i) findmax(mae_i) findmin(mae_i) findmax(mae_i_norm) findmin(mae_i_norm) findmin(grad_alpha) findmax(grad_alpha) n_regions n_points_u_step_min n_points_sample isbest
    @debug "callback_solve_mbspline_curves2" title n_iter k n_u_steps_aut0 n_u_steps_aut u loss_i mae_i mae_i_norm mbs mbs_norm grad grad_alpha

    if save_result
        mkpath(path_save)
        ft_n_iter = format_n_iter(n_iter)

        # save plot
        path_plot = path_save*"plots/"
        file_plot = path_plot*ft_n_iter*".png"
        mkpath(path_plot)
        lock(lock_plt) do # trying to add a lock here to prevent GR from segfaulting
            plt = plot_mbspline_curves_sample(mbs, curves, curves_sample_idxss)
            savefig(plt, file_plot)
            # save best plot until now
            if isbest
                file_plot_best_png = path_save * "plot_best.png"
                file_plot_best_pdf = path_save * "plot_best.pdf"
                savefig(plt, file_plot_best_png)
                savefig(plt, file_plot_best_pdf)
            end
        end

        ## save model
        path_jls = path_save
        mkpath(path_jls)
        file_mbs_jls = path_jls*"mbs.jls"
        serialize(file_mbs_jls, mbs)

        # save pretty much everything
        file_p_jls = path_save*"p.jls"
        serialize(file_p_jls, p)

        # JSON3 tidy version of the serialization

        @debug "callback_solve_mbspline_curves2 save_result" path_save file_plot file_mbs_jls file_p_jls
    end

    return false
end

find_best_idx(losses) = findmin(losses)[2]

function sieve_solve_mbspline_curves2(mbs, curves, atol)
    mae_i =  maximum_absolute_errors2(mbs, curves)

    return all(mae_i .< atol)
end

function report_solve_mbspline_curves2(p, save_result = true)
    @unpack start_time, losses, path_save, n_curves, losses, mbss, maes_i, curves, mpes_i_comp, mpes_comp, complete_check_after, maes_comp, maes_i_comp = p

    #idx_best = find_best_idx(losses)
    idx_best = find_best_idx(mpes_comp) * complete_check_after
    #mbs_best = mbss[idx_best]
    elapsed_time = now() - start_time
    n_iter_end = length(losses)
    idx_best = find_best_idx(losses)
    mbs_best = mbss[end]
    mae_i_best = maes_i_comp[end]
    mae_best, mae_i_best_comp = maximum_absolute_error_and_errors2(mbs_best, curves)
    mpe_best_comp, mpe_i_best_comp = maximum_percentage_error_and_errors2(mbs_best, curves)

    # cleanup
    path_plot = path_save*"plots/" # by now this should be a global variable
    cmd = `ffmpeg -f image2 -pattern_type glob -framerate 6 -i "$(path_plot)*.png" -s 1920x1080 $(path_save)training.mp4`# \&\& rm -r $(path_plot)`
    save_result && run(cmd) # generate video from plots and delete plots if the video was generated succesfully. i'm kinda scared of running a rm -r

    rep = (;p..., elapsed_time, n_iter_end, idx_best, mbs_best, mae_i_best, mae_i_best_comp, mpe_i_best_comp, mpe_best_comp)

    if save_result
        file_rep_jls = path_save * "report_complete.jls"
        serialize(file_rep_jls, rep)
        # I'd really like this in more transparent format, like JSON

        #touch(file_rep)
        #io = open(file_rep, "w+")
        #print(io, rep)
        
        format_rep = """
        Total number of iterations: $(n_iter_end)
        Total training time: $(elapsed_time)
        Maximum Absolute Errors (whole population): $(mae_i_best_comp)
        Maximum Percentage Errors (whole population): $(mpe_i_best_comp)
        

        k = $(mbs_best.k)
        mae_i_best = $(mae_i_best)
        mpes_i_comp = $(mpe_i_best_comp)
        mpes_comp = $(mpe_best_comp)
        losses = $(losses)
        mbs = $(mbs_best)
        """
        file_rep_txt = path_save * "report.txt"
        touch(file_rep_txt)
        io_rep = open(file_rep_txt, "w")
        write(io_rep, format_rep)
        close(io_rep)
    end

    @debug "report_solve_mbspline_curves2" losses mbs_best
    
    return rep
end

###########################################################################################
# Code not to be used anymore
###########################################################################################

"""
If you want the wild life of just chugging unnormalized curves inside the mse and have them normalized here.

Assumes that `u` comes from a secure place, like inside a MBSpline, and won't cause any trouble if used to calculate a MBSpline assuming `@inbounds`.
"""
function loss_and_loss_i_mbspline_curves2_normalize(u, p, mbs, curves)
    @unpack ys_min, ys_max, n_curves = p
    ns_points = length.(curves)
    k = mbs.k
    weightss = haskey(u, :weightss) ? u.weightss : mbs.weightss
    u_steps  = haskey(u, :u_steps ) ? u.u_steps  : mbs.u_steps

    return normalized_mean_squared_error_and_errors2(
        k, weightss, u_steps, curves, ys_min, ys_max, ns_points, n_curves
    )

    # Here the trained quantities can skip passing through the construction of a new MBSpline. This allows it's type to be way more flexible than a raw Float64. Parametrizing the MBSpline type with T<:Real wasn't that useful after all. But it's worth noticing that I didn't give up on the assertions that I do upon a MBSpline construction. The training parameters `u` are gotten from a MBSpline on the caller. From here on, I trust that the caller got those parameters from a decent and sanitized place, like a MBSpline. Please don't try to fill this directly by hand. I'd really enjoy if it was kept clean and inbounds.
end

curves_sample_solve_mbspline_curves2(curves, mbs) = curves_sample_u_steps(curves, mbs)


#Functors.@functor MBSpline (weights,)
#Functors.@functor MBSpline (weights, u_steps)

function save_res_p(res, p)
  
end

function normalize11_y_mbspline(mbs, ys_min, ys_max)
    weightss_norm = normalize11.(mbs.weightss, ys_min, ys_max)
    k = mbs.k
    mbs_norm = MBSpline(weightss_norm, u_steps, k)

    return mbs_norm 
end

function normalize11_y_curves(curves::ACurves, ys_min, ys_max)::ACurves
    curves_norm = similar(curves)

    for (i, (curve, y_min, y_max)) in enumerate(zip(curves, ys_min, ys_max))
        curves_norm[i] = Curve(
            curve.x,
            normalize_11.(curve.y, y_min, y_max),
            curve.xlabel,
            curve.ylabel,
        )
    end

    return curves_norm
end

unnormalize11(a_norm::Real, a_min::Real, a_max::Real) = a_norm * max(abs(a_min), abs(a_max))

function unnormalize11_x_mbspline(mbs::MBSpline, ys_min, ys_max)::MBSpline
    weightss = [
        unnormalize11.(weights, y_min, y_max)
        for (weights, y_min, y_max) in zip(mbs.weightss, ys_min, ys_max)
    ]
    u_steps = mbs.u_steps
    k = mbs.k

    return MBSpline(weightss, u_steps, k)
end



function mbs_from_u_mbs(u, mbs)
    weightss = haskey(u, :weightss) ? u.weightss : mbs.weightss
    u_steps  = haskey(u, :u_steps ) ? u.u_steps  : mbs.u_steps
    k = mbs.k

    return MBSpline(weightss, u_steps, k)
end


function normalized_mean_squared_error_and_errors2(
    k, weightss, u_steps, curves, ys_min, ys_max, ns_points, n_curves
)

    nmses = normalized_mean_squared_errors2(
        k, weightss, u_steps, curves, ys_min, ys_max, ns_points
    )
    nmse = (1/n_curves) * sum(nmses)

    return nmse, nmses
end

function normalized_mean_squared_errors2(
    k, weightss, u_steps, curves::ACurves, ys_min, ys_max, ns_points
)
    nmses = [
        normalized_mean_squared_error_i2(
            k, weightss, u_steps, curves, i, ys_min, ys_max, ns_points
        )
        for i in eachindex(curves)
    ]

    return nmses
end

function normalized_mean_squared_error_i2(
    k, weightss, u_steps, curves::ACurves, i::Int, ys_min, ys_max, ns_points
)
    y_min = ys_min[i]
    y_max = ys_max[i]
    n_points = ns_points[i]
    curve = curves[i]
    difference = curve.y - calc_mbspline(k, weightss, u_steps, curve.x)[i]
    normalized_difference = (1/n_points) * normalize.(difference, y_min, y_max)
    #@debug "normalized_mean_squared_error" curve.ylabel i y_min y_max n_points difference sum(difference) normalized_difference sum(normalized_difference) mse acc
    mse = sum(normalized_difference.^2)

    return mse
end

#=
function normalized_mean_squared_error2(
    k, weightss, u_steps, curves::ACurves, ys_min, ys_max,
    n_weightss, ns_points
)
     nmse = (1/n_weightss) * sum([
        normalized_mean_squared_error_i2(k, weightss, u_steps, curves, i, y_min, y_max, n_points)
        for (i,         (curve , y_min , y_max , n_points ))
        in enumerate(zip(curves, ys_min, ys_max, ns_points))
    ])

    return nmse
end
=#

#const adtype2 = AutoReverseDiff()

#solve_mbspline_curves = solve_mbspline_curves1

function solve_mbspline_curves(
    mbs0::MBSpline, curves::ACurves;

    alpha_n::Union{Real, AbstractVector{Real}}, maxiters::Int,

    title::AbstractString = "", 
    callback::Function = callback_solve_mbspline_curves, savefig::Bool = true,
    save_result::Bool = true, adtype, alg = Optimization.LBFGS(),
    solve_u_steps::Bool = false, 
)
    p = parameters_solve_mbspline_curves(
        mbs0, curves; alpha_n, maxiters, title, callback, savefig, save_result, solve_u_steps
    )
    @unpack n_weights0, n_weightss, k, ns_points, n_weights0_total, n_curves, x_min, x_max, ys_min, ys_max, format_start_time = p
    
    # Better safe than sorry
    @assert n_weightss == n_curves

    @info "solve_mbspline_curve" format_start_time n_curves k n_weights0 n_weights0_total findmin(ns_points) findmax(ns_points)
    @debug "solve_mbspline_curve" mbs0

    optf = Optimization.OptimizationFunction(
        (u, p) -> loss_mbspline_curves(u, p), adtype
    )

    u0 = trainable_parameters(mbs0, solve_u_steps)
    optprob = Optimization.OptimizationProblem(optf, u0, p)
    _callback(state, loss) = callback(state, loss, p)

    res = Optimization.solve(optprob, alg; callback = _callback, maxiters)
    u = res.u

    @info "solve_mbspline_curves" res.retcode

    #rp = report_solve_mbspline_curves(res, curves)
    
    mbs_norm = mbs_norm_from_u_p(u, p)
    mbs = unnormalize_mbspline(mbs_norm, x_min, x_max, ys_min, ys_max)

    return mbs_from_u_p(res.u, p)
end

#curves_sample_solve_mbspline_curves(curves, alpha_n) = curves_sample_random(curves, alpha_n)
curves_sample_solve_mbspline_curves(curves, alpha_n) = curves_sample_random(curves, alpha_n)

function parameters_solve_mbspline_curves(
    mbs0::MBSpline, curves::ACurves;

    alpha_n::Union{Real, AbstractVector{Real}}, maxiters::Int,

    title::AbstractString = "", 
    callback::Function = callback_solve_mbspline_curves, savefig::Bool = true,
    save_result::Bool = true,
    solve_u_steps::Bool = false, 
)
    # Where, when and why
    start_time = now(); format_start_time = Dates.format(start_time, "YYYY-mm-dd_HH-MM")
    path_save = "../results/solve_mbspline_curve/"*format_date_githash_title_now(title)*"/"
    
    # About mbs0
    n_weightss = length(mbs0.weightss)
    n_weights0 = length(mbs0.weightss[begin])
    n_u_steps0 = length(mbs0.u_steps)
    k = mbs0.k
    n_weights0_total = n_weights0 * n_weightss
    mbss = [mbs0]

    x_min, x_max, ys_min, ys_max = extrema_curves(curves)
    
    # normalize bspline
    mbs0_norm = normalize_mbspline(mbs0, x_min, x_max, ys_min, ys_max)
    n_out = n_points_outside_domain(mbs0)
    mbss_norm = [mbs0_norm]

    # About the curves
    curves_samples = [curves_sample_solve_mbspline_curves(curves, alpha_n)]
    curves_norm = normalize_curves(curves, x_min, x_max, ys_min, ys_max)
    curves_norm_samples = [curves_sample_solve_mbspline_curves(curves_norm, alpha_n)]
    ns_points = length.(curves)
    ns_points_sample = length.(curves_norm_samples[begin])
    n_curves = length(curves)

    losses = Float64[]
    losses_i = Vector{Float64}[]
    grads = [] # put a type in this please TASK
    us = [] # same here

    p = (;
        mbs0, mbs0_norm, curves, curves_norm, start_time, losses, title, solve_u_steps, format_start_time,
        path_save, alpha_n, curves_norm_samples, x_min, x_max, ys_min, ys_max,
        n_weights0, ns_points, ns_points_sample, k, n_weightss, savefig, n_out, n_u_steps0,
        n_weights0_total, n_curves, curves_samples, mbss, mbss_norm, losses_i, grads, us
    )
    
    return p
end

function solve_mbspline_curves(curves, n_weights_con, k; alpha_d, kwargs...)
    x_min, x_max, ys_min, ys_max = extrema_curves(curves)
    curves_norm = normalize_curves(curves, x_min, x_max, ys_min, ys_max)
    mbs0 = initialize_mbs0(curves, n_weights_con, k, alpha_d)

    return solve_mbspline_curves(mbs0, curves; kwargs...)
end

function parameters_solve_mbspline_curves(curves, n_weights_con, k; alpha_d, kwargs...)
    x_min, x_max, ys_min, ys_max = extrema_curves(curves)
    mbs0 = initialize_mbs0(curves, n_weights_con, k, alpha_d)
    mbs0_norm = normalize_mbspline(mbs0, x_min, x_max, ys_min, ys_max)

    return parameters_solve_mbspline_curves(mbs0, curves; kwargs...)
end


function callback_solve_mbspline_curves(state, loss, p)
    @unpack curves, title, losses, path_save, x_min, x_max, ys_min, ys_max, curves_norm, ns_points, curves_norm_samples, mbs0_norm, n_out, curves_samples, mbss, mbss_norm, alpha_n = p
    u = state.u
    push!(p.losses, loss)
    n_iter = length(losses)
    mbs_norm = mbs_norm_from_u_p(u, p)
    mbs_unnorm = unnormalize_mbspline(mbs_norm, x_min, x_max, ys_min, ys_max)
    mbs = mbs_from_u_p(u, p)
    loss_i = mean_squared_errors(mbs_norm, curves_norm, ns_points)
    u_steps_norm = haskey(u, :u_steps) ? u.u_steps : mbs0_norm.u_steps
    #n_points_u_step_min = min_curves_points_inside_u_step(curves_norm, u_steps_norm, n_out)
    n_u_steps = length(u_steps_norm)
    curves_sample = curves_samples[end]

    @info "callback_solve_mbspline_curves" n_iter loss findmax(loss_i) findmin(loss_i) n_u_steps #n_points_u_step_min
    #@show loss_i

    # Save stuff
    mkpath(path_save)

    ## save plot
    plt = plot_callback_solve_mbspline_curves(mbs, p)
    ft_n_iter = format_n_iter(n_iter)
    path_plot = path_save*"plots/"
    mkpath(path_plot)
    savefig(plt, path_plot*ft_n_iter*".png")

    ## save model
    path_jls = path_save*"MBSplines/"
    mkpath(path_jls)
    serialize(path_jls*ft_n_iter*".jls", mbs)

    # Update stuff
    ## update the curves_sample
    push!(curves_samples, curves_sample_solve_mbspline_curves(curves, alpha_n))
    push!(curves_norm_samples, curves_sample_solve_mbspline_curves(curves_norm, alpha_n))
    ## update the history of mbs
    push!(mbss, mbs)
    push!(mbss_norm, mbs_norm)

    return false
end

function loss_mbspline_curves(u, p)
    @unpack ys_min, ys_max, n_weightss, ns_points_sample, ns_points, curves_norm_samples, k, curves = p
    curves_norm_sample = curves_norm_samples[end]
    curves_norm_sample = curves_norm_samples[end]
    mbs_norm = mbs_norm_from_u_p(u, p)
    mbs = mbs_norm_from_u_p(u, p)

    #ret = mean_squared_error(mbs, curves_norm_sample, ns_points, n_weightss)
    ret = normalized_mean_squared_error(mbs, curves, ys_min, ys_max, n_weightss, ns_points)

    return ret
end

function mean_squared_error(mbs, curves, ns_points, n_weightss)

    return (1/n_weightss) * sum(mean_squared_errors(mbs, curves, ns_points))
end

mean_squared_errors(mbs, curves, ns_points) = [
    mean_squared_error_i(mbs, curves, ns_points, i) for i in eachindex(curves)
]

function mean_squared_error_i(mbs, curves, ns_points, i)

    return (1/ns_points[i]) * sum((curves[i].y - mbs(curves[i].x)[i]).^2)
end

function normalized_mean_squared_error(
    mbs::MBSpline, curves::ACurves, ys_min, ys_max,
    n_weightss = length(mbs.weightss),
    ns_points = length.(curves)
)
     nmse = (1/n_weightss) * sum([
        normalized_mean_squared_error_i(mbs, curves, i, y_min, y_max, n_points)
        for (i,         (curve , y_min , y_max , n_points ))
        in enumerate(zip(curves, ys_min, ys_max, ns_points))
    ])

    return nmse
end

function normalized_mean_squared_error_i(
    mbs::MBSpline, curves::ACurves, i::Int, y_min, y_max,
    n_points = length(curves[i])
)
    difference = curves[i].y - mbs(curves[i].x)[i]
    normalized_difference = (1/n_points) * normalize.(difference, y_min, y_max)
    #@debug "normalized_mean_squared_error" curve.ylabel i y_min y_max n_points difference sum(difference) normalized_difference sum(normalized_difference) mse acc
    mse = sum(normalized_difference.^2)

    return mse
end

function normalized_mean_squared_error_i(mbs, p, i)
    @unpack curves, ys_min, ys_max, n_points = p
    y_min = ys_min[i]; y_max = ys_max[i]

    return normalized_mean_squared_error_i(mbs, curves, y_min, y_max, n_points)
end

function normalized_mean_squared_errors(
    mbs::MBSpline, curves::ACurves, ys_min, ys_max,
    ns_points = length.(curves)
)
    nmses = [
        normalized_mean_squared_error_i(mbs, curves, i, y_min, y_max, n_points)
        for (i, (curve , y_min , y_max , n_points)) in enumerate(zip(curves, ys_min, ys_max, ns_points))
    ]

    return nmses
end

function normalized_mean_squared_errors(mbs::MBSpline, p)
    @unpack curves, ys_min, ys_max, ns_points = p
    
    return normalized_mean_squared_errors(mbs, curves, ys_min, ys_max)
end

format_n_iter(n_iter) = @sprintf "%05i" n_iter

function plot_callback_solve_mbspline_curves(mbs::MBSpline, p)
    plt = plot_mbspline_curves(mbs, p.curves)

    return plt
end

function report_solve_mbspline_curves(result, p)
    rp = """
    Inital loss:
    Minimal loss:
    Minimal loss n_iter:
    Minimal loss n_weights:
    Final loss:

    """

    return rp
end

#mbs_from_state(state, p) = mbs_from_trainable_parameters(state.u, p)

#=
function mbs_from_trainable_parameters(tp, p)
    if haskey(tp, :u_steps)
        return MBSpline(tp.weightss, tp.u_steps, p.mbs0.k)
    else
        return MBSpline(tp.weightss, p.mbs0.u_steps, p.mbs0.k)
    end
end
=#

"""
Assumes a normalized `u`
"""
function mbs_norm_from_u_p(u, p)
    @unpack mbss_norm = p
    mbs0_norm = p.mbs0_norm
    weightss_norm = haskey(u, :weightss) ? u.weightss : mbs0_norm.weightss
    u_steps_norm  = haskey(u, :u_steps ) ? u.u_steps  : mbs0_norm.u_steps
    k = mbs0_norm.k

    return MBSpline(weightss_norm, u_steps_norm, k)
end

"""
Assumes a non-normalized `u`
"""
function mbs_from_u_p(u, p)
    @unpack mbss = p
    mbs = mbss[end]
    weightss = haskey(u, :weightss) ? u.weightss : mbs.weightss
    u_steps  = haskey(u, :u_steps ) ? u.u_steps  : mbs.u_steps
    k = mbs.k

    return MBSpline(weightss, u_steps, k)
end

function trainable_parameters(mbs::MBSpline, solve_u_steps::Bool)
    if solve_u_steps
        return ComponentVector{Float64}((weightss = mbs.weightss, u_steps = mbs.u_steps))
    else
        return ComponentVector{Float64}((weightss = mbs.weightss,))
    end
end

# TASK clean up
normalize(a, a_min, a_max) = normalize_01(a, a_min, a_max)
normalize_01(a, a_min, a_max) = (a - a_min)/(a_max - a_min)
normalize_11(a, a_min, a_max) = a * (1/max(abs(a_min), abs(a_max)))
normalize11(a, a_min, a_max) = normalize_11(a, a_min, a_max)
normalize01(a, a_min, a_max) = normalize_01(a, a_min, a_max)

function normalize_curves(curves::ACurves, x_min, x_max, ys_min, ys_max)::ACurves
    curves_norm = similar(curves)

    for (i, (curve, y_min, y_max)) in enumerate(zip(curves, ys_min, ys_max))
        curves_norm[i] = Curve(
            normalize.(curve.y, y_min, y_max),
            normalize.(curve.x, x_min, x_max),
            curve.xlabel,
            curve.ylabel,
        )
    end

    return curves_norm
end

@inline unnormalize(a_norm::Real, a_min::Real, a_max::Real) = a_norm * (a_max-a_min) + a_min
unnormalize01(a_norm, a_min, a_max) = unnormalize(a_norm, a_min, a_max)

function unnormalize_mbspline(mbs::MBSpline, x_min, x_max, ys_min, ys_max)::MBSpline
    weightss = [
        unnormalize.(weights, y_min, y_max)
        for (weights, y_min, y_max) in zip(mbs.weightss, ys_min, ys_max)
    ]
    u_steps = unnormalize.(mbs.u_steps, x_min, x_max)
    k = mbs.k

    return MBSpline(weightss, u_steps, k)
end

function normalize_mbspline(mbs::MBSpline, x_min, x_max, ys_min, ys_max)::MBSpline
    weightss = [
        normalize.(weights, y_min, y_max)
        for (weights, y_min, y_max) in zip(mbs.weightss, ys_min, ys_max)
    ]
    u_steps = normalize.(mbs.u_steps, x_min, x_max)
    k = mbs.k

    return MBSpline(weightss, u_steps, k)
end

y_extrema(curve::Curve) = nm.extrema(curve.y)

#=
maximum_absolute_error(mbs::MBSpline, curves::ACurves) = maximum([
    nm.maximum([curve.y - mbs(curve.x) for (curve, ys_bs) in zip(curves, ys_mbs)])
])
=#

function initialize_mbs0(
    x_min ::Real          , x_max ::Real          ,
    ys_min::AbstractVector, ys_max::AbstractVector,
    n_u_steps_aut, k, alpha_d
)
    @assert length(ys_min) == length(ys_max)
    @assert 0 < alpha_d
    n_weightss = length(ys_min); n_u_steps_out = 2*k
    x_span = x_max - x_min
    dists_weights = [
        Normal((y_min+y_max)/2, 0)#(y_max-y_min))
        for (y_min, y_max) in zip(ys_min, ys_max)
    ]
    u_step_out_distance = x_span*alpha_d/n_u_steps_out
    u_steps = [
        [x_min - i*u_step_out_distance for i in k:-1:1]; # unauthorized points on the left
        LinRange(x_min, x_max, n_u_steps_aut)          ; # authorized points
        [x_max + i*u_step_out_distance for i in 1: 1:k]; # unauthorized points on the right 
    ]
    n_u_steps = length(u_steps)
    n_weights = n_u_steps - k - 1
    #weightss = [rand(dist_weights, n_weights) for dist_weights in dists_weights]
    weightss = [fill((y_min + y_max)/2, n_weights) for (y_min, y_max) in zip(ys_min, ys_max)]

    return MBSpline(weightss, u_steps, k)

    # Here I choose to initialize the weights with a normal distribution, because the profesor of Physics-Aware Machine Learning told to. There's a name for this method. 
end

function initialize_mbs0(curves::ACurves, n_u_steps_aut::Int, k::Int, alpha_d::Real)
    x_min, x_max, ys_min, ys_max = extrema_curves(curves)

    return initialize_mbs0(x_min, x_max, ys_min, ys_max, n_u_steps_aut, k, alpha_d)
end

function extrema_curves(curves::ACurves)
    x_min = nm.minimum([nm.minimum(curve.x) for curve in curves])
    x_max = nm.maximum([nm.maximum(curve.x) for curve in curves])
    ys_min = [nm.minimum(curve.y) for curve in curves]
    ys_max = [nm.maximum(curve.y) for curve in curves]

    return x_min, x_max, ys_min, ys_max
end


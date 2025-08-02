const adtype = AutoZygote()
#const adtype = AutoEnzyme()
#const adtype = AutoForwardDiff()

function initialize_bs0(x_cv, y_cv, n_weigths, k, n_out)
    x_cv_min = x_cv[begin]; x_cv_max = x_cv[end]
    x_cv_span = x_cv_max - x_cv_min
    y_cv_min = min(y_cv...); y_cv_max = max(y_cv...)
    n_u_steps = n_weigths+k+1
    x_bs_extrapolation = u_steps_mean_diff(n_out, n_u_steps, x_cv_span)
    x_bs_min = x_cv_min - x_bs_extrapolation
    x_bs_max = x_cv_max + x_bs_extrapolation
    bs0 = BSpline(n_weigths, k, (x_bs_min, x_bs_max), (y_cv_min, y_cv_max))

    return bs0
end

function u_steps_mean_diff(n_out, n_u_steps, x_cv_span)
    h = x_cv_span/(n_u_steps-2*n_out-1)
end

function save_solve_bspline_curve_results(dir, bs)
    #CSV
    serialize(dir*"BSpline.jll", bs)
    #plt = 
end

#=
function solve_bspline_curve(
    x_cv, y_cv,
    n_weigths::Int, k::Int, n_u_steps_out::Int,
    alphas;
    call_callback = true,
    title = "",
    maxiters = 1000
)

    bs0 = initialize_bs0(x_cv, y_cv, n_weigths, k, n_u_steps_out)
    solve_bspline_curve(
        bs0, x_cv, y_cv, n_weigths, k, alphas,
        savefig = savefig, maxiters = maxiters,
        call_callback = call_callback
    )
end
=#

function solve_bspline_curve(
    bs0::BSpline,
    x_cv, y_cv,
    alphas;
    savefig = true,
    save_bspline = true,
    callback = callback_savefig_bspline_curve_losses,
    title = "",
    maxiters,
    atol
)

    start_time = now(); started_at = Dates.format(start_time, "YYYY-mm-dd_HH-MM")
    losses = Float64[]
    p = (; bs0, x_cv, y_cv, start_time, losses, alphas, title, atol)

    n_weigths = length(bs0.weights); n_u_steps = length(bs0.u_steps); k = bs0.k
    @info "solve_bspline_curve" n_weigths n_u_steps k length(x_cv)

    optf = Optimization.OptimizationFunction(
        (u, p) -> loss_bspline_curve_mse(u, p),
        adtype#,
        #cons = (res, u, p) -> cons_bspline!(res, u, p)
    )
    u0 = trainable_parameters(bs0)
    optprob = Optimization.OptimizationProblem(
        optf, u0, p#,
        #lcons = repeat([-Inf], length(u0)),
        #ucons = repeat([0.0], length(u0))
    )
    _callback(state, losses) = callback(state, losses, p)

    res = Optimization.solve(
        optprob,
        Optimization.LBFGS(),
        callback = _callback,
        maxiters = maxiters
    )

    k = bs0.k; weights = res.u.weights; u_steps = res.u.u_steps
    bs = BSpline(weights, u_steps, k)

    return bs
end

function callback_savefig_bspline_curve_losses(state, loss, p)
    weights = state.u.weights
    u_steps = state.u.u_steps
    k = p.bs0.k
    bs = BSpline(weights, u_steps, k)
    x_cv = p.x_cv
    y_cv = p.y_cv
    y_bs = bs.(x_cv)
    title = p.title
    x_min = x_cv[begin]
    x_max = x_cv[end]
    start_time = p.start_time
    losses = p.losses
    started_at = Dates.format(start_time, "YYYY-mm-dd_HH-MM")
    dir = pkg_root_dir*"/results/solve_bspline_curve/"*date_title(started_at, title)*"/"
    aerr = max(abs.(y_bs - y_cv)...)
    atol = p.atol

    mkpath(dir)

    push!(p.losses, loss)
    n_iter = length(losses)

    @info "solve_bspline_curve" n_iter loss

    plt = plot_bspline_curve_losses(bs, x_cv, y_cv, losses)

    st_fm = Dates.format(start_time, "YYYY-mm-dd_HH-MM")
    dir_plt = dir*"/plot_bspline_curve_losses/"
    mkpath(dir_plt)
    n_iter_ft = @sprintf "%05i" n_iter
    savefig(plt, dir_plt*n_iter_ft*".png")

    return false
end

maximum_absolute_error(y_bs, y_cv) = max(abs.(y_bs-y_cv)...)

function maximum_absolute_error(bs::BSpline, x_cv, y_cv)
    y_bs = bs.(x_cv)

    return maximum_absolute_error(y_bs, y_cv)
end


#plot_bspline_fitting_losses() 

function loss_bspline_curve_mse(model_parameters, parameters)
    # get parameters to construct the spline
    bs0 = parameters.bs0
    k = bs0.k
    weights = model_parameters.weights
    u_steps = model_parameters.u_steps
    x_cv = parameters.x_cv
    y_cv = parameters.y_cv
    x_cv_min = x_cv[begin]
    x_cv_max = x_cv[end]
    n_u_steps = length(u_steps)
    n_x = length(x_cv)

    bs = BSpline(weights, u_steps, k) # construct the spline

    # calculate the prediction
    y_bs = bs.(x_cv)

    alpha_MSE = parameters.alphas.MSE
    MSE = (1/n_x)*sum((y_bs - y_cv).^2)

    return alpha_MSE*MSE
end

#=
function cons_bspline!(res, u, p)
    res = Vector{Float64}(undef, length(u))
    weights = u.weights
    u_steps = u.u_steps
    j_we_begin = 1
    j_we_end = length(weights)
    j_us_begin = length(weights)+1
    j_us_end = length(weights)+length(u_steps)
    res[j_we_begin:j_we_end] .= -Inf
    res[j_us_begin] = -Inf
    res[j_us_end] = -Inf
    res[(j_us_begin+1):(j_us_end-1)] .= [u[j]-u[j+1] for j in (j_us_begin+1):(j_us_end-1)]

    return res
end
=#

function loss_bspline_curve_mse_proximity_boundary(model_parameters, parameters)
    # get parameters to construct the spline
    bs0 = parameters.bs0
    k = bs0.k
    weights = model_parameters.weights
    u_steps = model_parameters.u_steps
    x_cv = parameters.x_cv
    y_cv = parameters.y_cv
    x_cv_min = x_cv[begin]
    x_cv_max = x_cv[end]
    n_u_steps = length(u_steps)
    n_x = length(x_cv)

    bs = BSpline(weights, u_steps, k) # construct the spline

    # calculate the prediction
    y_bs = bs.(x_cv)

    alpha_MSE = parameters.alphas.MSE
    MSE = (1/n_x)*sum((y_bs - y_cv).^2)

    alpha_proximity = parameters.alphas.proximity
    uss = sort(u_steps)
    L_proximity = (1/(n_x-1))*sum([1/(uss[j]-uss[j+1]).^2 for j in 1:n_u_steps-1])

    alpha_u_steps_boundary = 1
    n_out = k+1
    first_u_step_inside_domanin = 
    L_u_steps_boundary = [uss[j]-uss[j]]
    # make tem mirror the distance to x_cv_max from the inner points
    
    return alpha_MSE*MSE + alpha_proximity*L_proximity
end

function loss_bspline_curve_experimental(model_parameters, parameters)
    # get parameters to construct the spline
    bs0 = parameters.bs0
    k = bs0.k
    weights = model_parameters.weights
    u_steps = model_parameters.u_steps
    x_cv = parameters.x_cv
    y_cv = parameters.y_cv
    x_cv_min = x_cv[begin]
    x_cv_max = x_cv[end]
    n_u_steps = length(u_steps)
    n_x = length(x_cv)

    bs = BSpline(weights, u_steps, k) # construct the spline

    # calculate the prediction
    y_bs = bs.(x_cv)

    alpha_MSE = parameters.alphas.MSE
    MSE = (1/n_x)*sum((y_bs - y_cv).^2)

    alpha_proximity = parameters.alphas.proximity
    uss = sort(u_steps)
    L_proximity = (1/(n_x-1))*sum([1/(uss[j]-uss[j+1]).^2 for j in 1:n_u_steps-1])

    alpha_u_steps_boundary = 1
    L_u_steps_boundary = 1
    # make tem mirror the distance to x_cv_max from the inner points
    
    #=
    alpha_span = parameters.alphas.span
    L_span = 1/(uss[end]-uss[begin])^2

    alpha_boundary = parameters.alphas.boundary
    L_boundary = (sum(abs.(uss[1:(k+1)].-x_cv_min)) + sum(abs.(uss[(end-k-1):end].-x_cv_max)))^2
    =#

    alpha_u_steps_overshoot = 1
    L_u_steps_overshoot = 1

    return alpha_MSE*MSE + alpha_proximity*L_proximity
end



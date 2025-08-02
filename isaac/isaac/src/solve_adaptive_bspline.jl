function solve_adaptive_bspline_curve(
    x_cv, y_cv, k, alphas;
    maxiters_inside, maxiters_adaptive, atol, rtol, learning_ratio,
    title = ""
)

    @assert atol > zero(atol) && rtol > zero(rtol) && 0 < learning_ratio <= 1
   
    n_out = k+1
    n_weigths_start = 2*n_out+k+1
    bs = initialize_bs0(x_cv, y_cv, n_weigths_start, k, n_out)

    start_time = now()
    losses_adaptive = Float64[]
    pa = (; losses_adaptive, title, start_time)
    _callback(s, l, pp) = callback_solve_adaptive_bspline_curve(s, l, pp, pa)

    for i in 1:maxiters_adaptive
        bs = solve_bspline_curve(
            bs, x_cv, y_cv, alphas,
            callback = _callback,
            maxiters = maxiters_inside, atol = atol,
        )
  
        new_weights, new_u_steps = where_to_add_u_step(
            bs, x_cv, y_cv, atol, rtol, learning_ratio
        )
        trim_u_steps_left, trim_u_steps_right = where_to_trim_u_step(bs, x_cv, n_out)
        bs = recreate_bspline(
            bs, new_weights, new_u_steps,
            trim_u_steps_left, trim_u_steps_right
        )

        if length(new_weights) == 0
            aerr_max, aerr_mean, rerr_max, rerr_mean = error_bspline_curve(bs, x_cv, y_cv)
            @assert aerr_max <= atol && rerr_max <= rtol "Trying to exit without having achieved the desired tolerances."
            @info "solve_adaptive_bspline_curve achieved tolerances. Exiting..." aerr_max aerr_mean rerr_max rerr_mean
            println("Achieved tolerances. Exiting...")
            return bs
        end
    end

    return bs
end

function recreate_bspline(bs::BSpline, new_weights, new_u_steps, trim_u_steps_left, trim_u_steps_right)
    weights = bs.weights
    u_steps = bs.u_steps
    n_new_weights = length(new_weights)
    n_u_steps = length(u_steps)
    n_weigths = length(weights)
    k = bs.k
    #new_weights_and_u_steps = collect(zip(new_weights, new_u_steps))

    # add weights 
    for (new_weight, new_u_step) in zip(new_weights, new_u_steps)
        new_u_step_index = searchsortedfirst(u_steps, new_u_step)
        new_weight_index = begin
            if new_u_step_index <= k
                1
            elseif new_u_step_index < n_u_steps-k
                new_u_step_index-k
            else
              @assert new_u_step_index-k-1 == n_weigths "new_u_step_index = $(new_u_step_index); n_weigths = $(n_weigths); k = $(k)"
                new_u_step_index-k-1
            end
        end
        @info "Inserting new control points" new_u_step_index new_u_step new_weight_index new_weight
        insert!(u_steps, new_u_step_index, new_u_step)
        insert!(weights, new_weight_index, new_weight)
    end

    # remove weights
    u_steps = u_steps[trim_u_steps_left:trim_u_steps_right]
    weights = weights[trim_u_steps_left:trim_u_steps_right-k-1]

    new_bs = BSpline(weights, u_steps, k)

    return new_bs

    #=
    # Here I assume that it will never be the case that the `where_to_add_u_step` says that we should add more points outside the domain, and `where_to_rm_u_step` says the opossite. I assume both are excludent. Maybe I should refrase them to `adapt_boundaries` and `adapt_domain`, so this behaviour could be better enforced.
    =#
end

function callback_solve_adaptive_bspline_curve(state, loss, p, pa)
    weights = state.u.weights
    u_steps = state.u.u_steps
    k = p.bs0.k
    bs = BSpline(weights, u_steps, k)
    x_cv = p.x_cv
    y_cv = p.y_cv
    y_bs = bs.(x_cv)
    title = pa.title
    x_min = x_cv[begin]
    x_max = x_cv[end]
    start_time = pa.start_time
    losses = p.losses
    started_at = Dates.format(start_time, "YYYY-mm-dd_HH-MM")
    dir = pkg_root_dir*"/results/solve_adaptive_bspline_curve/"*date_title(started_at, title)*"/"
    aerr = max(abs.(y_bs - y_cv)...)
    atol = p.atol
    losses_adaptive = pa.losses_adaptive
    n_weigths = length(weights)
    
    mkpath(dir)

    push!(p.losses, loss)
    push!(losses_adaptive, loss)
    n_iter = length(losses)
    n_iter_adaptive = length(losses_adaptive)

    @info "solve_adaptive_bspline_curve" n_iter n_iter_adaptive loss 

    plt = plot_solve_adaptive_bspline_curve(bs, x_cv, y_cv, p, pa)

    st_fm = Dates.format(start_time, "YYYY-mm-dd_HH-MM")
    dir_plt = dir*"/training_plots/"
    mkpath(dir_plt)
    n_iter_ft = @sprintf "%05i" n_iter_adaptive
    n_weigths_ft = @sprintf "%03i" n_weigths
    savefig(plt, dir_plt*n_iter_ft*".png")

    return false
end

function plot_solve_adaptive_bspline_curve(bs, x_cv, y_cv, p, pa)
    n_weigths = length(bs.weights)
    aerr = maximum_absolute_error(bs, x_cv, y_cv)
    n_iter = length(pa.losses_adaptive)

    plt_bs_cv = plot_bspline_curve(bs, x_cv, y_cv)
    plt_info = plot(grid = false, showaxis = false)
    annotate!(
        plt_info,
        [
            (0/3, 2/2, ("Number of weights: $(n_weigths)", 8, :left)),
            (1/3, 2/2, ("Maximum absolute error: $(aerr)", 8, :left))
            #(0/3, 1/2), ("Maximum relative error: ")
        ]
    )

    plt = plot(
        plt_bs_cv, plt_info,
        layout = grid(2, 1, heights = [0.95, 0.05]),
        size = (600, 400), dpi = 1000
    )

    return plt
end

function error_bspline_curve(bs::BSpline, x_cv, y_cv)
    error("not implemented")
    u_steps = bs.u_steps
    u_steps_distances = Float64[]
    aerr_maxs = Float64[]
    rerrs = Float64[]
    rerrs_max = Float64

    y_bs = bs.(x_cv)
    aerr_mean = sum(abs.(y_cv-y_bs))
    rerr_mean = aerr_mean/u_steps_distance
    aerr_max = max(abs.(y_cv-y_bs)...)
    rerr_max = aerr_max/u_steps_distance

    return (; aerr_max, aerr_mean, rerr_max, rerr_mean)
end

function where_to_add_u_step(bs::BSpline, x_cv, y_cv, atol, rtol, learning_ratio)
    weights = bs.weights
    u_steps = bs.u_steps
    n_weigths = length(weights)
    n_u_steps = length(u_steps)
    new_u_steps = Float64[]
    new_weights = Float64[]
    u_steps_min = u_steps[begin]; u_steps_max = u_steps[end]
    x_cv_min = min(x_cv...); x_cv_max = max(x_cv...)
    new_weights_priority = Float64[]
    new_u_steps_priority = Float64[]
    u_steps_distances = Float64[]
    aerr_maxs = Float64[]
    rerr_mean = Float64[]
    rerr_maxs = Float64[]

    # adapt if the spline is smaller than the domain
    alpha = 1
    if x_cv_min < u_steps_min
        new_u_step = (1+alpha)*x_cv_min - alpha*u_steps_min
        push!(new_u_steps_priority, new_u_step)
        push!(new_weights_priority, y_cv[begin])
    end

    if x_cv_max > u_steps_max
        new_u_step = (1+alpha)*x_cv_max - alpha*u_steps_max
        push!(new_u_steps_priority, new_u_step)
        push!(new_weights_priority, y_cv[end])
    end

    for j in 1:n_u_steps-1
        x_cv_in_j_idxs = findall(x -> u_steps[j] <= x < u_steps[j+1], x_cv)
        length(x_cv_in_j_idxs) == 0 && continue # interval so small does not contain any x :(
        x_cv_in_j = x_cv[x_cv_in_j_idxs]
        y_cv_in_j = y_cv[x_cv_in_j_idxs]
        y_bs_in_j = bs.(x_cv_in_j)
        u_steps_distance = u_steps[j+1] - u_steps[j]
        aerr_mean = sum(abs.(y_cv_in_j-y_bs_in_j))
        rerr_mean = aerr_mean/u_steps_distance
        aerr_max = max(abs.(y_cv_in_j-y_bs_in_j)...)
        rerr_max = aerr_max/u_steps_distance

        if rerr_max >= rtol || aerr_max >= rtol
            new_u_step = (u_steps[j]+u_steps[j+1])/2
            new_weight = bs(new_u_step)
            push!(new_u_steps, new_u_step)
            push!(new_weights, new_weight)
            push!(u_steps_distances, u_steps_distance)
            push!(aerr_maxs, aerr_max)
            push!(rerr_maxs, rerr_max)
        end
    end

    u_steps_distances_perm = sortperm(u_steps_distances, rev = true)
    aerr_maxs_perm = sortperm(aerr_maxs, rev = true)
    rerr_maxs_perm = sortperm(rerr_maxs, rev = true)
    n_new_weights = length(new_weights)
    n_new_weights_sample = round(Int, n_new_weights*learning_ratio)
    new_weights_sample = new_weights[rerr_maxs_perm][1:n_new_weights_sample]
    new_u_steps_sample = new_u_steps[rerr_maxs_perm][1:n_new_weights_sample]

    #priority
    append!(new_weights_sample, new_weights_priority)
    append!(new_u_steps_sample, new_u_steps_priority)

    @assert length(new_weights) == length(new_u_steps) "Lengths for new_weights and new_u_steps differ!"

    return new_weights_sample, new_u_steps_sample
end

function where_to_trim_u_step(bs::BSpline, x_cv, n_out)
    weights = bs.weights
    u_steps = bs.u_steps
    n_weigths = length(weights)
    k = bs.k
    n_u_steps = length(u_steps)
    x_cv_min = min(x_cv...); x_cv_max = max(x_cv...)
    first_u_step_inside_domain_index = findfirst(
        x -> x > x_cv_min, u_steps
    )
    last_u_step_inside_domain_index = findfirst(
        x -> x < x_cv_max, u_steps
    )

    # find the indexes of u_steps that don't influence the domain
    trim_u_steps_left = max(1, first_u_step_inside_domain_index-(k+1))
    trim_u_steps_right = min(first_u_step_inside_domain_index+k+1,n_u_steps)

    return trim_u_steps_left, trim_u_steps_right
end



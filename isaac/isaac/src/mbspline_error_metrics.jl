# Error calculation

## Squared error

function mean_squared_error_and_errors2(
    k, weightss, u_steps, curves, ns_points, n_curves
)

    nmses = mean_squared_errors2(
        k, weightss, u_steps, curves, ns_points
    )
    nmse = (1/n_curves) * sum(nmses)

    return nmse, nmses
end

function mean_squared_error_and_errors2(mbs::MBSpline, curves::ACurves) 
    k = mbs.k
    weightss = mbs.weightss
    u_steps = mbs.u_steps
    n_curves = length(curves)
    ns_points = length.(curves)

    return mean_squared_error_and_errors2(k, weightss, u_steps, curves, ns_points, n_curves)
end

function mean_squared_errors2(
    k, weightss, u_steps, curves::ACurves, ns_points
)
    nmses = [
        mean_squared_error_i2(
            k, weightss, u_steps, curves, i, ns_points
        )
        for i in eachindex(curves)
    ]

    return nmses
end

function mean_squared_error_i2(
    k, weightss, u_steps, curves::ACurves, i::Int, ns_points
)
    n_points = ns_points[i]
    curve = curves[i]
    difference = curve.y - calc_mbspline(k, weightss, u_steps, curve.x)[i]
    #@debug "normalized_mean_squared_error" curve.ylabel i y_min y_max n_points difference sum(difference) normalized_difference sum(normalized_difference) mse acc
    mse = (1/n_points) * sum(difference.^2)

    return mse
end

# # Absolute Error

# ## Maximum absolute error

function maximum_absolute_error_and_errors2(
    k, weightss, u_steps, curves
)

    mae_i = maximum_absolute_errors2(
        k, weightss, u_steps, curves,
    )
    mae = maximum(mae_i)

    return mae, mae_i
end

function maximum_absolute_error_and_errors2(mbs::MBSpline, curves::ACurves)
    k = mbs.k
    weightss = mbs.weightss
    u_steps = mbs.u_steps

    return maximum_absolute_error_and_errors2(k, weightss, u_steps, curves)
end

function maximum_absolute_errors2(k, weightss, u_steps, curves::ACurves)
    mae_i = [
        maximum_absolute_error_i2(k, weightss, u_steps, curves, i)
        for i in eachindex(curves)
    ]

    return mae_i
end

function maximum_absolute_errors2(mbs::MBSpline, curves::ACurves)
    k = mbs.k
    weightss = mbs.weightss
    u_steps = mbs.u_steps

    return maximum_absolute_errors2(k, weightss, u_steps, curves)
end

function maximum_absolute_error_i2(
    k, weightss, u_steps, curves::ACurves, i::Int
)
    curve = curves[i]
    difference = curve.y - calc_mbspline(k, weightss, u_steps, curve.x)[i]
    #@debug "normalized_mean_squared_error" curve.ylabel i y_min y_max n_points difference sum(difference) normalized_difference sum(normalized_difference) mse acc
    mae = maximum(abs, difference)

    return mae
end

function maximum_absolute_error_i2(mbs, curves, i)

    return maximum_absolute_error_i2(mbs.k. mbs.weightss, mbs.u_steps, curves, i)
end

# # Percentage Error

function maximum_percentage_error_and_errors2(
    k, weightss, u_steps, curves, ys_min, ys_max
)

    mpe_i = maximum_percentage_errors2(
        k, weightss, u_steps, curves, ys_min, ys_max
    )
    mpe_idx = findmax(abs, mpe_i)[2]
    mpe = mpe_i[mpe_idx]

    return mpe, mpe_i
end

function maximum_percentage_error_and_errors2(mbs::MBSpline, curves::ACurves)
    k = mbs.k
    weightss = mbs.weightss
    u_steps = mbs.u_steps
    _, _, ys_min, ys_max = extrema_curves(curves)

    return maximum_percentage_error_and_errors2(k, weightss, u_steps, curves, ys_min, ys_max)
end

function maximum_percentage_errors2(k, weightss, u_steps, curves::ACurves, ys_min, ys_max)
    mpe_i = [
        maximum_percentage_error_i2(k, weightss, u_steps, curves, i, ys_min, ys_max)
        for i in eachindex(curves)
    ]

    return mpe_i
end

function maximum_percentage_errors2(mbs::MBSpline, curves::ACurves)
    k = mbs.k
    weightss = mbs.weightss
    u_steps = mbs.u_steps
    _, _, ys_min, ys_max = extrema_curves(curves)

    return maximum_percentage_errors2(k, weightss, u_steps, curves, ys_min, ys_max)
end

function maximum_percentage_error_i2(k, weightss, u_steps, curves, i, ys_min, ys_max)
    curve = curves[i]
    y_min = ys_min[i]
    y_max = ys_max[i]
    pe = 100.0 * (curve.y - calc_mbspline(k, weightss, u_steps, curve.x)[i]) ./ (y_max - y_min) # curve.y
    #@debug "normalized_mean_squared_error" curve.ylabel i y_min y_max n_points difference sum(difference) normalized_difference sum(normalized_difference) mse acc
    mpe_idx = findmax(abs, pe)[2]
    mpe = pe[mpe_idx]

    return mpe

end

# ### Regional

function maximum_percentage_error_and_errors_reg2(
    k, weightss, u_steps, curves, idx_reg, ys_min, ys_max
)

    mpe_i = maximum_percentage_errors_reg2(
        k, weightss, u_steps, curves, idx_reg, ys_min, ys_max
    )
    mpe_idx = findmax(abs, mpe_i)[2]
    mpe = mpe_i[mpe_idx]

    return mpe, mpe_i
end

function maximum_percentage_error_and_errors_reg2(mbs::MBSpline, curves::ACurves, idx_reg)
    k = mbs.k
    weightss = mbs.weightss
    u_steps = mbs.u_steps
    _, _, ys_min, ys_max = extrema_curves(curves)

    return maximum_percentage_error_and_errors_reg2(
        k, weightss, u_steps, curves, idx_reg, ys_min, ys_max
    )
end

function maximum_percentage_errors_reg2(
    k, weightss, u_steps, curves::ACurves, idx_reg, ys_min, ys_max
)
    mpe_i = [
        maximum_percentage_error_i_reg2(
            k, weightss, u_steps, curves, i, idx_reg, ys_min, ys_max
        )
        for i in eachindex(curves)
    ]

    return mpe_i
end

function maximum_percentage_errors_reg2(mbs::MBSpline, curves::ACurves, idx_reg)
    k = mbs.k
    weightss = mbs.weightss
    u_steps = mbs.u_steps
    _, _, ys_min, ys_max = extrema_curves(curves)

    return maximum_percentage_errors_reg2(
        k, weightss, u_steps, curves, idx_reg, ys_min, ys_max
    )
end

function maximum_percentage_error_i_reg2(
    k, weightss, u_steps, curves, i, idx_reg, ys_min, ys_max
)
    curve = curves[i]
    y_min = minimum(curve.x)
    y_max = maximum(curve.y)

    # What is inside the named region
    idxs_x_reg = findall_idxs_x_inside_idx_reg(u_steps, idx_reg, curve)
    length(idxs_x_reg) == 0 && return NaN # here we go...
    curve_reg = curve[idxs_x_reg]

    pe = 100.0 * (curve_reg.y - calc_mbspline(k, weightss, u_steps, curve_reg.x)[i] ./ (y_max - y_min))

    mpe_idx = findmax(abs, pe)[2]
    mpe = pe[mpe_idx]

    return mpe
end

# 

function findall_idxs_x_inside_idx_reg(u_steps, idx_reg, curve)
    idxs_x_reg = findall(x -> u_steps[idx_reg] <= x < u_steps[idx_reg + 1], curve.x)

    return idxs_x_reg
end


struct MBSpline{T<:Number}
    k::Int
    u_steps::Vector{T}
    weightss::Vector{Vector{T}}

    function MBSpline(weightss, u_steps, k)
        ns_weights = length.(weightss) # a vector of lenghts
        n_u_steps = length(u_steps)
 
        @assert k >= 0 "k must be positive"
        @assert all(n_u_steps .== ns_weights .+ k .+ 1) "n_weights + k + 1 != n_u_steps"
        @assert sum(u_steps) !== NaN
        @assert !any(isnan.(u_steps)) # redundant, but it's still passing the previous check, so worth the try
        @assert all(sum.(weightss) .!== NaN)

        return new{eltype(weightss[1])}(k, sort(u_steps), weightss)
    end
end

(mbs::MBSpline)(u::Real)                               = calc_mbspline(mbs, u )
(mbs::MBSpline)(us::AbstractVector{T}) where T <: Real = calc_mbspline(mbs, us)

calc_mbspline(k::Int, weightss, u_steps, u::Real)= [
    b_spline_definition(weights, k, u_steps, u)
    for weights in weightss
]

calc_mbspline(mbs::MBSpline, u::Real) = [
    b_spline_definition(weights, mbs.k, mbs.u_steps, u)
    for weights in mbs.weightss
]

calc_mbspline(k::Int, weightss, u_steps, us::AbstractVector{T}) where T <: Real = [
    [
        b_spline_definition(weights, k, u_steps, u)
        for u in us
    ]
    for weights in weightss
]

calc_mbspline(mbs::MBSpline, us::AbstractVector{T}) where T <: Real = [
    [
        b_spline_definition(weights, mbs.k, mbs.u_steps, u)
        for u in us
    ]
    for weights in mbs.weightss
]

control_points_coordinates(mbs::MBSpline) = [
    control_points_coordinates(BSpline(weights, mbs.u_steps, mbs.k))
    for weights in mbs.weightss
]

function calc_mbspline2(mbs, u)

end

function curves_sample_u_steps(curves::ACurves, mbs::MBSpline)
    n_out = n_u_steps_outside_domain(mbs)/2
    u_steps = mbs.u_steps

    return curves_sample_u_steps(curves, u_steps, n_out)
end

number_u_steps_outside_domain(mbs::MBSpline) = 2*(mbs.k + 1)

function number_u_steps_inside_domain(mbs::MBSpline)
    
    return length(mbs.u_steps) - number_u_steps_outside_domain(mbs) 
end

function min_curves_points_inside_u_step(mbs::MBSpline, curves::ACurves)
    n_out = Int(number_u_steps_outside_domain(mbs)/2)
    u_steps = mbs.u_steps

    return min_curves_points_inside_u_step(curves, u_steps, n_out)
end


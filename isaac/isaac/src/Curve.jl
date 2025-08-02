struct Curve <: AbstractVector{Float64}
    x::Vector{Float64}
    y::Vector{Float64}
    xlabel::String
    ylabel::String

    function Curve(
        x     ::AbstractVector, y     ::AbstractVector,
        xlabel::AbstractString, ylabel::AbstractString,
    )
        @assert length(x) == length(y)

        return new(x, y, xlabel, ylabel)
    end

    #Curve(x::Real, y::Real, xlabel, ylabel) = Curve([x], [y], xlabel, ylabel)
end

Base.size(curve::Curve) = (length(curve.x),)
Base.getindex(curve::Curve, i) = Curve(
    curve.x[i],
    curve.y[i],
    curve.xlabel,
    curve.ylabel,
)

#Base.getindex(curve::Curve, i) = (x = curve.x[i], y = curve.y[i])

Base.IndexStyle(::Type{<:Curve}) = IndexLinear()

function Base.show(io::IO, curve::Curve)

    return print(io, curve.x, ", ", curve.y, ", ", curve.xlabel, ", ", curve.ylabel)
end

ACurves = AbstractVector{Curve}

function curve_sample_random(curve::Curve, n_points_sample::Int)
    idxs = rand(CartesianIndices(curve), n_points_sample)

    return Curve(curve.x[idxs], curve.y[idxs], curve.xlabel, curve.ylabel)
end

function curve_sample_random_idxs(curve::Curve, n_points_sample::Int)
    idxs = rand(CartesianIndices(curve), n_points_sample)

    return idxs
end

function curves_sample_random_n_points_idxss(curves::ACurves, n_points_sample::Int)
    curves_sample_idxss = [
        curve_sample_random_idxs(curve, n_points_sample)
        for curve in curves
    ]

    return curves_sample_idxss
end

function curves_sample_random(curves::ACurves, alpha_n::Union{Real, AbstractVector{Real}})
    @assert all(0 .< alpha_n .<= 1)
    @assert length(alpha_n) == 1 || length(alpha_n) == length(curves)

    if alpha_n == 1
        return curves
    end

    ns_points = length.(curves)
    ns_points_sample = alpha_n .* ns_points .|> x -> Int(round(x, RoundUp))
    curves_sample = [
        curve_sample_random(curve, n_points_sample)
        for (curve, n_points_sample) in zip(curves, ns_points_sample)
    ]

    return curves_sample
end

function curves_sample_u_steps_min_idxss(
    curves::ACurves, u_steps::Vector, n_out::Int, n_points_per_u_step_max,
    clamp_n_points_u_steps
)
    n_points_u_step_min = min_curves_points_inside_u_step(curves, u_steps, n_out)
    n_points_per_u_step = begin
        if clamp_n_points_u_steps
            min(n_points_u_step_min, n_points_per_u_step_max)
        else
            n_points_per_u_step_max 
        end
    end

    return curves_sample_u_steps_idxss(curves, u_steps, n_points_per_u_step, n_out)
end

function curves_sample_u_steps_idxss(
    curves::ACurves, u_steps::Vector, n_points_u_step, n_out::Int,
)
    
    n_u_steps = length(u_steps)
    @assert n_points_u_step > 0 # this shouldn't happen, i guess
    curves_sample_idxss = Vector{Int}[]

    for (i, curve) in enumerate(curves)
        idxs_curve_sample = Int[]
        for i in n_out:(n_u_steps-n_out)
            idxs_curve_inside = findall(x -> u_steps[i] <= x < u_steps[i+1], curve.x)
            idxs_curve_inside_sample = shuffle(rng, idxs_curve_inside)
            append!(
                idxs_curve_sample,
                collect(Iterators.take(
                    Iterators.cycle(idxs_curve_inside_sample),
                    n_points_u_step
                ))
            )
        end
        push!(curves_sample_idxss, idxs_curve_sample)
    end

    return curves_sample_idxss

    # My aim with this function is to give every weight an equal importance. I think that currently I'm going into the issue that the curves are very densy at the edges of the domain, so the weights that govern this points contibute a lot to the loss function, even thought it's normalized at the end.
end

#=
function curves_sample_u_steps(
    curves::ACurves, u_steps::Vector, n_out::Int, n_points_u_step
)::ACurves

    idxs_curves_sample = curves_sample_u_steps_idxs(curves, u_steps, n_points_u_step, n_out)
    curves_sample = similar(curves)
    for (i, (curve, idxs_curve_sample)) in enumerate(curves, idxs_curves_sample)
        curves_sample[i] = Curve(
            curve.x[idxs_curve_sample],
            curve.y[idxs_curve_sample],
            curve.xlabel, curve.ylabel
        )
    end

    return curves_sample
end
=#

function curves_sample_u_steps(curves, mbs)
    k = mbs.k
    u_steps = mbs.u_steps
    n_out = k + 1

    return curves_sample_u_steps(curves, u_steps)
end

function min_curve_points_inside_u_step(curve::Curve, u_steps::AbstractVector, n_out::Int)
    n_u_steps = length(u_steps)
    ns_points_u_step = [
        begin
            #@show i u_steps[i] u_steps[i+1] maximum(curve.x) minimum(curve.x) length(u_steps) count(x -> u_steps[i] <= x < u_steps[i+1], curve.x)
            count(x -> u_steps[i] <= x < u_steps[i+1], curve.x)
        end
        for i in n_out:(n_u_steps-n_out)
    ]
    n_points_u_step_min = minimum(ns_points_u_step)

    #!(n_points_u_step_min > 0) && @warn "u_steps has regions without any points" curve.ylabel

    return n_points_u_step_min
end

function min_curves_points_inside_u_step(
    curves::ACurves, u_steps::AbstractVector, n_out::Int
)
    n_points_u_step_min = minimum([
        min_curve_points_inside_u_step(curve, u_steps, n_out)
        for curve in curves
    ])

    return n_points_u_step_min
end

function curves_sample_by_idxss(curves::ACurves, idxss::AbstractVector)
    @assert length(curves) == length(idxss)
    curves_sample = similar(curves)
    for i in eachindex(curves)
        curve = curves[i]
        idxs = idxss[i]
        curve_sample = curve[idxs]
        curves_sample[i] = curve_sample
    end

    return curves_sample
end

x_labels(curves::ACurves) = [curve.xlabel for curve in curves]

y_labels(curves::ACurves) = [curve.ylabel for curve in curves]

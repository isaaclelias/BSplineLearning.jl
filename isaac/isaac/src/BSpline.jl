"""
    BSpline

aqui eu devo pontuar o fato de usar as permutações pra organizar os elementos deu um grau de liberdade a mais pro solver pra poder brincar com os pontos como quiser. o solver não precisa nem ter contato com o fato de que no final os números são sorteados.
"""
struct BSpline #<: AbstractArray{Float64, 1}
    k::Int
    u_steps::Vector{Float64}
    weights::Vector{Float64}

    function BSpline(weights::AbstractArray, u_steps::AbstractArray, k::Int)
        if !(length(u_steps) == length(weights)+k+1)
            error("length(u_steps) !== length(weights)+k+1")
        else
            perm = sortperm(u_steps)
            return new(k, u_steps[perm], weights)#[perm]) subtrair a influência do ponto de um lado e somar do outro pra ter os pesos sortidos, já que cada u_step é influenciado por vários pesos
        end
    end

end


#=
Base.size(bs::BSpline) = length(bs.k) + length(bs.u_steps) + length(bs.weights)
Base.IndexStyle(::Type{BSpline}) = IndexLinear()

function Base.getindex(bs::BSpline, i::Int)
    if i < 1 || i > size(bs); throw(BoundsError(bs, i))
    elseif i == 1; return float(bs.k)
    elseif i <= length(bs.u_steps) + length(bs.k)
        return bs.u_steps[length(bs.k)+i]
    elseif i <= length(bs.k) + length(bs.u_steps) + length(bs.weights)
        return bs.weights[length(bs.k)+length(bs.u_steps)+i]
    end
end
=#

function BSpline(
    n_weights::Int, k::Int,
    x_lim::Tuple{Real, Real}, w_lim::Tuple{Real, Real}
)
        
    x_min = x_lim[1]; x_max = x_lim[2]
    w_min = w_lim[1]; w_max = w_lim[2]

    return BSpline(
        rand(LinRange(w_min, w_max, 1000), n_weights),
        LinRange(x_min, x_max, n_weights+k+1),
        k
    )
end

function (b::BSpline)(u)
    return b_spline_definition(b.weights, b.k, b.u_steps, u)
end

@inbounds function basis_function_definition(i, k, u_steps, u)
    us = u_steps
    bf = basis_function_definition

    if k == 0
        if us[i] <= u < us[i+1]; return 1.0
        else                   ; return 0.0
        end
    else
          return ((u-us[i])    /(us[i+k]  -us[i]))   * bf(i  , k-1, us, u) +
                 ((us[i+k+1]-u)/(us[i+k+1]-us[i+1])) * bf(i+1, k-1, us, u)
    end
end

function b_spline_definition(weights, k, u_steps, u)
    return sum([
        weights[i] * basis_function_definition(i, k, u_steps, u)
        for i in eachindex(weights)
    ])
    #=for (i, weight) in enumerate(weights)
        ret = ret + weight*basis_function_definition(i, k, u_steps, u)
    end=#

    #return ret
end

b_spline_definition(bs::BSpline, u) = b_spline_definition(bs.weights, bs.k, bs.u_steps, u)

function b_spline_coxdeboor(weights, k, u_steps, u)

    #=
    """Evaluates S(x).

    Arguments
    ---------
    i,k: Index of knot interval that contains x.
    u,x: Position.
    u_steps,t: Array of knot positions, needs to be padded as described above.
    weights,c: Array of control points.
    k,p: Degree of B-spline.
    """
    d = [c[j + k - p] for j in range(0, p + 1)] 

    for r in range(1, p + 1):
        for j in range(p, r - 1, -1):
            alpha = (x - t[j + k - p]) / (t[j + 1 + k - r] - t[j + k - p]) 
            d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]

    return d[p]
    =#

    #=
    d_buf = Zygote.Buffer(zeros(a), )
    d_buf = [weights[j+i-p] for j in range(0, p + 1)] 

    for r in 1:(k+1), j in k:(-1):(r-1)
        alpha = (u - u_steps[j+i-k]) / (u_steps[j+1+i-r] - u_steps[j+i-k])
        d_buf[j] = (1.0 - alpha)*d[j-1] + alpha*d[j]
    end
    
    return copy(d[k])
    =#

        
    #=
    dj
    d1 = [weights[j+i-k] for j in 1:(k+2)]
    alpha = [
        [
            (u - u_steps[j+i-k]) / (u_steps[j+1+i-r] - u_steps[j+i-k])
            for j in k:-1:r-1
        ]
        for r in 2:(k+1)
    ]
    d = [
        [
            [
                (one(u) - alpha[j, r] *)
                for
            ]
            for j in k:-1:r-1
        ]
        for r in 2:(k+1)
    ]

    d = for r in 2:(k+1),j in k:-1:r-1
        alpha = (u - u_steps[j+i-k]) / (u_steps[j+1+i-r] - u_steps[j+i-k])
        d = [(one(u)-alpha)*d[n]+ ]
    return d[k]
    =#
end

b_spline_coxdeboor(bs::BSpline, u) = b_spline_coxdeboor(bs.weights, bs.k, bs.u_steps, u)

BSplineControlPointCoordinates = Tuple{Vector{Float64}, Float64}

function control_point_coordinates(
    bs::BSpline,
    i::Int
)::BSplineControlPointCoordinates

    k     = bs.k
    y_cp  = bs.weights[i]
    xs_cp = [bs.u_steps[j] for j in i:1:(i+k+1)]

    return (xs_cp, y_cp)
end

function control_points_coordinates(
    bs::BSpline
)::Vector{BSplineControlPointCoordinates}

    ret = Vector{BSplineControlPointCoordinates}(undef, length(bs.weights))
    for (i, weight) in enumerate(bs.weights)
      ret[i] = control_point_coordinates(bs, i)
    end

    return ret
end

function trainable_parameters(bs::BSpline)
    tp = ComponentVector{Float64}((weights = bs.weights, u_steps = bs.u_steps))

    return tp
end

bspline_trainable_parameters_length(n_weights, k) = 2*n_weights + k + 1

struct BSpline2{K}
    weights::SVector
    u_steps::SVector

    function BSpline2{K}(weights::AbstractArray, u_steps::AbstractArray) where K
        n_u_steps = length(u_steps)
        n_weights = length(weights)
        @assert n_u_steps == n_weights+K+1
        perm = sortperm(u_steps)

        return new{K}(
            SVector{n_weights}(weights),
            SVector{n_u_steps}(u_steps[perm]),
        )
    end
end

(bs2::BSpline2)(u) = calc_bspline_gen(bs2, u)

# Lets get fancy
@generated function basis_function_definition2(i, ::Val{K}, us, u) where K
    if K == 0
        #return :(u_steps[I] <= u < u_steps[I+1] ? one(u) : zero(u))
        return :(us[i] <= u < us[i+1] ? 1 : 0) #:(one(u))
    else
        return quote
            ((u-us[i])/(us[i+K]-us[i]))                         * 
            basis_function_definition2(i, Val(K-1), us, u)   +
            ((us[i+K+1]-u)/(us[i+K+1]-us[i+1]))              *
            basis_function_definition2(i+1, Val(K-1), us, u)
        end
    end
end

function b_spline_definition2(weights, ::Val{K}, u_steps, u) where K
    is = 1:K+1
    #@show weights basis_function_definition2.(is, Val(K), u_steps, u)
    ret = sum([
        weights[i] * basis_function_definition2(i, Val(K), u_steps, u)
        for i in 1:K+1
    ])

    return ret
end

function weights_local(bs2::BSpline2{K}, u) where K
    u_steps = bs2.u_steps
    weights = bs2.weights

    u_step_u_left_idx  = findlast(x -> x < u, u_steps)
    u_step_u_right_idx = u_step_u_left_idx + 1
    u_step_u_begin_idx = u_step_u_left_idx - K + 1
    u_step_u_end_idx   = u_step_u_right_idx + K + 1
    weights_u_begin_idx = u_step_u_begin_idx
    weights_u_end_idx = u_step_u_left_idx + 1

    u_steps_u = u_steps[u_step_u_begin_idx:u_step_u_end_idx]
    weights_u = weights[weights_u_begin_idx:weights_u_end_idx]

    return weights_u, u_steps_u
end

function calc_bspline_gen(bs2::BSpline2{K}, u) where K
    weights_u, u_steps_u = weights_local(bs2, u)
    return b_spline_definition2(weights_u, Val(K), u_steps_u, u)
end

n_u_steps_at_domain_border(k) = k + 1
n_weights_that_influence_given_point(k) = k + 1


"""
    NMBSpline{T<:Number}

    
"""
struct NMBSpline{T<:Number}
    degree  ::Int
    nodess  ::Vector{Vector{T}} # I swear this notation makes sense to me
    weightss::Vector{Array{T}}

    function NMBSpline(degree, nodess, weightss)
        # basic info
        n_inputs  = length(nodess)
        n_outputs = length(weightss)
        size_weights = collect(size(weightss[1])) # NOT YET VERIFIED
        size_nodess = length.(nodess)
        size_weightss = collect.(size.(weightss))

        # isn't better to be safe than sorry? ~take on meee~
        # TASK throw proper exceptions instead of just AssertionError
        @assert degree >= 0 "Degree must be strictly positive (greater or equal to zero)"
        @assert all(all.([size_weightss_ .== size_weights_ for (size_weights_, size_weightss_) in zip(size_weights, size_weightss)])) "All weights arrays shall have the same size" # if it works, it works
        @show size_weights size_nodess size_weightss
        @assert all(size_weights .+ (degree+1) .== size_nodess) L"Every nodes Vector inside the Vector (yes, a Vector{Vector}) of nodess shall have $I_n + K + 1$ nodes"
        @assert length(nodess) == length(size_weights) "The number of node vectors shall equal the order of the weights tensor"
        @assert !any([any(isnan.(nodes)) for nodes in nodess]) "There shall be no NaN node"
        @assert !any([any(isnan.(weights)) for weights in weightss]) "There shall be no NaN weight"

        return new{typeof(weightss[1][1])}(degree, nodess, weightss)
    end
end

n_inputs(nmbs::NMBSpline)  = length(nmbs.nodess)
n_outputs(nmbs::NMBSpline) = length(nmbs.weightss)
size_weights(nmbs::NMBSpline) = collect(nmbs.size(weights[1]))
size_nodes(nmbs::NMBSpline)    = length.(nmbs.nodess)
sizes_weights(nmbs::NMBSpline) = collect.(size.(nmbs.weightss))

function (nmbs::NMBSpline)(u::AbstractVector)
    
    return calc_nmbs_naive(nmbs.degree, nmbs.nodess, nmbs.weightss, u)
end

#(nmbs::NMBSpline)(u...) = nmbs(collect(u))

function calc_nmbs_naive(degree, nodess, weightss, u)
    n_inputs = length(nodess)
    return [
        sum([
            weights[idx] * prod([
                basis_function_definition(idx[n], degree, nodess[n], u[n])
                for n in 1:n_inputs
            ])
            for idx in CartesianIndices(weights)
        ])
        for weights in weightss
    ]
end

calc_nmbs_naive(nmbs::NMBSpline, u) = calc_nmbs_naive(nmbs.degree, nmbs.nodess, nmbs.weightss, u)

function domain_start_node_idxs(nmbs::NMBSpline)
    degree = nmbs.degree
    _n_inputs = n_inputs(nmbs)
    idx = degree + 1

    return fill(idx, _n_inputs)
end

function domain_end_node_idxs(nmbs::NMBSpline)
    degree = nmbs.degree
    _n_inputs = n_inputs(nmbs)
    _size_nodess = size_nodess(nmbs)
    idxs = [size_nodes-degree for size_nodes in size_nodess]
end

function domain_start_nodes(nmbs::NMBSpline)
    nodess = nmbs.nodess
    idxs = domain_start_node_idxs(nmbs)

    return [nodes[idxs[n]] for (n, nodes) in enumerate(nodess)]
end

function domain_end_nodes(nmbs::NMBSpline)
    nodess = nmbs.nodess
    idxs = domain_end_node_idxs

    return [nodess[idxs[n]] for  (n, nodes) in enumerate(nodess)]
end

domain_start_region_idxs(nmbs) = domain_start_node_idxs(nmbs)

domain_end_region_idxs(nmbs) = domain_end_node_idxs(nmbs) .- 1

#=
function calc_nmbs_cox(degree, nodes, weightss, u)
    
    sum([
         weights[idx] * prod([
            
            for n in 1:N
        ])
        for idx in CartesianIndices(weights)
    ])
end

function region_that_contains_u(u, nodes)

end
=#

###########################################################################################
#
# An enemy I don't want to fight
#
###########################################################################################

struct SNMBSpline{N, M, I, T}
    #degree::Int
    #nodes::SVector{SVector}
    #weightss
end

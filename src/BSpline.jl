abstract type AbstractBSpline

BSpline = BSpline1

"""
one thesis style
"""
struct BSpline1 <: type
  
end

(bs::BSpline1)(u::AbstractVector) = calc_bspline1_1()

""""
most naive
"""
function calc_bspline1_1()
    
end

"""
the one cutting off the null basis functions
"""
function calc_bspline1_2(arguments)
  
end

struct BSpline2{T, K, N, M} <: AbstractBSpline
    weights::SVector{}
    grid   ::SVector{}

    function BSpline1()

    end
end

weights(bs::BSpline2) = bs.weights
grid(bs::BSpline2) = bs.grid
order(bs::BSpline2{K}) where K = K 

(bs::BSpline2)(u::AbstractVector) = calc_bspline2_1()

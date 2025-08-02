using MasterThesis

x_cv_min = 0
x_cv_max = 6*pi
x_cv = collect(LinRange(x_cv_min, x_cv_max, 100))
#y_cv = @. exp10(-x_cv) * cos.(exp10(x_cv)) + 1
y_cv = @. cos(x_cv)
y_cv_min = min(y_cv...)
y_cv_max = max(y_cv...)

alphas = (;
    MSE = 1,
    distance = 0,
    span = 0,
    boundary = 0
)

n_weights = 20
k = 2

bs0 = BSpline(n_weights, k, (1*pi, 5*pi), (y_cv_min, y_cv_max))

solve_bspline_curve(
    bs0, x_cv, y_cv, alphas, 
    maxiters = 300, atol = 10^-3,
    title = "cossine with weights just in the center more spreaded w$(n_weights) k$(k)"
)

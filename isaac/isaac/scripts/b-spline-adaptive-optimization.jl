using MasterThesis

x_cv_min = 0
x_cv_max = 2
x_cv = collect(LinRange(x_cv_min, x_cv_max, 500))
y_cv = @. exp10(-x_cv) * cos(exp10(x_cv)) + 1
y_cv_min = min(y_cv...)
y_cv_max = max(y_cv...)

alphas = (;
    MSE = 1,
    proximity = 10^-1
)

k = 6
atol = 0.001
rtol = 0.1
lratio = 0.5

solve_adaptive_bspline_curve(
    x_cv, y_cv, k, alphas,
    maxiters_inside = 5, maxiters_adaptive = 30, atol = atol, rtol = rtol,
    learning_ratio = lratio,
    title = "messing aroung with k$(k)"
)

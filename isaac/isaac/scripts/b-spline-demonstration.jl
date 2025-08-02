using MasterThesis
using Plots

weights = [1,-1,2,-2]
u_steps = [1,2,4,5,6,7,10,11]
k = 3
bs = BSpline(weights, u_steps, k)

x = 1:0.01:10
y = bs.(x)

plt = plot(x, y, title = "B-Spline demonstration")

savefig_with_date(plt, "b-spline-demonstration")

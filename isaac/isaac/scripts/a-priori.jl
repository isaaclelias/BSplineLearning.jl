#=
# This file contains the code for plotting the a priori analysis
=#

using MasterThesis
using Plots
using DataFrames
using Serialization
using Logging
using LoggingExtras
using Dates
using LaTeXStrings
using Printf
using Measures

# logging setup
log_dir = "../logs/"; mkpath(log_dir)
log_title = basename(@__FILE__) |> x -> replace(x, ".jl" => "")
log_format_date = Dates.format(now(), "yyyy-mm-dd_HH-MM")
log_path = log_dir * log_format_date * " " * log_title * ".log"
log_info_path  = log_path * ".info" ; touch(log_info_path)
log_debug_path = log_path * ".debug"; touch(log_debug_path)
const logger = TeeLogger(
    ConsoleLogger(stdout                    , Logging.Info ), # Console
    SimpleLogger( open(log_info_path , "w+"), Logging.Info ), # File for info
#    SimpleLogger( open(log_debug_path, "w+"), Logging.Debug), # File for debug
); global_logger(logger)
@info "Logging" log_info_path log_debug_path

# stuff to save the study results
_title = "k4n15hr"
filename = basename(@__FILE__) |> x -> replace(x, ".jl" => "")
title = MasterThesis.format_date_githash_title_now(_title)
path_save = "../results/" * filename * "/"*title*"/"; mkpath(path_save)

df = df_hoq_training_data_1d()

# Quantities

X = df[:, "X"]*10^3
T = df[:, "T"]
PV = df[:, "PV"]
rho = df[:, "rho"]
visc = df[:, "visc"]
lambda = df[:,"lambda"]
cpMean = df[:, "cpMean"]
tdot_OF = df[:, "tdot_OF"]
omega_yc = df[:, "omega_yc"]
#Teq_Term3_OF = df[:, "Teq_Term3_OF"]

## Min and max
X_min = minimum(X)
X_max = maximum(X)
X_fac = 100/(X_max - X_min)
T_min = minimum(T)
T_max = maximum(T)
T_fac = 100/(T_max - T_min)
PV_min = minimum(PV)
PV_max = maximum(PV)
PV_fac = 100/(PV_max - PV_min)
rho_min = minimum(rho)
rho_max = maximum(rho)
rho_fac = 100/(rho_max - rho_min)
visc_min = minimum(visc)
visc_max = maximum(visc)
visc_fac = 100/(visc_max - visc_min)
lambda_min = minimum(lambda)
lambda_max = maximum(lambda)
lambda_fac = 100/(lambda_max - lambda_min)
cpMean_min = minimum(cpMean)
cpMean_max = maximum(cpMean)
cpMean_fac = 100/(cpMean_max - cpMean_min)
tdot_OF_min = minimum(tdot_OF)
tdot_OF_max = maximum(tdot_OF)
tdot_OF_fac = 100/(tdot_OF_max - tdot_OF_min)
omega_yc_min = minimum(omega_yc)
omega_yc_max = maximum(omega_yc)
omega_yc_fac = 100/(omega_yc_max - omega_yc_min)

## Permutations
perm_PV = sortperm(PV)

# Labels

#label_T = L"T \ [\mathrm{K}]"
label_X = L"x \ [\mathrm{mm}]"
label_PV = L"PV \ [-]"
label_rho = L"\rho \ [\mathrm{kg}/\mathrm{m}^{3}]"
label_visc = L"\mu \ [\mathrm{Pa} \cdot \mathrm{s}]"
label_lambda = L"\lambda \ [\mathrm{W} / (\mathrm{m} \cdot \mathrm{K})]"
label_cpMean = L"c_p \ [\mathrm{J} /( \mathrm{kg} \cdot \mathrm{K})]"
label_tdot_OF = L"\dot{\omega}'_{T} \ [\mathrm{J} / (\mathrm{m}^{3} \cdot \mathrm{kg} \cdot \mathrm{s})]"
label_omega_yc = L"\dot{\omega}_{Y_{PV}} \ [\mathrm{kg} /( \mathrm{m}^{3} \cdot \mathrm{s})]"
#label_Teq_Term3_OF = L"D_{diff,s} \ [\text{kg} \cdot \text{s}^{-3}]"
label_RE = L"RE \ [\%]"
label_PDF = L"PDF \ [-]"

## For the relative errors
#label_T = L"T \ [\mathrm{K}]"
label_rho_err = L"\rho \ RE \ [\%]"
label_visc_err = L"\mu \ RE \ [\%]"
label_lambda_err = L"\lambda \ RE [\%]"
label_cpMean_err = L"c_p \ RE \ [\%]"
label_tdot_OF_err = L"\dot{\omega}'_{T} \ RE \ [\%]"
label_omega_yc_err = L"\dot{\omega}_{Y_{PV}} \ RE \ [\%]"
#label_Teq_Term3_OF = L"D_{diff,s} \ [\text{kg} \cdot \text{s}^{-3}]"
label_RE = L"RE \ [\%]"
label_PDF = L"PDF \ [-]"


# MBSpline prediction
mbs_25 = deserialize("../results/solve_mbspline_curve2/highlights/2025-03-11_19-47 5360c9bf_mod study-on-h-refinement mwhich-error mwhere-first_order_moment mhh-spline_prediction mi-2999 k-4 nusa0-2 rr-false hr-true maxiters-2999 npusm-100 a_opt-0.025 nir-200 nppusm-100/mbs.jls")
mbs_25_pred = mbs_25(PV)
rho_mbs_25 = mbs_25_pred[1]
lambda_mbs_25 = mbs_25_pred[2]
cpMean_mbs_25 = mbs_25_pred[3]
visc_mbs_25 = mbs_25_pred[4]
omega_yc_mbs_25 = mbs_25_pred[5]
tdot_OF_mbs_25 = mbs_25_pred[6]

@info "" issorted(X) issorted(PV)

# Flame structure on physical space

## Constants
xlims = (4, 6)
size = (400, 250)
dpi = 300
linewidth = 3
heights = [0.7, 0.3] 
nyticks_err = 4
color_RE = palette(:default)[4]

## Density
plt_rho_ = plot(xlabel = label_X, ylabel = label_rho)#, title = "Density")
plot!(plt_rho_, X, rho, label = "Simulated", linewidth = linewidth)
plot!(plt_rho_, X, rho_mbs_25, label = "Prediction", linewidth = linewidth)
plot!(plt_rho_; xlims)
plt_rho_err = plot(
    X, -(rho - rho_mbs_25)*rho_fac, ylabel = label_RE, label = "", xlims = xlims,
    linewidth = linewidth, color = color_RE
)
plot!(plt_rho_err, yticks = LinRange(yticks(plt_rho_err[1])[1][begin], yticks(plt_rho_err[1])[1][end], nyticks_err), yformatter = x -> (@sprintf "%.2f" x))
plt_rho = plot(
    plt_rho_, plt_rho_err, layout = grid(2, 1, heights = heights),
    size = size, dpi = dpi
)
savefig(plt_rho, path_save*"rho.png")
savefig(plt_rho, path_save*"rho.pdf")

## Heat conductivity
plt_lambda_ = plot(xlabel = label_X, ylabel = label_lambda)#, title = "Thermal conductivity")
plot!(plt_lambda_, X, lambda, label = "", linewidth = linewidth)
plot!(plt_lambda_, X, lambda_mbs_25, label = "", linewidth = linewidth)
plot!(plt_lambda_; xlims)
plt_lambda_err = plot(
    X, -(lambda - lambda_mbs_25)*lambda_fac, label = "", ylabel = label_RE, linewidth = linewidth, xlims = xlims, color = color_RE
)
plot!(plt_lambda_err, yticks = LinRange(yticks(plt_lambda_err[1])[1][begin], yticks(plt_lambda_err[1])[1][end], nyticks_err), yformatter = x -> (@sprintf "%.2f" x))
plt_lambda = plot(
    plt_lambda_, plt_lambda_err, size = size, dpi = dpi,
    layout = grid(2, 1, heights = heights)
)
savefig(plt_lambda, path_save*"lambda.png")
savefig(plt_lambda, path_save*"lambda.pdf")

## Specific heat capacity
plt_cpMean_ = plot(xlabel = label_X, ylabel = label_cpMean)#, title = "Specific heat capacity")
plot!(plt_cpMean_, X, cpMean, label = "", linewidth = linewidth)
plot!(plt_cpMean_, X, cpMean_mbs_25, label = "", linewidth = linewidth)
plot!(plt_cpMean_; xlims)
plt_cpMean_err = plot(X, -(cpMean - cpMean_mbs_25)*cpMean_fac, label = "", ylabel = label_RE, linewidth = linewidth, xlims = xlims, color = color_RE)
plot!(plt_cpMean_err, yticks = LinRange(yticks(plt_cpMean_err[1])[1][begin], yticks(plt_cpMean_err[1])[1][end], nyticks_err), yformatter = x -> (@sprintf "%.2f" x))
plt_cpMean = plot(plt_cpMean_, plt_cpMean_err, size = size, dpi = dpi, layout = grid(2, 1, heights = heights))
savefig(plt_cpMean, path_save*"cpMean.png")
savefig(plt_cpMean, path_save*"cpMean.pdf")

## Dynamic viscosity
plt_visc_ = plot(xlabel = label_X, ylabel = label_visc)#, title = "Viscosity")
plot!(plt_visc_, X, visc, label = "", linewidth = linewidth)
plot!(plt_visc_, X, visc_mbs_25, label = "", linewidth = linewidth)
plot!(plt_visc_; xlims)
plt_visc_err = plot(X, -(visc - visc_mbs_25)*visc_fac, label = "", ylabel = label_RE, linewidth = linewidth, xlims = xlims, color = color_RE)
plot!(plt_visc_err, yticks = LinRange(yticks(plt_visc_err[1])[1][begin], yticks(plt_visc_err[1])[1][end], nyticks_err), yformatter = x -> (@sprintf "%.2f" x))
plt_visc = plot(plt_visc_, plt_visc_err, dpi = dpi, size = size, layout = grid(2, 1, heights = heights))
savefig(plt_visc, path_save*"visc.png")
savefig(plt_visc, path_save*"visc.pdf")

## Progress variable source term
plt_omega_yc_ = plot(xlabel = label_X, ylabel = label_omega_yc)# title = "Progress variable source term")
plot!(plt_omega_yc_, X, omega_yc, label = "", linewidth = linewidth)
plot!(plt_omega_yc_, X, omega_yc_mbs_25, label = "", linewidth = linewidth)
plot!(plt_omega_yc_; xlims)
plt_omega_yc_err = plot(X, -(omega_yc - omega_yc_mbs_25)*omega_yc_fac, linewidth = linewidth, xlims = xlims, label = "", ylabel = label_RE, color = color_RE)
plot!(plt_omega_yc_err, yticks = LinRange(yticks(plt_omega_yc_err[1])[1][begin], yticks(plt_omega_yc_err[1])[1][end], nyticks_err), yformatter = x -> (@sprintf "%.2f" x))
plt_omega_yc = plot(plt_omega_yc_, plt_omega_yc_err, size = size, dpi = dpi, layout = grid(2, 1, heights = heights))
savefig(plt_omega_yc, path_save*"omega_yc.png")
savefig(plt_omega_yc, path_save*"omega_yc.pdf")

## Temperature source term
plt_tdot_OF_ = plot(xlabel = label_X, ylabel = label_tdot_OF)#, title = "Temperature source term")
plot!(plt_tdot_OF_, X, tdot_OF, label = "", linewidth = linewidth)
plot!(plt_tdot_OF_, X, tdot_OF_mbs_25, label = "", ylabel = label_tdot_OF, linewidth = linewidth)
plot!(plt_tdot_OF_; xlims)
plt_tdot_OF_err = plot(X, -(tdot_OF- tdot_OF_mbs_25)*tdot_OF_fac, linewidth = linewidth, xlims = xlims, label = "", ylabel = label_RE, color = color_RE)
plot!(plt_tdot_OF_err, yticks = LinRange(yticks(plt_tdot_OF_err[1])[1][begin], yticks(plt_tdot_OF_err[1])[1][end], nyticks_err), yformatter = x -> (@sprintf "%.2f" x))
plt_tdot_OF = plot(plt_tdot_OF_, plt_tdot_OF_err, size = size, dpi = dpi, layout = grid(2, 1, heights = heights))
savefig(plt_tdot_OF, path_save*"tdot_OF.png")
savefig(plt_tdot_OF, path_save*"tdot_OF.pdf")

# Comparison of data density and error in PV space

## Constants
xlims2 = (4, 6)
size2 = (400, 250)
heights2 = [0.7, 0.3] 
nyticks_err = 4
color_RE = palette(:default)[4]
#bins2 = range(PV_min, PV_max, length=20)
bins2 = range(PV_min, PV_max, length=20)
nyticks2_hist = 8

#=
## Temperature source term
plt2_tdot_OF_ = plot(xlabel = label_PV, ylabel = label_tdot_OF)#, title = "Temperature source term")
plot!(plt2_tdot_OF_, X, tdot_OF, label = "", linewidth = linewidth)
plot!(plt2_tdot_OF_, X, tdot_OF_mbs_25, label = "", ylabel = label_tdot_OF, linewidth = linewidth)
plot!(plt2_tdot_OF_; xlims)
plt2_tdot_OF_err = plot(X, (tdot_OF- tdot_OF_mbs_25)*tdot_OF_fac, linewidth = linewidth, xlims = xlims, label = "", ylabel = label_RE, color = color_RE)
plot!(plt_tdot_OF_err, yticks = LinRange(yticks(plt_tdot_OF_err[1])[1][begin], yticks(plt_tdot_OF_err[1])[1][end], nyticks_err), yformatter = x -> (@sprintf "%.2f" x))
plt_tdot_OF = plot(plt_tdot_OF_, plt_tdot_OF_err, size = size, dpi = dpi, layout = grid(2, 1, heights = heights))
savefig(plt_tdot_OF, path_save*"tdot_OF.png")
savefig(plt_tdot_OF, path_save*"tdot_OF.pdf")
=#


#=
## Histogram of PDF
plt2_hist = plot(xlabel = label_PV)
histogram!(
    plt2_hist, PV, bins = bins2, yscale = :log10, normalize = :pdf, ylabel = label_PDF,
    label = "", xlabel = label_PV,
)
plot!(
    plt2_hist, size = size2, dpi = dpi, yformatter = x -> (@sprintf "%.2f" x),
    yticks = [0.01, 0.1, 1, 10, 100]
    #yticks = logrange(
    #    yticks(plt2_hist[1])[1][begin], yticks(plt2_hist[1])[1][end],
    #    nyticks2_hist
    #),
)

plt2_hist1 = plot(plt2_hist)
plot!(plt2_hist1, xlabel = "", xticks = false)
plt2_hist2 = plot(plt2_hist)
#savefig(plt2_hist, path_save*"hist.png")
#savefig(plt2_hist, path_save*"hist.pdf")
=#

# Error in pv

## Density
plt2_rho_err = plot(
    PV, (rho_mbs_25 - rho)*rho_fac, linewidth = linewidth, color = color_RE,
    label = "", ylabel = label_rho_err
)

## Heat conductivity
plt2_lambda_err = plot(
    PV, (lambda_mbs_25 - lambda)*lambda_fac, linewidth = linewidth, color = color_RE,
    label = "", ylabel = label_lambda_err
)

## Specific heat capacity
plt2_cpMean_err = plot(
    PV, (cpMean_mbs_25 - cpMean)*cpMean_fac, linewidth = linewidth, color = color_RE,
    label = "", ylabel = label_cpMean_err
)

## Dynamic viscosity
plt2_visc_err = plot(
    PV, (visc_mbs_25 - visc)*visc_fac, linewidth = linewidth, color = color_RE,
    label = "", ylabel = label_visc_err
)

## Progress variable source term
plt2_omega_yc_err = plot(
    PV, (omega_yc_mbs_25 - omega_yc)*omega_yc_fac, linewidth = linewidth, color = color_RE,
    label = "", ylabel = label_omega_yc_err, xlabel = label_PV
)

## Temperature source term
plt2_tdot_OF_err = plot(
    PV, (-(tdot_OF - tdot_OF_mbs_25)*tdot_OF_fac), linewidth = linewidth, size = size2,
    color = color_RE, dpi = dpi, label = "", ylabel = label_tdot_OF_err, xlabel = label_PV
)
#savefig(plt2_tdot_OF_err, path_save*"tdot_OF_err.png")

## Merge all of them

plt2 = plot(
    plt2_rho_err, plt2_lambda_err, plt2_cpMean_err,
    plt2_visc_err, plt2_omega_yc_err, plt2_tdot_OF_err,
    layout = grid(3, 2), size = (640, 360), dpi = dpi, left_margin = 2mm,
)
savefig(plt2, path_save*"error_in_pv.png")
savefig(plt2, path_save*"error_in_pv.pdf")

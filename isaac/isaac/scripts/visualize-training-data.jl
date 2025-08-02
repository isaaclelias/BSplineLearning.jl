using MasterThesis
using Logging
using HDF5
using Dates
using LaTeXStrings
using CairoMakie
using preBurner
using DataFrames
using Measures

# Where to save
_title = "with 1d curves"
title = MasterThesis.date_title(Dates.format(now(), "YYYY-mm-dd_HH-MM"), _title)
dir = "../results/visualize-training-data/"*title*"/"

# What log level to print
logger = ConsoleLogger(stdout, Logging.Debug)
global_logger(logger)

# Read data
df_2d = preBurner.readHDF5("../resources/hoq_training_data_phi=1_Le=1_CH4.h5") |>
        ((df, meta_df),) -> df[
            :,
            [
                "t", "X", "PV", "omega_yc", "rho", "lambda", "Teq_Term3_OF",
                "CO2", "T", "yc", "CO", "cpMean", "tdot_OF", "visc"
            ]
        ]
df_1d = df_2d[df_2d[:, "t"] .== 0, :]
df = df_2d
print("df_2d "); println(describe(df_2d))
print("df_1d "); println(describe(df_1d))

#############################
# fig1 - every quantity colored by temperature
#############################

fig1 = Figure(size = (1490, 700), dpi = 300, fontsize = 24)
grd = fig1[1:3, 1:3] = GridLayout()
colsize!(fig1.layout, 1, Relative(1/3))
colsize!(fig1.layout, 2, Relative(1/3))
colsize!(fig1.layout, 3, Relative(1/3))
rowsize!(fig1.layout, 1, Relative(1/3))
rowsize!(fig1.layout, 2, Relative(1/3))
rowsize!(fig1.layout, 3, Relative(1/3))
x1_1d = df_1d[:, "PV"]; x1_2d = df_2d[:, "PV"]
xlabel1 = L"PV \ [-]"
colorbar_label1 = L"T \ [\text{K}]"
markersize1_1d = 5; markersize1_2d = 2
color1_1d = :blue; color1_2d = df_2d[:, "T"]
colormap1_2d = :thermal

# rho
ax1_rho = Axis(
    fig1[1,1], ylabel = L"\rho \ [\text{kg} \cdot \text{m}^{-3}]",
    yaxisposition = :left
)
scatter!( # 2d dataset
    ax1_rho, x1_2d, df_2d[:, "rho"],
    color = color1_2d, markersize = markersize1_2d,
    colormap = colormap1_2d
)
scatter!( # 1d dataset
    ax1_rho, x1_1d, df_1d[:, "rho"],
    color = color1_1d, markersize = markersize1_1d
)
hidexdecorations!(ax1_rho, grid = false)

# lambda
ax1_lambda = Axis(
    fig1[1,2],
    ylabel = L"\lambda \ [\text{W} \cdot \text{m}^{-1} \cdot \text{K}^{-1}]",
    yaxisposition = :left
)
scatter!( # 2d dataset
    ax1_lambda, x1_2d, df_2d[:, "lambda"],
    color = color1_2d, markersize = markersize1_2d,
    colormap = colormap1_2d,
)
scatter!( # 1d dataset
    ax1_lambda, x1_1d, df_1d[:, "lambda"],
    color = color1_1d, markersize = markersize1_1d
)
hidexdecorations!(ax1_lambda, grid = false)

# cpMean
ax1_cpMean = Axis(
    fig1[1,3],
    ylabel = L"c_p \ [\text{J} \cdot \text{kg}^{-1} \cdot \text{K}^{-1}]",
    yaxisposition = :left
)
scatter!( # 2d dataset
    ax1_cpMean, x1_2d, df_2d[:, "cpMean"],
    color = color1_2d, markersize = markersize1_2d,
    colormap = colormap1_2d
)
scatter!( # 1d dataset
    ax1_cpMean, x1_1d, df_1d[:, "cpMean"],
    color = color1_1d, markersize = markersize1_1d
)
hidexdecorations!(ax1_cpMean, grid = false)

# visc
ax1_visc = Axis(
    fig1[2,1], ylabel = L"\mu \ [\text{Pa} \cdot \text{s}]",
    yaxisposition = :left
)
scatter!( # 2d dataset
    ax1_visc, x1_2d, df_2d[:, "visc"],
    color = color1_2d, markersize = markersize1_2d,
    colormap = colormap1_2d
)
scatter!( # 1d dataset
    ax1_visc, x1_1d, df_1d[:, "visc"],
    color = color1_1d, markersize = markersize1_1d
)
hidexdecorations!(ax1_visc, grid = false)

# omega_yc
ax1_omega_yc = Axis(
    fig1[2,2],
    ylabel = L"\dot{\omega}_{Y_{PV}} \ [\text{kg} \cdot \text{m}^{-3} \cdot \text{s}^{-1}]",
    xlabel = xlabel1,
    yaxisposition = :left
)
scatter!( # 2d datset
    ax1_omega_yc, x1_2d, df_2d[:, "omega_yc"],
    color = color1_2d, markersize = markersize1_2d,
    colormap = colormap1_2d
)
scatter!( # 1d dataset
    ax1_omega_yc, x1_1d, df_1d[:, "omega_yc"],
    color = color1_1d, markersize = markersize1_1d
)
hidexdecorations!(ax1_omega_yc, grid = false)

# tdot_OF
ax1_tdot_OF = Axis(
    fig1[2, 3],
    ylabel = L"\dot{\omega}'_T \ [\text{J} \cdot \text{m}^{-3} \cdot \text{kg}^{-1} \cdot \text{s}^{-3}]",
    xlabel = xlabel1,
    yaxisposition = :left
)
scatter!( # 2d dataset
    ax1_tdot_OF, x1_2d, df_2d[:, "tdot_OF"],
    color = color1_2d, markersize = markersize1_2d,
    colormap = colormap1_2d
)
scatter!( # 1d dataset
    ax1_tdot_OF, x1_1d, df_1d[:, "tdot_OF"],
    color = color1_1d, markersize = markersize1_1d
)
hidexdecorations!(ax1_tdot_OF, grid = false)

# Teq_Term3_OF
ax1_Teq_Term3_OF = Axis(
    fig1[3,1],
    ylabel = L"D_{diff,s} \ [\text{J} \cdot \text{m}^{-2} \cdot \text{s}^{-1}]",
    xlabel = xlabel1,
    yaxisposition = :left
)
scatter!( # 2d dataset
    ax1_Teq_Term3_OF, x1_2d, df_2d[:, "Teq_Term3_OF"],
    color = color1_2d, markersize = markersize1_2d,
    colormap = colormap1_2d
)
sca1 = scatter!( # 1d dataset
    ax1_Teq_Term3_OF, x1_1d, df_1d[:, "Teq_Term3_OF"],
    color = color1_1d, markersize = markersize1_1d
)

Legend(
    fig1[3, 2],
    [sca1],
    [L"t = 0 \ \text{ms}"],
    haling = :left, valign = :bottom
)

Colorbar(
    fig1[3, 2:3],
    label = colorbar_label1,
    limits = extrema(color1_2d),
    vertical = false, colormap = colormap1_2d
)

linkxaxes!(
    ax1_rho, ax1_lambda, ax1_cpMean, ax1_visc, ax1_omega_yc, ax1_tdot_OF,
    ax1_Teq_Term3_OF
)
rowgap!(grd, 2)

mkpath(dir)
path_fig1 = dir*"fig1.png"
path_fig1_pdf = dir*"fig1.pdf"
@info "Saving fig1" path_fig1 path_fig1_pdf
save(path_fig1, fig1)
#save(path_fig1_pdf, fig1)

#############################
# fig2 - reproduce Jeremy's plot
#############################

fig2 = Figure(size = (1280/2, 720/2))

ax2_PV = Axis(
    fig2[1,1],
    xlabel = L"x \ [\text{mm}]",
    ylabel = L"t \ [\text{ms}]"
)
scatter!(
    ax2_PV, df_2d[:, "X"]*10^3, df_2d[:, "t"]*10^3,
    color = df_2d[:, "CO"], markersize = 2
)

ax2_T_PV = Axis(
    fig2[1, 2],
    xlabel = L"PV \ [-]",
    ylabel = L"T \ [\text{K}]"
)
scatter!(
    ax2_T_PV, df_2d[:, "PV"], df_2d[:, "T"],
    color = df_2d[:, "CO"], markersize = 2
)

Colorbar(
    fig2[2, 1:2],
    limits = (minimum(df_2d[:, "T"]), maximum(df_2d[:, "T"])),
    vertical = false, label = L"Y_{CO} \ [-]"
)

path_fig2 = dir*"fig2.png"
@info "Saving fig2" path_fig2
save(path_fig2, fig2)

#############################
# fig3
#############################

fig3 = Figure(size = (1000, 4*400))
colormap3 = :managua
markersize3 = 2

# rho
ax3_rho = Axis3(
    fig3[1, 1],
    xlabel = L"T \ [\text{K}]",
    ylabel = L"PV \ [-]",
    zlabel = L"\rho \ \ [\text{W} \cdot \text{m}^{-1} \cdot \text{K}^{-1}]"
)
scatter!(
    ax3_rho,
    df[:, "T"], df[:, "PV"], df[:, "rho"],
    color = df[:, "rho"],
    markersize = markersize3, colormap = colormap3
)

# lambda
ax3_lambda = Axis3(
    fig3[1, 2],
    xlabel = L"T \ [\text{K}]",
    ylabel = L"PV \ [-]",
    zlabel = L"\lambda \ [\text{kg} \cdot \text{m}^{-3}]"
)
scatter!(
    ax3_lambda,
    df[:, "T"], df[:, "PV"], df[:, "lambda"],
    color = df[:, "rho"],
    markersize = markersize3, colormap = colormap3
)

# cpMean
ax3_cpMean = Axis3(
    fig3[2, 1],
    xlabel = L"T \ [\text{K}]",
    ylabel = L"PV \ [-]",
    zlabel = L"c_p \ [\text{J} \cdot \text{kg}^{-1} \cdot \text{K}^{-1}]"
)
scatter!(
    ax3_cpMean,
    df[:, "T"], df[:, "PV"], df[:, "cpMean"],
    color = df[:, "cpMean"],
    markersize = markersize3, colormap = colormap3
)

# visc
ax3_visc = Axis3(
    fig3[2, 2],
    xlabel = L"T \ [\text{K}]",
    ylabel = L"PV \ [-]",
    zlabel = L"\mu \ [\text{Pa} \cdot \text{s}]",
)
scatter!(
    ax3_visc,
    df_2d[:, "T"], df_2d[:, "PV"], df_2d[:, "visc"],
    color = df_2d[:, "visc"],
    markersize = markersize3, colormap = colormap3
)

# omega_yc
ax3_omega_yc = Axis3(
    fig3[3,1],
    xlabel = L"T \ [\text{K}]",
    ylabel = L"PV \ [-]",
    zlabel = L"\dot{\omega}_{Y_{PV}} \ [-]"
)
scatter!(
    ax3_omega_yc,
    df[:, "T"], df[:, "PV"], df[:, "omega_yc"],
    color = df[:, "omega_yc"],
    markersize = markersize3, colormap = colormap3
)

# tdot_OF
ax3_tdot_OF = Axis3(
    fig3[3, 2],
    xlabel = L"T \ [\text{K}]",
    ylabel = L"PV \ [-]",
    zlabel = L"\dot{\omega}'_T"
)
scatter!(
    ax3_tdot_OF,
    df_2d[:, "T"], df_2d[:, "PV"], df_2d[:, "tdot_OF"],
    color = df_2d[:, "tdot_OF"],
    markersize = markersize3, colormap = colormap3
)

# Teq_Term3_OF
ax3_Teq_Term3_OF = Axis3(
    fig3[4,1],
    xlabel = L"T \ [\text{K}]",
    ylabel = L"PV \ [-]",
    zlabel = L"D_{diff,s} \ [?]"
)
scatter!(
    ax3_Teq_Term3_OF,
    df[:, "T"], df[:, "PV"], df[:, "Teq_Term3_OF"],
    color = df[:, "Teq_Term3_OF"],
    markersize = markersize3, colormap = colormap3
)

path_fig3 = dir*"fig3.png"
#@info "Saving fig3" path_fig3
#save(path_fig3, fig3)

#############################
# fig4 - temperature evolution
#############################

fig4 = Figure()

ax4_T = Axis(
    fig4[1,1],
    xlabel = L"x \ [\text{mm}]",
    ylabel = L"t \ [\text{ms}]"
)
scatter!(
    ax4_T, df_2d[:, "X"]*10^3, df_2d[:, "t"]*10^3,
    color = df_2d[:, "T"], markersize = 2, colormap = :thermal
)

Colorbar(
    fig4[2, 1],
    limits = (minimum(df_2d[:, "T"]), maximum(df_2d[:, "T"])),
    vertical = false, label = L"T \ [\text{K}]", colormap = :thermal
)

path_fig4 = dir*"fig4.png"
#@info "Saving fig4" path_fig4
#save(path_fig4, fig4)

#############################
# fig5 - experimental
#############################
#=
fig5 = Figure(size = (600, 1000))
grd5 = fig5[1:7, 1] = GridLayout()
x5_1d = df_1d[:, "PV"]; x5_2d = df_2d[:, "PV"]
xlabel5 = L"PV \ [-]"
colorbar_label5 = L"x \ [\text{mm}]"
markersize5_1d = 5; markersize5_2d = 2
color5_1d = :blue; color5_2d = df_2d[:, "X"]*10^3
colormap5_2d = :lajolla

# rho
ax5_rho = Axis(
    fig5[1,1], ylabel = L"\rho \ [\text{kg} \cdot \text{m}^{-3}]",
    yaxisposition = :left
)
scatter!( # 2d dataset
    ax5_rho, x5_2d, df_2d[:, "rho"],
    color = color5_2d, markersize = markersize5_2d,
    colormap = colormap5_2d
)
scatter!( # 1d dataset
    ax5_rho, x5_1d, df_1d[:, "rho"],
    color = color5_1d, markersize = markersize5_1d
)
hidexdecorations!(ax5_rho, grid = false)

# lambda
ax5_lambda = Axis(
    fig5[2,1],
    ylabel = L"\lambda \ [\text{W} \cdot \text{m}^{-1} \cdot \text{K}^{-1}]",
    yaxisposition = :left
)
scatter!( # 2d dataset
    ax5_lambda, x5_2d, df_2d[:, "lambda"],
    color = color5_2d, markersize = markersize5_2d,
    colormap = colormap5_2d
)
scatter!( # 1d dataset
    ax5_lambda, x5_1d, df_1d[:, "lambda"],
    color = color5_1d, markersize = markersize5_1d
)
hidexdecorations!(ax5_lambda, grid = false)

# cpMean
ax5_cpMean = Axis(
    fig5[3,1],
    ylabel = L"c_p \ [\text{J} \cdot \text{kg}^{-1} \cdot \text{K}^{-1}]",
    yaxisposition = :left
)
scatter!( # 2d dataset
    ax5_cpMean, x5_2d, df_2d[:, "cpMean"],
    color = color5_2d, markersize = markersize5_2d,
    colormap = colormap5_2d
)
scatter!( # 1d dataset
    ax5_cpMean, x5_1d, df_1d[:, "cpMean"],
    color = color5_1d, markersize = markersize5_1d
)
hidexdecorations!(ax5_cpMean, grid = false)

# visc
ax5_visc = Axis(
    fig1[4,1], ylabel = L"\mu \ [\text{Pa} \cdot \text{s}]",
    yaxisposition = :left
)
scatter!( # 2d dataset
    ax5_visc, x5_2d, df_2d[:, "visc"],
    color = color5_2d, markersize = markersize5_2d,
    colormap = colormap5_2d
)
scatter!( # 1d dataset
    ax5_visc, x5_1d, df_1d[:, "visc"],
    color = color5_1d, markersize = markersize5_1d
)
hidexdecorations!(ax5_visc, grid = false)

# omega_yc
ax5_omega_yc = Axis(
    fig5[5,1],
    ylabel = L"\dot{\omega}_{Y_{PV}} \ [\text{kg} \cdot \text{m}^{-3}] \cdot \text{s}^{-1}",
    yaxisposition = :left
)
scatter!( # 2d datset
    ax5_omega_yc, x5_2d, df_2d[:, "omega_yc"],
    color = color5_2d, markersize = markersize5_2d,
    colormap = colormap5_2d
)
scatter!( # 1d dataset
    ax5_omega_yc, x5_1d, df_1d[:, "omega_yc"],
    color = color5_1d, markersize = markersize5_1d
)
hidexdecorations!(ax5_omega_yc, grid = false)

# tdot_OF
ax5_tdot_OF = Axis(
    fig5[6, 1],
    ylabel = L"\dot{\omega}'_T \ [\text{kg} \cdot \text{m}^{-1} \cdot \text{s}^{-3}]",
    yaxisposition = :left
)
scatter!( # 2d dataset
    ax5_tdot_OF, x5_2d, df_2d[:, "tdot_OF"],
    color = color5_2d, markersize = markersize5_2d,
    colormap = colormap5_2d
)
scatter!( # 1d dataset
    ax5_tdot_OF, x5_1d, df_1d[:, "tdot_OF"],
    color = color5_1d, markersize = markersize5_1d
)
hidexdecorations!(ax5_tdot_OF, grid = false)

# Teq_Term3_OF
ax5_Teq_Term3_OF = Axis(
    fig5[7,1],
    ylabel = L"D_{diff,s} \ [\text{kg} \cdot \text{s}^{-3}]",
    xlabel = xlabel5,
    yaxisposition = :left
)
scatter!( # 2d dataset
    ax5_Teq_Term3_OF, x5_2d, df_2d[:, "Teq_Term3_OF"],
    color = color5_2d, markersize = markersize5_2d,
    colormap = colormap5_2d
)
sca5 = scatter!( # 1d dataset
    ax5_Teq_Term3_OF, x5_1d, df_1d[:, "Teq_Term3_OF"],
    color = color5_1d, markersize = markersize5_1d
)

Legend(
    fig5[1, 2],
    [sca5],
    [L"t = 0"]
)

Colorbar(
    fig5[2:7, 2],
    label = colorbar_label5,
    limits = (0, 40), #extrema(color5_2d),
    vertical = true, colormap = colormap5_2d
)

linkxaxes!(
    ax5_rho, ax5_lambda, ax5_cpMean, ax5_visc, ax5_omega_yc, ax5_tdot_OF,
    ax5_Teq_Term3_OF
)
rowgap!(grd5, 2)

path_fig5 = dir*"fig5.png"
@info "Saving fig5" path_fig5
save(path_fig5, fig5)
=#

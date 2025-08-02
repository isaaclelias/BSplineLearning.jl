using MasterThesis
using Plots
using Logging, LoggingExtras
using Dates
using LaTeXStrings
using Printf

# logging setup
log_dir = "../logs/"; mkpath(log_dir)
log_title = basename(@__FILE__) |> x -> replace(x, ".jl" => "")
log_format_date = Dates.format(now(), "yyyy-mm-dd_HH-MM")
log_path = log_dir * log_format_date * " " * log_title * ".log"
log_info_path  = log_path * ".info" ; touch(log_info_path)
log_debug_path = log_path * ".debug"; touch(log_debug_path)
const logger = TeeLogger(
    ConsoleLogger(stdout                    , Logging.Info ),
    SimpleLogger( open(log_info_path , "w+"), Logging.Info ),
    SimpleLogger( open(log_debug_path, "w+"), Logging.Debug),
); global_logger(logger)
@info "Logging" log_info_path log_debug_path

# stuff to save the study results
_title = "trying"
filename = basename(@__FILE__) |> x -> replace(x, ".jl" => "")
title = MasterThesis.format_date_githash_title_now(_title)
path_save = "../results/" * filename * "/"*title*"/"; mkpath(path_save)

# Labels
label_X = L"x \ [\mathrm{mm}]"
label_PV = L"PV \ [-]"
label_rho = L"\rho"
label_visc = L"\mu"
label_lambda = L"\lambda"
label_cpMean = L"c_p"
label_tdot_OF = L"\dot{\omega}'_{T}"
label_omega_yc = L"\dot{\omega}_{Y_{PV}}"
#label_Teq_Term3_OF = L"D_{diff,s} \ [\text{kg} \cdot \text{s}^{-3}]"
label_RE = L"|MRE| \ [\%]"
label_PDF = L"PDF \ [-]"
ylabels = [label_rho, label_lambda, label_cpMean, label_visc, label_omega_yc, label_tdot_OF]
label_n_reg = L"n_{reg} \ [-]"

# Plot with best for each output

nregs          = [ 5   , 10  , 15  , 20  ]
rho_bests      = [ 3.48, 1.22, 1.03, 1.55]
lambda_bests   = [ 0.14, 0.03, 0.02, 0.01]
cpMean_bests   = [ 1.11, 0.24, 0.15, 0.12]
visc_bests     = [ 0.40, 0.22, 0.08, 0.10]
omega_yc_bests = [10.04, 0.91, 0.11, 0.10]
tdot_OF_bests  = [10.32, 2.43, 0.08, 0.04]

#=
plt2_mpe_i = plot_heatmap_valuess(
    mpes_i_best_comp[idxs_wcm], mpes_i_best_comp[idxs_wcm], labels_nl[idxs_wcm],
    ylabels, label_plt = L"\mathrm{MRE} \ [\%]",
    size = (480*scale_plt1, 160*scale_plt1), bottom_margin = 13mm, left_margin = 4mm,
    right_margin = 10mm, top_margin = 3mm, yrotation = 45, xtickfontsize = 12,
    ytickfontsize = 6, annotation_size = 10
)
=#

#=
plt_bests = plot(
    fill(nregs, 6),
    [
        rho_bests, lambda_bests, cpMean_bests, visc_bests, omega_yc_bests,
        tdot_OF_bests
    ],
    labels = [
        label_rho, label_lambda, label_cpMean, label_visc, label_omega_yc,
        label_tdot_OF
    ],
    markers = [:circle, :rect, :star, :utriangle, :dtriangle, :star4],
)
=#
plt_bests = plot(
    xlabel = label_n_reg, ylabel = label_RE, yscale = :log10, xticks = nregs,
    yticks = [
        10   , 9   , 8   , 7   , 6   , 5   , 4   , 3   , 2   ,
         1   , 0.9 , 0.8 , 0.7 , 0.6 , 0.5 , 0.4 , 0.3 , 0.2 ,
         0.1 , 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02,
         0.01,
    ],
    yformatter = x -> (any(x .== [10, 1, 0.1, 0.01]) ? (@sprintf "%.2f" x) : ""),
    xformatter = x -> (@sprintf "%.0f" x),
    legend_title = L"\varphi", legend_position = :outerright, xscale = :log10,
    markersize = 6, legendfontsize = 12,
    #yminorgrid = true, yminorticks = 10,
)
plot!(plt_bests, nregs, rho_bests     , marker = :circle   , label = label_rho     )
plot!(plt_bests, nregs, lambda_bests  , marker = :rect     , label = label_lambda  )
plot!(plt_bests, nregs, cpMean_bests  , marker = :star     , label = label_cpMean  )
plot!(plt_bests, nregs, visc_bests    , marker = :utriangle, label = label_visc    )
plot!(plt_bests, nregs, omega_yc_bests, marker = :dtriangle, label = label_omega_yc)
plot!(plt_bests, nregs, tdot_OF_bests , marker = :star4    , label = label_tdot_OF )
plot!(plt_bests, size = (400, 250), dpi = 300)

savefig(plt_bests, path_save*"plot_bests.pdf")
savefig(plt_bests, path_save*"plot_bests.png")

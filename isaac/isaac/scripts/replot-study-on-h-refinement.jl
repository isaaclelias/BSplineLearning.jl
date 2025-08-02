using MasterThesis
using Serialization
using Plots
using DataFrames
using Dates
using Logging, LoggingExtras
using LaTeXStrings
using Measures

# Construct the cases
## Complete case
const _methods_which = [:error, :quadratic_error]
const _methods_where = [:first_order_moment, :second_order_moment]
const _methods_how_high = [:spline_prediction, :curve_middle]
## Construct vectors
methods_which    = Symbol[]
methods_where    = Symbol[]
methods_how_high = Symbol[]
for method_which    in _methods_which,
    method_where    in _methods_where,
    method_how_high in _methods_how_high
    
    push!(methods_which, method_which)
    push!(methods_where, method_where)
    push!(methods_how_high, method_how_high)
end

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
_title = "25 pontos"
filename = basename(@__FILE__) |> x -> replace(x, ".jl" => "")
title = MasterThesis.format_date_githash_title_now(_title)
path_save = "../results/" * filename * "/"*title*"/"; mkpath(path_save)

# data loading
curves = curves_hoq_training_data_1d()

# Load report
reps = deserialize("../results/study-on-h-refinement/highlights/2025-03-11_19-48 5360c9bf_mod maxiters-4999 npusm-100 a_opt-0.025 nir-200 nppusm-100/")

# Labels
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
ylabels = [label_rho, label_lambda, label_cpMean, label_visc, label_omega_yc, label_tdot_OF]

# Initialize containers
n_cases = length(methods_which)
lossess = Vector{Vector{Float64}}(undef, n_cases)
lossess_comp = Vector{Vector{Float64}}(undef, n_cases)
maess_comp = Vector{Vector{Float64}}(undef, n_cases)
maess_norm_comp = Vector{Vector{Float64}}(undef, n_cases)
maess_norm = Vector{Vector{Float64}}(undef, n_cases)
maess = Vector{Vector{Float64}}(undef, n_cases)
mpes_i_best_comp = Vector{Vector{Float64}}(undef, n_cases)
maes_i_best = Vector{Vector{Float64}}(undef, n_cases)
mpess_i_comp = Vector{Vector{Vector{Float64}}}(undef, n_cases)
mpess_comp = Vector{Vector{Float64}}(undef, n_cases)

## Start the computation batches

for (i, rep) in enumerate(reps)
    mbs = rep.mbss[end]
    _, mpes_i_best_comp[i] = MasterThesis.maximum_percentage_error_and_errors2(mbs, curves)
end

## Maximum Error heatmap

### Everything
labels_nl = [
    "- $(method_which), $(method_how_high),\n$(method_where)."
    for (method_which, method_where, method_how_high) in zip(methods_which, methods_where, methods_how_high)
]
scale_plt1 = 1.5
#labels_nl[1] = L"k = %$(ks[1]) \ , \ n_{reg} = %$(ns_u_steps_aut[1]-1)"
plt_mpe_i = plot_heatmap_valuess(
    mpes_i_best_comp, mpes_i_best_comp, labels_nl, y_labels(curves),
    label_plt = L"\mathrm{MRE} \ [\%]", size = (480*scale_plt1, 280*scale_plt1),
    bottom_margin = 12mm, left_margin = 8mm, right_margin = 10mm, top_margin = 3mm,
    yrotation = 45, xtickfontsize = 12, ytickfontsize = 6, annotation_size = 10
)
file_plot_mpe_i_png = path_save * "maximum-percentual-error.png"
file_plot_mpe_i_pdf = path_save * "maximum-percentual-error.pdf"
touch(file_plot_mpe_i_png)
touch(file_plot_mpe_i_pdf)
savefig(plt_mpe_i, file_plot_mpe_i_png)
savefig(plt_mpe_i, file_plot_mpe_i_pdf)

### Without `curve_middle`
idxs_wcm = [1, 3, 5, 7]
#@show labels_nl labels_nl[idxs_wcm]
plt2_mpe_i = plot_heatmap_valuess(
    mpes_i_best_comp[idxs_wcm], mpes_i_best_comp[idxs_wcm], labels_nl[idxs_wcm],
    ylabels, label_plt = L"\mathrm{MRE} \ [\%]",
    size = (480*scale_plt1, 160*scale_plt1), bottom_margin = 13mm, left_margin = 4mm,
    right_margin = 10mm, top_margin = 3mm, yrotation = 45, xtickfontsize = 12,
    ytickfontsize = 6, annotation_size = 10
)
file2_plot_mpe_i_png = path_save * "maximum-percentual-error-only-sp.png"
file2_plot_mpe_i_pdf = path_save * "maximum-percentual-error-only-sp.pdf"
touch(file2_plot_mpe_i_png)
touch(file2_plot_mpe_i_pdf)
savefig(plt2_mpe_i, file2_plot_mpe_i_png)
savefig(plt2_mpe_i, file2_plot_mpe_i_pdf)


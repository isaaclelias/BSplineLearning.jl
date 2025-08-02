using Interpolations
using LaTeXStrings

df = df_hoq_training_data_2d()

# Quantities
## xs
T = df[:, "T"]
PV = df[:, "PV"]
## ys
rho = df[:, "rho"]
visc = df[:, "visc"]
lambda = df[:,"lambda"]
cpMean = df[:, "cpMean"]
tdot_OF = df[:, "tdot_OF"]
omega_yc = df[:, "omega_yc"]
Teq_Term3_OF = df[:, "Teq_Term3_OF"]

# Labels
label_T = L"T \ [\mathrm{K}]"
label_PV = L"PV \ [-]"
label_rho = L"\rho \ [\mathrm{kg} \cdot \mathrm{m}^{-3}]"
label_visc = L"\mu \ [\mathrm{Pa} \cdot \mathrm{s}]"
label_lambda = L"\lambda \ [\mathrm{W} \cdot \mathrm{m}^{-1} \cdot \mathrm{K}^{-1}]"
label_cpMean = L"c_p \ [\mathrm{J} \cdot \mathrm{kg}^{-1} \cdot \mathrm{K}^{-1}]"
label_tdot_OF = L"\dot{\omega}'_{T} \ [\mathrm{J} \cdot \mathrm{m}^{-3} \cdot \mathrm{kg}^{-1} \cdot \mathrm{s}^{-1}]"
label_omega_yc = L"\dot{\omega}_{Y_{PV}} \ [\mathrm{kg} \cdot \mathrm{m}^{-3} \cdot \mathrm{s}^{-1}]"
label_Teq_Term3_OF = L"D_{diff,s} \ [\text{kg} \cdot \text{s}^{-3}]"

# Interpolation
interp_linear = linear_interpolation((PV, T), rho)

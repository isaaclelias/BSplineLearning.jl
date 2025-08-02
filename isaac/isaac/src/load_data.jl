function load_hoq_training_data()
    df_2d = nothing
    with_logger(NullLogger()) do
      df_2d = preBurner.readHDF5(pkg_root_dir*"/resources/hoq_training_data_phi=1_Le=1_CH4.h5") |>
            ((df, meta_df),) -> df[
:,
                [
                    "t", "X", "PV", "omega_yc", "rho", "lambda", "Teq_Term3_OF",
                    "CO2", "T", "yc", "CO", "cpMean", "tdot_OF", "visc"
                ]
               ]
end
    df_2d = replace_nan_with_missing.(df_2d)
    _completecases = completecases(df_2d)
    df_2d = df_2d[_completecases, :]
    df_1d = df_2d[df_2d[:, "t"] .== 0, :]

    @debug "load_hoq_training_data" count(_completecases)

    return df_1d, df_2d
end

replace_nan_with_missing(x) = (x isa Number && isnan(x)) ? missing : x

df_hoq_training_data() = load_hoq_training_data()

df_hoq_training_data_1d() = df_hoq_training_data()[1]
df_hoq_training_data_2d() = df_hoq_training_data()[2]

function curves_hoq_training_data_1d()::ACurves
    df = df_hoq_training_data_1d()
    x = df[:, "PV"]
    curves = [
        Curve(
            x, df[:, "rho"],
            "",
            L"\rho \ [\mathrm{kg} \cdot \mathrm{m}^{-3}]",
        ),
        Curve(
            x, df[:, "lambda"],
            "",
            L"\lambda \ [\mathrm{W} \cdot \mathrm{m}^{-1} \cdot \mathrm{K}^{-1}]",
        ),
        Curve(
            x, df[:, "cpMean"],
            "",
            L"c_p \ [\mathrm{J} \cdot \mathrm{kg}^{-1} \cdot \mathrm{K}^{-1}]",
        ),
        Curve(
            x, df[:, "visc"],
            "",
            L"\mu \ [\mathrm{Pa} \cdot \mathrm{s}]",
        ),
        Curve(
            x, df[:, "omega_yc"],
            L"PV \ [-]",
            L"\dot{\omega}_{Y_{PV}} \ [\mathrm{kg} \cdot \mathrm{m}^{-3} \cdot \mathrm{s}^{-1}]",
        ),
        Curve(
            x, df[:, "tdot_OF"],
            L"PV \ [-]",
            L"\dot{\omega}'_{T} \ [\mathrm{J} \cdot \mathrm{m}^{-3} \cdot \mathrm{kg}^{-1} \cdot \mathrm{s}^{-1}]",
        ),
        #=Curve(
            x, df[:, "Teq_Term3_OF"],
            L"\rho \ [\text{kg} \cdot \text{m}^{-3}]",
            L"D_{diff,s} \ [\text{kg} \cdot \text{s}^{-3}]",
        ),=#
        # let's assume that if only the first curve specifies a xlabel, then this xlabel is valid for all curves
    ]

    return curves
end

function load_hoq_training_data_complete()
    df_2d = nothing
    with_logger(NullLogger()) do
        df_2d, meta_df = preBurner.readHDF5(
            pkg_root_dir*"/resources/hoq_training_data_phi=1_Le=1_CH4.h5"
        )
    end
    df_2d = replace_nan_with_missing.(df_2d)
    _completecases = completecases(df_2d)
    df_2d = df_2d[_completecases, :]
    df_1d = df_2d[df_2d[:, "t"] .== 0, :]

    @debug "load_hoq_training_data" count(_completecases)

    return df_1d, df_2d
end

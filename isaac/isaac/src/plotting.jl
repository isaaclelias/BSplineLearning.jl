const n_dots::Int = 2000
lock_plt = ReentrantLock()

function savefig_with_date(plt, title::AbstractString)::Nothing
    dt = Dates.format(now(), "YYYY-mm-dd_HH_mm")
    filename = title*"_"*dt
    savefig(plt, pkg_root_dir*"/results/"*filename*".png")

    return nothing
end

function plot_bspline!(plt, bs::BSpline, range::AbstractRange)::Nothing
    x = range
    y = bs.(x)

    plot!(plt, x, y, label = "B-Spline") # plot the evaluated b-spline

    for (i, cpc) in enumerate(control_points_coordinates(bs))
        xs_cp = cpc[1]; y_cp = cpc[2]
        xm_cp = mean(xs_cp) # "center of mass" of xs_cp
        ys_cp = repeat([y_cp], length(xs_cp))

        scatter!(
            plt, xs_cp, ys_cp,
            label = "", markersize = 2, markercolor = :black,
            markershape = :vline, markeralpha = 0.4
        ) # the control points
        scatter!(
            plt, [xm_cp], [y_cp], 
            label = "", markersize = 4, markercolor = :black,
            markershape = :xcross, markeralpha = 0.6
        ) # control pointp center of mass
        plot!(
            plt, [xs_cp[begin], xs_cp[end]], [y_cp, y_cp],
            label = "", linecolor=:black, linealpha=0.2
        ) # line conecting all xs_cp
    end
    
    vline!(
        plt, bs.u_steps,
        label="", linecolor=:black, linealpha=0.2
    ) # vertical lines at u_steps

    return nothing
end

function plot_bspline(bs::BSpline, range::AbstractRange)
    plt = plot()
    plot_bspline!(plt, bs, range)

    return plt
end

function plot_bspline_curve_losses(bs::BSpline, x_cv, y_cv, losses)
    x_min = x_cv[begin]
    x_max = x_cv[end]
    plt_bs_cv = plot_bspline_curve(bs, x_cv, y_cv)
    plt_ls = plot_losses(losses)
    plt = plot(
        plt_bs_cv, plt_ls,
        layout = grid(2, 1, heights = [0.8, 0.2]),
        size = (600, 400), dpi = dpi
    )

    return plt
end

function plot_bspline_curve(bs::BSpline, x_cv, y_cv)
    x_min = bs.u_steps[begin]
    x_max = bs.u_steps[end]
    plt = plot(title = "B-Spline and targeted curve")
    scatter!(plt, x_cv, y_cv, markersize=1, label = "Data")
    plot_bspline!(plt, bs, LinRange(x_min, x_max, n_dots))

    return plt
end

function plot_losses(losses)
    plot(
        1:length(losses), losses,
        title = "Losses", xaxis = :log10, yaxis = :log10
    )
end

function plot_iterations_labels_log(ys, labels, ylabel)
    n = 1.2
plt = plot(
        size = (480*n, 220*n), dpi = dpi,
        bottom_margin = 5mm, left_margin = 10mm, right_margin = 3mm,
        #legend_position = :outerright,
    )
    for (y, label) in zip(ys, labels)
        plot!(
            plt, 1:length(y), y,
            xaxis = :log10, yaxis = :log10, label = label,
            xlabel = L"Iteration $[-]$", ylabel = ylabel
        )
    end

    return plt
end

function plot_iterations_labels_colors_linestyles_log(
    ys, labels, ylabel, colors, linestyles;
    n = 1, legend_column = 3, size = (480, 380), bottom_margin = -5mm,
    left_margin = 3mm, right_margin = 3mm, top_margin = 0mm
  )
    s = 1 #shold be here?
    plt = plot(;
        size, dpi = dpi*2,
        bottom_margin, left_margin, right_margin, top_margin,
        legend_position = :outerbottom, legend_column = legend_column,
    )
    for (y, label, linestyle, color) in zip(ys, labels, linestyles, colors)
        plot!(
            plt, collect(1:length(y)) .* n, y,
            xaxis = :log10, yaxis = :log10, label = label, color = color,
            linestyle = linestyle, xlabel = L"Iteration $[-]$", ylabel = ylabel,
        )
    end

    return plt
end

function plot_bspline_curve_sparameters(bs::BSpline, x_cv, y_cv, sp)
    plt_bs_cv = plot_bspline_curve(bs, x_cv, y_cv)
    plt_spar = plot()
    eltm = Dates.format(now()-sp.start_time)
    loss_reduction = sp.losses[end]/sp.losses[begin]
    annotate!(
        plt_spart,
        [
         (0/3, 2/2, ("Elapsed time: $(eltm)", 8, :left)),
         (0/3, 1/2, ("Iterations: $(sp.iter)", 8, :left)),
         (0/3, 1/2, ("Loss reduction: ", 8, :left)),
         (0/3, 1/2, ("n_weights: ", 8, :left)),
         (0/3, 1/2, ("k: ", 8, :left)),
         (0/3, 1/2, ("n_points: ", 8, :left))
        ]
    )
end

function date_title(st_fm, title)
    if title == ""; return st_fm
    else;           return st_fm*" "*title
    end
end

const length_gitshorthash::Int = 8

function format_pkg_githash()::String
    rp = LibGit2.GitRepo(pkg_root_dir)
    githash = LibGit2.head_oid(rp)
    gitshorthash = LibGit2.GitShortHash(githash, length_gitshorthash)
    format_gitshorthash = string(gitshorthash)

    return format_gitshorthash
end

function is_pkg_dirty()::Bool
    rp = LibGit2.GitRepo(pkg_root_dir)
    ret = LibGit2.isdirty(rp)

    return ret
end

function format_date_githash_title_now(title)
    fm_dt = Dates.format(now(), "YYYY-mm-dd_HH-MM")
    fm_gh = format_pkg_githash()

    fm = fm_dt * " " * fm_gh

    if is_pkg_dirty()
        fm = fm * "_mod"
    end

    if title !== ""
        fm = fm * " " * strip(title)
    end

    return fm
end

function plot_control_points!(plt, cps)
    for (i, cp) in enumerate(cps)
        xs_cp = cp[1]; y_cp = cp[2]
        xm_cp = mean(xs_cp) # "center of mass" of xs_cp
        ys_cp = repeat([y_cp], length(xs_cp))

        scatter!(
            plt, xs_cp, ys_cp,
            label = "", markersize = 2, markercolor = :black,
            markershape = :vline, markeralpha = 0.4
        ) # the control points
        scatter!(
            plt, [xm_cp], [y_cp], 
            label = "", markersize = 4, markercolor = :black,
            markershape = :xcross, markeralpha = 0.6
        ) # control pointp center of mass
        plot!(
            plt, [xs_cp[begin], xs_cp[end]], [y_cp, y_cp],
            label = "", linecolor=:black, linealpha=0.2
        ) # line conecting all xs_cp 
    end
end

function plot_control_points(cps)
    plt = plot()

    return plot_control_points!(plt, cps)
end

function plot_mbspline_curves(mbs::MBSpline, curves::ACurves)
    cpss = control_points_coordinates(mbs)
    x_min, x_max, _, _ = extrema_curves(curves)
    x_mbs = range(x_min, x_max, n_dots)
    ys_mbs = mbs(x_mbs)
    plts = [plot() for i in mbs.weightss]
    for (i, (y_mbs, cps, curve)) in enumerate(zip(ys_mbs, cpss, curves))
        plts[i] = plot()
        plot_control_points!(plts[i], cps)
        plot!(
            plts[i], x_mbs, y_mbs,
            xlabel = curve.xlabel, ylabel = curve.ylabel, label = ""
        )
        scatter!(plts[i], curve.x, curve.y, label = "")
    end

    n_plts = length(plts)
    plt = plot(
        plts..., layout = (length(plts), 1), size = (800, 200*n_plts),
        dpi = dpi, #bottom_margin = 5, left_margin = 5, right_margin = 3,
    )

    return plt
end

function plots_mbspline_curves_sample(mbs::MBSpline, curves::ACurves, curves_sample_idxss)
    cpss = control_points_coordinates(mbs)
    x_min, x_max, _, _ = extrema_curves(curves)
    x_mbs = range(x_min, x_max, n_dots)
    ys_mbs = mbs(x_mbs)
    plts = [plot() for i in mbs.weightss]
    curves_sample = curves_sample_by_idxss(curves, curves_sample_idxss)
    for (i, (y_mbs, cps, curve, curve_sample)) in enumerate(zip(ys_mbs, cpss, curves, curves_sample))
        plts[i] = plot()
        scatter!(plts[i], curve.x, curve.y, label = "", color = :blue) # plot the original curve
        scatter!( # plot the sample
            plts[i], curve_sample.x, curve_sample.y, label = "", color = :yellow
        )
        plot!( # Plot the mbspline
            plts[i], x_mbs, y_mbs,
            xlabel = curve.xlabel, ylabel = curve.ylabel, label = "", color = :red,
            linewidth = 3
        )
        plot_control_points!(plts[i], cps) # plot the control points
    end

    return plts
end

function plot_mbspline_curves_sample(mbs::MBSpline, curves::ACurves, curves_sample_idxs)
    plts = plots_mbspline_curves_sample(mbs, curves, curves_sample_idxs)
    n_plts = length(plts)
    plt = plot(
        plts..., layout = (3, 2), size = (960, 540),
        dpi = dpi, bottom_margin = 5mm, left_margin = 10mm, right_margin = 3mm,
    )

    return plt
end

function plot_mbspline(mbs::MBSpline)
    cpss = control_points_coordinates(mbs)
    k = mbs.k; u_steps = mbs.u_steps; weightss = mbs.weightss
    x_min = u_steps[k+1]; x_max = u_steps[end-(k+1)+1]
    x_mbs = range(x_min, x_max, n_dots)
    ys_mbs = mbs(x_mbs)
    plts = [plot() for i in mbs.weightss]
    for (i, (y_mbs, cps)) in enumerate(zip(ys_mbs, cpss))
        plts[i] = plot()
        plot_control_points!(plts[i], cps)
        plot!(plts[i], x_mbs, y_mbs, label = "")
    end

    n_plts = length(plts)
    plt = plot(plts..., layout = (n_plts, 1))

    return plt
end

function plot_heatmap_valuess(
    valuess, labelss, xs_labels, ys_labels;
    label_plt = "", size = (400, 2), bottom_margin = 5mm, left_margin = 0mm,
    right_margin = 8mm, top_margin = 0mm, xrotation = -30, yrotation = 0, xtickfontsize = 8,
    ytickfontsize = 8, annotation_size = 5,
)
    n_xs_labels = length(xs_labels)
    n_ys_labels = length(ys_labels)
    _values = transpose(hcat(valuess...))
    _labels = hcat([labels[end:-1:begin] for labels in labelss]...)
    plt = heatmap(
        ys_labels, xs_labels, _values;
        size = size, colorbar_title = label_plt, dpi = dpi*2, bottom_margin,
        left_margin, right_margin, xrotation, yrotation, c = :redsblues,
        clims = x -> (-maximum(abs.(x)), maximum(abs.(x))),
        colorbar_titlefonthalign = :right, xtickfontsize, ytickfontsize,
        #series_annotations = _annotations,
    )
    for (i, labels) in enumerate(eachrow(_labels))
        for (j, label) in enumerate(labels)
            #println("$i $j $label")
            #label_str = (@sprintf "%.3e" label) |> x -> replace(x, "e-0" => "e-", "e+0" => "e+")
            label_str = (@sprintf "%.2f" label) #|> x -> replace(x, "e-0" => "e-", "e+0" => "e+")
            annotate!(
                plt, n_ys_labels-i+0.05, j-0.5, (label_str, annotation_size, :black, :left)
            )
        end
    end
            
    #annotate!(plt, annotations)

    return plt
end

function unpack_reps(reps)
    n_cases = length(reps)
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

    for (i, rep) in enumerate(reps)
        lossess[i] = rep.losses
        lossess_comp[i] = rep.losses_comp
        maess_norm[i] = rep.maes_norm
        maess_norm_comp[i] = rep.maes_norm_comp
        maess[i] = rep.maes
        maess_comp[i] = rep.maes_comp
        mpes_i_best_comp[i] = rep.mpe_i_best_comp
        mpess_i_comp[i] = rep.mpes_i_comp
        mpess_comp[i] = rep.mpes_comp
        maes_i_best[i] = rep.mae_i_best
    end

    return (; lossess, lossess_comp, maess_norm, maess_norm_comp, maess, maess_comp, mpes_i_best_comp, mpess_i_comp, mpess_comp, maes_i_best)
end

function plot_nmbs_2d!(plt, nmbs::NMBSpline)
    minimum_inputs(nmbs)
    maximum_inputs(nmbs)
    for (n, (minimum_input, maximum_input)) in enumerate(zip(
        minimum_inputs, maximum_inputs)
    )
    end
end

function minimum_input(nmbs::NMBSpline)
  
end

function maximum_input(nmbs::NMBSpline)
  
end

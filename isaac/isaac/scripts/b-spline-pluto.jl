### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ bc287936-7d2d-11ef-09bf-2b6a902a80cd
begin
	using Pkg
	Pkg.activate(".")
end

# ╔═╡ 6ebb745d-8093-443a-a0f2-2ec6a6adcdd4
begin
	using MasterThesis
	using PlutoUI
	using Plots
end

# ╔═╡ 876e8329-be08-4038-ba8a-7ddaf9051db7
md"""
w1: $(@bind w1 Slider(-1:0.01:1)) w2: $(@bind w2 Slider(-1:0.01:1))

w3: $(@bind w3 Slider(-1:0.01:1)) w4: $(@bind w4 Slider(-1:0.01:1))
"""

# ╔═╡ 88fe7ebc-10dd-4f07-b1e2-6d40a777b5fd
begin
	k = 3
	bs = BSpline([w1, w2, w3, w4], LinRange(1,4+k+1, 4+k+1), k)
end 

# ╔═╡ fa5b36ac-4845-4ace-b254-de4e9bda42ea
begin
	plt = plot_bspline(bs, LinRange(0, 4+k+2, 300))
	plot!(plt, ylims=(-1.2,1.2))
end

# ╔═╡ Cell order:
# ╠═bc287936-7d2d-11ef-09bf-2b6a902a80cd
# ╠═6ebb745d-8093-443a-a0f2-2ec6a6adcdd4
# ╟─876e8329-be08-4038-ba8a-7ddaf9051db7
# ╟─88fe7ebc-10dd-4f07-b1e2-6d40a777b5fd
# ╟─fa5b36ac-4845-4ace-b254-de4e9bda42ea

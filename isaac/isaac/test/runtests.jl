using MasterThesis
using Test
using ForwardDiff
using Enzyme

const n_us_sample::Int = 10

df_1d = df_hoq_training_data_1d()
curves = curves_hoq_training_data_1d()

@testset "BSpline.jl" begin
    bs3 = BSpline(
        [1, -2, 3, -4],
        [1, 2, 4, 5, 6, 7, 10, 11],
        3
    )

    bs23 = BSpline2{3}(
        [1, -2, 3, -4],
        [1, 2, 4, 5, 6, 7, 10, 11]
    )

    @testset "BSpline" begin
        @test bs3(1) == 0
        @test bs3(3) == 0.375
        @test bs3(10) == -0.2
    end

    @testset "b_spline_definition" begin
        @test b_spline_definition(bs3, 1) == 0
        @test b_spline_definition(bs3, 3) == 0.375
        @test b_spline_definition(bs3, 10) == -0.2
    end

    @testset "b_spline_coxdeboor" begin
        @test_broken b_spline_coxdeboor(bs3, 1) == 0
        @test_broken b_spline_coxdeboor(bs3, 3) == 0.375
        @test_broken b_spline_coxdeboor(bs3, 10) == -0.2
    end

    @testset "control_point_coordinates()" begin
        @test control_point_coordinates(bs3, 2) == ([2,4,5,6,7], -2)
    end

    @testset "BSpline2" begin
        bs23_1 = BSpline2{3}(
            [1, -2, 3, -4],
            [1, 2, 3, 4, 5, 6, 7, 8] # only fully controlled from 4 to 5
        )
        bs23_2 = BSpline2{3}(
            [0, 1, -2, 3, -4, 0],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] # fully controlled from 3 to 6?
        )

        us_1 = LinRange(4, 5, n_us_sample)
        @test_broken all(bs23_1.(us_1) .== bs3.(us_1))

        us_2 = LinRange(3, 6, n_us_sample)
        @test_broken all(bs23_2.(us_2) .== bs3.(us_2))
    end

    @testset "BSpline ForwardDiff"
        
    end
end

@testset "MBSpline.jl" begin
    weights_bs1 = [1, pi, -5, -8, 1.5, exp(1)]
    weights_bs2 = [4, 8 , 15, 16, 23 , 42    ]
    u_steps = -6:1:5
    k = 5 # let's try higher and higher orders
    us_trial = LinRange(minimum(u_steps), maximum(u_steps), n_us_sample)
    bs1 = BSpline(weights_bs1                , u_steps, k)
    bs2 = BSpline(weights_bs2                , u_steps, k)
    mbs = MBSpline([weights_bs1, weights_bs2], u_steps, k)

    @testset "MBSpline" begin
        @test mbs(us_trial) ==  [bs1.(u_trial), bs2(u_trial)]
    end

    @testset "calc_mbspline" begin
        @test calc_mbspline(mbs, us_trial) == [[bs1.(u_trial), bs2.(u_trial)]]
    end
end

@testset "solve_mbspline.jl"
    n_weights_con = 5
    k = 3
    alpha_d = 0.3
    mbs0 = initialize_mbs0(curves, n_weights_con, k, alpha_d)
    p = parameters_solve_mbspline_curves(curves, n_weights_con, k; alpha, kwargs...)
    u = trainable_parameters(mbs0, solve_u_steps = true)
 
    @testset "loss_solve_mbspline_curves" begin
        @show ForwardDiff.gradient(x -> loss_solve_msbpline_curves(x, p), u)
    end

    @testset "normalized_mean_square_error" begin
        @show MasterThesis.normalized_mean_square_error(mbs0, p)
    end

    @testset "normalized_mean_square_errors" begin
        @show MasterThesis.normalized_mean_square_errors(mbs0, p)
    end

    @testset "normalized_mean_square_error ForwardDiff" begin
        nmse(x) = normalized_mean_square_error(x, p)
        @show ForwardDiff.gradient()
    end

    @testset "normalized_mean_square_errors ForwardDiff" begin

    end


    begin "NMBSpline"
        weightss_nmbs1 = [
            [1, -2 ,3, -4, 5]
        ]
       nodess_nmbs1 = [
            [1,2,3,4,5,6,7,8,9],
            [1,2,3,4,5,6,7,8,9]
        ]

        weightss_nmbs2 = [
            [1 2 3 4 5;
             2 3 4 5 6;
             3 4 5 6 7;
             4 5 6 7 8;
             5 6 7 8 9],
            [1 -2 3 -4 5;
             -2 3 -4 5 -6;
             3 -4 5 -6 7;
             -4 5 -6 7 -8;
             5 -6 7 -8 9]
        ]
        nodess_nmbs2 = [
            [1,2,3,4,5,6,7,8,9],
            [1,2,3,4,5,6,7,8,9]
        ]
        degree = 3

        nmbs1 = NMBSpline()
        mbs_nmbs = MBSpline(degree, nodess_nmbs[1], weightss_nmbs[1])
    end
end

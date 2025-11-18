using SeeToDee
using Test
using StaticArrays
using ForwardDiff
# using NonlinearSolve
using FastGaussQuadrature
using JET

# This has to be defined outside of the testset
function cartesian_pendulum(U, inp, p, t)
    x,y,u,v,τ = U
    SA[u;
    v;
    - τ*x;
    - (τ*y - 1);
    x^2 + y^2 - 1]
end

function cartesian_pendulum(dU, U, inp, p, t)
    x,y,u,v,τ = U
    SA[u;
    v;
    - τ*x;
    - (τ*y - 1);
    x^2 + y^2 - 1] - [dU[1:4]; 0]
end

function cartpole(x, u, p, _=0)
    T = promote_type(eltype(x), eltype(u))
    mc, mp, l, g = 1.0, 0.2, 0.5, 9.81

    q  = x[SA[1, 2]]
    qd = x[SA[3, 4]]

    s = sin(q[2])
    c = cos(q[2])

    H = @SMatrix [mc+mp mp*l*c; mp*l*c mp*l^2]
    C = @SMatrix [0 -mp*qd[2]*l*s; 0 0]
    G = @SVector [0, mp * g * l * s]
    B = @SVector [1, 0]
    qdd = -H \ (C * qd + G - B * u[1])
    return [qd; qdd]::SVector{4, T}
end

function cartpole_implicit(dx, x, u, p, _=0)
    T = promote_type(eltype(x), eltype(u))
    mc, mp, l, g = 1.0, 0.2, 0.5, 9.81

    q  = x[SA[1, 2]]
    qd = x[SA[3, 4]]

    s = sin(q[2])
    c = cos(q[2])

    H = @SMatrix [mc+mp mp*l*c; mp*l*c mp*l^2]
    C = @SMatrix [0 -mp*qd[2]*l*s; 0 0]
    G = @SVector [0, mp * g * l * s]
    B = @SVector [1, 0]
    qdd = (C * qd + G - B * u[1])
    return [qd; -qdd] - [dx[SA[1, 2]]; H*dx[SA[3, 4]]]
end

@testset "SeeToDee.jl" begin
    n = 5
    N = 100
    Ts = 30/N
    discrete_dynamics = SimpleColloc(cartesian_pendulum, Ts, 4, 1, 0; n, abstol=1e-9, nodetype=gausslobatto)
    
    @test length(discrete_dynamics.τ) == n
    @test size(discrete_dynamics.D) == (n, n)
    
    θ0 = π/3
    x = [sin(θ0), -cos(θ0), 0, 0, 0.1]
    @inferred discrete_dynamics(x, 0, 0, 0)
    
    X = [x]
    for i = 2:N
        # @show i
        x = discrete_dynamics(x, 0, 0, 0)
        push!(X, x)
    end
    Xm = reduce(hcat, X)
    
    Xm_bench = [  0.884916  -0.998009   -0.288137   0.948136   -0.941448    0.344263   0.999548   -0.880882   0.812853   0.920481
    -0.465751  -0.0630642   0.957589  -0.317864   -0.337159    0.938873  -0.0300569  -0.473336   0.582469   0.390788
     0.121897   0.0589605   1.63498    0.191854    0.192427    1.59272    0.0291361   0.109384   0.857049  -0.521628
     0.231601  -0.932957    0.49198    0.572272   -0.537307   -0.583989   0.96907    -0.203565  -1.19603    1.22865
    -0.397285   0.811478    3.87114    0.0465319  -0.0112136   3.81449    0.910313   -0.419928   2.74741    2.1731]
    
    @test Xm[:, 2:10:end] ≈ Xm_bench atol=1e-5

    # Test that it's possible to differentiate through
    A = ForwardDiff.jacobian(x -> discrete_dynamics(x, 0, 0, 0), x)
    @test size(A) == (5, 5)
    B = ForwardDiff.jacobian(u -> discrete_dynamics(x, u, 0, 0), [0.0])
    @test size(B) == (5, 1)

    # Test residual formulation
    discrete_dynamics = SimpleColloc(cartesian_pendulum, Ts, 4, 1, 0; n, abstol=1e-7, residual=true, nodetype=gausslobatto)

    θ0 = π/3
    x = [sin(θ0), -cos(θ0), 0, 0, 0.1]
    @inferred discrete_dynamics(x, 0, 0, 0)
    
    X = [x]
    for i = 2:N
        # @show i
        x = discrete_dynamics(x, 0, 0, 0)
        push!(X, x)
    end
    Xm = reduce(hcat, X)
    @test Xm[:, 2:10:end] ≈ Xm_bench atol=1e-5
    

    # Cartpole

    n = 5
    discrete_dynamics = SeeToDee.SimpleColloc(cartpole, Ts, 4, 0, 1; n, abstol=1e-10, residual=false, scale_x=[1,2,3,4])#, solver=NonlinearSolve.NewtonRaphson())
    discrete_dynamics_implicit = SeeToDee.SimpleColloc(cartpole_implicit, Ts, 4, 0, 1; n, abstol=1e-10, residual=true, scale_x=[1,2,3,4])
    discrete_dynamics_rk = SeeToDee.Rk4(cartpole, Ts; supersample=3)
    discrete_dynamics_rk_ss = SeeToDee.Rk4(cartpole, Ts; supersample=200)
    discrete_dynamics_rk3 = SeeToDee.Rk3(cartpole, Ts; supersample=3)
    discrete_dynamics_fe = SeeToDee.ForwardEuler(cartpole, Ts; supersample=3)
    discrete_dynamics_heun = SeeToDee.Heun(cartpole, Ts; supersample=3)
    discrete_dynamics_trapz = SeeToDee.Trapezoidal(cartpole, Ts, 4, 0, 1; abstol=1e-10, residual=false, scale_x=[1,2,3,4])
    
    discrete_dynamics_backeuler = SeeToDee.SuperSampler(SeeToDee.BackwardEuler(cartpole, Ts, 4, 0, 1; abstol=1e-10, residual=false, scale_x=[1,2,3,4]), 5)
    discrete_dynamics_backeuler_implicit = SeeToDee.SuperSampler(SeeToDee.BackwardEuler(cartpole_implicit, Ts, 4, 0, 1; abstol=1e-10, residual=true, scale_x=[1,2,3,4]), 5)
    
    discrete_dynamics_backeuler_oop = SeeToDee.SuperSampler(SeeToDee.BackwardEuler(cartpole, Ts, 4, 0, 1; abstol=1e-10, residual=false, scale_x=[1,2,3,4], inplace=false), 5)
    discrete_dynamics_backeuler_implicit_oop = SeeToDee.SuperSampler(SeeToDee.BackwardEuler(cartpole_implicit, Ts, 4, 0, 1; abstol=1e-10, residual=true, scale_x=[1,2,3,4], inplace=false), 5)
    discrete_dynamics_rkc2 = SeeToDee.RKC2(cartpole, Ts; supersample=3)

    x = SA[1.0, 2.0, 3.0, 4.0]
    u = SA[1.0]

    @inferred discrete_dynamics(x, u, 0, 0)
    @inferred discrete_dynamics_implicit(x, u, 0, 0)
    @inferred discrete_dynamics_rk(x, u, 0, 0)
    @inferred discrete_dynamics_rk_ss(x, u, 0, 0)
    @inferred discrete_dynamics_rk3(x, u, 0, 0)
    @inferred discrete_dynamics_fe(x, u, 0, 0)
    @inferred discrete_dynamics_heun(x, u, 0, 0)
    @inferred discrete_dynamics_trapz(x, u, 0, 0)
    @inferred discrete_dynamics_backeuler(x, u, 0, 0)
    @inferred discrete_dynamics_backeuler_implicit(x, u, 0, 0)
    @inferred discrete_dynamics_rkc2(x, u, 0, 0)


    # @test_opt discrete_dynamics(x, u, 0, 0.0)
    # @test_opt discrete_dynamics_implicit(x, u, 0, 0.0)
    @test_opt discrete_dynamics_rk(x, u, 0, 0.0)
    @test_opt discrete_dynamics_rk_ss(x, u, 0, 0.0)
    @test_opt discrete_dynamics_rk3(x, u, 0, 0.0)
    @test_opt discrete_dynamics_fe(x, u, 0, 0.0)
    @test_opt discrete_dynamics_heun(x, u, 0, 0.0)
    @test_opt discrete_dynamics_rkc2(x, u, 0, 0.0)


    # @report_call discrete_dynamics_rk(x, u, 0, 0)
    # @report_call discrete_dynamics_rk_ss(x, u, 0, 0)
    # @report_call discrete_dynamics_rk3(x, u, 0, 0)
    # @report_call discrete_dynamics_fe(x, u, 0, 0)
    # @report_call discrete_dynamics_heun(x, u, 0, 0)

    # Test that the Static version is used despite input x being a normal vector
    @test discrete_dynamics_rk(Vector(x), u, 0, 0) isa SVector{4, Float64}
    @test discrete_dynamics_rk3(Vector(x), u, 0, 0) isa SVector{4, Float64}
    @test discrete_dynamics_fe(Vector(x), u, 0, 0) isa SVector{4, Float64}
    @test discrete_dynamics_heun(Vector(x), u, 0, 0) isa SVector{4, Float64}
    # @test discrete_dynamics_trapz(Vector(x), u, 0, 0) isa SVector{4, Float64}
    # @test discrete_dynamics_rkc2(Vector(x), u, 0, 0) isa SVector{4, Float64}

    x1 = discrete_dynamics(x, u, 0, 0)
    x2 = discrete_dynamics_implicit(x, u, 0, 0)
    x3 = discrete_dynamics_rk(x, u, 0, 0)
    x4 = discrete_dynamics_rk_ss(x, u, 0, 0)
    x5 = discrete_dynamics_rk3(x, u, 0, 0)
    x6 = discrete_dynamics_fe(x, u, 0, 0)
    x7 = discrete_dynamics_heun(x, u, 0, 0)
    x8 = discrete_dynamics_trapz(x, u, 0, 0)
    x9 = discrete_dynamics_rkc2(x, u, 0, 0)
    x10 = discrete_dynamics_backeuler(x, u, 0, 0)
    x11 = discrete_dynamics_backeuler_implicit(x, u, 0, 0)
    x12 = discrete_dynamics_backeuler_oop(x, u, 0, 0)
    x13 = discrete_dynamics_backeuler_implicit_oop(x, u, 0, 0)

    @test x1 ≈ x2 atol=1e-9
    @test x1 ≈ x3 atol=2e-3
    @test x1 ≈ x4 atol=1e-5
    @test x1 ≈ x5 atol=1e-2
    @test x1 ≈ x6 rtol=5e-2
    @test x1 ≈ x7 rtol=4e-2
    @test x1 ≈ x8 rtol=5e-2
    @test x1 ≈ x9 rtol=4e-3
    @test x1 ≈ x10 rtol=5e-2
    @test x1 ≈ x11 rtol=5e-2
    @test x1 ≈ x12 rtol=5e-2
    @test x1 ≈ x13 rtol=5e-2

    # using BenchmarkTools
    # @btime $discrete_dynamics($x, $u, 0, 0);
    # @btime $discrete_dynamics_implicit($x, $u, 0, 0); # Maybe a tiny improvement on this example
    # @btime $discrete_dynamics_rk($x, $u, 0, 0); # 200x faster, 346.237 ns (0 allocations: 0 bytes)
    # @btime $discrete_dynamics_rk3($x, $u, 0, 0); # 241.779 ns (0 allocations: 0 bytes)
    # @btime $discrete_dynamics_fe($x, $u, 0, 0); # 83.720 ns (0 allocations: 0 bytes)
    # @btime $discrete_dynamics_heun($x, $u, 0, 0); # 167.138 ns (0 allocations: 0 bytes)
    # @btime $discrete_dynamics_rkc2($x, $u, 0, 0); # 842.014 ns (0 allocations: 0 bytes)


    @testset "batch" begin
        @info "Testing batch"
        include("test_batch.jl")
    end


    @testset "time" begin
        @info "Testing time"
        include("test_time.jl")
    end

    @testset "linearization" begin
        @info "Testing linearization"
        include("test_linearization.jl")
    end

    @testset "SwitchingIntegrator" begin
        @info "Testing SwitchingIntegrator"

        Ts = 0.1
        x = SA[1.0, 2.0, 3.0, 4.0]
        u = SA[1.0]

        # Create two different integrators
        fast_integrator = SeeToDee.ForwardEuler(cartpole, Ts; supersample=1)
        accurate_integrator = SeeToDee.Rk4(cartpole, Ts; supersample=3)

        # Test condition: use fast integrator when norm(x) < 5, otherwise use accurate one
        cond = (x, u, p, t) -> norm(x) < 5.0

        switching_integrator = SeeToDee.SwitchingIntegrator(fast_integrator, accurate_integrator, cond)

        # Test that it's an AbstractIntegrator
        @test switching_integrator isa SeeToDee.AbstractIntegrator

        # Test when condition is true (norm(x) = ~5.48)
        x_small = SA[1.0, 1.0, 1.0, 1.0]  # norm = 2.0
        result_true = switching_integrator(x_small, u, 0, 0)
        expected_true = fast_integrator(x_small, u, 0, 0)
        @test result_true ≈ expected_true

        # Test when condition is false
        x_large = SA[10.0, 10.0, 10.0, 10.0]  # norm = 20.0
        result_false = switching_integrator(x_large, u, 0, 0)
        expected_false = accurate_integrator(x_large, u, 0, 0)
        @test result_false ≈ expected_false

        # Test type inference
        @inferred switching_integrator(x_small, u, 0, 0)
        @inferred switching_integrator(x_large, u, 0, 0)

        # Test that it properly forwards kwargs
        result_with_ts = switching_integrator(x_small, u, 0, 0; Ts=0.05)
        expected_with_ts = fast_integrator(x_small, u, 0, 0; Ts=0.05)
        @test result_with_ts ≈ expected_with_ts

        # Test with different condition (time-based switching)
        time_cond = (x, u, p, t) -> t < 0.5
        time_switching = SeeToDee.SwitchingIntegrator(fast_integrator, accurate_integrator, time_cond)

        result_early = time_switching(x, u, 0, 0.2)
        @test result_early ≈ fast_integrator(x, u, 0, 0.2)

        result_late = time_switching(x, u, 0, 1.0)
        @test result_late ≈ accurate_integrator(x, u, 0, 1.0)

        # Test with parameter-based condition
        param_cond = (x, u, p, t) -> p > 0
        param_switching = SeeToDee.SwitchingIntegrator(fast_integrator, accurate_integrator, param_cond)

        result_pos = param_switching(x, u, 1.0, 0)
        @test result_pos ≈ fast_integrator(x, u, 1.0, 0)

        result_neg = param_switching(x, u, -1.0, 0)
        @test result_neg ≈ accurate_integrator(x, u, -1.0, 0)

        # Test switching during simulation
        X = [x_small]
        for i = 1:10
            x_current = X[end]
            x_next = switching_integrator(x_current, u, 0, (i-1)*Ts)
            push!(X, x_next)
        end
        @test length(X) == 11
        @test all(x -> x isa SVector{4, Float64}, X)
    end

    @testset "AdaptiveStep" begin
        @info "Testing AdaptiveStep"

        # Test with explicit integrators (these support supersample and Ts keywords)
        Ts_base = 0.1
        discrete_rk4 = SeeToDee.Rk4(cartpole, Ts_base; supersample=2)
        discrete_rk3 = SeeToDee.Rk3(cartpole, Ts_base; supersample=2)
        discrete_fe = SeeToDee.ForwardEuler(cartpole, Ts_base; supersample=2)
        discrete_heun = SeeToDee.Heun(cartpole, Ts_base; supersample=2)
        
        # Create adaptive wrappers
        adaptive_rk4 = SeeToDee.AdaptiveStep(discrete_rk4)
        adaptive_rk3 = SeeToDee.AdaptiveStep(discrete_rk3)
        adaptive_fe = SeeToDee.AdaptiveStep(discrete_fe)
        adaptive_heun = SeeToDee.AdaptiveStep(discrete_heun)
        
        x = SA[1.0, 2.0, 3.0, 4.0]
        u = SA[1.0]
        
        # Test that AdaptiveStep correctly calculates largest_Ts
        @test adaptive_rk4.largest_Ts ≈ Ts_base / 2  # Since supersample=2
        @test adaptive_rk3.largest_Ts ≈ Ts_base / 2
        @test adaptive_fe.largest_Ts ≈ Ts_base / 2
        @test adaptive_heun.largest_Ts ≈ Ts_base / 2
        
        # Test that base step size works identically
        @test adaptive_rk4(x, u, 0, 0; Ts=Ts_base) ≈ discrete_rk4(x, u, 0, 0; Ts=Ts_base)
        @test adaptive_rk3(x, u, 0, 0; Ts=Ts_base) ≈ discrete_rk3(x, u, 0, 0; Ts=Ts_base)
        @test adaptive_fe(x, u, 0, 0; Ts=Ts_base) ≈ discrete_fe(x, u, 0, 0; Ts=Ts_base)
        @test adaptive_heun(x, u, 0, 0; Ts=Ts_base) ≈ discrete_heun(x, u, 0, 0; Ts=Ts_base)
        
        # Test with larger step sizes (should use supersample internally)
        Ts_large = 0.4  # 8x larger than largest_Ts (0.05)
        
        # Test that larger steps work (using internal supersample mechanism)
        x_adaptive_rk4 = adaptive_rk4(x, u, 0, 0; Ts=Ts_large)
        x_adaptive_rk3 = adaptive_rk3(x, u, 0, 0; Ts=Ts_large)
        x_adaptive_fe = adaptive_fe(x, u, 0, 0; Ts=Ts_large)
        x_adaptive_heun = adaptive_heun(x, u, 0, 0; Ts=Ts_large)
        
        @test x_adaptive_rk4 isa SVector{4, Float64}
        @test x_adaptive_rk3 isa SVector{4, Float64}
        @test x_adaptive_fe isa SVector{4, Float64}
        @test x_adaptive_heun isa SVector{4, Float64}
        
        # Test with smaller step sizes (should work without subdivision)
        Ts_small = 0.05  # Equal to largest_Ts
        @test adaptive_rk4(x, u, 0, 0; Ts=Ts_small) ≈ discrete_rk4(x, u, 0, 0; Ts=Ts_small, supersample=1)
        @test adaptive_rk3(x, u, 0, 0; Ts=Ts_small) ≈ discrete_rk3(x, u, 0, 0; Ts=Ts_small, supersample=1)
        
        # Test type inference
        @inferred adaptive_rk4(x, u, 0, 0; Ts=Ts_base)
        @inferred adaptive_rk3(x, u, 0, 0; Ts=Ts_base)
        @inferred adaptive_fe(x, u, 0, 0; Ts=Ts_base)
        @inferred adaptive_heun(x, u, 0, 0; Ts=Ts_base)
        
        # Test with very large step sizes
        Ts_verylarge = 1.0  # 20x larger than largest_Ts
        x_verylarge = adaptive_rk4(x, u, 0, 0; Ts=Ts_verylarge)
        @test x_verylarge isa SVector{4, Float64}
        
        # Test edge case: exactly matching largest_Ts
        @test adaptive_rk4(x, u, 0, 0; Ts=adaptive_rk4.largest_Ts) ≈ discrete_rk4(x, u, 0, 0; Ts=adaptive_rk4.largest_Ts, supersample=1)
        
        # Test with different supersample in base integrator
        discrete_rk4_ss = SeeToDee.Rk4(cartpole, 0.2; supersample=5)
        adaptive_rk4_ss = SeeToDee.AdaptiveStep(discrete_rk4_ss)
        @test adaptive_rk4_ss.largest_Ts ≈ 0.04  # 0.2/5
        
        # Test large step with supersampled base integrator
        result_ss = adaptive_rk4_ss(x, u, 0, 0; Ts=0.8)  # Should use supersample=20
        @test result_ss isa SVector{4, Float64}
        
        # Test that AdaptiveStep works correctly when Ts > largest_Ts
        # For a step that requires supersample=10 (0.4/0.04 = 10)
        result_large = adaptive_rk4_ss(x, u, 0, 0; Ts=0.4)
        @test result_large isa SVector{4, Float64}
        
        # Test consistency: same result for equivalent calls
        x1 = adaptive_rk4(x, u, 0, 0; Ts=0.2)  # Uses supersample=4
        x2 = discrete_rk4(x, u, 0, 0; Ts=0.2, supersample=4)
        @test x1 ≈ x2
    end

    @testset "SuperSampler" begin
        @info "Testing SuperSampler"

        # Test with implicit integrators that don't have built-in supersample
        Ts_base = 0.1
        discrete_trapz = SeeToDee.Trapezoidal(cartpole, Ts_base, 4, 0, 1; abstol=1e-10)
        discrete_backeuler = SeeToDee.BackwardEuler(cartpole, Ts_base, 4, 0, 1; abstol=1e-10)
        # discrete_colloc = SeeToDee.SimpleColloc(cartpole, Ts_base, 4, 0, 1; n=5, abstol=1e-10)

        # Create supersampled versions
        supersampled_trapz = SeeToDee.SuperSampler(discrete_trapz, 5)
        supersampled_backeuler = SeeToDee.SuperSampler(discrete_backeuler, 5)
        # supersampled_colloc = SeeToDee.SuperSampler(discrete_colloc, 5)

        x = SA[1.0, 2.0, 3.0, 4.0]
        u = SA[1.0]

        # Test that SuperSampler is an AbstractIntegrator
        @test supersampled_trapz isa SeeToDee.AbstractIntegrator
        @test supersampled_backeuler isa SeeToDee.AbstractIntegrator
        # @test supersampled_colloc isa SeeToDee.AbstractIntegrator

        # Test constructor validation
        @test_throws ArgumentError SeeToDee.SuperSampler(discrete_trapz, 0)
        @test_throws ArgumentError SeeToDee.SuperSampler(discrete_trapz, -1)

        # Test that supersampling produces different (hopefully better) results than single step
        x_single_trapz = discrete_trapz(x, u, 0, 0)
        x_super_trapz = supersampled_trapz(x, u, 0, 0)
        @test x_single_trapz != x_super_trapz  # Should be different

        # Test type inference
        @inferred supersampled_trapz(x, u, 0, 0)
        @inferred supersampled_backeuler(x, u, 0, 0)
        # @inferred supersampled_colloc(x, u, 0, 0)

        # Test that supersampling matches manual stepping
        x_manual = x
        Ts_step = Ts_base / 5
        t_manual = 0.0
        for i in 1:5
            x_manual = discrete_trapz(x_manual, u, 0, t_manual; Ts=Ts_step)
            t_manual += Ts_step
        end
        x_super = supersampled_trapz(x, u, 0, 0)
        @test x_manual ≈ x_super

        # Test with different supersample values
        supersampled_trapz_10 = SeeToDee.SuperSampler(discrete_trapz, 10)
        x_super_10 = supersampled_trapz_10(x, u, 0, 0)
        @test x_super_10 isa SVector{4, Float64}

        # Test that higher supersampling gives different results
        @test x_super_trapz != x_super_10

        # Test with custom Ts at call time
        x_custom_ts = supersampled_trapz(x, u, 0, 0; Ts=0.05)
        @test x_custom_ts isa SVector{4, Float64}
        # With Ts=0.05 and supersample=5, each internal step is 0.01
        x_manual_custom = x
        for i in 1:5
            x_manual_custom = discrete_trapz(x_manual_custom, u, 0, (i-1)*0.01; Ts=0.01)
        end
        @test x_custom_ts ≈ x_manual_custom

        # Test that supersampling improves accuracy for BackwardEuler
        # Create a reference with very fine stepping
        discrete_trapz_fine = SeeToDee.Trapezoidal(cartpole, Ts_base/100, 4, 0, 1; abstol=1e-10)
        x_ref = x
        for i in 1:100
            x_ref = discrete_trapz_fine(x_ref, u, 0, (i-1)*Ts_base/100; Ts=Ts_base/100)
        end

        # Single step should be less accurate than supersampled
        x_single_be = discrete_backeuler(x, u, 0, 0)
        x_super_be = SeeToDee.SuperSampler(discrete_backeuler, 10)(x, u, 0, 0)

        using LinearAlgebra
        err_single = norm(x_ref - x_single_be)
        err_super = norm(x_ref - x_super_be)

        # Supersampling should reduce error (though not guaranteed for all systems)
        @test err_super < err_single
    end

end


# Accuracy test
# using FastGaussQuadrature
# using OrdinaryDiffEq



# Ts = 0.8

# prob = ODEProblem((x,p,t)->cartpole(x, u, p, t), x, (0.0, Ts))
# sol = solve(prob, Vern9(), reltol=1e-15, abstol=1e-15) # 333.078 μs (14004 allocations: 558.35 KiB)
# xa = sol.u[end]

# discrete_dynamics_rk = SeeToDee.Rk4(cartpole, Ts; supersample=100)
# xrk = discrete_dynamics_rk(x, u, 0, 0) # 10.895 μs (0 allocations: 0 bytes)


# ##
# n = 11
# ns = 2:40
# err_l = Float64[]
# err_r = Float64[]
# for n in ns
#     discrete_dynamics_l = SeeToDee.SimpleColloc(cartpole, Ts, 4, 0, 1; n, abstol=1e-14, nodetype = gausslobatto)
#     discrete_dynamics_r = SeeToDee.SimpleColloc(cartpole, Ts, 4, 0, 1; n, abstol=1e-14, nodetype = gaussradau)
#     xl = discrete_dynamics_l(x, u, 0, 0)
#     xr = discrete_dynamics_r(x, u, 0, 0)
#     push!(err_l, norm(xa-xl))
#     push!(err_r, norm(xa-xr))
# end

# plot(ns, err_l, label="Gauss-Lobatto", yscale=:log10)
# plot!(ns, err_r, label="Gauss-Radau")
# hline!([norm(xa-xrk)], lab="RK4")
# display(current())


# ## with super sampling

# function supersample(f::F, n) where F
#     function (x, u, p, t)
#         for _ = 1:n
#             x = f(x, u, p, t)
#         end
#         x
#     end
# end

# n = 11
# ns = 2:20
# err_l = Float64[]
# err_r = Float64[]
# n_ss = 5
# for n in ns
#     discrete_dynamics_l = supersample(SeeToDee.SimpleColloc(cartpole, Ts/n_ss, 4, 0, 1; n, abstol=1e-14, nodetype = gausslobatto), n_ss)
#     discrete_dynamics_r = supersample(SeeToDee.SimpleColloc(cartpole, Ts/n_ss, 4, 0, 1; n, abstol=1e-14, nodetype = gaussradau), n_ss)
#     xl = discrete_dynamics_l(x, u, 0, 0)
#     xr = discrete_dynamics_r(x, u, 0, 0)
#     push!(err_l, norm(xa-xl))
#     push!(err_r, norm(xa-xr))
# end

# scatter(ns, err_l, label="Gauss-Lobatto", yscale=:log10, xlabel="Number of collocation points", ylabel="Accuracy")
# scatter!(ns, err_r, label="Gauss-Radau")
# hline!([norm(xa-xrk)], lab="RK4")
# display(current())



# Robertsons

# using StaticArrays
# function rob(x, u, p, t)
#     SA[
#         - 0.04x[1] + 1e4x[2] * x[3]
#         0.04x[1] - 1e4x[2] * x[3] - 3e7x[2]^2
#         3e7x[2]^2
#     ]
# end

# ##
# x0 = SA[
#     1.0, 1e-6, 1e-6
# ]

# X = [x0]
# T = [Ts]
# x = x0


# function sim(x, X, T)
#     Ts = 1e-5
#     drob = SeeToDee.SimpleColloc(rob, Ts, 3, 0, 0; n=5)
#     t = Ts
#     for i = 1:10000
#         x = drob(x, 0, 0, 0)
#         t += Ts
#         push!(X, x)
#         push!(T, t)
#     end

#     Ts = 1e-3
#     drob = SeeToDee.SimpleColloc(rob, Ts, 3, 0, 0; n=5)
#     for i = 1:10000
#         x = drob(x, 0, 0, 0)
#         t += Ts
#         push!(X, x)
#         push!(T, t)
#     end

#     Ts = 1e-0
#     drob = SeeToDee.SimpleColloc(rob, Ts, 3, 0, 0; n=5)
#     for i = 1:100000
#         x = drob(x, 0, 0, 0)
#         t += Ts
#         push!(X, x)
#         push!(T, t)
#     end
# end
# sim(x, X, T)

# Xm = reduce(hcat, X)'
# using Plots
# plot(T, Xm, layout=(3, 1), xscale=:log10, legend=false, xlabel="Time", ylabel="Concentration", title=["A" "B" "C"])
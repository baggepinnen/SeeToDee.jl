using SeeToDee
using Test
using StaticArrays
using ForwardDiff
# using NonlinearSolve
using FastGaussQuadrature

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
    discrete_dynamics = SeeToDee.SimpleColloc(cartpole, Ts, 4, 0, 1; n, abstol=1e-10, residual=false)#, solver=NonlinearSolve.NewtonRaphson())
    discrete_dynamics_implicit = SeeToDee.SimpleColloc(cartpole_implicit, Ts, 4, 0, 1; n, abstol=1e-10, residual=true)
    discrete_dynamics_rk = SeeToDee.Rk4(cartpole, Ts; supersample=3)
    discrete_dynamics_rk_ss = SeeToDee.Rk4(cartpole, Ts; supersample=200)

    x = SA[1.0, 2.0, 3.0, 4.0]
    u = SA[1.0]

    @inferred discrete_dynamics(x, u, 0, 0)
    @inferred discrete_dynamics_implicit(x, u, 0, 0)
    @inferred discrete_dynamics_rk(x, u, 0, 0)
    @inferred discrete_dynamics_rk_ss(x, u, 0, 0)

    # Test that the Static version is used despite input x being a normal vector
    @test discrete_dynamics_rk(Vector(x), u, 0, 0) isa SVector{4, Float64}

    x1 = discrete_dynamics(x, u, 0, 0)
    x2 = discrete_dynamics_implicit(x, u, 0, 0)
    x3 = discrete_dynamics_rk(x, u, 0, 0)
    x4 = discrete_dynamics_rk_ss(x, u, 0, 0)

    @test x1 ≈ x2 atol=1e-9
    @test x1 ≈ x3 atol=1e-2
    @test x1 ≈ x4 atol=1e-5

    # using BenchmarkTools
    # @btime $discrete_dynamics($x, $u, 0, 0);
    # @btime $discrete_dynamics_implicit($x, $u, 0, 0); # Maybe a tiny improvement on this example
    # @btime $discrete_dynamics_rk($x, $u, 0, 0); # 200x faster


    @testset "batch" begin
        @info "Testing batch"
        include("test_batch.jl")
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
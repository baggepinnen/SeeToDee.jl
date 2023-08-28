using SeeToDee
using Test
using StaticArrays
using ForwardDiff

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

@testset "SeeToDee.jl" begin
    n = 5
    N = 100
    Ts = 30/N
    discrete_dynamics = SimpleColloc(cartesian_pendulum, Ts, 4, 1; n, abstol=1e-7)
    
    @test length(discrete_dynamics.taupoints) == n
    @test size(discrete_dynamics.D) == (n, n)
    
    θ0 = π/3
    x = [sin(θ0), -cos(θ0), 0, 0, 0.1]
    discrete_dynamics(x, 0, 0, 0)
    
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


    # Test residual formulation
    discrete_dynamics = SimpleColloc(cartesian_pendulum, Ts, 4, 1; n, abstol=1e-7, residual=true)

    θ0 = π/3
    x = [sin(θ0), -cos(θ0), 0, 0, 0.1]
    discrete_dynamics(x, 0, 0, 0)
    
    X = [x]
    for i = 2:N
        # @show i
        x = discrete_dynamics(x, 0, 0, 0)
        push!(X, x)
    end
    Xm = reduce(hcat, X)
    @test Xm[:, 2:10:end] ≈ Xm_bench atol=1e-5
    
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

n = 5
discrete_dynamics = SimpleColloc(cartpole, Ts, 4, 0; n, abstol=1e-10, residual=false)
discrete_dynamics_implicit = SimpleColloc(cartpole_implicit, Ts, 4, 0; n, abstol=1e-10, residual=true)
discrete_dynamics_rk = rk4(cartpole, Ts; supersample=2)

x = SA[1.0, 2.0, 3.0, 4.0]
u = SA[1.0]


x1 = discrete_dynamics(x, u, 0, 0)
x2 = discrete_dynamics_implicit(x, u, 0, 0)
# x3 = discrete_dynamics_rk(x, u, 0, 0)

@test x1 ≈ x2 atol=1e-9
# @test x1 ≈ x3 atol=1e-2

# @btime $discrete_dynamics($x, $u, 0, 0);
# @btime $discrete_dynamics_implicit($x, $u, 0, 0); # Maybe a tiny improvement on this example
# @btime $discrete_dynamics_rk($x, $u, 0, 0); # 200x faster

using SeeToDee, LinearAlgebra

function dynamics(x, u, p, t)
    xp = p.A*x
    mul!(xp, p.B, u, 1, 1)
end

nx, nu = 10, 2
p = (A=randn(Float32, nx, nx), B=randn(Float32, nx, nu))

Ts = 0.05f0
N = 10000
x = randn(Float32, nx, N)
u = randn(Float32, nu, N)
t = 0
xd = dynamics(x, u, p, t)


discrete_dynamics = SeeToDee.Rk4(dynamics, Ts; supersample=2)
xp = discrete_dynamics(x, u, p, t)

@test xp[:,1] ≈ discrete_dynamics(x[:, 1], u[:, 1], p, t)
@test xp[:,end] ≈ discrete_dynamics(x[:, end], u[:, end], p, t)

# using BenchmarkTools
# @btime $discrete_dynamics($x, $u, $p, $t)


# using CUDA

# xgpu = cu(x)
# ugpu = cu(u)
# pgpu = cu(p)
# xpgpu = discrete_dynamics(xgpu, ugpu, pgpu, t)


# @btime $discrete_dynamics($xgpu, $ugpu, $pgpu, t)
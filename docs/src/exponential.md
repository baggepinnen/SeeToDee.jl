# Exponential Runge-Kutta Methods

Exponential Runge-Kutta (ETDRK) methods are designed for semilinear ODEs of the form

```math
\dot{x} = Lx + N(x, u, p, t)
```

where ``L`` is a constant linear operator (matrix) and ``N`` is the nonlinear remainder.
The key idea is to integrate the linear part **exactly** via the matrix exponential
``\exp(hL)``, while applying a Runge-Kutta-like correction for the nonlinear part.
This eliminates the stability restriction imposed by the stiff eigenvalues of ``L``.

SeeToDee provides three ETDRK variants:

| Method | Order | Reference |
|--------|-------|-----------|
| [`ETDRK2`](@ref SeeToDee.ETDRK2) | 2 | Cox & Matthews (2002) |
| [`ETDRK3`](@ref SeeToDee.ETDRK3) | 3 | Krogstad (2005) |
| [`ETDRK4`](@ref SeeToDee.ETDRK4) | 4 | Cox & Matthews (2002) |

All three share the same interface as every other integrator in this package:
`x_next = integrator(x, u, p, t)`.

## When to use ETDRK

Use ETDRK when:

- The system is **semilinear**: stiffness comes exclusively from a known constant linear
  operator ``L``, while the nonlinear part ``N`` is non-stiff.
- Classical explicit methods (Rk4, Rk3, …) require an impractically small step size to
  remain stable, because ``|\lambda_{\max}(L)| \cdot h`` exceeds the stability boundary.
- Implicit methods (Trapezoidal, BackwardEuler, SimpleColloc) work but their nonlinear
  solve overhead is unnecessary when stiffness is purely linear.

Typical examples: reaction-diffusion PDEs after spatial discretization, linearized fluid
dynamics, stiff chemical kinetics with a dominant linear decay term.

## Interface

```@example ETDRK_interface
using SeeToDee, LinearAlgebra

# Semilinear dynamics:  ẋ = L*x + N(x, u, p, t)
L_ex = [-10.0  1.0; -1.0 -10.0]   # stiff linear part

f_ex(x, u, p, t) = L_ex * x + [5*cos(t) + 0.1*x[2]^2;
                                 3*sin(t) - 0.1*x[1]^2]

Ts = 0.1
integ = SeeToDee.ETDRK4(f_ex, L_ex, Ts)

x0 = [1.0, 0.0]
integ(x0, Float64[], nothing, 0.0)
```

The matrix exponential and all φ-function matrices are **precomputed at construction**,
so repeated calls inside a simulation loop are cheap.

`supersample` is supported: `SeeToDee.ETDRK4(f, L, Ts; supersample=4)` divides
`Ts` into 4 sub-steps, each using the full ETDRK4 scheme.

## Stiffness demonstration

The test problem has eigenvalues ``-\lambda \pm i`` with ``\lambda = 10``.  RK4 is
stable on the real axis only when ``|\lambda h| \leq 2.79``; at ``T_s = 0.4`` we have
``|\lambda T_s| = 4.0``, so RK4 diverges immediately while ETDRK4 tracks the reference.

!!! details "Simulation and plot code"
    ```julia
    using SeeToDee, LinearAlgebra, Plots

    const λ = 10.0
    const L = [-λ 1.0; -1.0 -λ]

    dynamics(x, u, p, t) = L * x + [5*cos(t) + 0.1*x[2]^2;
                                      3*sin(t) - 0.1*x[1]^2]

    x0 = [1.0, 0.0]
    u  = Float64[]

    Ts_traj = 0.4
    N_traj  = 10
    t_traj  = range(0.0, N_traj * Ts_traj; length = N_traj + 1)

    ref_traj    = SeeToDee.Rk4(dynamics, Ts_traj; supersample=2000)
    rk4_traj    = SeeToDee.Rk4(dynamics, Ts_traj)
    etdrk4_traj = SeeToDee.ETDRK4(dynamics, L, Ts_traj)

    function simulate(integ, x0, Ts, N)
        X = Matrix{Float64}(undef, length(x0), N + 1)
        X[:, 1] = x0
        x = copy(x0)
        for k in 1:N
            x = integ(x, u, nothing, (k-1)*Ts)
            X[:, k+1] = x
        end
        X
    end

    X_ref    = simulate(ref_traj,    x0, Ts_traj, N_traj)
    X_rk4    = simulate(rk4_traj,    x0, Ts_traj, N_traj)
    X_etdrk4 = simulate(etdrk4_traj, x0, Ts_traj, N_traj)

    T_conv   = 1.0
    h_values = 10 .^ range(log10(0.004), log10(0.8); length=60)
    h_stab   = 2.79 / λ

    ref_conv  = SeeToDee.Rk4(dynamics, T_conv; supersample=50_000)
    x_ref_end = ref_conv(x0, u, nothing, 0.0)

    function final_error(integ, x_ref_end, x0, Ts, N)
        x = copy(x0)
        for k in 1:N
            x = integ(x, u, nothing, (k-1)*Ts)
            any(!isfinite, x) && return NaN
        end
        norm(x - x_ref_end)
    end

    e_rk4    = [final_error(SeeToDee.Rk4(dynamics, h),       x_ref_end, x0, h, round(Int, T_conv/h)) for h in h_values]
    e_etdrk4 = [final_error(SeeToDee.ETDRK4(dynamics, L, h), x_ref_end, x0, h, round(Int, T_conv/h)) for h in h_values]
    
    p1 = plot(; title = "x₁(t)  —  Ts = $(Ts_traj), |λ·Ts| = $(λ*Ts_traj) > 2.79",
    xlabel = "time  t", ylabel = "x₁", ylims = (-5, 5))
    plot!(p1, t_traj, X_ref[1, :];    color = :black,     lw = 2.5,              label = "Reference")
    plot!(p1, t_traj, clamp.(X_rk4[1, :], -5.0, 5.0); color = :red, lw = 2.0, ls = :dash, label = "RK4")
    plot!(p1, t_traj, X_etdrk4[1, :]; color = :royalblue, lw = 2.0, ls = :dash, label = "ETDRK4")

    p2 = plot(; title = "x₂(t)  —  Ts = $(Ts_traj), |λ·Ts| = $(λ*Ts_traj) > 2.79",
        xlabel = "time  t", ylabel = "x₂", ylims = (-5, 5), legend = false)
    plot!(p2, t_traj, X_ref[2, :];    color = :black,     lw = 2.5)
    plot!(p2, t_traj, clamp.(X_rk4[2, :], -5.0, 5.0); color = :red, lw = 2.0, ls = :dash)
    plot!(p2, t_traj, X_etdrk4[2, :]; color = :royalblue, lw = 2.0, ls = :dash)

    fig1 = plot(p1, p2; layout = (1, 2), size = (900, 400))

    rk4_mask = isfinite.(e_rk4)

    plot(; title = "Convergence: global error at T = $(T_conv) vs step size h",
        xlabel = "step size  h", ylabel = "‖x(T) − x_ref(T)‖",
        xscale = :log10, yscale = :log10, size = (800, 500), legend = :topleft)
    vspan!([h_stab], [maximum(h_values)]; color = :red, alpha = 0.08, label = nothing)
    vline!([h_stab]; color = :red, alpha = 0.5, ls = :dash, lw = 1.5, label = nothing)
    annotate!(h_stab * 1.2, 1e-1, text("RK4 unstable\n(|λ·h| > 2.79)", 8, :red))
    plot!(h_values[rk4_mask], e_rk4[rk4_mask]; color = :red,       lw = 2, ls = :dash, label = "RK4")
    fig2 = plot!(h_values,           e_etdrk4;         color = :royalblue, lw = 2, ls = :dash, label = "ETDRK4")
    ```



![fig1](https://github.com/user-attachments/assets/eac1b182-6926-4d17-8cce-0a2b1453fe00)

ETDRK4 tracks the reference closely while RK4 diverges within the first step (clipped at ±5).

![fig2](https://github.com/user-attachments/assets/6518a396-2b34-494e-a1b5-98c1c63a67f5)

Both methods are 4th-order accurate for small ``h``, but RK4 produces `NaN` once
``h > 2.79/\lambda`` (red shaded region) while ETDRK4 remains stable across the full range.


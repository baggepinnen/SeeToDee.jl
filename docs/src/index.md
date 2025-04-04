# SeeToDee

[![Build Status](https://github.com/baggepinnen/SeeToDee.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/baggepinnen/SeeToDee.jl/actions/workflows/CI.yml?query=branch%3Amain)


SeeToDee implements low-overhead, nonlinear variants of the classical [`c2d`](https://juliacontrol.github.io/ControlSystems.jl/dev/lib/synthesis/#ControlSystemsBase.c2d) function from [ControlSystems.jl](https://github.com/JuliaControl/ControlSystems.jl).

Given a continuous-time dynamics function
```math
\dot x = f(x, u, p, t)
```
this package contains *integrators* that convert the continuous-time dynamics into a discrete-time dynamics function
```math
x_{t+T_s} = f(x_t, u_t, p, t)
```
that advances the state from time ``t`` to time ``t+T_s``, with a [Zero-order-Hold (ZoH)](https://en.wikipedia.org/wiki/Zero-order_hold) assumption on the input ``u``.

The integrators in this package focus on
- **Inputs are first class**, i.e., the signature of the dynamics take input signals (such as control signals or disturbance inputs) as arguments. This is in contrast to the DifferentialEquations ecosystem, where there are [several different ways of handling inputs](https://help.juliahub.com/juliasimcontrol/dev/simulation/), none of which are quite first class.
- **Low overhead** for single-step integration, i.e., no solution handling, no interpolation, nothing fancy at all.
- **Fixed time step**. All integrators are non-adaptive, i.e., the integrators do not change their step size using error control. This typically makes the integrator have a more **predictable runtime**. It also reduces overhead without affecting accuracy in situations when the fixed step-size is small in relation to what would be required to meet the desired accuracy, a situation which is common when simulating control systems. It also means that the user is responsible for checking the accuracy for the chosen step size.
- **Dirt-simple interface**, i.e., you literally use the integrator as a function `x⁺ = f(x, u, p, t)` that you can call in a loop etc. to perform simulations.
- Most things are **manual**. Want to simulate a trajectory? Write a loop!



## Available discretization methods
The following methods are available
- [`SeeToDee.Rk4`](@ref) An explicit 4th order Runge-Kutta integrator with ZoH input. Supports non-stiff differential equations only. If called with StaticArrays, this method is allocation free.
- [`SeeToDee.Rk3`](@ref) An explicit 3rd order Runge-Kutta integrator with ZoH input. Supports non-stiff differential equations only. If called with StaticArrays, this method is allocation free.
- [`SeeToDee.Heun`](@ref) An explicit 2nd order Runge-Kutta integrator with ZoH input. Supports non-stiff differential equations only. If called with StaticArrays, this method is allocation free.
- [`SeeToDee.ForwardEuler`](@ref) An explicit 1st order Runge-Kutta integrator with ZoH input. Supports non-stiff differential equations only. If called with StaticArrays, this method is allocation free.
- [`SeeToDee.SimpleColloc`](@ref) A [textbook](https://www.equalsharepress.com/media/NMFSC.pdf) implementation of a direct collocation method with ZoH input. Supports stiff differential-algebraic equations (DAE) and fully implicit form $0 = F(ẋ, x, u, p, t)$.
- [`SeeToDee.Trapezoidal`](@ref) Trapezoidal integration with ZoH input. Supports stiff differential-algebraic equations (DAE).

See their docstrings for more details.


## When is this package useful?
Control systems are most of the time implemented in discrete time a computer, but they may interact with the continuous-time world around them. Simulation of the combined system including both the discrete-time controller and the continuous-time world is therefore a common task. The typically fixed sample interval of the controller, $T_s$, is thus presenting an upper bound to the length of the integration step in the simulation, an adaptive solver cannot take longer steps than this since the input is discountinuous with the invocation of the controller. If the sample time is short, a good adaptive algorithm will thus always take a step of length $T_s$, but figuring this out incurs unnecessary overhead. 

The invocation of the controller also means that some additional code, the implementation of the controller, has to be run at a fixed interval, but not in-between. The DifferentialEquations ecosystem provides _callbacks_ for this purpose. Making use of the callback is associated with a [very large amount of boilerplate](https://help.juliahub.com/juliasimcontrol/dev/simulation/#Use-of-a-discrete-controller-in-a-continuous-time-simulation) already for simple control-system simulations.

It's also common to want to do _more than just simulation_, for example, linearize the dynamics or state estimation. In both of those cases, one may be interested in integrating a single step of length $T_s$ only, starting from an arbitrary initial condition. In such situations, a one-liner function call is then preferable to a 10+ lines configuration of a traditional ODE solver, combined with the appropriate calls to `remake` etc. in-between calls.

### When should you not use this package?
- If inputs that change at a fixed interval is not an integral part of your dynamics.
- Your system requires a very special integrator for simulation.
- Your system is very large, e.g., a discretized PDE.
- If the sample interval $T_s$ is long in relation to the time constants of the system, then an adaptive solver may be more efficient despite the discontinuities.

## Example
The example below defines a dynamics function `cartpole` and then discretizes this using [`SeeToDee.Rk4`](@ref) and propagates the state forward one time step
```@example STEP
using SeeToDee, StaticArrays

function cartpole(x, u, p, t)
    T = promote_type(eltype(x), eltype(u))
    mc, mp, l, g = p

    q  = x[SA[1, 2]]
    qd = x[SA[3, 4]]

    s = sin(q[2])
    c = cos(q[2])

    H = @SMatrix [mc+mp mp*l*c; mp*l*c mp*l^2]
    C = @SMatrix [0 -mp*qd[2]*l*s; 0 0]
    G = @SVector [0, mp * g * l * s]
    B = @SVector [1, 0]
    qdd = -H \ (C * qd + G - B * u[1])
    return [qd; qdd]
end

Ts = 0.01
discrete_dynamics_rk = SeeToDee.Rk4(cartpole, Ts; supersample=2)

x0 = SA[1.0, 2.0, 3.0, 4.0]
u0 = SA[1.0]
p = mc, mp, l, g = 1.0, 0.2, 0.5, 9.81

x1_rk4 = discrete_dynamics_rk(x0, u0, p, 0)
```

The function `discrete_dynamics_rk` is now the discretized dynamics $x^+ = f(x, u, p, t)$, and `x1_rk4` is the state at time $0+T_s$. 

Next, we do the same but with [`SeeToDee.SimpleColloc`](@ref) instead of [`SeeToDee.Rk4`](@ref).
```@example STEP
n  = 5 # Number of collocation points
nx = 4 # Number of differential state variables
na = 0 # Number of algebraic variables
nu = 1 # Number of inputs

discrete_dynamics_colloc = SeeToDee.SimpleColloc(cartpole, Ts, nx, na, nu; n, abstol=1e-10)
x1_colloc = discrete_dynamics_colloc(x0, u0, p, 0)

using Test
@test x1_rk4 ≈ x1_colloc atol=1e-2 # Test the it's roughly the same as the output of RK4
```

If we benchmark these two methods
```julia
@btime $discrete_dynamics_rk($x0, $u0, $p, 0);     # 203.633 ns (0 allocations: 0 bytes)
@btime $discrete_dynamics_colloc($x0, $u0, $p, 0); # 22.072 μs (80 allocations: 50.23 KiB)
```
the explicit RK4 method is *much* faster in this case.

### Using fully implicit dynamics
Below, we define the same dynamics as above, but this time in the fully implicit form
```math
0 = F(ẋ, x, u, p, t)
```
This is occasionally useful in order to avoid inverting large coordinate-dependent mass matrices for mechanical systems etc. In the `cartpole` function, the mass matrix is called `H`, and it's of size 2×2

```@example STEP
function cartpole_implicit(dx, x, u, p, _=0)
    mc, mp, l, g = 1.0, 0.2, 0.5, 9.81

    q  = x[SA[1, 2]]
    qd = x[SA[3, 4]]

    s = sin(q[2])
    c = cos(q[2])

    H = @SMatrix [mc+mp mp*l*c; mp*l*c mp*l^2]
    C = @SMatrix [0 -mp*qd[2]*l*s; 0 0]
    G = @SVector [0, mp * g * l * s]
    B = @SVector [1, 0]
    Hqdd = (C * qd + G - B * u[1]) # Acceleration times mass matrix H
    return [qd; -Hqdd] - [dx[SA[1, 2]]; H*dx[SA[3, 4]]] # We multiply H here instead of inverting H like above
end

discrete_dynamics_implicit = SimpleColloc(cartpole_implicit, Ts, nx, na, nu; n, abstol=1e-10, residual=true)

x1_implicit = discrete_dynamics_implicit(x0, u0, p, 0)

@test x1_implicit ≈ x1_colloc atol=1e-9
```

```julia
@btime $discrete_dynamics_implicit($x0, $u0, 0, 0); # 21.911 μs (84 allocations: 50.39 KiB)
```
For this system, the solve time is almost identical to the explicit collocation case, but for larger systems, the implicit form can be faster.

## Simulate whole trajectories
Simulation is done by implementing the loop manually, for example (pseudocode)
```julia
discrete_dynamics = SeeToDee.Rk4(cartpole, Ts; supersample=2)

x = x0
X = [x]
U = []
for i = 1:T
    u = compute_control_input(x)      # State feedback, MPC, etc.
    x = discrete_dynamics(x, u, p, t) # Propagate state forward one step
    push!(X, x) # Log data
    push!(U, u)
end
```


## Batch propagation and GPU support
Using [`SeeToDee.Rk4`](@ref), it's possible to propagate a batch of `N` states forward in time by writing the dynamics to accept matrices `x` and `u` where `size(x) = (nx, N)`. No further changes are required. This also works if `x, u` are GPU arrays, such as those from CUDA.jl. For best performance on the GPU, make sure to construct the `Rk4` object using a sample time of type `Float32` and make sure that your `x, u` are also of element type `Float32`. `N` has to be rather large for this to be worthwhile. Propagating the state in batches can be useful when implementing, e.g., a particle filter.


## Usage in the wild
This package is used in the following places
- [DiscretePIDs.jl](https://github.com/JuliaControl/DiscretePIDs.jl#example-using-seetodee)
- [LowLevelParticleFilters.jl](https://github.com/baggepinnen/LowLevelParticleFilters.jl), see [tutorial](https://juliahub.com/ui/Notebooks/fredrik-carlson2/controlsystems/dae_stateest.jl).
- [ODE parameter calibration (video)](https://youtu.be/GKl8Tz9n2gs?si=V4ubuJsRTtSjgVxb)
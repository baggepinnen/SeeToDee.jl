# SeeToDee

[![Build Status](https://github.com/baggepinnen/SeeToDee.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/baggepinnen/SeeToDee.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Documentation, stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://baggepinnen.github.io/SeeToDee.jl/stable)
[![Documentation, latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://baggepinnen.github.io/SeeToDee.jl/dev)


SeeToDee implements low-overhead, nonlinear variants of the classical [`c2d`](https://juliacontrol.github.io/ControlSystems.jl/dev/lib/synthesis/#ControlSystemsBase.c2d-Tuple{AbstractStateSpace{%3C:Continuous},%20AbstractMatrix,%20Real}) function from [ControlSystems.jl](https://github.com/JuliaControl/ControlSystems.jl).

Given a continuous-time dynamics function
```math
\dot x = f(x, u, p, t)
```
this package contains *integrators* that convert the continuous-time dynamics into a discrete-time dynamics function
```math
x_{t+T_s} = f(x_t, u_t, p, t)
```
that advances the state from time $t$ to time $t+T_s$, with a [Zero-order-Hold (ZoH)](https://en.wikipedia.org/wiki/Zero-order_hold) assumption on the input $u$.

The integrators in this package focus on
- **Low overhead** for single-step integration, i.e., no solution handling, no interpolation, nothing fancy at all.
- **Fixed time step**. All integrators are non-adaptive, i.e., the integrators do not change their step size using error control. This typically makes the integrator have a more **predictable runtime**. It also reduces overhead without affecting accuracy in situations when the fixed step-size is small in relation to what would be required to meet the desired accuracy.
- **Dirt-simple interface**, i.e., you literally use the integrator as a function `x⁺ = f(x, u, p, t)` that you can call in a loop etc. to perform simulations.
- **Inputs are first class**, i.e., the signature of the dynamics take input signals (such as controlled inputs or disturbance inputs) as arguments. This is in contrast to the DifferentialEquations ecosystem, where there are [several different ways of handling inputs](https://help.juliahub.com/juliasimcontrol/dev/simulation/), none of which are first class.
- Most things are **manual**. Want to simulate a trajectory? Write a loop!




## Available methods
The following methods are available
- `SeeToDee.Rk4` A 4th order Runge-Kutta integrator with ZoH input. Supports differential equations only. If called with StaticArrays, this method is allocation free.
- `SeeToDee.SimpleColloc` A [textbook](https://www.equalsharepress.com/media/NMFSC.pdf) implementation of a direct collocation method (includes trapezoidal integration as a special case) with ZoH input. Supports differential-algebraic equations (DAE) and fully implicit form `x⁺ = f(ẋ, x, u, p, t)`.


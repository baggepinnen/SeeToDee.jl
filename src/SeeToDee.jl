module SeeToDee

using FastGaussQuadrature, SimpleNonlinearSolve, PreallocationTools, LinearAlgebra, ForwardDiff, StaticArrays

export SimpleColloc, AdaptiveStep
# public Rk4, Rk3, ForwardEuler, Heun, Trapezoidal

abstract type AbstractIntegrator <: Function end


"""
    A,B = linearize(f, x0, u0, p, t)

Linearize dynamics function `f(x, u, p, t)` w.r.t., state `x`, input `u`. Returns Jacobians `A,B` in
```math
ẋ = A\\, Δx + B\\, Δu
```
Works for both continuous and discrete-time dynamics.
"""
function linearize(f, x, u, args...)
    A = ForwardDiff.jacobian(x->f(x, u, args...), x)
    if u isa SVector
        B = ForwardDiff.jacobian(u->f(x, u, args...), u)
    else
        B = ForwardDiff.jacobian(u->f(convert(typeof(u), x), u, args...), u)
    end
    A, B
end

"""
    AdaptiveStep(integrator)

A wrapper that enables automatic step subdivision for taking arbitrary-length steps with any integrator.

When the requested step size `Ts` is larger than the integrator's effective step size (`largest_Ts`), 
`AdaptiveStep` automatically subdivides the step using the integrator's internal supersample mechanism 
(for explicit integrators) or manual stepping (for implicit integrators).

# Fields
- `integ`: The wrapped integrator
- `largest_Ts`: The largest step size the integrator can take in a single call (`Ts / supersample`)

# Usage
```julia
# Wrap any integrator to enable automatic step subdivision
base_integrator = Rk4(dynamics, 0.1; supersample=2)  # largest_Ts = 0.05
adaptive_integrator = AdaptiveStep(base_integrator)

# Take arbitrary step sizes - automatically subdivides when needed
x_next = adaptive_integrator(x, u, p, t; Ts=0.3)  # Uses supersample=6 internally
```

# Notes
- This wrapper does NOT use error control - it only ensures step sizes never exceed `largest_Ts`
- For explicit integrators (`Rk4`, `Rk3`, `ForwardEuler`, `Heun`), uses built-in supersample mechanism
- For implicit integrators (`SimpleColloc`, `Trapezoidal`), performs manual step subdivision
- When `Ts ≤ largest_Ts`, calls the integrator directly without subdivision

# Examples
```julia
using SeeToDee, StaticArrays

# Define dynamics
function simple_dynamics(x, u, p, t)
    return -x + u
end

# Create base integrator with supersample=3
base = SeeToDee.Rk4(simple_dynamics, 0.1; supersample=3)  # largest_Ts = 0.1/3 ≈ 0.033

# Wrap with AdaptiveStep
adaptive = SeeToDee.AdaptiveStep(base)

x0 = SA[1.0]
u = SA[0.5]

# Small step - no subdivision needed
x1 = adaptive(x0, u, 0, 0; Ts=0.02)  # Direct call

# Large step - automatic subdivision
x2 = adaptive(x0, u, 0, 0; Ts=0.5)   # Uses supersample=15 internally
```
"""
struct AdaptiveStep{I,T} <: AbstractIntegrator
    integ::I
    largest_Ts::T
end

AdaptiveStep(integ) = AdaptiveStep(integ, integ.Ts/(hasproperty(integ, :supersample) ? integ.supersample : 1.0))

function (as::AdaptiveStep)(x, u, p, t, args...; Ts=as.integ.Ts, kwargs...)
    if Ts > as.largest_Ts
        supersample = ceil(Int, Ts / as.largest_Ts)
        if hasproperty(as.integ, :supersample)
            # In this case we turn supersampling off, since this is already taken into account in the largest Ts
            return as.integ(x, u, p, t, args...; Ts, supersample, kwargs...)
        else
            new_Ts = as.largest_Ts / supersample
            for i in 1:supersample-1
                x = as.integ(x, u, p, t, args...; Ts=new_Ts, kwargs...)
                t += new_Ts
            end
            return x
        end
    else
        if hasproperty(as.integ, :supersample)
            # In this case we turn supersampling off, since this is already taken into account in the largest Ts
            return as.integ(x, u, p, t, args...; Ts, supersample=1, kwargs...)
        else
            return as.integ(x, u, p, t, args...; Ts, kwargs...)
        end
    end
end

## SwitchingIntegrator =========================================================
"""
    SwitchingIntegrator(int_true, int_false, cond)

Create an integrator that switches between two different integrators based on a condition.
- `int_true`: Integrator to use when `cond(...)` is true
- `int_false`: Integrator to use when `cond(...)` is false
- `cond`: A function that takes the same arguments as the integrator and returns a `Bool`

This can be used to, e.g., use a faster integrator when the state is in a certain region and a more accurate (but slower) integrator otherwise.
"""
struct SwitchingIntegrator{D,S,C} <: AbstractIntegrator
    dyn::D
    ss::S
    cond::C
end

function (DI::SwitchingIntegrator)(args...; kwargs...)
    if DI.cond(args...)
        return DI.dyn(args...; kwargs...)
    else
        return DI.ss(args...; kwargs...)
    end
end

## RK4 =========================================================================

"""
    f_discrete = Rk4(f, Ts; supersample = 1)

Discretize a continuous-time dynamics function `f` using RK4 with sample time `Tₛ`. 
`f` is assumed to have the signature `f : (x,u,p,t)->ẋ` and the returned function `f_discrete : (x,u,p,t)->x(t+Tₛ)`.

`supersample` determines the number of internal steps, 1 is often sufficient, but this can be increased to make the integration more accurate. `u` is assumed constant during all steps.

If called with StaticArrays, this integrator is allocation free.
"""
struct Rk4{F,TS} <: AbstractIntegrator
    f::F
    Ts::TS
    supersample::Int
    function Rk4(f::F, Ts; supersample::Integer = 1) where {F}
        supersample ≥ 1 || throw(ArgumentError("supersample must be positive."))
        new{F, typeof(Ts / 1)}(f, Ts / 1, supersample) # Divide by one to floatify ints
    end
end


function (integ::Rk4{F})(x, u, p, t, args...; Ts=integ.Ts, supersample=integ.supersample) where F
    f = integ.f
    f1 = f(x, u, p, t, args...)
    _inner_rk4(integ, f1, x, u, p, t, args...; Ts, supersample) # Dispatch depending on return type of dynamics
end


function _inner_rk4(integ::Rk4{F}, f1::SArray, x, u, p, t, args...; Ts=integ.Ts, supersample=integ.supersample) where F
    Ts2 = Ts / supersample
    f = integ.f
    f2 = f(x + Ts2 / 2 * f1, u, p, t + Ts2 / 2, args...)
    f3 = f(x + Ts2 / 2 * f2, u, p, t + Ts2 / 2, args...)
    f4 = f(x + Ts2 * f3, u, p, t + Ts2, args...)
    add = Ts2 / 6 .* (f1 .+ 2 .* f2 .+ 2 .* f3 .+ f4)
    # This gymnastics with changing the name to y is to ensure type stability when x + add is not the same type as x. The compiler is smart enough to figure out the type of y
    y = x + add
    for i in 2:supersample
        t += Ts2
        f1 = f(y, u, p, t, args...)
        f2 = f(y + Ts2 / 2 * f1, u, p, t + Ts2 / 2, args...)
        f3 = f(y + Ts2 / 2 * f2, u, p, t + Ts2 / 2, args...)
        f4 = f(y + Ts2 * f3, u, p, t + Ts2, args...)
        add = Ts2 / 6 .* (f1 .+ 2 .* f2 .+ 2 .* f3 .+ f4)
        y += add
    end
    return y
end

function _inner_rk4(integ::Rk4{F}, f1, x, u, p, t, args...; Ts=integ.Ts, supersample=integ.supersample) where F
    Ts2 = Ts / supersample
    f = integ.f
    xi = x .+ (Ts2 / 2) .* f1
    f2 = f(xi, u, p, t + Ts2 / 2, args...)
    xi .= x .+ (Ts2 / 2) .* f2
    f3 = f(xi, u, p, t + Ts2 / 2, args...)
    xi .= x .+ Ts2 .* f3
    f4 = f(xi, u, p, t + Ts2, args...)
    xi .= (Ts2 / 6) .* (f1 .+ 2 .* f2 .+ 2 .* f3 .+ f4)
    # This gymnastics with changing the name to y is to ensure type stability when x + add is not the same type as x. The compiler is smart enough to figure out the type of y
    y = _mutable(x + xi) # If x is non-static but xi isn't, we get a static array and adding to y won't work
    for i in 2:supersample
        t += Ts2
        f1 = f(y, u, p, t, args...)
        xi = y .+ (Ts2 / 2) .* f1
        f2 = f(xi, u, p, t + Ts2 / 2, args...)
        xi .= y .+ (Ts2 / 2) .* f2
        f3 = f(xi, u, p, t + Ts2 / 2, args...)
        xi .= y .+ Ts2 .* f3
        f4 = f(xi, u, p, t + Ts2, args...)
        xi .= (Ts2 / 6) .* (f1 .+ 2 .* f2 .+ 2 .* f3 .+ f4)
        y .+= xi
    end
    return y
end

_mutable(x::StaticArray) = Array(x)
_mutable(x::AbstractArray) = x

## RK3 ==========================================================================
"""
    f_discrete = Rk3(f, Ts; supersample = 1)

Discretize a continuous-time dynamics function `f` using RK3 with sample time `Tₛ`.
`f` is assumed to have the signature `f : (x,u,p,t)->ẋ` and the returned function `f_discrete : (x,u,p,t)->x(t+Tₛ)`.

`supersample` determines the number of internal steps, 1 is often sufficient, but this can be increased to make the integration more accurate. `u` is assumed constant during all steps.

If called with StaticArrays, this integrator is allocation free.
"""
struct Rk3{F,TS} <: AbstractIntegrator
    f::F
    Ts::TS
    supersample::Int
    function Rk3(f::F, Ts; supersample::Integer = 1) where {F}
        supersample ≥ 1 || throw(ArgumentError("supersample must be positive."))
        new{F, typeof(Ts / 1)}(f, Ts / 1, supersample) # Divide by one to floatify ints
    end
end


function (integ::Rk3{F})(x, u, p, t, args...; Ts=integ.Ts, supersample=integ.supersample) where F
    f = integ.f
    f1 = f(x, u, p, t, args...)
    _inner_rk3(integ, f1, x, u, p, t, args...; Ts, supersample) # Dispatch depending on return type of dynamics
end

function _inner_rk3(integ::Rk3{F}, f1, x, u, p, t, args...; Ts=integ.Ts, supersample=integ.supersample) where F
    Ts2 = Ts / supersample
    f = integ.f
    f2 = f(x + Ts2 / 2 * f1, u, p, t + Ts2 / 2, args...)
    f3 = f(x - Ts2 * f1 .+ 2 * Ts2 * f2, u, p, t + Ts2, args...)
    add = Ts2 / 6 .* (f1 .+ 4 .* f2 .+ f3)
    # This gymnastics with changing the name to y is to ensure type stability when x + add is not the same type as x. The compiler is smart enough to figure out the type of y
    y = x + add
    for i in 2:supersample
        t += Ts2
        f1 = f(y, u, p, t, args...)
        f2 = f(y + Ts2 / 2 * f1, u, p, t + Ts2 / 2, args...)
        f3 = f(y - Ts2 * f1 .+ 2 * Ts2 * f2, u, p, t + Ts2, args...)
        add = Ts2 / 6 .* (f1 .+ 4 .* f2 .+ f3)
        y += add
    end
    return y
end



## Forward Euler ===============================================================
"""
    f_discrete = ForwardEuler(f, Ts; supersample = 1)

Discretize a continuous-time dynamics function `f` using forward Euler with sample time `Tₛ`. 
`f` is assumed to have the signature `f : (x,u,p,t)->ẋ` and the returned function `f_discrete : (x,u,p,t)->x(t+Tₛ)`.

`supersample` determines the number of internal steps, this can be increased to make the integration more accurate, but it might be favorable to choose a higher-order method instead. `u` is assumed constant during all steps.

If called with StaticArrays, this integrator is allocation free.
"""
struct ForwardEuler{F,TS} <: AbstractIntegrator
    f::F
    Ts::TS
    supersample::Int
    function ForwardEuler(f::F, Ts; supersample::Integer = 1) where {F}
        supersample ≥ 1 || throw(ArgumentError("supersample must be positive."))
        new{F, typeof(Ts / 1)}(f, Ts / 1, supersample) # Divide by one to floatify ints
    end
end


function (integ::ForwardEuler{F})(x, u, p, t, args...; Ts=integ.Ts, supersample=integ.supersample) where F
    f = integ.f
    f1 = f(x, u, p, t, args...)
    _inner_forwardeuler(integ, f1, x, u, p, t, args...; Ts, supersample) # Dispatch depending on return type of dynamics
end


function _inner_forwardeuler(integ::ForwardEuler{F}, f1, x, u, p, t, args...; Ts=integ.Ts, supersample=integ.supersample) where F
    Ts2 = Ts / supersample
    f = integ.f
    add = Ts2 .* f1
    # This gymnastics with changing the name to y is to ensure type stability when x + add is not the same type as x. The compiler is smart enough to figure out the type of y
    y = x + add
    for i in 2:supersample
        t += Ts2
        f2 = f(y, u, p, t, args...)
        add = Ts2 .* f2
        y += add
    end
    return y
end


## Heun #######=================================================================

"""
    f_discrete = Heun(f, Ts; supersample = 1)

Discretize a continuous-time dynamics function `f` using Heun's method with sample time `Tₛ`.
`f` is assumed to have the signature `f : (x,u,p,t)->ẋ` and the returned function `f_discrete : (x,u,p,t)->x(t+Tₛ)`.

`supersample` determines the number of internal steps, this can be increased to make the integration more accurate, but it might be favorable to choose a higher-order method instead. `u` is assumed constant during all steps.

If called with StaticArrays, this integrator is allocation free.
"""
struct Heun{F,TS} <: AbstractIntegrator
    f::F
    Ts::TS
    supersample::Int
    function Heun(f::F, Ts; supersample::Integer = 1) where {F}
        supersample ≥ 1 || throw(ArgumentError("supersample must be positive."))
        new{F, typeof(Ts / 1)}(f, Ts / 1, supersample) # Divide by one to floatify ints
    end
end

function (integ::Heun{F})(x, u, p, t, args...; Ts=integ.Ts, supersample=integ.supersample) where F
    f = integ.f
    f1 = f(x, u, p, t, args...)
    _inner_heun(integ, f1, x, u, p, t, args...; Ts, supersample) # Dispatch depending on return type of dynamics
end

function _inner_heun(integ::Heun{F}, f1, x, u, p, t, args...; Ts=integ.Ts, supersample=integ.supersample) where F
    Ts2 = Ts / supersample
    f = integ.f
    f2 = f(x + Ts2 * f1, u, p, t + Ts2, args...)
    add = (Ts2 / 2) .* (f1 .+ f2)
    y = x + add
    for i in 2:supersample
        t += Ts2
        f1 = f(y, u, p, t, args...)
        f2 = f(y .+ Ts2 .* f1, u, p, t + Ts2, args...)
        add = (Ts2 / 2) .* (f1 .+ f2)
        y += add
    end
    return y
end

## =============================================================================

function simple_mul!(C, A, B)
    @inbounds @fastmath for m ∈ axes(A,1), n ∈ axes(B,2)
        Cmn = zero(eltype(C))
        for k ∈ axes(A,2)
            Cmn += A[m,k] * B[k,n]
        end
        C[m,n] = Cmn
    end
end

simple_mul!(C::AbstractMatrix{Any}, A, B) = mul!(C,A,B)

function diffoperator(n, Ts::T, nodetype::typeof(gausslobatto)) where T
    τ, w = nodetype(n+1)
    τ = (big.(τ)[2:end] .+ 1) ./ 2
    A = τ.^(0:n-1)'.*(1:n)'
    B = τ.^(1:n)'
    T.((A/B) ./ Ts), T.(τ), w
end

function diffoperator(n, Ts::T, ::typeof(gaussradau) = gaussradau) where T
    τ, w = gaussradau(n)
    τ = reverse(-τ) # We reverse the nodes to get the representation that includes the end-point rather than the initial point
    τ = (big.(τ) .+ 1) ./ 2
    A = τ.^(0:n-1)'.*(1:n)'
    B = τ.^(1:n)'
    T.((A/B) ./ Ts), T.(τ), w
end

struct SimpleColloc{F,T,X,A,DT,TP,CT,NP,S,SX} <: AbstractIntegrator
    dyn::F
    Ts::T
    nx::Int
    x_inds::X
    a_inds::A
    nu::Int
    D::DT
    τ::TP
    abstol::Float64
    cache::CT
    nlproblem::NP
    solver::S
    residual::Bool
    scale_x::SX
end

function get_cache!(integ::SimpleColloc, x::AbstractArray{T}) where T#::Tuple{Vector{T}, Matrix{T}, Matrix{T}, Matrix{T}} where T
    return get_tmp(integ.cache[1], x), get_tmp(integ.cache[2], x), get_tmp(integ.cache[3], x), get_tmp(integ.cache[4], x)
end


"""
    SimpleColloc(dyn, Ts, nx, na, nu; n = 5, abstol = 1.0e-8, solver=SimpleNewtonRaphson(), residual=false)
    SimpleColloc(dyn, Ts, x_inds, a_inds, nu; n = 5, abstol = 1.0e-8, solver=SimpleNewtonRaphson(), residual=false)

A simple direct-collocation integrator that can be stepped manually, similar to the function returned by [`SeeToDee.Rk4`](@ref).

This integrator supports differential-algebraic equations (DAE), the dynamics is expected to be on either of the forms 
- `nx,na` provided: `(xz,u,p,t)->[ẋ; res]` where `xz` is a vector `[x; z]` contaning the differential state `x` and the algebraic variables `z` in this order. `res` is the algebraic residuals, and `u` is the control input. The algebraic residuals are thus assumed to be the last `na` elements of of the arrays returned by the dynamics (the convention used by ModelingToolkit).
- `x_inds, a_inds` provided: `(xz,u,p,t)->xzd` where `xzd[x_inds] = ẋ` and `xzd[a_inds] = res`.

The returned function has the signature `f_discrete : (x,u,p,t)->x(t+Tₛ)`. 

This integrator also supports a fully implicit form of the dynamics
```math
0 = F(ẋ, x, u, p, t)
```
When using this interface, the dynamics is called using an additional input `ẋ` as the first argument, and the return value is expected to be the residual of the entire state descriptor. To use the implicit form, pass `residual = true`.


A Gauss-Radau collocation method is used to discretize the dynamics. The resulting nonlinear problem is solved using (by default) a Newton-Raphson method. This method handles stiff dynamics.


# Arguments:
- `dyn`: Dynamics function (continuous time)
- `Ts`: Sample time
- `nx`: Number of differential state variables
- `na`: Number of algebraic variables
- `x_inds, a_inds`: If indices are provided instead of `nx` and `na`, the mass matrix is assumed to be diagonal, with ones located at `x_inds` and zeros at `a_inds`. For maximum efficiency, provide these indices as unit ranges or static arrays.
- `nu`: Number of inputs
- `n`: Number of collocation points. `n=2` corresponds to trapezoidal integration.
- `abstol`: Tolerance for the root finding algorithm
- `residual`: If `true` the dynamics function is assumed to return the residual of the entire state descriptor and have the signature `(ẋ, x, u, p, t) -> res`. This is sometimes called "fully implicit form".
- `solver`: Any compatible SciML Nonlinear solver to use for the root finding problem
- `scale_x`: If provided, the state variables are scaled by this vector before being passed to the nonlinear solver. This can improve convergence for states with very different magnitudes. The scaling is applied as `res .= res ./ scale_x` before being passed to the solver.

# Extended help
- Super-sampling is not supported by this integrator, but you can trivially wrap it in a function that does super-sampling by stepping `supersample` times in a loop with the same input and sample time `Ts / supersample`.
"""
function SimpleColloc(dyn, Ts::T0, nx::Int, na::Int, args...; kwargs...) where T0 <: Real
    x_inds = 1:nx
    a_inds = nx+1:nx+na
    SimpleColloc(dyn, Ts, x_inds, a_inds, args...; kwargs...)
end

function SimpleColloc(dyn, Ts::T0, x_inds::AbstractVector{Int}, a_inds, nu::Int; n=5, abstol=1e-8, solver=SimpleNewtonRaphson(), residual=false, nodetype=gaussradau, scale_x=nothing) where T0 <: Real
    T = float(T0)
    D, τ = diffoperator(n, float(Ts), nodetype)
    nx = length(x_inds)
    na = length(a_inds)
    cv = zeros(T, (nx+na)*n)
    x = zeros(T, nx+na, n)
    ẋ = zeros(T, nx+na, n)
    cache = (DiffCache(cv, levels=2), DiffCache(x, levels=2), DiffCache(ẋ, levels=2), DiffCache(copy(x), levels=2))

    problem = NonlinearProblem{false}(coldyn,x,SciMLBase.NullParameters())

    SimpleColloc(dyn, float(Ts), nx, x_inds, a_inds, nu, T.(D), T.(τ*Ts), abstol, cache, problem, solver, residual, scale_x)
end

function coldyn(xv::AbstractArray{T}, (integ, x0, u, p, t, args...)) where T
    (; dyn, x_inds, a_inds, D, τ, residual) = integ
    nx, na = length(x_inds), length(a_inds)
    cv, x_cache, ẋ, x = get_cache!(integ, xv)

    # @show typeof(cv)
    # println()
    # @show typeof(x_cache)
    # println()
    # @show typeof(ẋ)
    # println()
    # @show typeof(x)
    # println("\n"^3)
    # x = reshape(xv, nx+na, :)#::Matrix{T} # Use cache of correct size instead to avoid allocations
    copyto!(x, xv) # Reshape but allocation free

    n_c = length(τ)
    inds_c      = x_inds
    inds_x      = x_inds
    inds_alg    = a_inds

    x_cache .= x .- x0
    # ẋ = reshape(cv, nx+na, n_c) # This renames cv to ẋ
    simple_mul!(ẋ, x_cache, D') # Applying the diff operator computes ẋ
    copyto!(cv, ẋ) # This renames ẋ back to cv

    allinds = 1:(nx+na)
    if residual
        @views for k in 1:n_c
            res          = dyn(ẋ[:,k], x[:,k], u, p, t+τ[k], args...)
            if integ.scale_x === nothing
                cv[allinds] .= res
            else
                cv[allinds] .= res ./ integ.scale_x
            end
            allinds      = allinds .+ (nx+na)
        end
    else
        @views for k in 1:n_c
            temp_dyn      = dyn(x[:,k], u, p, t+τ[k], args...)
            cv[inds_c]  .-= temp_dyn[inds_x]
            inds_c        = inds_c .+ (nx+na)

            cv[inds_alg] .=  temp_dyn[nx+1:end]
            inds_alg      = inds_alg .+ (nx+na)
            if integ.scale_x !== nothing
                cv[allinds] ./= integ.scale_x
                allinds      = allinds .+ (nx+na)
            end
        end
    end
    if cv isa Base.ReinterpretArray
        copy(cv)
    else
        cv
    end
end

function (integ::SimpleColloc)(x0::T, u, p, t, args...; abstol=integ.abstol, kwargs...)::T where T
    nx, na = length(integ.x_inds), length(integ.a_inds)
    n_c = length(integ.τ)
    _, _, _, u00 = get_cache!(integ, x0)
    u0 = vec(u00)
    nx0 = length(x0)
    for i = 1:nx0, j = 1:n_c
        u0[i + (j-1)*nx0] = x0[i]
    end
    problem = SciMLBase.remake(integ.nlproblem, u0=u0,p=(integ, x0, u, p, t, args...))
    solution = solve(problem, integ.solver; abstol, kwargs...)
    if !SciMLBase.successful_retcode(solution)
        @warn "Nonlinear solve failed to converge" solution.retcode# maxlog=10
    end
    @views T(solution.u[end-nx-na+1:end])
end


"""
    initialize(integ, x0, u, p, t = 0.0; solver=integ.solver, abstol=integ.abstol)

Given the differential state variables in `x0`, initialize the algebraic variables by solving the nonlinear problem `f(x,u,p,t) = 0` using the provided solver.

# Arguments:
- `integ`: An intergrator like [`SeeToDee.SimpleColloc`](@ref)
- `x0`: Initial state descriptor (differential and algebraic variables, where the algebraic variables comes last)
"""
function initialize(integ, x0, u, p, t=0.0, args...; solver = integ.solver, abstol = integ.abstol)
    (; dyn, nx, nu) = integ
    # u = zeros(nu)
    x0 = copy(x0)
    diffinds = integ.x_inds
    alginds = integ.a_inds
    res0 = dyn(x0, u, p, t, args...)
    norm(res0[alginds]) < abstol && return x0
    res = function (z, _)
        x0T = eltype(z).(x0)
        x0T[alginds] .= z
        dyn(x0T, u, p, t, args...)[alginds]
    end
    problem = NonlinearProblem(res, x0[alginds], p)
    solution = solve(problem, solver; abstol)
    x0[alginds] .= solution.u
    x0
end


## =============================================================================

struct Trapezoidal{F,T,X,A,NP,S,SX} <: AbstractIntegrator
    dyn::F
    Ts::T
    nx::Int
    x_inds::X
    a_inds::A
    nu::Int
    abstol::Float64
    nlproblem::NP
    solver::S
    residual::Bool
    scale_x::SX
end


"""
    Trapezoidal(dyn, Ts, nx, na, nu; abstol = 1.0e-8, solver=SimpleNewtonRaphson(), residual=false)
    Trapezoidal(dyn, Ts, x_inds, a_inds, nu; abstol = 1.0e-8, solver=SimpleNewtonRaphson(), residual=false)

A simple trapezoidal integrator that can be stepped manually, similar to the function returned by [`SeeToDee.Rk4`](@ref).

This integrator supports differential-algebraic equations (DAE), the dynamics is expected to be on either of the forms 
- `nx,na` provided: `(xz,u,p,t)->[ẋ; res]` where `xz` is a vector `[x; z]` contaning the differential state `x` and the algebraic variables `z` in this order. `res` is the algebraic residuals, and `u` is the control input. The algebraic residuals are thus assumed to be the last `na` elements of of the arrays returned by the dynamics (the convention used by ModelingToolkit).
- `x_inds, a_inds` provided: `(xz,u,p,t)->xzd` where `xzd[x_inds] = ẋ` and `xzd[a_inds] = res`.

The returned function has the signature `f_discrete : (x,u,p,t)->x(t+Tₛ)`. 

# Arguments:
- `dyn`: Dynamics function (continuous time)
- `Ts`: Sample time
- `nx`: Number of differential state variables
- `na`: Number of algebraic variables
- `x_inds, a_inds`: If indices are provided instead of `nx` and `na`, the mass matrix is assumed to be diagonal, with ones located at `x_inds` and zeros at `a_inds`. For maximum efficiency, provide these indices as unit ranges or static arrays.
- `nu`: Number of inputs
- `abstol`: Tolerance for the root finding algorithm
- `residual`: If `true` the dynamics function is assumed to return the residual of the entire state descriptor and have the signature `(ẋ, x, u, p, t) -> res`. This is sometimes called "fully implicit form".
- `solver`: Any compatible SciML Nonlinear solver to use for the root finding problem
- `scale_x`: If provided, the residual is scaled by this vector before being passed to the nonlinear solver, `res ./ scale_x`. This can help with convergence if the state variables have very different magnitudes.

# Extended help
- Super-sampling is not supported by this integrator, but you can trivially wrap it in a function that does super-sampling by stepping `supersample` times in a loop with the same input and sample time `Ts / supersample`.
"""
function Trapezoidal(dyn, Ts::T0, nx::Int, na::Int, args...; kwargs...) where T0 <: Real
    x_inds = 1:nx
    a_inds = nx+1:nx+na
    Trapezoidal(dyn, Ts, x_inds, a_inds, args...; kwargs...)
end

function Trapezoidal(dyn, Ts::T0, x_inds::AbstractVector{Int}, a_inds, nu::Int; abstol=1e-8, solver=SimpleNewtonRaphson(), residual=false, inplace=true, scale_x=nothing) where T0 <: Real
    T = float(T0)
    nx = length(x_inds)
    na = length(a_inds)
    x = SVector(zeros(T, nx+na)...)

    if inplace
        problem = NonlinearProblem{true}(coldyn_trapz,x,SciMLBase.NullParameters())
    else
        problem = NonlinearProblem{false}(coldyn_trapz_oop,x,SciMLBase.NullParameters())
    end

    Trapezoidal(dyn, Ts, nx, x_inds, a_inds, nu, abstol, problem, solver, residual, scale_x)
end

function coldyn_trapz(res, x::AbstractArray{T}, (integ, x0, u, p, t, args...)) where T
    (; dyn, x_inds, a_inds, residual, Ts) = integ

    x1 = x isa MVector ? SVector(x) : x

    if residual
        error("Not yet implemented")
        # allinds = 1:(nx+na)
        # res          = dyn(ẋ[:,k], x[:,k], u, p, t+τ[k], args...)
        # res[allinds] .= res
    else
        f0 = dyn(x0, u, p, t,    args...)
        f1 = dyn(x1, u, p, t+Ts, args...)
        if isempty(a_inds)
            res .= x0 .- x1 .+ (Ts/2) .* (f0 .+ f1)
        else
            @views res[x_inds]  .= x0[x_inds] .- x1[x_inds] .+ (Ts/2) .* (f0[x_inds] .+ f1[x_inds])
            @views res[a_inds] .=  f1[a_inds]
        end
    end
    if integ.scale_x !== nothing
        res .= res ./ integ.scale_x
    end
    res
end

function coldyn_trapz_oop(x::AbstractArray{T}, (integ, x0, u, p, t, args...)) where T
    (; dyn, x_inds, a_inds, Ts) = integ


    f0 = dyn(x0, u, p, t,    args...)
    f1 = dyn(x,  u, p, t+Ts, args...)
    if isempty(a_inds)
        if ingteg.scale_x !== nothing
            return (x0 .- x .+ (Ts/2) .* (f0 .+ f1)) ./ integ.scale_x
        else
            return x0 .- x .+ (Ts/2) .* (f0 .+ f1)
        end
    else
        res = [x0[x_inds] .- x[x_inds] .+ (Ts/2) .* (f0[x_inds] .+ f1[x_inds])
        f1[a_inds]]
        if integ.scale_x !== nothing
            res .= res ./ integ.scale_x
        end
        if x isa Union{SVector, MVector} && !(res isa SVector)
            return SVector{length(x)}(res)
        else
            return res
        end
    end
end

function (integ::Trapezoidal)(x0::T, u, p, t, args...; abstol=integ.abstol)::T where T
    problem = SciMLBase.remake(integ.nlproblem, u0=x0, p=(integ, x0, u, p, t, args...))
    # integ.nlproblem = problem
    # problem.u0 .= x0
    # problem.p[2] .= x0
    # problem.p[3] .= u
    # problem.p[6] .= d

    solution = solve(problem, integ.solver; abstol)
    if !SciMLBase.successful_retcode(solution)
        @warn "Nonlinear solve failed to converge" solution.retcode maxlog=10
    end
    T(solution.u)
end


end

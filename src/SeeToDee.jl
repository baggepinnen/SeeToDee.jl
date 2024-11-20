module SeeToDee

using FastGaussQuadrature, SimpleNonlinearSolve, PreallocationTools, LinearAlgebra, ForwardDiff, StaticArrays

export SimpleColloc

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
    B = ForwardDiff.jacobian(u->f(convert(typeof(u), x), u, args...), u)
    A, B
end

"""
    f_discrete = Rk4(f, Ts; supersample = 1)

Discretize a continuous-time dynamics function `f` using RK4 with sample time `Tₛ`. 
`f` is assumed to have the signature `f : (x,u,p,t)->ẋ` and the returned function `f_discrete : (x,u,p,t)->x(t+Tₛ)`.

`supersample` determines the number of internal steps, 1 is often sufficient, but this can be increased to make the integration more accurate. `u` is assumed constant during all steps.

If called with StaticArrays, this integrator is allocation free.
"""
struct Rk4{F,TS}
    f::F
    Ts::TS
    supersample::Int
    function Rk4(f::F, Ts; supersample::Integer = 1) where {F}
        supersample ≥ 1 || throw(ArgumentError("supersample must be positive."))
        new{F, typeof(Ts / 1)}(f, Ts / 1, supersample) # Divide by one to floatify ints
    end
end


function (integ::Rk4{F})(x, u, p, t; Ts=integ.Ts, supersample=integ.supersample) where F
    f = integ.f
    f1 = f(x, u, p, t)
    _inner_rk4(integ, f1, x, u, p, t; Ts, supersample) # Dispatch depending on return type of dynamics
end


function _inner_rk4(integ::Rk4{F}, f1::SArray, x, u, p, t; Ts=integ.Ts, supersample=integ.supersample) where F
    Ts2 = Ts / supersample
    f = integ.f
    f2 = f(x + Ts2 / 2 * f1, u, p, t + Ts2 / 2)
    f3 = f(x + Ts2 / 2 * f2, u, p, t + Ts2 / 2)
    f4 = f(x + Ts2 * f3, u, p, t + Ts2)
    add = Ts2 / 6 .* (f1 .+ 2 .* f2 .+ 2 .* f3 .+ f4)
    # This gymnastics with changing the name to y is to ensure type stability when x + add is not the same type as x. The compiler is smart enough to figure out the type of y
    y = x + add
    for i in 2:supersample
        f1 = f(y, u, p, t)
        f2 = f(y + Ts2 / 2 * f1, u, p, t + Ts2 / 2)
        f3 = f(y + Ts2 / 2 * f2, u, p, t + Ts2 / 2)
        f4 = f(y + Ts2 * f3, u, p, t + Ts2)
        add = Ts2 / 6 .* (f1 .+ 2 .* f2 .+ 2 .* f3 .+ f4)
        y += add
    end
    return y
end

function _inner_rk4(integ::Rk4{F}, f1, x, u, p, t; Ts=integ.Ts, supersample=integ.supersample) where F
    Ts2 = Ts / supersample
    f = integ.f
    xi = x .+ (Ts2 / 2) .* f1
    f2 = f(xi, u, p, t + Ts2 / 2)
    xi .= x .+ (Ts2 / 2) .* f2
    f3 = f(xi, u, p, t + Ts2 / 2)
    xi .= x .+ Ts2 .* f3
    f4 = f(xi, u, p, t + Ts2)
    xi .= (Ts2 / 6) .* (f1 .+ 2 .* f2 .+ 2 .* f3 .+ f4)
    # This gymnastics with changing the name to y is to ensure type stability when x + add is not the same type as x. The compiler is smart enough to figure out the type of y
    y = _mutable(x + xi) # If x is non-static but xi isn't, we get a static array and adding to y won't work
    for i in 2:supersample
        f1 = f(y, u, p, t)
        xi = y .+ (Ts2 / 2) .* f1
        f2 = f(xi, u, p, t + Ts2 / 2)
        xi .= y .+ (Ts2 / 2) .* f2
        f3 = f(xi, u, p, t + Ts2 / 2)
        xi .= y .+ Ts2 .* f3
        f4 = f(xi, u, p, t + Ts2)
        xi .= (Ts2 / 6) .* (f1 .+ 2 .* f2 .+ 2 .* f3 .+ f4)
        y .+= xi
    end
    return y
end

_mutable(x::StaticArray) = Array(x)
_mutable(x::AbstractArray) = x

## RK3 ==========================================================================
struct Rk3{F,TS}
    f::F
    Ts::TS
    supersample::Int
    function Rk3(f::F, Ts; supersample::Integer = 1) where {F}
        supersample ≥ 1 || throw(ArgumentError("supersample must be positive."))
        new{F, typeof(Ts / 1)}(f, Ts / 1, supersample) # Divide by one to floatify ints
    end
end


function (integ::Rk3{F})(x, u, p, t; Ts=integ.Ts, supersample=integ.supersample) where F
    f = integ.f
    f1 = f(x, u, p, t)
    _inner_rk3(integ, f1, x, u, p, t; Ts, supersample) # Dispatch depending on return type of dynamics
end

function _inner_rk3(integ::Rk3{F}, f1, x, u, p, t; Ts=integ.Ts, supersample=integ.supersample) where F
    Ts2 = Ts / supersample
    f = integ.f
    f2 = f(x + Ts2 / 2 * f1, u, p, t + Ts2 / 2)
    f3 = f(x - Ts2 * f1 .+ 2 * Ts2 * f2, u, p, t + Ts2)
    add = Ts2 / 6 .* (f1 .+ 4 .* f2 .+ f3)
    # This gymnastics with changing the name to y is to ensure type stability when x + add is not the same type as x. The compiler is smart enough to figure out the type of y
    y = x + add
    for i in 2:supersample
        f1 = f(y, u, p, t)
        f2 = f(y + Ts2 / 2 * f1, u, p, t + Ts2 / 2)
        f3 = f(y - Ts2 * f1 .+ 2 * Ts2 * f2, u, p, t + Ts2)
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

`supersample` determines the number of internal steps, 1 is often sufficient, but this can be increased to make the integration more accurate. `u` is assumed constant during all steps.

If called with StaticArrays, this integrator is allocation free.
"""
struct ForwardEuler{F,TS}
    f::F
    Ts::TS
    supersample::Int
    function ForwardEuler(f::F, Ts; supersample::Integer = 1) where {F}
        supersample ≥ 1 || throw(ArgumentError("supersample must be positive."))
        new{F, typeof(Ts / 1)}(f, Ts / 1, supersample) # Divide by one to floatify ints
    end
end


function (integ::ForwardEuler{F})(x, u, p, t; Ts=integ.Ts, supersample=integ.supersample) where F
    f = integ.f
    f1 = f(x, u, p, t)
    _inner_forwardeuler(integ, f1, x, u, p, t; Ts, supersample) # Dispatch depending on return type of dynamics
end


function _inner_forwardeuler(integ::ForwardEuler{F}, f1, x, u, p, t; Ts=integ.Ts, supersample=integ.supersample) where F
    Ts2 = Ts / supersample
    f = integ.f
    add = Ts2 .* f1
    # This gymnastics with changing the name to y is to ensure type stability when x + add is not the same type as x. The compiler is smart enough to figure out the type of y
    y = x + add
    for i in 2:supersample
        f2 = f(y, u, p, t)
        add = Ts2 .* f2
        y += add
    end
    return y
end


## Heun #######=================================================================
struct Heun{F,TS}
    f::F
    Ts::TS
    supersample::Int
    function Heun(f::F, Ts; supersample::Integer = 1) where {F}
        supersample ≥ 1 || throw(ArgumentError("supersample must be positive."))
        new{F, typeof(Ts / 1)}(f, Ts / 1, supersample) # Divide by one to floatify ints
    end
end

function (integ::Heun{F})(x, u, p, t; Ts=integ.Ts, supersample=integ.supersample) where F
    f = integ.f
    f1 = f(x, u, p, t)
    _inner_heun(integ, f1, x, u, p, t; Ts, supersample) # Dispatch depending on return type of dynamics
end

function _inner_heun(integ::Heun{F}, f1, x, u, p, t; Ts=integ.Ts, supersample=integ.supersample) where F
    Ts2 = Ts / supersample
    f = integ.f
    f2 = f(x + Ts2 * f1, u, p, t + Ts2)
    add = (Ts2 / 2) .* (f1 .+ f2)
    y = x + add
    for i in 2:supersample
        f1 = f(y, u, p, t)
        f2 = f(y .+ Ts2 .* f1, u, p, t + Ts2)
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

struct SimpleColloc{F,T,X,A,DT,TP,CT,NP,S}
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

# Extended help
- Super-sampling is not supported by this integrator, but you can trivially wrap it in a function that does super-sampling by stepping `supersample` times in a loop with the same input and sample time `Ts / supersample`.
- To use trapezoidal integration, set `n=2` and `nodetype=SeeToDee.FastGaussQuadrature.gausslobatto`.
"""
function SimpleColloc(dyn, Ts::T0, nx::Int, na::Int, args...; kwargs...) where T0 <: Real
    x_inds = 1:nx
    a_inds = nx+1:nx+na
    SimpleColloc(dyn, Ts, x_inds, a_inds, args...; kwargs...)
end

function SimpleColloc(dyn, Ts::T0, x_inds::AbstractVector{Int}, a_inds, nu::Int; n=5, abstol=1e-8, solver=SimpleNewtonRaphson(), residual=false, nodetype=gaussradau) where T0 <: Real
    T = float(T0)
    D, τ = diffoperator(n, Ts, nodetype)
    nx = length(x_inds)
    na = length(a_inds)
    cv = zeros(T, (nx+na)*n)
    x = zeros(T, nx+na, n)
    ẋ = zeros(T, nx+na, n)
    cache = (DiffCache(cv), DiffCache(x), DiffCache(ẋ), DiffCache(copy(x)))

    problem = NonlinearProblem(coldyn,x,SciMLBase.NullParameters())

    SimpleColloc(dyn, Ts, nx, x_inds, a_inds, nu, T.(D), T.(τ), abstol, cache, problem, solver, residual)
end

function coldyn(xv::AbstractArray{T}, (integ, x0, u, p, t)) where T
    (; dyn, x_inds, a_inds, D, τ, residual) = integ
    nx, na = length(x_inds), length(a_inds)
    cv, x_cache, ẋ, x = get_cache!(integ, xv)
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

    if residual
        allinds = 1:(nx+na)
        @views for k in 1:n_c
            res          = dyn(ẋ[:,k], x[:,k], u, p, t+τ[k])
            cv[allinds] .= res
            allinds      = allinds .+ (nx+na)
        end
    else
        @views for k in 1:n_c
            temp_dyn      = dyn(x[:,k], u, p, t+τ[k])
            cv[inds_c]  .-= temp_dyn[inds_x]
            inds_c        = inds_c .+ (nx+na)

            cv[inds_alg] .=  temp_dyn[nx+1:end]
            inds_alg      = inds_alg .+ (nx+na)
        end
    end
    cv
end

function (integ::SimpleColloc)(x0::T, u, p, t; abstol=integ.abstol)::T where T
    nx, na = length(integ.x_inds), length(integ.a_inds)
    n_c = length(integ.τ)
    _, _, _, u00 = get_cache!(integ, x0)
    u0 = vec(u00)
    nx0 = length(x0)
    for i = 1:nx0, j = 1:n_c
        u0[i + (j-1)*nx0] = x0[i]
    end
    problem = SciMLBase.remake(integ.nlproblem, u0=u0,p=(integ, x0, u, p, t))
    solution = solve(problem, integ.solver; abstol)
    if !SciMLBase.successful_retcode(solution)
        @warn "Nonlinear solve failed to converge" solution.retcode maxlog=10
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
function initialize(integ, x0, u, p, t=0.0; solver = integ.solver, abstol = integ.abstol)
    (; dyn, nx, nu) = integ
    # u = zeros(nu)
    x0 = copy(x0)
    diffinds = integ.x_inds
    alginds = integ.a_inds
    res0 = dyn(x0, u, p, t)
    norm(res0[alginds]) < abstol && return x0
    res = function (z, _)
        x0T = eltype(z).(x0)
        x0T[alginds] .= z
        dyn(x0T, u, p, t)[alginds]
    end
    problem = NonlinearProblem(res, x0[alginds], p)
    solution = solve(problem, solver; abstol)
    x0[alginds] .= solution.u
    x0
end


end

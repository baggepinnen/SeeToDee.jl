module SeeToDee

using FastGaussQuadrature, SimpleNonlinearSolve, PreallocationTools, LinearAlgebra, ForwardDiff

export SimpleColloc

"""
    A,B = linearize(f, x0, u0, p, t)

Linearize dynamics function `f` w.r.t., state `x`, input `u`. Returns Jacobians `A,B` in
```math
ẋ = A\\, Δx + B\\, Δu
```
Works for both continuous and discrete-time dynamics.
"""
function linearize(f, x, u, p, t)
    A = ForwardDiff.jacobian(x->f(x, u, p, t), x)
    B = ForwardDiff.jacobian(u->f(x, u, p, t), u)
    A, B
end


"""
    f_discrete = Rk4(f, Ts; supersample = 1)

Discretize a continuous-time dynamics function `f` using RK4 with sample time `Tₛ`. 
`f` is assumed to have the signature `f : (x,u,p,t)->ẋ` and the returned function `f_discrete : (x,u,p,t)->x(t+Tₛ)`.

`supersample` determins the number of internal steps, 1 is often sufficient, but this can be increased to make the interation more accurate. `u` is assumed constant during all steps.
"""
function Rk4(f::F, Ts0; supersample::Integer = 1) where {F}
    supersample ≥ 1 || throw(ArgumentError("supersample must be positive."))
    # Runge-Kutta 4 method
    Ts = Ts0 / supersample # to preserve type stability in case Ts0 is an integer
    let Ts = Ts
        function (x, u, p, t)
            T = typeof(x)
            f1 = f(x, u, p, t)
            f2 = f(x + Ts / 2 * f1, u, p, t + Ts / 2)
            f3 = f(x + Ts / 2 * f2, u, p, t + Ts / 2)
            f4 = f(x + Ts * f3, u, p, t + Ts)
            add = Ts / 6 * (f1 + 2 * f2 + 2 * f3 + f4)
            # This gymnastics with changing the name to y is to ensure type stability when x + add is not the same type as x. The compiler is smart enough to figure out the type of y
            y = x + add
            for i in 2:supersample
                f1 = f(y, u, p, t)
                f2 = f(y + Ts / 2 * f1, u, p, t + Ts / 2)
                f3 = f(y + Ts / 2 * f2, u, p, t + Ts / 2)
                f4 = f(y + Ts * f3, u, p, t + Ts)
                add = Ts / 6 * (f1 + 2 * f2 + 2 * f3 + f4)
                y += add
            end
            return y
        end
    end
end


# ==============================================================================

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

function diffoperator(n, Ts)
    τ, _ = gausslobatto(n+1)
    τ = (τ[2:end] .+ 1) ./ 2
    A = τ.^(0:n-1)'.*(1:n)'
    B = τ.^(1:n)'
    (A/B) ./ Ts, τ
end

struct SimpleColloc{F,T,DT,TP,CT,NP,S}
    dyn::F
    Ts::T
    nx::Int
    na::Int
    nu::Int
    D::DT
    τ::TP
    abstol::Float64
    cache::CT
    nlproblem::NP
    solver::S
    residual::Bool
end

function get_cache!(integ::SimpleColloc, x::Array{T}) where T#::Tuple{Vector{T}, Matrix{T}, Matrix{T}, Matrix{T}} where T
    return get_tmp(integ.cache[1], x), get_tmp(integ.cache[2], x), get_tmp(integ.cache[3], x), get_tmp(integ.cache[4], x)
end


"""
    SimpleColloc(dyn, Ts, nx, na; n = 5, abstol = 1.0e-8, solver=SimpleNewtonRaphson(), residual=false)

A simple direct-collocation integrator that can be stepped manually, similar to the function returned by [`Rk4`](@ref).

This integrator supports algebraic equations (DAE), the dynamics is expected to be on the form `(x,u,p,t)->[ẋ; res]` where `x` is the differential state, `res` are the algebraic residuals, `u` is the control input. The algebraic residuals are thus assumed to be the last `na` elements of of the arrays returned by the dynamics (the convention used by ModelingToolkit). The returned function has the signature `f_discrete : (x,u,p,t)->x(t+Tₛ)`. 

A Gauss-Lobatto collocation method is used to discretize the dynamics. The resulting nonlinear problem is solved using (by default) a Newton-Raphson method. This method handles stiff dynamics.

# Arguments:
- `dyn`: Dynamics function (continuous time)
- `Ts`: Sample time
- `nx`: Number of differential state variables
- `na`: Number of algebraic variables
- `n`: Number of collocation points. `n=2` corresponds to trapezoidal integration.
- `abstol`: Tolerance for the root finding algorithm
- `residual`: If `true` the dynamics function is assumed to return the residual of the entire state descriptor and have the signature `(ẋ, x, u, p, t) -> res`. This is sometimes called "fully implicit form".
- `solver`: Any compatible SciML Nonlinear solver to use for the root finding problem
"""
function SimpleColloc(dyn, Ts, nx, na, nu; n=5, abstol=1e-8, solver=SimpleNewtonRaphson(), residual=false)
    D, τ = diffoperator(n, Ts)
    cv = zeros((nx+na)*n)
    x = zeros(nx+na, n)
    ẋ = zeros(nx+na, n)
    cache = (DiffCache(cv), DiffCache(x), DiffCache(ẋ), DiffCache(copy(x)))

    problem = NonlinearProblem(coldyn,x,SciMLBase.NullParameters())

    SimpleColloc(dyn, Ts, nx, na, nu, D, τ, abstol, cache, problem, solver, residual)
end

function coldyn(xv::Array{T}, (integ, x0, u, p, t)) where T
    (; dyn, nx, na, D, τ, residual) = integ
    cv, x_cache, ẋ, x = get_cache!(integ, xv)
    # x = reshape(xv, nx+na, :)#::Matrix{T} # Use cache of correct size instead to avoid allocations
    copyto!(x, xv) # Reshape but allocation free

    n_c = length(τ)
    inds_c      = (1:nx)   
    inds_x      = (1:nx)
    inds_alg    = (nx+1:nx+na)

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
    (; nx, na) = integ
    n_c = length(integ.τ)
    problem = SciMLBase.remake(integ.nlproblem, u0=vec(x0*ones(1, n_c)),p=(integ, x0, u, p, t))
    solution = solve(problem, integ.solver, abstol)
    @views T(solution.u[end-nx-na+1:end])
end


"""
    initialize(integ, x0, p, t = 0.0; solver=integ.solver, abstol=integ.abstol)

Given the differential state variables in `x0`, initialize the algebraic variables by solving the nonlinear problem `f(x,u,p,t) = 0` using the provided solver.

# Arguments:
- `integ`: An intergrator like [`SimpleColloc`](@ref)
- `x0`: Initial state descriptor (differential and algebraic variables, where the algebraic variables comes last)
"""
function initialize(integ, x0, p, t=0.0; solver = integ.solver, abstol = integ.abstol)
    (; dyn, nx, na, nu) = integ
    u = zeros(nu)
    diffinds = 1:nx
    alginds = nx+1:nx+na
    res0 = dyn(x0, u, p, t)
    norm(res0[alginds]) < abstol && return x0
    res = (z, _) -> dyn([x0[diffinds]; z], u, p, t)[alginds]
    problem = NonlinearProblem(res, x0[alginds], p)
    solution = solve(problem, solver, abstol)
    [x0[diffinds]; solution.u]
end

end

module SimpleCollocation

using FastGaussQuadrature, SimpleNonlinearSolve, PreallocationTools

export SimpleColloc

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
    taupoints, _ = gausslobatto(n+1)
    taupoints = (taupoints[2:end] .+ 1) ./ 2
    A = taupoints.^(0:n-1)'.*(1:n)'
    B = taupoints.^(1:n)'
    (A/B) ./ Ts, taupoints
end

struct SimpleColloc{F,T,DT,TP,CT,NP,S}
    dyn::F
    Ts::T
    nx::Int
    na::Int
    D::DT
    taupoints::TP
    abstol::Float64
    cache::CT
    nlproblem::NP
    solver::S
end

function get_cache!(integ::SimpleColloc, x::Array{T})::Tuple{Vector{T}, Matrix{T}} where T
    return get_tmp(integ.cache[1], x), get_tmp(integ.cache[2], x)
end


"""
SimpleColloc(dyn, Ts, nx, na; n = 5, abstol = 1.0e-8, solver=SimpleNewtonRaphson())

A simple direct collocation integrator that can be stepped manually, similar to the function returned by [`MPC.rk4`](@ref).

This integrator supports algebraic equations (DAE), the dynamics is expected to be on the form `(x,u,p,t)->[ẋ; res]` where `x` is the differential state, `res` are the algebraic residuals, `u` is the control input. The algebraic residuals are thus assumed to be the last `na` elements of of the arrays returned by the dynamics (the convention used by ModelingToolkit).

A Gauss-Lobatto collocation method is used to discretize the dynamics. The resulting nonlinear problem is solved using a Newton-Raphson method.

# Arguments:
- `dyn`: Dynamics function (continuous time)
- `Ts`: Sample time
- `nx`: Number of differential state variables
- `na`: Number of algebraic variables
- `n`: Number of collocation points
- `abstol`: Tolerance for the root finding algorithm
"""
function SimpleColloc(dyn, Ts, nx, na; n=5, abstol=1e-8, solver=SimpleNewtonRaphson())
    D, taupoints = diffoperator(n, Ts)
    cv = zeros((nx+na)*n)
    x = zeros(nx+na, n)
    cache = (DiffCache(cv), DiffCache(x))

    problem = NonlinearProblem(coldyn,x,SciMLBase.NullParameters())

    SimpleColloc(dyn, Ts, nx, na, D, taupoints, abstol, cache, problem, solver)
end

function coldyn(xv::Array{T}, (integ, x0, u, p, t)) where T
    (; dyn, nx, na, D, taupoints) = integ
    cv, x_cache = get_cache!(integ, xv)
    x = reshape(xv, nx+na, :)::Matrix{T} # NOTE: there are some allocations here that can be ommitted by not reshaping and instead index xv with x_inds, change the mul to operate on vector form and take care of the .-

    n_c = length(taupoints)
    inds_c      = (1:nx)   
    inds_x      = (1:nx)
    inds_alg    = (nx+1:nx+na)

    x_cache .= x .- x0
    simple_mul!(reshape(cv, nx+na, n_c), x_cache, D')

    @views for k in 1:n_c
        temp_dyn      = dyn(x[:,k], u, p, t+taupoints[k])
        cv[inds_c]  .-= temp_dyn[inds_x]
        inds_c        = inds_c .+ (nx+na)

        cv[inds_alg] .=  temp_dyn[nx+1:end]
        inds_alg      = inds_alg .+ (nx+na)
    end
    cv
end

function (integ::SimpleColloc)(x0::T, u, p, t)::T where T
    (; abstol, nx, na) = integ
    n_c = length(integ.taupoints)
    problem = SciMLBase.remake(integ.nlproblem, u0=vec(x0*ones(1, n_c)),p=(integ, x0, u, p, t))
    solution = solve(problem, integ.solver, abstol)
    T(solution.u[end-nx-na+1:end])
end




end

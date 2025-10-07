using SeeToDee
using Test
using ForwardDiff
using StaticArrays

"""
    A, B = linearize(f, x, u, args...)

Linearize dynamics ``ẋ = f(x, u, args...)`` around operating point ``(x,u,args...)`` using ForwardDiff. `args` can be empty, or contain, e.g., parameters and time `(p, t)` like in the SciML interface.
This function can also be used to linearize an output equation `C, D = linearize(h, x, u, args...)`.
"""
function linearize(f, xi::AbstractVector, ui::AbstractVector, args...)
    A = ForwardDiff.jacobian(x -> f(x, ui, args...), xi)
    # For B, we need to handle cases where x and u have different dimensions
    if xi isa SVector && ui isa SVector
        B = ForwardDiff.jacobian(u -> f(xi, u, args...), ui)
    else
        B = ForwardDiff.jacobian(u -> f(convert(typeof(u), xi), u, args...), ui)
    end
    A, B
end


# Define trivial linear dynamics: dx = -x + u
# Continuous-time: A_c = -1, B_c = 1
function simple_dynamics(x, u, p, t)
    return -x .+ u
end

Ts = 0.1
x0 = SA[1.0]
u0 = SA[0.5]

# Compute exact discrete-time A, B for simple_dynamics
# dx = -x + u can be written as dx = A_c*x + B_c*u where A_c = -1, B_c = 1
# The exact discretization is: [A B; 0 0] = exp(Ts*[A_c B_c; 0 0])
AB_true = exp(Ts*[-1 1; 0 0])
A_true = AB_true[1,1]
B_true = AB_true[1,2]

# Exact discretization for vec_dynamics: dx = [-1 0; 0 -2]*x + [1; 1]*u
AB_vec_true = exp(Ts*[-1 0 1; 0 -2 1; 0 0 0])
A_vec_true = AB_vec_true[1:2, 1:2]
B_vec_true = AB_vec_true[1:2, 3:3]

@testset "Explicit Integrators" begin
    @testset "Rk4" begin
        integrator = SeeToDee.Rk4(simple_dynamics, Ts; supersample=1)

        # Test that we can compute Jacobians
        A, B = linearize(integrator, x0, u0, 0, 0)

        # Verify dimensions
        @test size(A) == (1, 1)
        @test size(B) == (1, 1)

        # Verify against exact solution (RK4 should be very accurate)
        @test A[1,1] ≈ A_true rtol=1e-6
        @test B[1,1] ≈ B_true rtol=1e-6

        # Test with vector states
        x0_vec = SA[1.0, 2.0]
        u0_vec = SA[0.5]
        function vec_dynamics(x, u, p, t)
            return SA[-x[1] + u[1], -2*x[2] + u[1]]
        end


        integrator_vec = SeeToDee.Rk4(vec_dynamics, Ts; supersample=1)
        A_vec, B_vec = linearize(integrator_vec, x0_vec, u0_vec, 0, 0)
        @test size(A_vec) == (2, 2)
        @test size(B_vec) == (2, 1)
        @test A_vec ≈ A_vec_true rtol=1e-5
        @test B_vec ≈ B_vec_true rtol=1e-5
    end

    @testset "Rk3" begin
        integrator = SeeToDee.Rk3(simple_dynamics, Ts; supersample=1)

        A, B = linearize(integrator, x0, u0, 0, 0)

        @test size(A) == (1, 1)
        @test size(B) == (1, 1)
        # RK3 should also be very accurate
        @test A[1,1] ≈ A_true rtol=1e-4
        @test B[1,1] ≈ B_true rtol=1e-4
    end

    @testset "ForwardEuler" begin
        integrator = SeeToDee.ForwardEuler(simple_dynamics, Ts; supersample=1)

        A, B = linearize(integrator, x0, u0, 0, 0)

        @test size(A) == (1, 1)
        @test size(B) == (1, 1)

        # For Forward Euler, we can verify the exact discretization
        # x(k+1) = x(k) + Ts*(-x(k) + u(k)) = (1-Ts)*x(k) + Ts*u(k)
        @test A[1,1] ≈ 1 - Ts atol=1e-10
        @test B[1,1] ≈ Ts atol=1e-10

        # Forward Euler is less accurate than the exact solution
        @test A[1,1] ≈ A_true rtol=0.01
        @test B[1,1] ≈ B_true rtol=0.05
    end

    @testset "Heun" begin
        integrator = SeeToDee.Heun(simple_dynamics, Ts; supersample=1)

        A, B = linearize(integrator, x0, u0, 0, 0)

        @test size(A) == (1, 1)
        @test size(B) == (1, 1)
        # Heun (RK2) should be reasonably accurate
        @test A[1,1] ≈ A_true rtol=1e-3
        @test B[1,1] ≈ B_true rtol=0.03
    end
end

@testset "Implicit Integrators" begin
    # Implicit integrators need regular vectors, not StaticArrays, for ForwardDiff
    x0_vec = [1.0]
    u0_vec = [0.5]

    @testset "SimpleColloc" begin
        integrator = SeeToDee.SimpleColloc(simple_dynamics, Ts, 1, 0, 1; n=3,
            # solver=NonlinearSolve.NewtonRaphson()
            # solver=SimpleNewtonRaphson()
        )

        A, B = linearize(integrator, x0_vec, u0_vec, 0, 0)

        @test size(A) == (1, 1)
        @test size(B) == (1, 1)
        # SimpleColloc with n=3 should be quite accurate
        @test A[1,1] ≈ A_true rtol=1e-6
        @test B[1,1] ≈ B_true rtol=1e-6

        # Test with higher order - should be very accurate
        integrator_high = SeeToDee.SimpleColloc(simple_dynamics, Ts, 1, 0, 1; n=5)
        A_high, B_high = linearize(integrator_high, x0_vec, u0_vec, 0, 0)
        @test size(A_high) == (1, 1)
        @test size(B_high) == (1, 1)
        @test A_high[1,1] ≈ A_true rtol=1e-8
        @test B_high[1,1] ≈ B_true rtol=1e-8
    end

    @testset "Trapezoidal" begin
        integrator = SeeToDee.Trapezoidal(simple_dynamics, Ts, 1, 0, 1)

        A, B = linearize(integrator, x0_vec, u0_vec, 0, 0)

        @test size(A) == (1, 1)
        @test size(B) == (1, 1)
        # Trapezoidal should be reasonably accurate
        @test A[1,1] ≈ A_true rtol=1e-4
        @test B[1,1] ≈ B_true rtol=1e-3
    end
end



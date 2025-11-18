# Test that time is handled correctly during integration

function timedynamics(x, u, p, t)
    SA[cos(t) + 1/(t+1)]
end

Ts = 0.02
integrators = [
    SeeToDee.ForwardEuler(timedynamics, Ts; supersample=3)
    SeeToDee.Heun(timedynamics, Ts; supersample=3)
    SeeToDee.Rk3(timedynamics, Ts; supersample=3)
    SeeToDee.Rk4(timedynamics, Ts; supersample=3)
    SeeToDee.SimpleColloc(timedynamics, Ts, 1, 0, 0; n=5, abstol=1e-10, residual=false)#, solver=NonlinearSolve.NewtonRaphson())
    SeeToDee.Trapezoidal(timedynamics, Ts, 1, 0, 0; abstol=1e-10)
    SeeToDee.RKC2(timedynamics, Ts; supersample=3)
    SeeToDee.SuperSampler(SeeToDee.BackwardEuler(timedynamics, Ts, 1, 0, 0; abstol=1e-10), 3)
]


function integrate(discrete_dynamics)
    x = SA[0.0]
    t = 0.0
    errors = map(1:round(Int, pi/Ts)) do i
        x = discrete_dynamics(x, 0, 0, t)
        t += Ts
        x[] - (sin(t) + log1p(t) - log1p(0))
    end
end

errors = map(integrate, integrators)
mse = sum.(abs2, errors)

@test mse[5] <= mse[4] <= mse[2] <= mse[1]
@test mse[4] ≈ mse[3] # Same order in time, so same error

@test mse[1] <= 0.005587911599153512 * 1.01
@test mse[2] <= 2.1391759085458752e-10 * 1.01
@test mse[3] <= 2.799702690248203e-21 * 1.01
@test mse[5] <= 8.887497432691419e-29 * 1.01
@test mse[6] <= 1.732582431712323e-8 * 1.01
@test mse[7] <= 1.4638806535776506e-10 * 1.01
@test mse[8] <= 0.005585246945595866 * 1.01


# using Plots
# plot(reduce(hcat, errors))
# plot(1:5, mse, yscale=:log10) 
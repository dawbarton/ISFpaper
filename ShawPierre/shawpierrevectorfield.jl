
function vfrhs!(x, y, p, t)
    k = 1.0
    kappa = 0.5
    c = 0.003

    x[1] = y[3]
    x[2] = y[4]
    x[3] = - c*y[3] - k*y[1] - kappa*y[1]^3 + k*(y[2]-y[1]) + c*(y[4]-y[3])
    x[4] = - c*y[4] - k*y[2] - k*(y[2]-y[1]) - c*(y[4]-y[3])
    return nothing
end

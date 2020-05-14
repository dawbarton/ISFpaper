module s

    using PyPlot
    using DifferentialEquations

    include("shawpierrevectorfield.jl")

    function generate()
        xs = []
        ys = []
        zs = []
        nruns = 100
        npoints = 16
        Tstep = 0.8
        for j=1:nruns
            u0 = 2*rand(4) .- ones(4)
            u0 = (u0*(sqrt(sum(u0.^2)/length(u0)))^2)/5
            tspan = (0.0,Tstep*(npoints-1)) # 51 intervals with T=0.8 as in Proc Roy Soc Paper
            prob = ODEProblem(vfrhs!, u0, tspan)
            sol = solve(prob, Vern7(), saveat = Tstep/10, abstol = 1e-12, reltol = 1e-12)
            trange = range(sol.t[1], sol.t[end], length = npoints)
            dsol = [sol(t) for t in trange]
            dsol1 = [sol(t)[1] for t in trange]
            xs = vcat(xs, dsol[1:end-1])
            ys = vcat(ys, dsol[2:end])
            if j==1
                figure(3)
                plot(sol.t, sol.u)
                scatter(trange, dsol1)
            end
        end
        return xs, ys, zs
    end
    
    using BSON: @load, @save
    xs, ys = generate()
    @save "ShawPierreDataTrain.bson" xs ys
    xs, ys = generate()
    @save "ShawPierreDataTest.bson" xs ys
    
    xpts = [a[1] for a in xs]
    ypts = [a[3] for a in xs]
    zpts = [a[4] for a in xs]
    figure(1)
    scatter3D(ypts, zpts, xpts)
end

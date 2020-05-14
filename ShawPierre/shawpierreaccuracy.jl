module u

using DynamicPolynomials
using MultivariatePolynomials
using TaylorSeries
using LinearAlgebra
using ForwardDiff

include("../src/vfmethods.jl")
include("shawpierrevectorfield.jl")
include("../src/isfpolymodel.jl")

function vfrhsPar(y)
    x = zero(y)
    vfrhs!(x, y, 0, 0)
    return x
end

# duplicated from dataromopt02.jl
function Restore(U, Ut, vars, varst)
    # must have the same exponents
    @assert maximum(abs.(U.mexp .- Ut.mexp)) == 0
    @assert size(U.W,1) + size(Ut.W,1) == size(U.mexp,1)
    # the composite submersion
    Uc = PolyModel(U.mexp, zeros(typeof(U.W[1]), size(U.W,1) + size(Ut.W,1), size(U.W,2)) )
    Uc.W[vars,:] = U.W
    Uc.W[varst,:] = Ut.W
    # Uc must have identity as the linear part!
    UcA = PolyGetLinearPart(Uc)
    @show UcA
    UcN = PolyModel(Uc.mexp, - inv(UcA) * Uc.W)
    PolySetLinearPart!(UcN, zero(UcA))

    res0 = PolyModel(size(Uc.mexp,1), size(Uc.W,1), PolyOrder(U.mexp))
    PolySetLinearPart!(res0, inv(UcA))
    res1 = PolyCopy(res0)
    for k=1:PolyOrder(res0.mexp)+1
        PolySubs!(res1, UcN, res0)
        PolySetLinearPart!(res1, inv(UcA))
        err = maximum(abs.(res0.W .- res1.W))
        res0.W .= res1.W
        if err < 1e-12
            break
        end
    end
    return res0
end

function makeUhat(Utup)
    lens = zeros(Int, length(Utup))
    begs = zeros(Int, length(Utup))
    ends = zeros(Int, length(Utup))
    begs[1] = 1
    lens[1] = size(Utup[1].W,1)
    ends[1] = begs[1] + lens[1] - 1
    for k=2:length(Utup)
        lens[k] = size(Utup[k].W,1)
        begs[k] = begs[k-1] + lens[k]
        ends[k] = begs[k] + lens[k] - 1
    end
    Uc = PolyModel(Utup[1].mexp, zeros(typeof(Utup[1].W[1]), ends[end], size(Utup[1].W,2)))
    for k=1:length(Utup)
        Uc.W[begs[k]:ends[k],:] = Utup[k].W
    end
    return Uc
end

function RestoreNewton(Utup, x0; init=nothing)
    Uc = makeUhat(Utup)
    jac = xx -> ForwardDiff.jacobian(Uc, xx)

    A = PolyGetLinearPart(Uc)
    if init == nothing
        W0 = A\x0
    else
        W0 = copy(init)
    end
    it = 1
    while true
        deltaW = jac(W0)\(x0 - Uc(W0))
        W0 .+= deltaW
        if maximum(abs.(deltaW)) < 1e-8
            break
        end
        it += 1
        if it > 300
            println("Wz: max iteration exceeded. Error = ", maximum(abs.(deltaW)))
            W0 .= NaN
            break
        end
    end
    return W0
end

function SplitData(xs, DF)
    # splitting the data:
    # We need to use left eigenvectors. [the complex conjugate pairs should be treated together.]
    vals, adjvecs = eigen(collect(DF'))
    vals, vecs = eigen(DF)

    # assuming that adjvecs are complex conjugate pairs
    adjproj = vcat(real(adjvecs[:,1])', imag(adjvecs[:,1])', real(adjvecs[:,3])', imag(adjvecs[:,3])')
    xs2 = [adjproj*xs[k] for k=1:length(xs)]
    return xs2
end

using DifferentialEquations
# using JLD2, FileIO
using BSON: @load
using LinearAlgebra
using PyPlot
 
ndim = 4
zdim = 2
orders = [3, 5, 7]


figure(4)
cla()
figure(5)
cla()

for order in orders
    
    qmcolor = "orange"
    if order == 3
        qmcolor = "red"
    elseif order == 5
        qmcolor = "green"
    elseif order == 7
        qmcolor = "blue"
    elseif order == 9
        qmcolor = "orange"
    elseif order == 11
        qmcolor = "black"
    else
        qmcolor = "yellow"
    end    
    
    SIGMA = 3
    @load "MapFitShawPierreS$(SIGMA)O$(order).bson" DT Wo1 Ro1 Ws1 Rs1 W1 R1 U1 S1 mpar1 mexp1 Wo2 Ro2 Ws2 Rs2 W2 R2 U2 S2 mpar2 mexp2

    qU = PolyModel(U1.mexp, U1.W)
    qS = PolyModel(S1.mexp, S1.W)
    qUt = PolyModel(U2.mexp, U2.W)
    qSt = PolyModel(S2.mexp, S2.W)
#     qIUd = PolyModel(IUd.mexp, IUd.W) # for the direct fitting
#     qBMd = PolyModel(BMd.mexp, BMd.W) # for the direct fitting
    qWo = PolyModel(Wo1.mexp, Wo1.W)
    qRo = PolyModel(Ro1.mexp, Ro1.W)
    qWot = PolyModel(Wo2.mexp, Wo2.W)
    qRot = PolyModel(Ro2.mexp, Ro2.W)
#     qIU = PolyModel(IU.mexp, IU.W) # for the already fitted map
#     qBM = PolyModel(BM.mexp, BM.W) # for the already fitted map
#     qA = DF
    
    # this makes a polynomial of the model
    F = PolyModel(order, ndim, vfrhsPar)

    Wo, Ro, W, R = ISFVFCalc(F, [1, 2], [])

    Wot, Rot, Wt, Rt = ISFVFCalc(F, [3, 4], [])

    IU = Restore(Wo, Wot, [1, 2], [3, 4])
    qIU = Restore(qWo, qWot, [1, 2], [3, 4])
    qIUd = Restore(qU, qUt, [1, 2], [3, 4])
# UNCOMMENT IF WANT A MORE ACCURATE RECONSTRAUCTION
#     IU = x-> RestoreNewton((Wo, Wot), x)
#     qIU = x-> RestoreNewton((qWo, qWot), x)
#     qIUd = x-> RestoreNewton((qU, qUt), x)
    
    npoints = 32
    Tstep = 0.8

    AA = PolyGetLinearPart(F)
    vals, vecs = eigen(AA)
    ics = hcat(real(vecs[:,1]), real(vecs[:,3]))/5

    u0 = ics*[cos(pi/3); sin(pi/3)]/2
    @show u0
    tspan = (0.0,Tstep*(npoints-1)) # 51 intervals with T=0.8 as in Proc Roy Soc Paper

    probfull = ODEProblem(vfrhs!, u0, tspan)
    solfull = solve(probfull, RadauIIA5(), saveat = Tstep/10, abstol = 1e-10, reltol = 1e-10)

    function rhsred1!(x, y, p, t)
        x[:] = Ro(y)
        return nothing
    end

    function rhsred2!(x, y, p, t)
        x[:] = Rot(y)
        return nothing
    end

    prob1 = ODEProblem(rhsred1!, Wo(u0), tspan)
    prob2 = ODEProblem(rhsred2!, Wot(u0), tspan)

    sol1 = solve(prob1, RadauIIA5(), saveat = Tstep/10, abstol = 1e-10, reltol = 1e-10)
    sol2 = solve(prob2, RadauIIA5(), saveat = Tstep/10, abstol = 1e-10, reltol = 1e-10)
    
    # ODE
    ODEfull = zeros(4, length(sol1.u))
    ODErecon = zeros(4, length(sol1.u))
    fwODEfull = zeros(4, length(sol1.u))
    fwODErecon = zeros(4, length(sol1.u))

    for k=1:size(ODEfull,2)
        # the full ODE solution
        ODEfull[:,k] = solfull.u[k]
        # the reconstructed ODE solution from reduced model
        ODErecon[:,k] = IU(vcat(sol1.u[k], sol2.u[k]))
        # the ODE solution mapped into the reduced 
        fwODEfull[:,k] = vcat(Wo(ODEfull[:,k]), Wot(ODEfull[:,k]))
        fwODErecon[:,k] = vcat(sol1.u[k], sol2.u[k])
    end

    # differences for ODE
    # original space: ODEfull - ODErecon
    # reduced space: fwODEfull - fwODErecon

    # MAP
    xs = solfull.u[1:10:end]
    MAPfull = zeros(4, length(xs))
    MAPrecon = zeros(4, length(xs))
    fwMAPfull = zeros(4, length(xs))
    fwMAPrecon = zeros(4, length(xs))
    
    # initial conditions of the MAP
    z0 = qWo(xs[1])
    zt0 = qWot(xs[1])
    MAPfull[:,1] = xs[1]
    MAPrecon[:,1] = qIU(vcat(qWo(xs[1]), qWot(xs[1])))
    fwMAPfull[:,1] = vcat(z0, zt0)
    fwMAPrecon[:,1] = vcat(z0, zt0)
    for k=2:size(MAPfull,2)
        MAPfull[:,k] = xs[k]
        z0 .= qRo(z0)
        zt0 .= qRot(zt0)
        MAPrecon[:,k] = qIU(vcat(z0, zt0))
        fwMAPfull[:,k] = vcat(qWo(MAPfull[:,k]), qWot(MAPfull[:,k]))
        fwMAPrecon[:,k] = vcat(z0, zt0)
    end
    
    # differences for MAP
    # original space: MAPfull - MAPrecon
    # reduced space: fwMAPfull - fwMAPrecon

    # DATA
    DATAfull = zeros(4, length(xs))
    DATArecon = zeros(4, length(xs))
    fwDATAfull = zeros(4, length(xs))
    fwDATArecon = zeros(4, length(xs))
    
    # initial conditions of the DATA
    z0 = qU(xs[1])
    zt0 = qUt(xs[1])
    DATAfull[:,1] = xs[1]
    DATArecon[:,1] = qIUd(vcat(qU(xs[1]), qUt(xs[1])))
    fwDATAfull[:,1] = vcat(z0, zt0)
    fwDATArecon[:,1] = vcat(z0, zt0)
    for k=2:size(DATAfull,2)
        DATAfull[:,k] = xs[k]
        z0 .= qS(z0)
        zt0 .= qSt(zt0)
        DATArecon[:,k] = qIUd(vcat(z0, zt0))
        fwDATAfull[:,k] = vcat(qU(DATAfull[:,k]), qUt(DATAfull[:,k]))
        fwDATArecon[:,k] = vcat(z0, zt0)
    end
    
    # differences for DATA
    # original space: DATAfull - DATArecon
    # reduced space: fwDATAfull - fwDATArecon

    figure(4)
    # cla()
    fsn = sqrt.(sum(ODEfull .^ 2,dims=1)')
    semilogy(sol1.t, sqrt.(sum((fwODEfull - fwODErecon) .^ 2,dims=1)')./fsn, "-", color = qmcolor)
    semilogy(sol1.t[1:10:end], sqrt.(sum((fwMAPfull - fwMAPrecon) .^ 2,dims=1)')./fsn[1:10:end], "x", color = qmcolor)
    semilogy(sol1.t[1:10:end], sqrt.(sum((fwDATAfull - fwDATArecon) .^ 2,dims=1)')./fsn[1:10:end], "v", markersize=8, fillstyle="none", color = qmcolor)
    ax = gca()
    ax.tick_params(labelsize=16)
    ylim([1e-8,1e-2])
    xlim([-Tstep/2,Tstep*npoints])
    xlabel("time [s]",size=16)
    ylabel("Relative error",size=16)
    subplots_adjust(top=0.95, bottom=0.15, left=0.15, right=0.75)
    legend(bbox_to_anchor=[1.05, 1], loc=2, borderaxespad=0, ("VF O(3)", "MAP O(3)", "DATA O(3)", "VF O(5)", "MAP O(5)", "DATA O(5)", "VF O(7)", "MAP O(7)", "DATA O(7)", "VF O(9)", "MAP O(9)", "DATA O(9)"))
    text(21, 0.003, "(a)", size=18)
    savefig("ShawPierreReconstructErrorFw.pdf", format="pdf")

    figure(5)
    # cla()
    fsn = sqrt.(sum(ODEfull .^ 2,dims=1)')
    semilogy(sol1.t, sqrt.(sum((ODEfull - ODErecon) .^ 2,dims=1)')./fsn, "-", color = qmcolor)
    semilogy(sol1.t[1:10:end], sqrt.(sum((MAPfull - MAPrecon) .^ 2,dims=1)')./fsn[1:10:end], "x", color = qmcolor)
    semilogy(sol1.t[1:10:end], sqrt.(sum((DATAfull - DATArecon) .^ 2,dims=1)')./fsn[1:10:end], "v", markersize=8, fillstyle="none", color = qmcolor)
    ax = gca()
    ax.tick_params(labelsize=16)
    ylim([1e-8,1e-2])
    xlim([-Tstep/2,Tstep*npoints])
    xlabel("time [s]",size=16)
    ylabel("Relative error",size=16)
    subplots_adjust(top=0.95, bottom=0.15, left=0.15, right=0.75)
    legend(bbox_to_anchor=[1.05, 1], loc=2, borderaxespad=0, ("VF O(3)", "MAP O(3)", "DATA O(3)", "VF O(5)", "MAP O(5)", "DATA O(5)", "VF O(7)", "MAP O(7)", "DATA O(7)", "VF O(9)", "MAP O(9)", "DATA O(9)"))
    text(21, 0.003, "(b)", size=18)
    savefig("ShawPierreReconstrucBack.pdf", format="pdf")

end # for order ...

end # module

module u

using Printf
using PyPlot
using DifferentialEquations
using LinearAlgebra

include("../src/polymethods.jl")
include("shawpierrevectorfield.jl")

function vfrhsPar(y)
    x = zero(y)
    vfrhs!(x, y, 0, 0)
    return x
end

include("../src/isfpolymodel.jl")
include("../src/vfmethods.jl")

function ISFPlot(Ua, Ub, a, b, St, DF)
    vals, adjvecs = eigen(collect(DF'))
    vals, vecs = eigen(DF)
    
    # assuming that adjvecs are complex conjugate pairs
    adjproj = vcat(real(adjvecs[:,1])', imag(adjvecs[:,1])', real(adjvecs[:,3])', imag(adjvecs[:,3])')

    function redstep(z0)
        return St(z0)
    end
    
    function fullstep(x0)
        Tstep = 0.8
        tspan = (0.0,Tstep) # 200 intervals with T=0.8 as in Proc Roy Soc Paper
        prob = ODEProblem(vfrhs!, x0, tspan)
        sol = solve(prob, RadauIIA5(), saveat = Tstep/10, abstol = 1e-10, reltol = 1e-10)
        return sol(Tstep)
    end

    yrng = range(-.005, .005, length=12)
    srng = range(-.005, .005, length=12)
    z0 = [0.08, 0.0]
    ndim = 4
    nstep = 4
    # array with forward simulation
    xarr = zeros(length(srng),nstep+1, ndim)
    # array with forward mapping
    parr = zeros(length(yrng),nstep+1, ndim)
    
    # The initial condition must be on a single leaf
    # However we need this to be centered about the SSM
    for p=1:length(srng)
        xarr[p,1,:] = adjproj*a([z0[1], z0[2], srng[p], 0])
    end
    for p=1:length(yrng)
        parr[p,1,:] = adjproj*a([z0[1], z0[2], yrng[p], 0])
    end
    
    # plot3D(x3arr, x4arr, x1arr, color=[0,1,0,1])
    # scatter3D(x3arr, x4arr, x1arr, color=[0,1,0,1])
    for k = 2:nstep+1
        for p=1:length(yrng)
            xarr[p,k,:] = adjproj*fullstep(inv(adjproj)*xarr[p,k-1,:])
        end
        z1 = redstep(z0)
        # @show z0
        for p=1:length(yrng)
            parr[p,k,:] = adjproj*a([z1[1],z1[2], yrng[p], 0])
        end
        z0[:] = z1
    end
    for k = 1:nstep+1
        scatter3D(xarr[:,k,3], xarr[:,k,4], xarr[:,k,1], color=[1,0,0,1])
        plot3D(parr[:,k,3], parr[:,k,4], parr[:,k,1], color=[0,0,1,1])
    end

    # now the invariant manifold
    
    xrng = range(-.2, .2, length=12)
    yrng = range(-.2, .2, length=12)
    xarr = zeros(length(xrng), length(yrng), ndim)
    for p=1:length(xrng)
        for q=1:length(yrng)
            xarr[p,q,:] = (adjproj*b([0,0,xrng[p],yrng[q]]))
#             xarr[p,q,2] = (adjproj*b([0,0,xrng[p],yrng[q]]))[2]
#             xarr[p,q,3] = xrng[p]
#             xarr[p,q,4] = yrng[q]
        end
    end
    surf(xarr[:,:,3], xarr[:,:,4], xarr[:,:,1], color=[0,1,0,0.5])
    xlabel("\$x_3\$",fontsize=16)
    ylabel("\$x_4\$",fontsize=16)
    zlabel("\$x_1\$",fontsize=16)
#     zlim([-0.02,0.02])
end

function fRealImag(S, r)
    Sr = S([r, 0.0])
    return Sr[1]/r, Sr[2]/r
end

# frequency, damping
function freqdamp(S, r, T)
    fr, fi = fRealImag(S, r)
    omega = abs(atan(fi/fr)/T)
    dm = -log(sqrt(fr^2 + fi^2))/T/omega
    return omega, dm
end

# amplitude
function ISFamplitude(Wz, r)
    ndim = size(Wz.mexp,1)
    arg = zero(Wz.W[:,1])
    maxamp = 0.0
    for q in range(0, 2*pi, length=24)
        arg[1] = r*cos(q)
        arg[2] = r*sin(q)
#         @show arg
        nr = maximum(abs.(Wz(arg)))
        if nr > maxamp
            maxamp = nr
        end
    end
    return maxamp
end

# amplitude
function SSMamplitude(Wz, r)
    ndim = size(Wz.mexp,1)
    maxamp = 0.0
    for q in range(0, 2*pi, length=24)
        nr = maximum(abs.(Wz([ r*cos(q), r*sin(q)])))
        if nr > maxamp
            maxamp = nr
        end
    end
    return maxamp
end

function backbones(S, Wz, r, T)
    amps = zero(r)
    freq = zero(r)
    damp = zero(r)
    for k=1:length(r)
        amps[k] = ISFamplitude(Wz, r[k])
        freq[k], damp[k] = freqdamp(S, r[k], T)
    end
    return amps, freq, damp
end

function VFfreqdamp(S, r)
    Sr = real(S([r, 0.0])/r)
    omega = abs(Sr[2])
    dm = -Sr[1]/omega
    return omega, dm
end

function VFISFbackbones(S, Wz, r)
    amps = zero(r)
    freq = zero(r)
    damp = zero(r)
    for k=1:length(r)
        amps[k] = ISFamplitude(Wz, r[k])
        freq[k], damp[k] = VFfreqdamp(S, r[k])
    end
    return amps, freq, damp
end

function VFSSMbackbones(S, Wz, r)
    amps = zero(r)
    freq = zero(r)
    damp = zero(r)
    for k=1:length(r)
        amps[k] = SSMamplitude(Wz, r[k])
        freq[k], damp[k] = VFfreqdamp(S, r[k])
    end
    return amps, freq, damp
end

# This makes a submersion into a set of graphs parametrised by z
function SubmersionToGraph(U)
    U = PolyModel(U.mexp, U.W)
    UL = PolyGetLinearPart(U)
    ULH = vcat(UL, zeros(typeof(UL[1]), size(UL,2) - size(UL,1), size(UL,2)))
    F = svd(ULH)
    Vpar = F.Vt[1:size(UL,1),:]'
    Vpar = Vpar*inv(UL*Vpar)
    Vperp = F.Vt[size(UL,1)+1:end,:]'
    
    # nonlinear part
    UN = PolyCopy(U)
    PolySetLinearPart!(UN, zero(UL))
    
#     order = max(PolyOrder(U.mexp)+2, 9)
    order = PolyOrder(U.mexp)
    # first variable is z, second variable is y
    # the identity in y 
    yID = PolyModel(size(UL,2), size(UL,2) - size(UL,1), order)
    yIDlin = zero(PolyGetLinearPart(yID))
    @show size(yIDlin)
    @show size(yIDlin[:,size(UL,1)+1:end])
    yIDlin[:,size(UL,1)+1:end] .= one(yIDlin[:,size(UL,1)+1:end])
    PolySetLinearPart!(yID, yIDlin)
    # the identity in z
    zID = PolyModel(size(UL,2), size(UL,1), order)
    zIDlin = zero(UL)
    zIDlin[:,1:size(UL,1)] .= one(zIDlin[:,1:size(UL,1)])
    PolySetLinearPart!(zID, zIDlin)
    
    # define the g function
    g = PolyModel(size(UL,2), size(UL,1), order)
    glin = zero(UL)
    glin[1:size(UL,1),1:size(UL,1)] = one(glin[1:size(UL,1),1:size(UL,1)])
    PolySetLinearPart!(g, glin)

    Wz = PolyModel(size(UL,2), size(UL,2), order)
    UNWz = PolyZero(g)
    gi = PolyZero(g)
    # out, out, in2
    multabUNWz = mulTable(UNWz.mexp, UNWz.mexp, Wz.mexp)
    # iteration starts
    for k=1:order+1
        Wz.W .= Vperp*yID.W + Vpar*g.W
        PolySubs!(UNWz, UN, Wz, multabUNWz)
        gi.W .= zID.W - UNWz.W
        @show maximum(abs.(gi.W .- g.W))
        g.W .= gi.W
    end
    return Wz
end

# using FileIO
using BSON: @load

figure(3)
cla()
figure(4)
cla()
figure(5)
cla()
figure(6)
cla()


@load "ShawPierreDataTest.bson" xs ys

xst = [xs[k] for k=1:length(xs)]
yst = [ys[k] for k=1:length(ys)]

@load "ShawPierreDataTrain.bson" xs ys

xs = [xs[k] for k=1:length(xs)]
ys = [ys[k] for k=1:length(ys)]
DT = 0.8

let
# list of legends
legs = []

orders = [7 5]

for order in orders
for SIGMA in [2 3]
    @load "MapFitShawPierreS$(SIGMA)O$(order).bson" DT Wo1 Ro1 Ws1 Rs1 W1 R1 U1 S1 mpar1 mexp1 Wo2 Ro2 Ws2 Rs2 W2 R2 U2 S2 mpar2 mexp2

    println("ISF O=$(order) S=$(SIGMA) DATA TRAIN residuals")
    println("Mode 1 ",(@sprintf "%.4e" sum([sqrt(sum((U2(ys[k]) - S2(U2(xs[k]))).^2)./dot(xs[k],xs[k])) for k=1:length(xs)])/length(xs)))
    println("Mode 2 ",(@sprintf "%.4e" sum([sqrt(sum((U1(ys[k]) - S1(U1(xs[k]))).^2)./dot(xs[k],xs[k])) for k=1:length(xs)])/length(xs)))
    println("ISF O=$(order) S=$(SIGMA) DATA TEST residuals")
    println("Mode 1 ",(@sprintf "%.4e" sum([sqrt(sum((U2(yst[k]) - S2(U2(xst[k]))).^2)./dot(xst[k],xst[k])) for k=1:length(xst)])/length(xst)))
    println("Mode 2 ",(@sprintf "%.4e" sum([sqrt(sum((U1(yst[k]) - S1(U1(xst[k]))).^2)./dot(xst[k],xst[k])) for k=1:length(xst)])/length(xst)))

    qmcolor = "orange"
    if order == 3 && SIGMA == 2
        qmcolor = "gray"
    elseif order == 3 && SIGMA == 3
        qmcolor = "yellow"
    elseif order == 5 && SIGMA == 2
        qmcolor = "red"
    elseif order == 7 && SIGMA == 2
        qmcolor = "green"
    elseif order == 5 && SIGMA == 3
        qmcolor = "blue"
    elseif order == 7 && SIGMA == 3
        qmcolor = "orange"
    elseif order == 11
        qmcolor = "black"
    else
        qmcolor = "yellow"
    end    
 
    hd1 = SubmersionToGraph(Wo1)
    hd2 = SubmersionToGraph(Wo2)
    h1 = SubmersionToGraph(U1)
    h2 = SubmersionToGraph(U2)
    rad = range(1e-6, 0.1, length=24)

    if order == 7 && SIGMA == 2
        
        # this makes a polynomial of the model
        F = PolyModel(order, 4, vfrhsPar)

        vfWo, vfRo, vfW, vfR = ISFVFCalc(F, [1, 2], [])
        vfWot, vfRot, vfWt, vfRt = ISFVFCalc(F, [3, 4], [])
        
        vfWz = SubmersionToGraph(vfWo)
        vfWzt = SubmersionToGraph(vfWot)

        vfWout, vfRout = SSMVFCalc(F, [1, 2], [])
        vfWoutt, vfRoutt = SSMVFCalc(F, [3, 4], [])

        fig, ax = subplots(num=3, clear=true, figsize=(6,4))
        amps, freq, damp = VFISFbackbones(vfRo, vfWz, rad)
        plot(freq, amps, "x-", color=qmcolor)
        legs = vcat(legs, ["O($(order)) VF ISF"])
        amps, freq, damp = VFSSMbackbones(vfRout, vfWout, rad)
        plot(freq, amps, "d-", color=qmcolor)
        legs = vcat(legs, ["O($(order)) VF SSM"])
        text(-0.2, 1.0, "(b)", horizontalalignment="left", verticalalignment="top", transform=ax.transAxes, fontsize=18)
#         @load "isfShawPierreOK_RAND5_NN_elu_M1_L3_N12_S2_CURVES.bson" amps freq damp
#         plot(abs.(freq), amps, "--", color="black")
#         legs = vcat(legs, ["S($(SIGMA)) NN ISF"])

        fig, ax = subplots(num=4, clear=true, figsize=(6,4))
        amps, freq, damp = VFISFbackbones(vfRot, vfWzt, rad)
        plot(freq, amps, "x-", color=qmcolor)
        amps, freq, damp = VFSSMbackbones(vfRoutt, vfWoutt, rad)
        plot(freq, amps, "d-", color=qmcolor)
        text(-0.2, 1.0, "(a)", horizontalalignment="left", verticalalignment="top", transform=ax.transAxes, fontsize=18)
#         @load "isfShawPierreOK_RAND5_NN_elu_M3_L3_N12_S2_CURVES.bson" amps freq damp
#         plot(abs.(freq), amps, "--", color="black")


        fig, ax = subplots(num=5, clear=true, figsize=(6,4))
        amps, freq, damp = VFISFbackbones(vfRo, vfWz, rad)
        plot(damp, amps, "x-", color=qmcolor)
        amps, freq, damp = VFSSMbackbones(vfRout, vfWout, rad)
        plot(damp, amps, "d-", color=qmcolor)
        text(-0.2, 1.0, "(d)", horizontalalignment="left", verticalalignment="top", transform=ax.transAxes, fontsize=18)
#         @load "isfShawPierreOK_RAND5_NN_elu_M1_L3_N12_S2_CURVES.bson" amps freq damp
#         plot(damp, amps, "--", color="black")


        fig, ax = subplots(num=6, clear=true, figsize=(6,4))
        amps, freq, damp = VFISFbackbones(vfRot, vfWzt, rad)
        plot(damp, amps, "x-", color=qmcolor)
        amps, freq, damp = VFSSMbackbones(vfRoutt, vfWoutt, rad)
        plot(damp, amps, "d-", color=qmcolor)
        text(-0.2, 1.0, "(c)", horizontalalignment="left", verticalalignment="top", transform=ax.transAxes, fontsize=18)
#         @load "isfShawPierreOK_RAND5_NN_elu_M3_L3_N12_S2_CURVES.bson" amps freq damp
#         plot(damp, amps, "--", color="black")
    end
    
    figure(3, figsize=(6,4))
    amps, freq, damp = backbones(S1, h1, rad, 0.8)
    plot(freq, amps, color=qmcolor)
    legs = vcat(legs, ["O($(order)) σ=$(SIGMA) DATA ISF"])
    xlabel("Frequency (rad/s)",fontsize=16)
    ylabel("Amplitude",fontsize=16)
    ax = gca()
    ax.tick_params(labelsize=16)
    legend(legs)
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment="right")
    tight_layout()
    savefig("ShawPierreBackBoneModeTwo$(order).pdf", format="pdf")
    
    figure(4, figsize=(6,4))
    amps, freq, damp = backbones(S2, h2, rad, 0.8)
    plot(freq, amps, color=qmcolor)
    xlabel("Frequency (rad/s)",fontsize=16)
    ylabel("Amplitude",fontsize=16)
    ax = gca()
    ax.tick_params(labelsize=14)
    legend(legs)
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment="right")
    tight_layout()
    savefig("ShawPierreBackBoneModeOne$(order).pdf", format="pdf")

    figure(5, figsize=(6,4))
    amps, freq, damp = backbones(S1, h1, rad, 0.8)
    plot(damp, amps, color=qmcolor)
    xlabel("Damping ratio",fontsize=16)
    ylabel("Amplitude",fontsize=16)
    ax = gca()
    ax.tick_params(labelsize=14)
    legend(legs)
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment="right")
    tight_layout()
    savefig("ShawPierreDampingModeTwo$(order).pdf", format="pdf")

    figure(6, figsize=(6,4))
    amps, freq, damp = backbones(S2, h2, rad, 0.8)
    plot(damp, amps, color=qmcolor)
    xlabel("Damping ratio",fontsize=16)
    ylabel("Amplitude",fontsize=16)
    ax = gca()
    ax.tick_params(labelsize=14)
    legend(legs)
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment="right")
    tight_layout()
    savefig("ShawPierreDampingModeOne$(order).pdf", format="pdf")

end
end
end
end # module

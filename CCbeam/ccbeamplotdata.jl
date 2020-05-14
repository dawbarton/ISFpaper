using Printf
using PyPlot
using DifferentialEquations
using LinearAlgebra
using ForwardDiff

include("../src/polymethods.jl")
# include("shawpierrevectorfield.jl")
# 
# function vfrhsPar(y)
#     x = zero(y)
#     vfrhs!(x, y, 0, 0)
#     return x
# end

include("../src/isfpolymodel.jl")
# include("vfmethods.jl")

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

function Wz(U, x; init=nothing)
    # Linear part
    UL = ForwardDiff.jacobian(U, zero(x))
    # stack a zero to make it square 
    ULH = vcat(UL, zeros(typeof(UL[1]), size(UL,2) - size(UL,1), size(UL,2)))
    F = svd(ULH)
    Vpar = F.Vt[1:size(UL,1),:]'
    Vpar = Vpar*inv(UL*Vpar)
    Vperp = F.Vt[size(UL,1)+1:end,:]'

#     @show UL*Vpar
#     @show UL*Vperp
    jac = xx -> ForwardDiff.jacobian(U, xx)
    z = x[1:size(UL,1)]
    y = x[size(UL,1)+1:end]
    if init == nothing
        gzy0 = copy(z)
    else
        gzy0 = init
    end
    deltagzy = zero(gzy0)
    it = 1
    while true
        x1 = Vperp*y + Vpar*gzy0
#             @show jac(x1)*Vpar
        deltagzy .= (jac(x1)*Vpar) \ (U(x1) .- z)
        gzy0 .-= deltagzy
        if maximum(abs.(deltagzy)) < 1e-8
            break
        end
        it += 1
        if it > 1200
            println("Wz: max iteration exceeded. Error = ", maximum(abs.(deltagzy)))
            x1 .= NaN
            break
        end
    end
    x1 = Vperp*y + Vpar*gzy0
    if init == nothing
        return x1
    else
        return x1, gzy0
    end
end

function WzSSM(U1, U2, U3, z; init=nothing)
    Uc = PolyModel(U1.mexp, zeros(typeof(U1.W[1]), size(U1.W,1) + size(U2.W,1) + size(U3.W,1), size(U1.W,2)))
    Uc.W[1:size(U1.W,1),:] = U1.W
    Uc.W[1+size(U1.W,1):size(U1.W,1)+size(U2.W,1),:] = U2.W
    Uc.W[1+size(U1.W,1)+size(U2.W,1):end,:] = U3.W
    jac = xx -> ForwardDiff.jacobian(Uc, xx)
    x0 = zeros(size(U1.W,1) + size(U2.W,1) + size(U3.W,1))
    x0[1:size(U1.W,1)] = z
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
    
function fRealImag(S, r)
    Sr = S([r, 0.0])
    return Sr[1]/r, -Sr[2]/r
end

# frequency, damping
function freqdamp(S, r, T)
    fr, fi = fRealImag(S, r)
    omega = angle(fr+1im*fi)/T
    dm = -log(sqrt(fr^2 + fi^2))/T/abs(omega)
    return omega, dm
end

function ISFLeafAmplitude(Wz, r, ndim)
    amps = zero(r)
    arg = zeros(ndim)
    gprev = zeros(2)
    for k=1:length(r)
        maxamp = 0.0
        for q in range(0, 2*pi, length=24)
            arg[1] = r[k]*cos(q)
            arg[2] = r[k]*sin(q)
            x, gprev = Wz(arg)
            nr = maximum(abs.(x))
            if nr > maxamp
                maxamp = nr
            end
        end
        amps[k] = maxamp
    end
    return amps
end

function ISFManifAmplitude(U1, U2, U3, r)
    amps = zero(r)
    arg = zeros(2)
    thetas = range(0, 2*pi, length=24)
    x = WzSSM(U1, U2, U3, [0.0 0.0])
    xprev = [zero(x) for k=1:24]
    for k=1:length(r)
        maxamp = 0.0
        for l=1:length(thetas)
            q = thetas[l]
            arg[1] = r[k]*cos(q)
            arg[2] = r[k]*sin(q)
            if any(isnan.(xprev[l]))
                WzSSM(U1, U2, U3, arg)
            else
                x .= WzSSM(U1, U2, U3, arg; init = xprev[l])
            end
            xprev[l] .= x
            nr = maximum(abs.(x))
            if nr > maxamp
                maxamp = nr
            end
        end
        amps[k] = maxamp
    end
    return amps
end

function ISFamplitude(U, r, ndim)
    amps = zero(r)
    arg = zeros(ndim)
    gprev = zeros(2)
    for k=1:length(r)
        maxamp = 0.0
        for q in range(0, 2*pi, length=24)
            arg[1] = r[k]*cos(q)
            arg[2] = r[k]*sin(q)
            x, gprev = Wz(U, arg, init=gprev)
            nr = maximum(abs.(x))
            if nr > maxamp
                maxamp = nr
            end
        end
        amps[k] = maxamp
    end
    return amps
end

function ISFLeafBackbones(S, Wz, r, T; ndim = 6)
    amps = ISFLeafAmplitude(Wz, r, ndim)
    freq = zero(r)
    damp = zero(r)
    for k=1:length(r)
        freq[k], damp[k] = freqdamp(S, r[k], T)
    end
    return amps, freq, damp
end

function ISFManifBackbones(S, U1, U2, U3, r, T)
    amps = ISFManifAmplitude(U1, U2, U3, r)
    freq = zero(r)
    damp = zero(r)
    for k=1:length(r)
        freq[k], damp[k] = freqdamp(S, r[k], T)
    end
    return amps, freq, damp
end

function ISFbackbones(S, U, r, T; ndim = 6)
    amps = ISFamplitude(U, r, ndim)
    freq = zero(r)
    damp = zero(r)
    for k=1:length(r)
        freq[k], damp[k] = freqdamp(S, r[k], T)
    end
    return amps, freq, damp
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

function SSMbackbones(S, Wz, r, T)
    amps = zero(r)
    freq = zero(r)
    damp = zero(r)
    for k=1:length(r)
        amps[k] = SSMamplitude(Wz, r[k])
        freq[k], damp[k] = freqdamp(S, r[k], T)
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
using BSON: @load, @save

figure(3)
cla()
figure(4)
cla()
figure(5)
cla()
figure(6)
cla()
figure(7)
cla()
figure(8)
cla()

include("../src/mapmethods.jl")
    @load "CCbeamData.bson" nnm1 nnm2 nnm3

    # individal points
    dat1 = [nnm1[2,1+k:6+k] for k=0:size(nnm1,2)-6]
    dat2 = [nnm2[2,1+k:6+k] for k=0:size(nnm2,2)-6]
    dat3 = [nnm3[2,1+k:6+k] for k=0:size(nnm3,2)-6]
    DT = nnm1[1,2] - nnm1[1,1]
    xs = vcat(dat1[1:end-1], dat2[1:end-1], dat3[1:end-1])
    ys = vcat(dat1[2:end], dat2[2:end], dat3[2:end])


let
# list of legends
legs = []

@load "FreeDecay.bson" decay
@load "NNM1.bson" dec1 appr1
# convert [mm] into [m/s]
dec1 = hcat(dec1[:,1]*2*pi, 2*pi*dec1[:,2].*dec1[:,1]/1000)
appr1 = hcat(appr1[:,1]*2*pi, 2*pi*appr1[:,2].*appr1[:,1]/1000)

@load "NNM3.bson" dec3 appr3
dec3 = hcat(dec3[1,:]*2*pi, 2*pi*(dec3[2,:].*dec3[1,:])/1000)
appr3 = hcat(appr3[1,:]*2*pi, 2*pi*(appr3[2,:].*appr3[1,:])/1000)

POLYINV = true
MANIF = false
order = 3
SIGMA = 1

F = PolyModel(order, xs, ys, sigma = SIGMA)
Wo1, Ro1, W1, R1 = ISFCalc(F, [1, 2], [])
Wo2, Ro2, W2, R2 = ISFCalc(F, [3, 4], [])
Wo3, Ro3, W3, R3 = ISFCalc(F, [5, 6], [])
Ws1, Rs1 = SSMCalc(F, [1, 2])
Ws2, Rs2 = SSMCalc(F, [3, 4])
Ws3, Rs3 = SSMCalc(F, [5, 6])

println("ISF MAP residuals")
println("Mode 3 ",(@sprintf "%.4e" sum([sqrt(sum((Wo1(ys[k]) - Ro1(Wo1(xs[k]))).^2)./dot(xs[k],xs[k])^SIGMA) for k=1:length(xs)])/length(xs)))
println("Mode 2 ",(@sprintf "%.4e" sum([sqrt(sum((Wo2(ys[k]) - Ro2(Wo2(xs[k]))).^2)./dot(xs[k],xs[k])^SIGMA) for k=1:length(xs)])/length(xs)))
println("Mode 1 ",(@sprintf "%.4e" sum([sqrt(sum((Wo3(ys[k]) - Ro3(Wo3(xs[k]))).^2)./dot(xs[k],xs[k])^SIGMA) for k=1:length(xs)])/length(xs)))

function PrintFD(n, R, DT)
    rmat = PolyGetLinearPart(PolyModel(R.mexp,R.W))
    freq = abs.(angle.(eigvals(rmat))/DT)
    damp = -log.(abs.(eigvals(rmat)))./(DT*freq)
    println("Mode $(n) freq = ", (@sprintf "%.6e" freq[1]), " damp = ", (@sprintf "%.6e" damp[1]))
end

function spectralCoeff(R1, R2, R3)
    rmat1 = PolyGetLinearPart(PolyModel(R1.mexp,R1.W))
    rmat2 = PolyGetLinearPart(PolyModel(R2.mexp,R2.W))
    rmat3 = PolyGetLinearPart(PolyModel(R3.mexp,R3.W))
    leg1 = log.(abs.(eigvals(rmat1)))
    leg2 = log.(abs.(eigvals(rmat2)))
    leg3 = log.(abs.(eigvals(rmat3)))
    beth1 = minimum(leg1)/maximum([leg1 leg2 leg3])
    beth2 = minimum(leg2)/maximum([leg1 leg2 leg3])
    beth3 = minimum(leg3)/maximum([leg1 leg2 leg3])
    println("SPC1 = ", (@sprintf "%.4e" beth1), " SPC2 = ", (@sprintf "%.4e" beth2), " SPC3 = ", (@sprintf "%.6e" beth3))
end

println("SSM MAP frequencies")
PrintFD(3, Rs1, DT)
PrintFD(2, Rs2, DT)
PrintFD(1, Rs3, DT)
println("ISF MAP frequencies")
PrintFD(3, R1, DT)
PrintFD(2, R2, DT)
PrintFD(1, R3, DT)
spectralCoeff(R3, R2, R1)

if POLYINV
    hm1 = SubmersionToGraph(Wo1)
    hm2 = SubmersionToGraph(Wo2)
    hm3 = SubmersionToGraph(Wo3)
    algname = "PolyInv"
elseif MANIF
    algname = "Manif"
else
    algname = "PolyRoot"
end

    rad = range(1e-6, 0.8, length=60)
   
    fig, ax = subplots(num=3, clear=true, figsize=(6,4))
    plot(dec3[:,1], dec3[:,2], "-.", color="gray")
    legs = vcat(legs, ["Decay"])
    plot(appr3[:,1], appr3[:,2], "v", color="gray")
    legs = vcat(legs, ["Forcing"])
    xl3 = xlim()
    yl3 = ylim()
    amps, freq, damp = SSMbackbones(Rs1, Ws1, rad, DT)
    plot(freq, amps, ".", color="gray")
    legs = vcat(legs, ["O($(order)) σ=$(SIGMA) MAP SSM"])
    if POLYINV
        lab = "(b)"
        amps, freq, damp = ISFLeafBackbones(Ro1, hm1, rad, DT)
    elseif MANIF
        lab = "(f)"
        amps, freq, damp = ISFManifBackbones(Ro1, Wo1, Wo2, Wo3, rad, DT)
    else
        lab = "(d)"
        amps, freq, damp = ISFbackbones(Ro1, Wo1, rad, DT)
    end
    plot(freq, amps, "--", color="gray")
    text(-0.16, 1.0, lab, horizontalalignment="left", verticalalignment="top", transform=ax.transAxes, fontsize=18)
    legs = vcat(legs, ["O($(order)) σ=$(SIGMA) MAP ISF"])
    
    figure(4, figsize=(6,4))
    amps, freq, damp = SSMbackbones(Rs2, Ws2, rad, DT)
    plot(freq, amps, ".", color="gray")
    xl4 = xlim()
    yl4 = ylim()
    if POLYINV
        amps, freq, damp = ISFLeafBackbones(Ro2, hm2, rad, DT)
    elseif MANIF
        amps, freq, damp = ISFManifBackbones(Ro2, Wo2, Wo1, Wo3, rad, DT)
    else
        amps, freq, damp = ISFbackbones(Ro2, Wo2, rad, DT)
    end
    plot(freq, amps, "--", color="gray")

    fig, ax = subplots(num=5, clear=true, figsize=(6,4))
    plot(dec1[:,1], dec1[:,2], "-.", color="gray")
    plot(appr1[:,1], appr1[:,2], "v", color="gray")
    xl5 = xlim()
    yl5 = ylim()
    amps, freq, damp = SSMbackbones(Rs3, Ws3, rad, DT)
    plot(freq, amps, ".", color="gray")
    if POLYINV
        lab = "(a)"
        amps, freq, damp = ISFLeafBackbones(Ro3, hm3, rad, DT)
    elseif MANIF
        lab = "(e)"
        amps, freq, damp = ISFManifBackbones(Ro3, Wo3, Wo1, Wo2, rad, DT)
    else
        lab = "(c)"
        amps, freq, damp = ISFbackbones(Ro3, Wo3, rad, DT)
    end
    plot(freq, amps, "--", color="gray")
    text(-0.16, 1.0, lab, horizontalalignment="left", verticalalignment="top", transform=ax.transAxes, fontsize=18)

    figure(6, figsize=(6,4))
    amps, freq, damp = SSMbackbones(Rs1, Ws1, rad, DT)
    plot(damp, amps, ".", color="gray")
    xl6 = xlim()
    yl6 = ylim()
#     if POLYINV
#         amps, freq, damp = ISFLeafBackbones(Ro1, hm1, rad, DT)
#         plot(damp, amps, "--", color="gray")
#     else
#         amps, freq, damp = ISFbackbones(Ro1, Wo1, rad, DT)
#         plot(damp, amps, "--", color="gray")
#     end

    figure(7, figsize=(6,4))
    amps, freq, damp = SSMbackbones(Rs2, Ws2, rad, DT)
    plot(damp, amps, ".", color="gray")
    xl7 = xlim()
    yl7 = ylim()
#     if POLYINV
#         amps, freq, damp = ISFLeafBackbones(Ro2, hm2, rad, DT)
#         plot(damp, amps, "--", color="gray")
#     else
#         amps, freq, damp = ISFbackbones(Ro2, Wo2, rad, DT)
#         plot(damp, amps, "--", color="gray")
#     end

    figure(8, figsize=(6,4))
    amps, freq, damp = SSMbackbones(Rs3, Ws3, rad, DT)
    plot(damp, amps, ".", color="gray")
    xl8 = xlim()
    yl8 = ylim()
#     if POLYINV
#         amps, freq, damp = ISFLeafBackbones(Ro3, hm3, rad, DT)
#         plot(damp, amps, "--", color="gray")
#     else
#         amps, freq, damp = ISFbackbones(Ro3, Wo3, rad, DT)
#         plot(damp, amps, "--", color="gray")
#     end

for order in [3 5 7]
for SIGMA in [1]
    @load "MapFitCCbeamS$(SIGMA)O$(order).bson" DT Wo1 Ro1 Ws1 Rs1 W1 R1 U1 S1 mpar1 mexp1 Wo2 Ro2 Ws2 Rs2 W2 R2 U2 S2 mpar2 mexp2 Wo3 Ro3 Ws3 Rs3 W3 R3 U3 S3 mpar3 mexp3
    
#     @save "MapFitCCbeamS$(SIGMA)O$(order).bson" DT Wo1 Ro1 Ws1 Rs1 W1 R1 U1 S1 mpar1 mexp1 Wo2 Ro2 Ws2 Rs2 W2 R2 U2 S2 mpar2 mexp2 Wo3 Ro3 Ws3 Rs3 W3 R3 U3 S3 mpar3 mexp3
    
    println("ISF O=$(order) S=$(SIGMA) DATA residuals")
    println("Mode 1 ",(@sprintf "%.4e" sum([sqrt(sum((U3(ys[k]) - S3(U3(xs[k]))).^2)./dot(xs[k],xs[k])^SIGMA) for k=1:length(xs)])/length(xs)))
    println("Mode 2 ",(@sprintf "%.4e" sum([sqrt(sum((U2(ys[k]) - S2(U2(xs[k]))).^2)./dot(xs[k],xs[k])^SIGMA) for k=1:length(xs)])/length(xs)))
    println("Mode 3 ",(@sprintf "%.4e" sum([sqrt(sum((U1(ys[k]) - S1(U1(xs[k]))).^2)./dot(xs[k],xs[k])^SIGMA) for k=1:length(xs)])/length(xs)))
    println("ISF O=$(order) S=$(SIGMA) DATA frequencies")
    PrintFD(3, S1, DT)
    PrintFD(2, S2, DT)
    PrintFD(1, S3, DT)
    spectralCoeff(S3, R2, S1)

    qmcolor = "orange"
    if order == 3 && SIGMA == 1
        qmcolor = "red"
    elseif order == 5 && SIGMA == 1
        qmcolor = "green"
    elseif order == 7 && SIGMA == 1
        qmcolor = "blue"
    elseif order == 3 && SIGMA == 2
        qmcolor = "orange"
    elseif order == 5 && SIGMA == 2
        qmcolor = "black"
    elseif order == 7 && SIGMA == 2
        qmcolor = "purple"
    elseif order == 11
        qmcolor = "black"
    else
        qmcolor = "black"
    end    
    
    if POLYINV
        hd1 = SubmersionToGraph(U1)
        hd2 = SubmersionToGraph(U2)
        hd3 = SubmersionToGraph(U3)
    end
    
    rad = range(1e-6, 0.8, length=100)
   
    figure(3, figsize=(6,4))
    if POLYINV
        amps, freq, damp = ISFLeafBackbones(S1, hd1, rad, DT)
    elseif MANIF
        amps, freq, damp = ISFManifBackbones(S1, U1, U2, U3, rad, DT)
    else
        amps, freq, damp = ISFbackbones(S1, U1, rad, DT)
    end
    plot(freq, amps, color=qmcolor)
    legs = vcat(legs, ["O($(order)) σ=$(SIGMA) DATA ISF"])
    xlabel("Frequency (rad/s)",fontsize=16)
    ylabel("Amplitude (m/s)",fontsize=16)
    ax = gca()
    ax.tick_params(labelsize=16)
    legend(legs)
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment="right")
    xlim(xl3)
    ylim(yl3)
    tight_layout()
    savefig("CCbeam$(algname)BackBoneModeThree.pdf", format="pdf")
    
    figure(4, figsize=(6,4))
    if POLYINV
        amps, freq, damp = ISFLeafBackbones(S2, hd2, rad, DT)
    elseif MANIF
        amps, freq, damp = ISFManifBackbones(S2, U2, U1, U3, rad, DT)
    else
        amps, freq, damp = ISFbackbones(S2, U2, rad, DT)
    end
    plot(freq, amps, color=qmcolor)
    xlabel("Frequency (rad/s)",fontsize=16)
    ylabel("Amplitude (m/s)",fontsize=16)
    ax = gca()
    ax.tick_params(labelsize=16)
    legend(legs[3:end])
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment="right")
    xlim(xl4)
    ylim(yl4)
    tight_layout()
    savefig("CCbeam$(algname)BackBoneModeTwo.pdf", format="pdf")

    figure(5, figsize=(6,4))
    if POLYINV
        amps, freq, damp = ISFLeafBackbones(S3, hd3, rad, DT)
    elseif MANIF
        amps, freq, damp = ISFManifBackbones(S3, U3, U1, U2, rad, DT)
    else
        amps, freq, damp = ISFbackbones(S3, U3, rad, DT)
    end
    plot(freq, amps, color=qmcolor)
    xlabel("Frequency (rad/s)",fontsize=16)
    ylabel("Amplitude (m/s)",fontsize=16)
    ax = gca()
    ax.tick_params(labelsize=16)
    legend(legs)
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment="right")
    xlim(xl5)
    ylim(yl5)
    tight_layout()
    savefig("CCbeam$(algname)BackBoneModeOne.pdf", format="pdf")

    figure(6, figsize=(6,4))
    if POLYINV
        amps, freq, damp = ISFLeafBackbones(S1, hd1, rad, DT)
    elseif MANIF
        amps, freq, damp = ISFManifBackbones(S1, U1, U2, U3, rad, DT)
    else
        amps, freq, damp = ISFbackbones(S1, U1, rad, DT)
    end
    plot(damp, amps, color=qmcolor)
    xlabel("Damping ratio",fontsize=16)
    ylabel("Amplitude",fontsize=16)
    ax = gca()
    ax.tick_params(labelsize=16)
    legend(legs)
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment="right")
    xlim(xl6)
    ylim(yl6)
    tight_layout()
    savefig("CCbeam$(algname)DampingModeThree.pdf", format="pdf")

    figure(7, figsize=(6,4))
    if POLYINV
        amps, freq, damp = ISFLeafBackbones(S2, hd2, rad, DT)
    elseif MANIF
        amps, freq, damp = ISFManifBackbones(S2, U2, U1, U3, rad, DT)
    else
        amps, freq, damp = ISFbackbones(S2, U2, rad, DT)
    end
    plot(damp, amps, color=qmcolor)
    xlabel("Damping ratio",fontsize=16)
    ylabel("Amplitude",fontsize=16)
    ax = gca()
    ax.tick_params(labelsize=16)
    legend(legs[3:end])
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment="right")
    xlim(xl7)
    ylim(yl7)
    tight_layout()
    savefig("CCbeam$(algname)DampingModeTwo.pdf", format="pdf")

    figure(8, figsize=(6,4))
    if POLYINV
        amps, freq, damp = ISFLeafBackbones(S3, hd3, rad, DT)
    elseif MANIF
        amps, freq, damp = ISFManifBackbones(S3, U3, U1, U2, rad, DT)
    else
        amps, freq, damp = ISFbackbones(S3, U3, rad, DT)
    end
    plot(damp, amps, color=qmcolor)
    xlabel("Damping ratio",fontsize=16)
    ylabel("Amplitude",fontsize=16)
    ax = gca()
    ax.tick_params(labelsize=16)
    legend(legs)
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment="right")
    xlim(xl8)
    ylim(yl8)
    tight_layout()
    savefig("CCbeam$(algname)DampingModeOne.pdf", format="pdf")

end
end
end

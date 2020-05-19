module u
using Printf
using PyPlot
using DifferentialEquations
using LinearAlgebra
using ForwardDiff

include("../src/polymethods.jl")
include("../src/isfpolymodel.jl")
include("../src/postprocess.jl")
include("../src/mapmethods.jl")

NAME = "CCbeam"

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

@load "NNM1.bson" dec1 appr1
# convert [mm] into [m/s]
dec1 = hcat(dec1[:,1]*2*pi, 2*pi*dec1[:,2].*dec1[:,1]/1000)
appr1 = hcat(appr1[:,1]*2*pi, 2*pi*appr1[:,2].*appr1[:,1]/1000)

@load "NNM3.bson" dec3 appr3
dec3 = hcat(dec3[1,:]*2*pi, 2*pi*(dec3[2,:].*dec3[1,:])/1000)
appr3 = hcat(appr3[1,:]*2*pi, 2*pi*(appr3[2,:].*appr3[1,:])/1000)

POLYINV = false # this inverts the polynomials
MANIF = true # this computes SSM by newton iteration
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

    rad = range(1e-6, 0.7, length=60)
   
    fig, ax = subplots(num=3, clear=true, figsize=(6,4))
    plot(dec3[:,1], dec3[:,2], "-.", color="gray")
    legs = vcat(legs, ["Decay"])
    plot(appr3[:,1], appr3[:,2], "v", color="gray")
    legs = vcat(legs, ["Forcing"])
    xl3 = xlim()
    yl3 = ylim()
    amps, freq, damp = SSMBackbones(Rs1, Ws1, rad, DT)
    plot(freq, amps, ".", color="gray")
    legs = vcat(legs, ["O($(order)) σ=$(SIGMA) MAP SSM"])
    if POLYINV
        lab = "(b)"
#         amps, freq, damp = ISFGraphBackbones(Ro1, hm1, rad, DT)
    elseif MANIF
        lab = "(f)"
#         amps, freq, damp = ISFManifNewtonBackbones(Ro1, (Wo1, Wo2, Wo3), rad, DT)
    else
        lab = "(d)"
#         amps, freq, damp = ISFGraphNewtonBackbones(Ro1, Wo1, rad, DT)
    end
#     plot(freq, amps, "--", color="gray")
#     legs = vcat(legs, ["O($(order)) σ=$(SIGMA) MAP ISF"])
    text(-0.16, 1.0, lab, horizontalalignment="left", verticalalignment="top", transform=ax.transAxes, fontsize=18)
    
    figure(4, figsize=(6,4))
    amps, freq, damp = SSMBackbones(Rs2, Ws2, rad, DT)
    plot(freq, amps, ".", color="gray")
    xl4 = xlim()
    yl4 = ylim()
#     if POLYINV
#         amps, freq, damp = ISFGraphBackbones(Ro2, hm2, rad, DT)
#     elseif MANIF
#         amps, freq, damp = ISFManifNewtonBackbones(Ro2, (Wo2, Wo1, Wo3), rad, DT)
#     else
#         amps, freq, damp = ISFGraphNewtonBackbones(Ro2, Wo2, rad, DT)
#     end
#     plot(freq, amps, "--", color="gray")

    fig, ax = subplots(num=5, clear=true, figsize=(6,4))
    plot(dec1[:,1], dec1[:,2], "-.", color="gray")
    plot(appr1[:,1], appr1[:,2], "v", color="gray")
    xl5 = xlim()
    yl5 = ylim()
    amps, freq, damp = SSMBackbones(Rs3, Ws3, rad, DT)
    plot(freq, amps, ".", color="gray")
    if POLYINV
        lab = "(a)"
#         amps, freq, damp = ISFGraphBackbones(Ro3, hm3, rad, DT)
    elseif MANIF
        lab = "(e)"
#         amps, freq, damp = ISFManifNewtonBackbones(Ro3, (Wo3, Wo1, Wo2), rad, DT)
    else
        lab = "(c)"
#         amps, freq, damp = ISFGraphNewtonBackbones(Ro3, Wo3, rad, DT)
    end
#     plot(freq, amps, "--", color="gray")
    text(-0.16, 1.0, lab, horizontalalignment="left", verticalalignment="top", transform=ax.transAxes, fontsize=18)

    figure(6, figsize=(6,4))
    amps, freq, damp = SSMBackbones(Rs1, Ws1, rad, DT)
    plot(damp, amps, ".", color="gray")
    xl6 = xlim()
    yl6 = ylim()
#     if POLYINV
#         amps, freq, damp = ISFGraphBackbones(Ro1, hm1, rad, DT)
#         plot(damp, amps, "--", color="gray")
#     else
#         amps, freq, damp = ISFGraphNewtonBackbones(Ro1, Wo1, rad, DT)
#         plot(damp, amps, "--", color="gray")
#     end

    figure(7, figsize=(6,4))
    amps, freq, damp = SSMBackbones(Rs2, Ws2, rad, DT)
    plot(damp, amps, ".", color="gray")
    xl7 = xlim()
    yl7 = ylim()
#     if POLYINV
#         amps, freq, damp = ISFGraphBackbones(Ro2, hm2, rad, DT)
#         plot(damp, amps, "--", color="gray")
#     else
#         amps, freq, damp = ISFGraphNewtonBackbones(Ro2, Wo2, rad, DT)
#         plot(damp, amps, "--", color="gray")
#     end

    figure(8, figsize=(6,4))
    amps, freq, damp = SSMBackbones(Rs3, Ws3, rad, DT)
    plot(damp, amps, ".", color="gray")
    xl8 = xlim()
    yl8 = ylim()
#     if POLYINV
#         amps, freq, damp = ISFGraphBackbones(Ro3, hm3, rad, DT)
#         plot(damp, amps, "--", color="gray")
#     else
#         amps, freq, damp = ISFGraphNewtonBackbones(Ro3, Wo3, rad, DT)
#         plot(damp, amps, "--", color="gray")
#     end

for order in [3 5]
for SIGMA in [1]
    @load "MapFit$(NAME)S$(SIGMA)O$(order).bson" DT Wo1 Ro1 Ws1 Rs1 W1 R1 U1 S1 mpar1 mexp1 Wo2 Ro2 Ws2 Rs2 W2 R2 U2 S2 mpar2 mexp2 Wo3 Ro3 Ws3 Rs3 W3 R3 U3 S3 mpar3 mexp3
    
#     @save "MapFit$(NAME)S$(SIGMA)O$(order).bson" DT Wo1 Ro1 Ws1 Rs1 W1 R1 U1 S1 mpar1 mexp1 Wo2 Ro2 Ws2 Rs2 W2 R2 U2 S2 mpar2 mexp2 Wo3 Ro3 Ws3 Rs3 W3 R3 U3 S3 mpar3 mexp3
    
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
        amps, freq, damp = ISFGraphBackbones(S1, hd1, rad, DT)
    elseif MANIF
        amps, freq, damp = ISFManifNewtonBackbones(S1, (U1, U2, U3), rad, DT)
    else
        amps, freq, damp = ISFGraphNewtonBackbones(S1, U1, rad, DT)
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
    savefig("$(NAME)$(algname)BackBoneModeThree.pdf", format="pdf")
    
    figure(4, figsize=(6,4))
    if POLYINV
        amps, freq, damp = ISFGraphBackbones(S2, hd2, rad, DT)
    elseif MANIF
        amps, freq, damp = ISFManifNewtonBackbones(S2, (U2, U1, U3), rad, DT)
    else
        amps, freq, damp = ISFGraphNewtonBackbones(S2, U2, rad, DT)
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
    savefig("$(NAME)$(algname)BackBoneModeTwo.pdf", format="pdf")

    figure(5, figsize=(6,4))
    if POLYINV
        amps, freq, damp = ISFGraphBackbones(S3, hd3, rad, DT)
    elseif MANIF
        amps, freq, damp = ISFManifNewtonBackbones(S3, (U3, U1, U2), rad, DT)
    else
        amps, freq, damp = ISFGraphNewtonBackbones(S3, U3, rad, DT)
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
    savefig("$(NAME)$(algname)BackBoneModeOne.pdf", format="pdf")

    figure(6, figsize=(6,4))
    if POLYINV
        amps, freq, damp = ISFGraphBackbones(S1, hd1, rad, DT)
    elseif MANIF
        amps, freq, damp = ISFManifNewtonBackbones(S1, (U1, U2, U3), rad, DT)
    else
        amps, freq, damp = ISFGraphNewtonBackbones(S1, U1, rad, DT)
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
    savefig("$(NAME)$(algname)DampingModeThree.pdf", format="pdf")

    figure(7, figsize=(6,4))
    if POLYINV
        amps, freq, damp = ISFGraphBackbones(S2, hd2, rad, DT)
    elseif MANIF
        amps, freq, damp = ISFManifNewtonBackbones(S2, (U2, U1, U3), rad, DT)
    else
        amps, freq, damp = ISFGraphNewtonBackbones(S2, U2, rad, DT)
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
    savefig("$(NAME)$(algname)DampingModeTwo.pdf", format="pdf")

    figure(8, figsize=(6,4))
    if POLYINV
        amps, freq, damp = ISFGraphBackbones(S3, hd3, rad, DT)
    elseif MANIF
        amps, freq, damp = ISFManifNewtonBackbones(S3, (U3, U1, U2), rad, DT)
    else
        amps, freq, damp = ISFGraphNewtonBackbones(S3, U3, rad, DT)
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
    savefig("$(NAME)$(algname)DampingModeOne.pdf", format="pdf")

end
end
end
end

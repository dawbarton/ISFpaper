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
include("../src/postprocess.jl")

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

orders = [7 5 3]

for order in orders
for SIGMA in [2 3]
    @load "MapFitShawPierreS$(SIGMA)O$(order).bson" DT Wo1 Ro1 Ws1 Rs1 W1 R1 U1 S1 mpar1 mexp1 Wo2 Ro2 Ws2 Rs2 W2 R2 U2 S2 mpar2 mexp2

    qmcolor = "orange"
    if order == 3 && SIGMA == 2
        qmcolor = "gray"
    elseif order == 3 && SIGMA == 3
        qmcolor = "purple"
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
        amps, freq, damp = VFISFGraphBackbones(vfRo, vfWz, rad)
        plot(freq, amps, "x-", color=qmcolor)
        legs = vcat(legs, ["O($(order)) VF ISF"])
        amps, freq, damp = VFSSMBackbones(vfRout, vfWout, rad)
        plot(freq, amps, "d-", color=qmcolor)
        legs = vcat(legs, ["O($(order)) VF SSM"])
        text(-0.2, 1.0, "(b)", horizontalalignment="left", verticalalignment="top", transform=ax.transAxes, fontsize=18)
#         @load "isfShawPierreOK_RAND5_NN_elu_M1_L3_N12_S2_CURVES.bson" amps freq damp
#         plot(abs.(freq), amps, "--", color="black")
#         legs = vcat(legs, ["S($(SIGMA)) NN ISF"])

        fig, ax = subplots(num=4, clear=true, figsize=(6,4))
        amps, freq, damp = VFISFGraphBackbones(vfRot, vfWzt, rad)
        plot(freq, amps, "x-", color=qmcolor)
        amps, freq, damp = VFSSMBackbones(vfRoutt, vfWoutt, rad)
        plot(freq, amps, "d-", color=qmcolor)
        text(-0.2, 1.0, "(a)", horizontalalignment="left", verticalalignment="top", transform=ax.transAxes, fontsize=18)
#         @load "isfShawPierreOK_RAND5_NN_elu_M3_L3_N12_S2_CURVES.bson" amps freq damp
#         plot(abs.(freq), amps, "--", color="black")


        fig, ax = subplots(num=5, clear=true, figsize=(6,4))
        amps, freq, damp = VFISFGraphBackbones(vfRo, vfWz, rad)
        plot(damp, amps, "x-", color=qmcolor)
        amps, freq, damp = VFSSMBackbones(vfRout, vfWout, rad)
        plot(damp, amps, "d-", color=qmcolor)
        text(-0.2, 1.0, "(d)", horizontalalignment="left", verticalalignment="top", transform=ax.transAxes, fontsize=18)
#         @load "isfShawPierreOK_RAND5_NN_elu_M1_L3_N12_S2_CURVES.bson" amps freq damp
#         plot(damp, amps, "--", color="black")


        fig, ax = subplots(num=6, clear=true, figsize=(6,4))
        amps, freq, damp = VFISFGraphBackbones(vfRot, vfWzt, rad)
        plot(damp, amps, "x-", color=qmcolor)
        amps, freq, damp = VFSSMBackbones(vfRoutt, vfWoutt, rad)
        plot(damp, amps, "d-", color=qmcolor)
        text(-0.2, 1.0, "(c)", horizontalalignment="left", verticalalignment="top", transform=ax.transAxes, fontsize=18)
#         @load "isfShawPierreOK_RAND5_NN_elu_M3_L3_N12_S2_CURVES.bson" amps freq damp
#         plot(damp, amps, "--", color="black")
    end
    
    figure(3, figsize=(6,4))
    amps, freq, damp = ISFGraphBackbones(S1, h1, rad, 0.8)
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
    amps, freq, damp = ISFGraphBackbones(S2, h2, rad, 0.8)
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
    amps, freq, damp = ISFGraphBackbones(S1, h1, rad, 0.8)
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
    amps, freq, damp = ISFGraphBackbones(S2, h2, rad, 0.8)
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

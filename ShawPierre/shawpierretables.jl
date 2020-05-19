module u
using Printf
using PyPlot
using DifferentialEquations
using LinearAlgebra
using ForwardDiff

include("../src/polymethods.jl")
include("../src/isfpolymodel.jl")
include("../src/postprocess.jl")
include("../src/vfmethods.jl")

NAME = "ShawPierre"

include("shawpierrevectorfield.jl")

function vfrhsPar(y)
    x = zero(y)
    vfrhs!(x, y, 0, 0)
    return x
end

# using FileIO
using BSON: @load, @save

function toLatexExp(a, d)
#     @show a
    ex = Int(floor(log10(abs(a))))
    frac = a/(10.0^ex)
#     @show frac
    res = Int(round(frac * 10^d))
    str = string(res)
    if ex == 0
        return "\$" * str[1] * "." * str[2:end] * "\$"
    else
        return "\$" * str[1] * "." * str[2:end] * " \\times 10^{" * (@sprintf "%d" ex) * "} \$"
    end
end

function spectralCoeffs(Rtup, DT; vf=false)
    leg = zeros(length(Rtup))
    freq = zeros(length(Rtup))
    damp = zeros(length(Rtup))
    for k=1:length(Rtup)
        rmat = PolyGetLinearPart(PolyModel(Rtup[k].mexp,Rtup[k].W))
        if vf
            freq[k] = minimum(abs.(imag.(eigvals(rmat))))
            damp[k] = minimum(real.(eigvals(rmat))./freq[k])
            leg[k] = minimum(real.(eigvals(rmat)))
        else
            freq[k] = minimum(abs.(angle.(eigvals(rmat))/DT))
            damp[k] = minimum(-log.(abs.(eigvals(rmat)))./(DT*freq[k]))
            leg[k] = minimum(log.(abs.(eigvals(rmat))))
        end
    end
    for fq in freq
        print(" & ", toLatexExp(fq, 4))
    end
    for da in damp
        print(" & ", toLatexExp(da, 4))
    end
    for lg in leg
        print(" & ", toLatexExp(lg/maximum(leg), 3))
    end
    println("\\\\")
end

function residual(S, U, xs, ys)
    return sum([sqrt(sum((U(ys[k]) - S(U(xs[k]))).^2)./dot(xs[k],xs[k])) for k=1:length(xs)])/length(xs)
end

@load "ShawPierreDataTest.bson" xs ys

xst = [xs[k] for k=1:length(xs)]
yst = [ys[k] for k=1:length(ys)]

@load "ShawPierreDataTrain.bson" xs ys

xs = [xs[k] for k=1:length(xs)]
ys = [ys[k] for k=1:length(ys)]
DT = 0.8

order = 7
SIGMA = 1

F = PolyModel(order, 4, vfrhsPar)

vfWo, vfRo, vfW, vfR = ISFVFCalc(F, [1, 2], [])
vfWot, vfRot, vfWt, vfRt = ISFVFCalc(F, [3, 4], [])

println("\\begin{tabular}{l | l | l | l | l | l }\n & mode 1 & mode 2 \\\\ \n\\hline")
print("VF O($(order)) & ")
println(toLatexExp(residual(vfRot, vfWot, xs, ys), 4), " & ",
        toLatexExp(residual(vfRo, vfWo, xs, ys), 4), " \\\\")
for order in [3 5 7]
    for SIGMA in [2 3]
        @load "MapFit$(NAME)S$(SIGMA)O$(order).bson" DT Wo1 Ro1 Ws1 Rs1 W1 R1 U1 S1 mpar1 mexp1 Wo2 Ro2 Ws2 Rs2 W2 R2 U2 S2 mpar2 mexp2
        print("DATA O($(order)) \$\\sigma=$(SIGMA)\$ & ")
        println(toLatexExp(residual(S2, U2, xs, ys), 4), " & ",
            toLatexExp(residual(S1, U1, xs, ys), 4), " & ",
            toLatexExp(residual(S2, U2, xst, yst), 4), " & ",
            toLatexExp(residual(S1, U1, xst, yst), 4)," \\\\")

    end
end
println("\\end{tabular}")

println("\\begin{tabular}{l | l | l | l | l | l | l }")
println(" & \$\\omega_1\$ & \$\\omega_2\$ & \$\\zeta_1\$ & \$\\zeta_2\$ & \$\\beth_1\$ & \$\\beth_2\$ \\\\ \n\\hline")
print("VF O($(order))")
spectralCoeffs((vfRot, vfRo,), DT, vf=true)
for order in [3 5 7]
    for SIGMA in [2 3]
        @load "MapFit$(NAME)S$(SIGMA)O$(order).bson" DT Wo1 Ro1 Ws1 Rs1 W1 R1 U1 S1 mpar1 mexp1 Wo2 Ro2 Ws2 Rs2 W2 R2 U2 S2 mpar2 mexp2
        print("DATA O($(order)) \$\\sigma=$(SIGMA)\$ ")
        spectralCoeffs((S2, S1,), DT)
    end
end
println("\\end{tabular}")

end

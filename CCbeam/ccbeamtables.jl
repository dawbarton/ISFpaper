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

function PrintFD(n, R, DT)
    rmat = PolyGetLinearPart(PolyModel(R.mexp,R.W))
    freq = abs.(angle.(eigvals(rmat))/DT)
    damp = -log.(abs.(eigvals(rmat)))./(DT*freq)
    println("Mode $(n) freq = ", (@sprintf "%.6e" freq[1]), " damp = ", (@sprintf "%.6e" damp[1]))
end

function toLatexExp(a, d)
    frac = a/10^floor(log10(a))
    res = Int(round(frac * 10^d))
    str = string(res)
    if Int(floor(log10(a))) == 0
        return "\$" * str[1] * "." * str[2:end] * "\$"
    else
        return "\$" * str[1] * "." * str[2:end] * " \\times 10^{" * (@sprintf "%d" Int(floor(log10(a)))) * "} \$"
    end
end

function spectralCoeffs(Rtup, DT)
    leg = zeros(length(Rtup))
    freq = zeros(length(Rtup))
    damp = zeros(length(Rtup))
    for k=1:length(Rtup)
        rmat = PolyGetLinearPart(PolyModel(Rtup[k].mexp,Rtup[k].W))
        freq[k] = minimum(abs.(angle.(eigvals(rmat))/DT))
        damp[k] = minimum(-log.(abs.(eigvals(rmat)))./(DT*freq[k]))
        leg[k] = minimum(log.(abs.(eigvals(rmat))))
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

@load "CCbeamData.bson" nnm1 nnm2 nnm3

# individal points
dat1 = [nnm1[2,1+k:6+k] for k=0:size(nnm1,2)-6]
dat2 = [nnm2[2,1+k:6+k] for k=0:size(nnm2,2)-6]
dat3 = [nnm3[2,1+k:6+k] for k=0:size(nnm3,2)-6]
DT = nnm1[1,2] - nnm1[1,1]
xs = vcat(dat1[1:end-1], dat2[1:end-1], dat3[1:end-1])
ys = vcat(dat1[2:end], dat2[2:end], dat3[2:end])

order = 3
SIGMA = 1

F = PolyModel(order, xs, ys, sigma = SIGMA)
Wo1, Ro1, W1, R1 = ISFCalc(F, [1, 2], [])
Wo2, Ro2, W2, R2 = ISFCalc(F, [3, 4], [])
Wo3, Ro3, W3, R3 = ISFCalc(F, [5, 6], [])

println("\\begin{tabular}{l | l | l | l }\n & mode 1 & mode 2 & mode 3 \\\\ \n\\hline")
print("MAP O($(order)) & ")
println(toLatexExp(residual(Ro3, Wo3, xs, ys), 4), " & ",
        toLatexExp(residual(Ro2, Wo2, xs, ys), 4), " & ",
        toLatexExp(residual(Ro1, Wo1, xs, ys), 4), " \\\\")
for order in [3 5 7]
    for SIGMA in [1]
        @load "MapFit$(NAME)S$(SIGMA)O$(order).bson" DT Wo1 Ro1 Ws1 Rs1 W1 R1 U1 S1 mpar1 mexp1 Wo2 Ro2 Ws2 Rs2 W2 R2 U2 S2 mpar2 mexp2 Wo3 Ro3 Ws3 Rs3 W3 R3 U3 S3 mpar3 mexp3
        print("DATA O($(order)) & ")
        println(toLatexExp(residual(S3, U3, xs, ys), 4), " & ",
            toLatexExp(residual(S2, U2, xs, ys), 4), " & ",
            toLatexExp(residual(S1, U1, xs, ys), 4), " \\\\")

    end
end
println("\\end{tabular}")

println("\\begin{tabular}{l | l | l | l | l | l | l | l | l | l}")
println(" & \$\\omega_1\$ & \$\\omega_2\$ & \$\\omega_3\$ & \$\\zeta_1\$ & \$\\zeta_2\$ & \$\\zeta_3\$ & \$\\beth_1\$ & \$\\beth_2\$ & \$\\beth_3\$ \\\\ \n\\hline")
print("MAP O($(order))")
spectralCoeffs((Ro3, Ro2, Ro1,), DT)
for order in [3 5 7]
    for SIGMA in [1]
        @load "MapFit$(NAME)S$(SIGMA)O$(order).bson" DT Wo1 Ro1 Ws1 Rs1 W1 R1 U1 S1 mpar1 mexp1 Wo2 Ro2 Ws2 Rs2 W2 R2 U2 S2 mpar2 mexp2 Wo3 Ro3 Ws3 Rs3 W3 R3 U3 S3 mpar3 mexp3
        print("DATA O($(order))")
        spectralCoeffs((S3, S2, S1,), DT)
    end
end
println("\\end{tabular}")

end

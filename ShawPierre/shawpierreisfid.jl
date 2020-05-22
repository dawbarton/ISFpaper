module u

using BSON: @load, @save

NAME = "ShawPierre"

include("../src/isfpolymodel.jl")
include("../src/mapmethods.jl")

@load "$(NAME)DataTrain.bson" xs ys
xs = [xs[k] for k=1:length(xs)]
ys = [ys[k] for k=1:length(ys)]
DT = 0.8
    
for SIGMA in [2 3]
    for order in [3 5 7]
        println("SIGMA=", SIGMA, " ORDER=", order)
        F = PolyModel(order, xs, ys, sigma = SIGMA)
        # make sure that equilibriums is at zero (no constant part)
        F.W[:,end] .= 0.0
        println("RESIDUAL MAP")
        @show sum([ sum((ys[k] - F(xs[k]))) for k=1:length(xs)])

        DF = PolyGetLinearPart(F)
        @show eigvals(DF)
        
        Wo1, Ro1, W1, R1 = ISFCalc(F, [1, 2], [])
        println("residual of MAP ISF calculation")
        @show sum([sum((Wo1(ys[k]) - Ro1(Wo1(xs[k]))).^2)./dot(xs[k],xs[k])^SIGMA for k=1:length(xs)])
        Ws1, Rs1 = SSMCalc(F, [1, 2])
        #----------------------------------------------
        # OPTIMISATION STARTS
        #----------------------------------------------
        U1, S1, mpar10, mexp10 = ISFData(xs, ys, DF, [1,2], order, order, SIGMA, dt=DT, rmax=0.2, onlyU = true, steps = 300)
        U1, S1, mpar1, mexp1 = ISFData(xs, ys, DF, [1,2], order, order, SIGMA, dt=DT, rmax=0.2, mpar0 = mpar10, mexp0 = mexp10)
        @show sum([sum((U1(ys[k]) - S1(U1(xs[k]))).^2)./dot(xs[k],xs[k])^2 for k=1:length(xs)])
        @save "MapFit$(NAME)S$(SIGMA)O$(order).bson" DT Wo1 Ro1 Ws1 Rs1 W1 R1 U1 S1 mpar1 mexp1
        
        Wo2, Ro2, W2, R2 = ISFCalc(F, [3, 4], [])
        println("residual of MAP ISF calculation")
        @show sum([sum((Wo2(ys[k]) - Ro2(Wo2(xs[k]))).^2)./dot(xs[k],xs[k])^SIGMA for k=1:length(xs)])
        Ws2, Rs2 = SSMCalc(F, [3, 4])
        #----------------------------------------------
        # OPTIMISATION STARTS
        #----------------------------------------------
        U2, S2, mpar20, mexp20 = ISFData(xs, ys, DF, [3,4], order, order, SIGMA, dt=DT, rmax=0.2, onlyU = true, steps = 300)
        U2, S2, mpar2, mexp2 = ISFData(xs, ys, DF, [3,4], order, order, SIGMA, dt=DT, rmax=0.2, mpar0 = mpar20, mexp0 = mexp20)
        @show sum([sum((U2(ys[k]) - S2(U2(xs[k]))).^2)./dot(xs[k],xs[k])^2 for k=1:length(xs)])
        @save "MapFit$(NAME)S$(SIGMA)O$(order).bson" DT Wo1 Ro1 Ws1 Rs1 W1 R1 U1 S1 mpar1 mexp1 Wo2 Ro2 Ws2 Rs2 W2 R2 U2 S2 mpar2 mexp2

    end
end

end

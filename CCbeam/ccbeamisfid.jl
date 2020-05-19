module u

using BSON: @load, @save

NAME = "CCbeam"

include("../src/isfpolymodel.jl")
include("../src/mapmethods.jl")

@load "CCbeamData.bson" nnm1 nnm2 nnm3
# by 6 points at a time

# individal points
dat1 = [nnm1[2,1+k:6+k] for k=0:size(nnm1,2)-6]
dat2 = [nnm2[2,1+k:6+k] for k=0:size(nnm2,2)-6]
dat3 = [nnm3[2,1+k:6+k] for k=0:size(nnm3,2)-6]
DT = nnm1[1,2] - nnm1[1,1]

xs = vcat(dat1[1:end-1], dat2[1:end-1], dat3[1:end-1])
ys = vcat(dat1[2:end], dat2[2:end], dat3[2:end])
    
for SIGMA in [1]
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
        U1, S1, mpar10, mexp10 = ISFData(xs, ys, DF, [1,2], order, SIGMA, dt=DT, rmax=0.7, onlyU = true, steps = 300)
        U1, S1, mpar1, mexp1 = ISFData(xs, ys, DF, [1,2], order, SIGMA, dt=DT, rmax=0.7, mpar0 = mpar10, mexp0 = mexp10)
        @show sum([sum((U1(ys[k]) - S1(U1(xs[k]))).^2)./dot(xs[k],xs[k])^2 for k=1:length(xs)])
        @save "MapFit$(NAME)S$(SIGMA)O$(order).bson" DT Wo1 Ro1 Ws1 Rs1 W1 R1 U1 S1 mpar1 mexp1
        
        Wo2, Ro2, W2, R2 = ISFCalc(F, [3, 4], [])
        println("residual of MAP ISF calculation")
        @show sum([sum((Wo2(ys[k]) - Ro2(Wo2(xs[k]))).^2)./dot(xs[k],xs[k])^SIGMA for k=1:length(xs)])
        Ws2, Rs2 = SSMCalc(F, [3, 4])
        #----------------------------------------------
        # OPTIMISATION STARTS
        #----------------------------------------------
        U2, S2, mpar20, mexp20 = ISFData(xs, ys, DF, [3,4], order, SIGMA, dt=DT, rmax=0.7, onlyU = true, steps = 300)
        U2, S2, mpar2, mexp2 = ISFData(xs, ys, DF, [3,4], order, SIGMA, dt=DT, rmax=0.7, mpar0 = mpar20, mexp0 = mexp20)
        @show sum([sum((U2(ys[k]) - S2(U2(xs[k]))).^2)./dot(xs[k],xs[k])^2 for k=1:length(xs)])
        @save "MapFit$(NAME)S$(SIGMA)O$(order).bson" DT Wo1 Ro1 Ws1 Rs1 W1 R1 U1 S1 mpar1 mexp1 Wo2 Ro2 Ws2 Rs2 W2 R2 U2 S2 mpar2 mexp2

        Wo3, Ro3, W3, R3 = ISFCalc(F, [5, 6], [])
        println("residual of MAP ISF calculation")
        @show sum([sum((Wo3(ys[k]) - Ro3(Wo3(xs[k]))).^2)./dot(xs[k],xs[k])^SIGMA for k=1:length(xs)])
        Ws3, Rs3 = SSMCalc(F, [5, 6])
        #----------------------------------------------
        # OPTIMISATION STARTS
        #----------------------------------------------
        U3, S3, mpar30, mexp30 = ISFData(xs, ys, DF, [5,6], order, SIGMA, dt=DT, rmax=0.7, onlyU = true, steps = 300)
        U3, S3, mpar3, mexp3 = ISFData(xs, ys, DF, [5,6], order, SIGMA, dt=DT, rmax=0.7, mpar0 = mpar30, mexp0 = mexp30)
        @show sum([sum((U3(ys[k]) - S3(U3(xs[k]))).^2)./dot(xs[k],xs[k])^2 for k=1:length(xs)])

        @save "MapFit$(NAME)S$(SIGMA)O$(order).bson" DT Wo1 Ro1 Ws1 Rs1 W1 R1 U1 S1 mpar1 mexp1 Wo2 Ro2 Ws2 Rs2 W2 R2 U2 S2 mpar2 mexp2 Wo3 Ro3 Ws3 Rs3 W3 R3 U3 S3 mpar3 mexp3
    end
end

end

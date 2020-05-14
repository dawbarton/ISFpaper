module u

using Printf
using LinearAlgebra
using Optim, LineSearches
using BSON: @load, @save
using Zygote
using ForwardDiff

SIGMA = 3

jrhs = zeros(6,6)

using DynamicPolynomials
using MultivariatePolynomials

include("../src/isfpolymodel.jl")
include("../src/polymethods.jl")

# convert to proper polynomials
function ISFtoPolyModel(mpar, mexp)
    uorder = maximum(sum(mexp.U,dims=1))
    sorder = 1 + 2*maximum(sum(mexp.f,dims=1))
    ndim = size(mexp.U,1)
    zdim = size(mpar.U,1)
    U = PolyModel(ndim, zdim, uorder)
    S = PolyModel(zdim, zdim, sorder)
    U.W .= 0.0
    for k=1:size(mexp.U, 2)
        p = PolyFindIndex(U.mexp, mexp.U[:,k])
        U.W[:,p] .= mpar.U[:,k]
    end
    S.W .= 0.0
    for k=1:size(mexp.f, 2)
        for p=0:mexp.f[1,k]
            iexp1 = [1+2*p, 2*(mexp.f[1,k]-p)]
            iexp2 = [2*p, 1+2*(mexp.f[1,k]-p)]
            q1 = PolyFindIndex(S.mexp, iexp1)
            q2 = PolyFindIndex(S.mexp, iexp2)
            S.W[1,q1] += binomial(mexp.f[1,k], p)*mpar.fr[k]
            S.W[1,q2] -= binomial(mexp.f[1,k], p)*mpar.fi[k]
            S.W[2,q1] += binomial(mexp.f[1,k], p)*mpar.fi[k]
            S.W[2,q2] += binomial(mexp.f[1,k], p)*mpar.fr[k]
        end
    end
    return U, S
end

function PolyModelToISF(U, S)
    DU = PolyGetLinearPart(U)
    vsr = DU[1,:]
    vsi = DU[2,:]
    DS = PolyGetLinearPart(S)
    mur = DS[1,1]
    mui = DS[2,1]
    mpar, mexp = ISFconstruct(size(U.mexp,1), PolyOrder(U.mexp), vsr, vsi, mur, mui)
    for k=1:size(mexp.U, 2)
        p = PolyFindIndex(U.mexp, mexp.U[:,k])
        mpar.U[:,k] .= U.W[:,p]
    end
    for k=1:size(mexp.f, 2)
#         for p=0:mexp.f[1,k]
        p = 0
        iexp1 = [1+2*p, 2*(mexp.f[1,k]-p)]
        iexp2 = [2*p, 1+2*(mexp.f[1,k]-p)]
        q1 = PolyFindIndex(S.mexp, iexp1)
        q2 = PolyFindIndex(S.mexp, iexp2)
        mpar.fr[k] = S.W[1,q1]/binomial(mexp.f[1,k], p)
        mpar.fi[k] = S.W[2,q1]/binomial(mexp.f[1,k], p)
    end
    return mpar, mexp
end

function constraintLoss(mpar, At, Bt)
    loss2 = 0.0
    for k=1:size(Bt,3)
        res1 = 0.0
        res2 = 0.0
        for l=1:size(Bt,2)
            tmp = mpar * Bt[:,l,k]
            res1 += dot(At[1,:,l], tmp)
            res2 += dot(At[2,:,l], tmp)
        end
#         @show (res1/r, res2/r)
        loss2 += (res1 - 0.5*size(Bt,2))^2 + (res2)^2
    end
    return loss2
end

function lossISFmodel(mpar, mexp, xTOm, yTOm, At, Bt)
    # unit norm of linear part of U
    linidx = findall(dropdims(sum(mexp.U, dims=1), dims=1) .== 1)
    loss1 = 0.0
#     loss1 = 10.0*size(xTOm,2)*(sum(mpar.U[:,linidx].^2) - 1.0)^2
    for s = 1:size(xTOm,2)
        u1 = mpar.U * xTOm[:,s]
        u2 = mpar.U * yTOm[:,s]
        rsq = u1[1]^2+u1[2]^2
        rsqpow = rsq .^ mexp.f[1,:]
        fr = mpar.fr * rsqpow
        fi = mpar.fi * rsqpow
        res = sum((u2 .- [u1[1]*fr[1] - u1[2]*fi[1], u1[1]*fi[1] + u1[2]*fr[1]]).^2)
        loss1 += res / dot(xTOm[linidx,s], xTOm[linidx,s])^SIGMA
    end
    loss2 = constraintLoss(mpar.U, At, Bt)
#     for k=1:size(Bt,3)
#         res1 = 0.0
#         res2 = 0.0
#         for l=1:size(Bt,2)
#             tmp = mpar.U * Bt[:,l,k]
#             res1 += dot(At[1,:,l], tmp)
#             res2 += dot(At[2,:,l], tmp)
#         end
# #         @show (res1/r, res2/r)
#         loss2 += (res1 - 0.5*size(Bt,2))^2 + (res2)^2
#     end
#     @show loss1
#     @show loss2
    return loss1 + 10*size(xTOm,2)*loss2
end

function delta(i,j)
    if i == j
        return 1
    else
        return 0
    end
    return 0
end

function lossISFmodelGrad!(mparGrad, mpar, mexp, xTOm, yTOm, At, Bt)
    # unit norm of linear part of U
    linidx = findall(dropdims(sum(mexp.U, dims=1), dims=1) .== 1)
    loss = 0.0
#     loss = 10.0*size(xTOm,2)*(sum(mpar.U[:,linidx].^2) - 1.0)^2
    mparGrad.U[:] .= 0.0
#     mparGrad.U[:,linidx] .+= 2.0*10.0*size(xTOm,2)*(sum(mpar.U[:,linidx].^2) - 1.0)*2.0*mpar.U[:,linidx]
#     @show mparGrad.U[:,linidx]
    mparGrad.fr[:] .= 0.0
    mparGrad.fi[:] .= 0.0
    for s = 1:size(xTOm,2)
        u1 = mpar.U * xTOm[:,s]
        u2 = mpar.U * yTOm[:,s]
        rsq = u1[1]^2+u1[2]^2
        rsqpow = rsq .^ mexp.f[1,:]
        fr = mpar.fr * rsqpow
        fi = mpar.fi * rsqpow
        res = u2 .- [u1[1]*fr[1] - u1[2]*fi[1], u1[1]*fi[1] + u1[2]*fr[1]]
        loss += sum(res.^2) / dot(xTOm[linidx,s], xTOm[linidx,s])^2
        # gradient
        rsqpowderi = mexp.f[1,:] .* (rsq .^ (mexp.f[1,:] .- 1))       
        frderi = mpar.fr * rsqpowderi
        fideri = mpar.fi * rsqpowderi
        for k=1:size(mpar.U,1)
            for l=1:size(mpar.U,2)
                mparGrad.U[k,l] += (2.0*res[1]*
                        (delta(1,k)*yTOm[l,s] - 
                            (  delta(1,k)*xTOm[l,s]*fr[1] + u1[1]*frderi[1]*2.0*
                                                            (u1[1]*delta(1,k)*xTOm[l,s] + u1[2]*delta(2,k)*xTOm[l,s])
                             - delta(2,k)*xTOm[l,s]*fi[1] - u1[2]*fideri[1]*2.0*
                                                            (u1[1]*delta(1,k)*xTOm[l,s] + u1[2]*delta(2,k)*xTOm[l,s])))
                                  + 2.0*res[2]*
                        (delta(2,k)*yTOm[l,s] - 
                            (  delta(1,k)*xTOm[l,s]*fi[1] + u1[1]*fideri[1]*2.0*
                                                            (u1[1]*delta(1,k)*xTOm[l,s] + u1[2]*delta(2,k)*xTOm[l,s])
                             + delta(2,k)*xTOm[l,s]*fr[1] + u1[2]*frderi[1]*2.0*
                                                            (u1[1]*delta(1,k)*xTOm[l,s] + u1[2]*delta(2,k)*xTOm[l,s])))) /  dot(xTOm[linidx,s], xTOm[linidx,s])^SIGMA
            end
        end
        for p=1:size(mpar.fr,2)
            mparGrad.fr[1,p] += (2.0*res[1]*( -u1[1]*rsqpow[p] ) + 2.0*res[2]*(-u1[2]*rsqpow[p])) / dot(xTOm[linidx,s], xTOm[linidx,s])^SIGMA
            mparGrad.fi[1,p] += (2.0*res[1]*( u1[2]*rsqpow[p] ) + 2.0*res[2]*(-u1[1]*rsqpow[p])) / dot(xTOm[linidx,s], xTOm[linidx,s])^SIGMA
        end
    end

    loss2 = 0.0    
    for m=1:size(Bt,3)
        res1 = 0.0
        res2 = 0.0
        for k=1:size(Bt,2)
            tmp = mpar.U * Bt[:,k,m]
            res1 += dot(At[1,:,k], tmp)
            res2 += dot(At[2,:,k], tmp)
        end
        for k=1:size(Bt,2)
            for p=1:size(mpar.U,1)
                for q=1:size(mpar.U,2)
                    mparGrad.U[p,q] += 10*size(xTOm,2)*2*(res1 - 0.5*size(Bt,2)) * At[1,p,k] * Bt[q,k,m]
                    mparGrad.U[p,q] += 10*size(xTOm,2)*2*res2 * At[2,p,k] * Bt[q,k,m]
                end
            end
        end
        loss2 += (res1 - 0.5*size(Bt,2))^2 + (res2)^2
    end
    return loss + 10*size(xTOm,2)*loss2
end

function Smod(mpar, mexp, u1)
    rsq = u1[1]^2+u1[2]^2
    rsqpow = rsq .^ mexp.f[1,:]
    fr = mpar.fr * rsqpow
    fi = mpar.fi * rsqpow
    return [u1[1]*fr[1] - u1[2]*fi[1], u1[1]*fi[1] + u1[2]*fr[1]]
end

function Umod(mpar, mexp, x)
    xspow = x .^ mexp.U
    return mpar.U * dropdims(prod(xspow, dims=1), dims=1)
end

function constraints(mexp, mpar, vr, vi)
    # to constrain the parametrisation  
    rlen = 12
    thetalen = 24
    At = zeros(2,2,thetalen)
    Bt = zeros(size(mpar,2),thetalen,rlen)
    for l=1:thetalen
        theta = 2*pi*l/thetalen
        At[:,:,l] = [cos(theta) sin(theta); sin(theta) (-cos(theta))]
    end
    for k=1:rlen
        r = 0.7*k/rlen
        for l=1:thetalen
            theta = 2*pi*l/thetalen
            Bt[:,l,k] = dropdims(prod( (vr*r*cos(theta) - vi*r*sin(theta)) .^ mexp, dims=1), dims=1) / r
        end
    end
    return At, Bt
end

# sets values of vector x from struct a
function setX!(x, a)
    pt = 0
    for name in fieldnames(typeof(a))
        sz = prod(size(getfield(a,name)))
        x[1+pt:pt+sz] .= getfield(a,name)[:]
        pt += sz
    end  
    return nothing
end

function setX!(x, a, fns)
    pt = 0
    for name in fns
        sz = prod(size(getfield(a,name)))
        x[1+pt:pt+sz] .= getfield(a,name)[:]
        pt += sz
    end  
    return nothing
end

# sets values of struct a from vector x
function setA!(a, x)
    pt = 0
    for name in fieldnames(typeof(a))
        sz = prod(size(getfield(a,name)))
        getfield(a,name)[:] .= x[1+pt:pt+sz]
        pt += sz
    end  
    return nothing
end

function setA!(a, x, fns)
    pt = 0
    for name in fns
        sz = prod(size(getfield(a,name)))
        getfield(a,name)[:] .= x[1+pt:pt+sz]
        pt += sz
    end  
    return nothing
end

# creates a new vector based on a
function newX(a)
    pt = 0
    for name in fieldnames(typeof(a))
        sz = prod(size(getfield(a,name)))
        pt += sz
    end
    x = zeros(pt)
    setX!(x, a)
    return x    
end

function newX(a, fns)
    pt = 0
    for name in fns
        sz = prod(size(getfield(a,name)))
        pt += sz
    end
    x = zeros(pt)
    setX!(x, a, fns)
    return x    
end

# U is the submersion
# R is the conjugate dynamics
function ISFData(xs, ys, DF, vars, order, U, S)

    # the linear part
    sel = vars[1]
    # need the right eigenvector, too
    rightvals, rightvecs = eigen(DF)
    if maximum(abs.(rightvecs[:,sel] .- conj(rightvecs[:,vars[2]]))) > 1e-12
        println("ERROR: the two eigenvectors are not conjugate pairs!")
    end
    invrightvecs = inv(rightvecs)
    vsr = real(invrightvecs[sel,:])
    vsi = imag(invrightvecs[sel,:])
    
    vr = real(rightvecs[:,sel])
    vi = imag(rightvecs[:,sel])
    
    mpar, mexp = ISFconstruct(length(xs[1]), order, vsr, vsi, real(rightvals[sel]), imag(rightvals[sel]))
    # copy over the analytic result     
#     mpar, mexp = PolyModelToISF(U, S)
    println("residual of INIT ISF")
    @show sum([sum((Umod(mpar, mexp,ys[k]) - Smod(mpar, mexp,Umod(mpar, mexp,xs[k]))).^2)./dot(xs[k],xs[k])^SIGMA for k=1:length(xs)])

#     return
    
    xTOm = DataToExp(xs, mexp.U)
    yTOm = DataToExp(ys, mexp.U)
    
    # to constrain the parametrisation
    At, Bt = constraints(mexp.U, mpar.U, vr, vi)
        
    @show dot(invrightvecs[sel,:]', rightvecs[:,sel])
    @show dot(vsr, vr)
    @show dot(vsr, vi)
    @show dot(vsi, vr)
    @show dot(vsi, vi)
    
    rcost = zeros(size(xs))
    rsint = zeros(size(xs))
    vsincos = deepcopy(xs)
    for k=1:length(xs)
        rcost[k] = dot(vsr,xs[k])
        rsint[k] = dot(vsi,xs[k])
        vsincos[k] = vr*rcost[k] - vi*rsint[k]
    end
    vproj = DataToExp(vsincos, mexp.U)
    
    function fg!(f, g, x, fns)
        # setting the papameters
        setA!(mpar, x, fns)
        if g != nothing
            # code to compute gradient here
            # writing the result to the vector G
            mparGrad = ISFmodel(zero(mpar.U), zero(mpar.fr), zero(mpar.fi))
            @views f = lossISFmodelGrad!(mparGrad, mpar, mexp, xTOm, yTOm, At, Bt)
            # store the gradient
            setX!(g, mparGrad, fns)
        end
        if f != nothing
            # value = ... code to compute objective function
            @views f = lossISFmodel(mpar, mexp, xTOm, yTOm, At, Bt)
        end
        return f
    end
#     fv = zeros(1)
#     gv = zero(allpar)
    
    function cb(x)
        rmat = ForwardDiff.jacobian(x -> Smod(mpar, mexp, x), Umod(mpar, mexp, zero(xs[1])))
        umat = ForwardDiff.jacobian(x -> Umod(mpar, mexp, x), zero(xs[1]))
        println("       umat norm:      ", (sum(umat.^2) - 1.0)^2)
        println("       constr loss:    ", constraintLoss(mpar.U, At, Bt))
        freq = abs.(angle.(eigvals(rmat))/DT)
        damp = -log.(abs.(eigvals(rmat)))./(DT*freq)
        println("       DATA freq = ", (@sprintf "%.6e" freq[1]), " damp = ", (@sprintf "%.6e" damp[1]))
        freq = abs.(angle.(eigvals(jrhs)[sel:sel+1])/DT)
        damp = -log.(abs.(eigvals(jrhs)[sel:sel+1]))./(DT*freq)
        println("       JAC freq = ", (@sprintf "%.6e" freq[1]), " damp = ", (@sprintf "%.6e" damp[1]))
        return false
    end
#     println("Gradient Tolerance = ", length(allpar)*(1e-6))
    opt = BFGS(linesearch = BackTracking(order=3))
    fns = fieldnames(typeof(mpar))
    allpar = newX(mpar, fns)
    Optim.optimize(Optim.only_fg!((f, g, x)->fg!(f, g, x, fns)), allpar, opt, Optim.Options(;callback = cb,
                      g_tol = length(allpar)*(1e-9),
                      iterations = 4000,
                      store_trace = true,
                      show_trace = true))
    
    U2, S2 = ISFtoPolyModel(mpar, mexp)
    return U2, S2, mpar, mexp
end

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

    for sig in [1]
    global SIGMA = sig
    for order in [3 5 7]
        println("SIGMA=", sig, " ORDER=", order)
        F = PolyModel(order, xs, ys, sigma = SIGMA)
        # make sure that equilibriums is at zero (no constant part)
        F.W[:,end] .= 0.0
        println("RESIDUAL MAP")
        @show sum([ sum((ys[k] - F(xs[k]))) for k=1:length(xs)])

        DF = PolyGetLinearPart(F)
        jrhs .= DF
        @show eigvals(DF)
        
        Wo1, Ro1, W1, R1 = ISFCalc(F, [1, 2], [])
        println("residual of MAP ISF calculation")
        @show sum([sum((Wo1(ys[k]) - Ro1(Wo1(xs[k]))).^2)./dot(xs[k],xs[k])^SIGMA for k=1:length(xs)])
        Ws1, Rs1 = SSMCalc(F, [1, 2])
        #----------------------------------------------
        # OPTIMISATION STARTS
        #----------------------------------------------
        U1, S1, mpar1, mexp1 = ISFData(xs, ys, DF, [1,2], order, Wo1, Ro1)
        @show sum([sum((U1(ys[k]) - S1(U1(xs[k]))).^2)./dot(xs[k],xs[k])^2 for k=1:length(xs)])

        @save "MapFitCCbeamS$(SIGMA)O$(order).bson" DT Wo1 Ro1 Ws1 Rs1 W1 R1 U1 S1 mpar1 mexp1
        
        Wo2, Ro2, W2, R2 = ISFCalc(F, [3, 4], [])
        println("residual of MAP ISF calculation")
        @show sum([sum((Wo2(ys[k]) - Ro2(Wo2(xs[k]))).^2)./dot(xs[k],xs[k])^SIGMA for k=1:length(xs)])
        Ws2, Rs2 = SSMCalc(F, [3, 4])
        #----------------------------------------------
        # OPTIMISATION STARTS
        #----------------------------------------------
        U2, S2, mpar2, mexp2 = ISFData(xs, ys, DF, [3, 4], order, Wo2, Ro2)
        @show sum([sum((U2(ys[k]) - S2(U2(xs[k]))).^2)./dot(xs[k],xs[k])^2 for k=1:length(xs)])

        @save "MapFitCCbeamS$(SIGMA)O$(order).bson" DT Wo1 Ro1 Ws1 Rs1 W1 R1 U1 S1 mpar1 mexp1 Wo2 Ro2 Ws2 Rs2 W2 R2 U2 S2 mpar2 mexp2

        Wo3, Ro3, W3, R3 = ISFCalc(F, [5, 6], [])
        println("residual of MAP ISF calculation")
        @show sum([sum((Wo3(ys[k]) - Ro3(Wo3(xs[k]))).^2)./dot(xs[k],xs[k])^SIGMA for k=1:length(xs)])
        Ws3, Rs3 = SSMCalc(F, [5, 6])
        #----------------------------------------------
        # OPTIMISATION STARTS
        #----------------------------------------------
        U3, S3, mpar3, mexp3 = ISFData(xs, ys, DF, [5, 6], order, Wo3, Ro3)
        @show sum([sum((U3(ys[k]) - S3(U3(xs[k]))).^2)./dot(xs[k],xs[k])^2 for k=1:length(xs)])

        @save "MapFitCCbeamS$(SIGMA)O$(order).bson" DT Wo1 Ro1 Ws1 Rs1 W1 R1 U1 S1 mpar1 mexp1 Wo2 Ro2 Ws2 Rs2 W2 R2 U2 S2 mpar2 mexp2 Wo3 Ro3 Ws3 Rs3 W3 R3 U3 S3 mpar3 mexp3

    end
    end

end

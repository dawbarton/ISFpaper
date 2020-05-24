include("polymethods.jl")

using Printf
using DynamicPolynomials
using MultivariatePolynomials
using LinearAlgebra
using Optim, LineSearches
using Zygote
using ForwardDiff

# Here we define the polynomial ISF model
function ISFPolyExps(in::Integer, omin::Integer, omax::Integer)
    @polyvar x[1:in]
    mx0 = monomials(x, omin:omax)
    return hcat([exponents(m) for m in mx0]...)
end

struct ISFexps
    U::Array{Int64,2}
    f::Array{Int64,2}
    rmax::Float64
    Nr::Int64
    Ntheta::Int64
end

function ISFexps(ndim::Integer, Uorder::Integer, Sorder::Integer, rmax, Nr, Ntheta)
    return ISFexps(ISFPolyExps(ndim, 1, Uorder), ISFPolyExps(1, 0, div(Sorder-1,2)), rmax, Nr, Ntheta)
end

struct ISFmodel
    U::Array{Float64,2}
    fr::Array{Float64,2}
    fi::Array{Float64,2}
end

function DataToExp(xs, mexp)
    xTOm = zeros(typeof(xs[1][1]), size(mexp,2), length(xs))
    for k=1:length(xs)
        zp = xs[k].^mexp
        xTOm[:,k] = prod(zp, dims=1)
    end
    return xTOm
end

# Two dimesional ISF
function ISFconstruct(ndim, Uorder, Sorder, vsr, vsi, lr, li; rmax, Nr, Ntheta)
    mexp = ISFexps(ndim, Uorder, Sorder, rmax, Nr, Ntheta)
    mpar = ISFmodel(
        zeros(2, size(mexp.U, 2)), # U
        zeros(1, size(mexp.f, 2)),    # fr
        zeros(1, size(mexp.f, 2)) )   # fi
    # setting linear part of U
    linidx = findall(dropdims(sum(mexp.U, dims=1), dims=1) .== 1)
    for p in linidx
        id = findfirst(mexp.U[:,p] .== 1)
        mpar.U[1,p] = vsr[id]
        mpar.U[2,p] = vsi[id]
    end
    # setting the constant part of fr, fi
    mpar.fr[1,end] = lr
    mpar.fi[1,end] = li
    return mpar, mexp
end

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

function PolyModelToISF(U; rmax, Nr, Ntheta)
    DU = PolyGetLinearPart(U)
    vsr = DU[1,:]
    vsi = DU[2,:]
    DS = PolyGetLinearPart(S)
    mur = DS[1,1]
    mui = DS[2,1]
    mpar, mexp = ISFconstruct(size(U.mexp,1), PolyOrder(U.mexp), vsr, vsi, mur, mui, rmax=rmax, Nr=Nr, Ntheta=Ntheta)
    for k=1:size(mexp.U, 2)
        p = PolyFindIndex(U.mexp, mexp.U[:,k])
        mpar.U[:,k] .= U.W[:,p]
    end
    return mpar, mexp
end

function ISFSetConjugateDynamics!(mpar, mexp, S)
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
    return nothing
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

function lossISFmodel(mpar, mexp, xTOm, yTOm, At, Bt, SIGMA)
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

function lossISFmodelGrad!(mparGrad, mpar, mexp, xTOm, yTOm, At, Bt, SIGMA)
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
    rlen = mexp.Nr
    thetalen = mexp.Ntheta
    At = zeros(2,2,thetalen)
    Bt = zeros(size(mpar.U,2),thetalen,rlen)
    for l=1:thetalen
        theta = 2*pi*l/thetalen
        At[:,:,l] = [cos(theta) sin(theta); sin(theta) (-cos(theta))]
    end
    for k=1:rlen
        r = mexp.rmax*k/rlen
        for l=1:thetalen
            theta = 2*pi*l/thetalen
            Bt[:,l,k] = dropdims(prod( (vr*r*cos(theta) - vi*r*sin(theta)) .^ mexp.U, dims=1), dims=1) / r
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
function ISFData(xs, ys, DF, vars, Uorder, Sorder, SIGMA; U=nothing, S=nothing, dt=1.0, rmax=0.2, Nr=12, Ntheta=24, steps=32000, onlyU = false, onlyS=false, 
                mpar0 = nothing, mexp0 = nothing)
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
    
    if mpar0 != nothing && mexp0 != nothing
        mpar = deepcopy(mpar0)
        mexp = deepcopy(mexp0)
    elseif U == nothing
        mpar, mexp = ISFconstruct(length(xs[1]), Uorder, Sorder, vsr, vsi, real(rightvals[sel]), imag(rightvals[sel]), rmax=rmax, Nr=Nr, Ntheta=Ntheta)
    else
        mpar, mexp = PolyModelToISF(U, rmax=rmax, Nr=Nr, Ntheta=Ntheta)
    end
    if S != nothing
        ISFSetConjugateDynamics!(mpar, mexp, S)
    end
    println("residual of INIT ISF")
    @show sum([sum((Umod(mpar, mexp,ys[k]) - Smod(mpar, mexp,Umod(mpar, mexp,xs[k]))).^2)./dot(xs[k],xs[k])^SIGMA for k=1:length(xs)])

    xTOm = DataToExp(xs, mexp.U)
    yTOm = DataToExp(ys, mexp.U)
    
    # to constrain the parametrisation
    At, Bt = constraints(mexp, mpar, vr, vi)

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
            @views f = lossISFmodelGrad!(mparGrad, mpar, mexp, xTOm, yTOm, At, Bt, SIGMA)
            # store the gradient
            setX!(g, mparGrad, fns)
        end
        if f != nothing
            # value = ... code to compute objective function
            @views f = lossISFmodel(mpar, mexp, xTOm, yTOm, At, Bt, SIGMA)
        end
        return f
    end
    
    function cb(x)
        rmat = ForwardDiff.jacobian(x -> Smod(mpar, mexp, x), Umod(mpar, mexp, zero(xs[1])))
        umat = ForwardDiff.jacobian(x -> Umod(mpar, mexp, x), zero(xs[1]))
        println("       umat norm = ", (@sprintf "%.6e" sqrt(sum(umat.^2))), " S=", SIGMA, " OU=", PolyOrder(mexp.U), " OS=", 2*PolyOrder(mexp.f)+1)
        println("       constraint= ", (@sprintf "%.6e" constraintLoss(mpar.U, At, Bt)))
        freq = abs.(angle.(eigvals(rmat))/dt)
        damp = -log.(abs.(eigvals(rmat)))./(dt*freq)
        println("       DATA freq = ", (@sprintf "%.6e" freq[1]), " damp = ", (@sprintf "%.6e" damp[1]))
        freq = abs.(angle.(eigvals(DF)[sel:sel+1])/dt)
        damp = -log.(abs.(eigvals(DF)[sel:sel+1]))./(dt*freq)
        println("       JAC freq  = ", (@sprintf "%.6e" freq[1]), " damp = ", (@sprintf "%.6e" damp[1]))
        return false
    end
    opt = BFGS(linesearch = BackTracking(order=3))
    if onlyU
        fns = (:U,)
    elseif onlyS
        fns = (:fr,:fi)
    else
        fns = fieldnames(typeof(mpar))
    end
    allpar = newX(mpar, fns)
    Optim.optimize(Optim.only_fg!((f, g, x)->fg!(f, g, x, fns)), allpar, opt, Optim.Options(;callback = cb,
                      g_tol = length(allpar)*(1e-9),
                      iterations = steps,
                      store_trace = true,
                      show_trace = true))
    
    U2, S2 = ISFtoPolyModel(mpar, mexp)
    return U2, S2, mpar, mexp
end

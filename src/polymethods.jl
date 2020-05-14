using DynamicPolynomials
using MultivariatePolynomials
using TaylorSeries

# -------------------------------------------------------
# VECTOR VALUED MODELS AND HELPER FUNCTIONS
# -------------------------------------------------------

function PolyExps(in::Integer, order::Integer)
    @polyvar x[1:in]
    mx0 = monomials(x, 0:order)
    return hcat([exponents(m) for m in mx0]...)
end

function PolyOrder(mexp::AbstractArray)
    return maximum(sum(mexp, dims=1))
end

# find the indices of "order' order terms.
function PolyOrderIndices(mexp::AbstractArray, order::Integer)
    return findall(isequal(order), dropdims(sum(mexp,dims=1),dims=1))
end

struct PolyModel{T} 
    mexp
    W::AbstractArray{T}
end

function PolyModel(T, in::Integer, out::Integer, order::Integer)
    mx1 = PolyExps(in, order)
    pars = zeros(T, out,size(mx1,2))
    # set only the linear parameters
#     pars[:,findall(isequal(1), dropdims(sum(mx1,dims=1),dims=1))] = randn(out, in)
    return PolyModel(mx1, pars)
end

function PolyModel(in::Integer, out::Integer, order::Integer)
    return PolyModel(Float64, in, out, order)
end

# the real version
function PolyMonomial(zRe, mexp, p)
    rre = one(zRe[1])
    for k=1:size(mexp,1)
        rre = rre*zRe[k]^mexp[k,p]
    end
    return rre
end

# evaluates the polynomial at z
function (a::PolyModel)(z)
    zp = z.^a.mexp
    zpprod = zp[1, :]
    for i = 2:size(zp, 1)
        zpprod = zpprod .* zp[i, :]
    end
    return a.W*zpprod
end

# here order is the indicies that should be included
function PolyEvalOrder(z, W, mexp, order)
    zp = z.^mexp[:,order]
    zpprod = prod(zp, dims=1)
    return W[:,order]*zpprod'
end


# the real version
function PolyEval(zRe, Wre, mexp)
    zprodRe = zero(Wre[1,:])
    for p=1:size(mexp,2)
        rre = PolyMonomial(zRe, mexp, p)
        zprodRe[p] = rre
    end
    # zprodFlat = reshape(collect(Iterators.flatten(zprod)),2,size(mexp,2))
    return Wre*zprodRe   
end

function PolyMaxOrders(S)
    ord = maximum(sum(S.mexp,dims=1))
    orders = [[findfirst(dropdims(sum(S.mexp, dims=1),dims=1) .== 0)]]
    for k=1:ord
        push!(orders, findall(dropdims(sum(S.mexp, dims=1),dims=1) .<= k))
    end
    return orders
end

function PolyEvalSubs(x, SW, Sexp, UW, Uexp, Uorders)
    zprodRe = zero(SW[1,:])
    ord = PolyOrder(Sexp)
    for p=1:size(Sexp,2)
        xi = PolyEvalOrder(x, UW, Uexp, Uorders[1+div(ord, max(1, sum(Sexp[:,p])))] )
        rre = xi.^Sexp[:,p]
        zprodRe[p] = prod(rre)
    end
    # zprodFlat = reshape(collect(Iterators.flatten(zprod)),2,size(mexp,2))
    return SW*zprodRe
end

# depending on the column of S.W a different set of exponent are evaluated in U.W
# output has same number elements as columns of S.W
# each element of output is a list of indices for the columns of U.W
function PolyMaxOrders(S, U)
    maxord = PolyOrder(U.mexp)
    rmo0 = div(maxord, max(1, sum(S.mexp[:,1])))
    orders = [ findall(dropdims(sum(U.mexp, dims=1), dims=1) .<= rmo0) ]
    for p=2:size(S.mexp,2)
        # the required maximum order to substitute
        rmo = div(maxord, max(1, sum(S.mexp[:,p])))
        push!(orders, findall(dropdims(sum(U.mexp, dims=1), dims=1) .<= rmo))
    end
    return orders
end

# depending on the column of S.W a different set of exponent are evaluated in U.W
# output has same number elements as columns of S.W
# each element of output is a list of indices for the columns of U.W
function PolyMaxOrdersS(S, U)
    maxord = PolyOrder(U.mexp)
    rmo0 = div(maxord, max(1, sum(S.mexp[:,1])))
    orders = [ findall(dropdims(sum(U.mexp, dims=1), dims=1) .<= rmo0) ]
    for p=2:size(S.mexp,2)
        # the required maximum order to substitute
        rmo = div(maxord, max(1, sum(S.mexp[:,p])))
        push!(orders, findall(dropdims(sum(U.mexp, dims=1), dims=1) .<= rmo))
    end
    return orders
end

# z^m is already supplied
function PolyEval!(res, zTOm, W)
    res .= W * zTOm
    return nothing
end

function PolyEvalOrder!(res, zTOm, W, order)
    res .= W[:,order] * zTOm[order]
    return nothing
end

function PolyEvalSubs!(res, xTOm, SW, Sexp, UW, Uexp, orders)
    zprod = zero(SW[1,:])
    xi = zero(SW[:,1])
    for p=1:size(Sexp,2)
        xi .= UW[:,orders[p]] * zTOm[orders[p]]
        rre = xi.^Sexp[:,p]
        zprod[p] = prod(rre)
    end
    res .= SW*zprod
    return nothing
end

function PolyZero(a::PolyModel{T}) where T
    return PolyModel(a.mexp, zero(a.W))
end

function PolyCopy(a::PolyModel{T}) where T
    return PolyModel(a.mexp, deepcopy(a.W))
end

import Base.-
import Base.+

function -(a::PolyModel, b::PolyModel)
    # needs an assert
    return PolyModel(a.mexp, a.W - b.W)
end

function +(a::PolyModel, b::PolyModel)
    # needs an assert
    return PolyModel(a.mexp, a.W + b.W)
end

# -------------------------------------------------------
# MATRIX VALUED MODELS
# -------------------------------------------------------

# this has the same struture, except W is a 3 index array. The last index is the poly coefficient
struct PolyMatrixModel{T} 
    mexp
    W::AbstractArray{T}
end

function PolyMatrixModel(T, npar::Integer, order::Integer, row::Integer, col::Integer)
    mx1 = PolyExps(npar, order)
    pars = zeros(T, row, col, size(mx1,2))
    # set only the linear parameters
#     pars[:,findall(isequal(1), dropdims(sum(mx1,dims=1),dims=1))] = randn(out, in)
    return PolyMatrixModel(mx1, pars)
end

function PolyMatrixCopy(x::PolyMatrixModel)
    return PolyMatrixModel(x.mexp, deepcopy(x.W))
end

function PolyMatrixZero(x::PolyMatrixModel)
    return PolyMatrixModel(x.mexp, zero(x.W))
end

function (a::PolyMatrixModel)(z)
    zp = z.^a.mexp
    zpprod = zp[1, :]
    for i = 2:size(zp, 1)
        zpprod = zpprod .* zp[i, :]
    end
    rMAT = zero(a.W[:,:,1])
    for i=1:size(a.W,3)
        rMAT .+= a.W[:,:,i] * zpprod[i]
    end
    return rMAT
end

# -------------------------------------------------------
# END MATRIX VALUED MODELS
# -------------------------------------------------------

# This substitutes into the vector of monomials: x are the values of variables
function PhiVector(m::PolyModel, x::AbstractArray)
    xp = x.^m.mexp
    # zpprod is the set of monomials evaluated at z
    xpprod = xp[1, :]
    for i = 2:size(xp, 1)
        xpprod = xpprod .* xp[i, :]
    end
    return xpprod
end

# provides the matrix and vector for parameter estimation from data
function PolyAandB(m::PolyModel, x::AbstractArray, y::AbstractArray, sigma)
    phi = PhiVector(m, x)
    A = (kron(phi', phi))./dot(x,x)^sigma
    b = (phi.*transpose(y))./dot(x,x)^sigma
    return A, b
end

# estimates the parameters fo the polynomial from data
function PolyEstimate(m, x::AbstractArray, y::AbstractArray, sigma)
    A, b = PolyAandB(m, x[1], y[1], sigma)
    # assert length(x) != length(y)
    for k = 2:length(x)
        Ap, bp = PolyAandB(m, x[k], y[k], sigma)
        A .+= Ap
        b .+= bp
    end
    return (A\b)'
end

# This one creates a polynomial model from data
function PolyModel(order::Integer, 
            x::Array{Array{T,1},1}, y::Array{Array{T,1},1}; sigma=0) where T
    in = length(x[1])
    out = length(y[1])
    m0 = PolyModel(T, in, out, order)
    W = PolyEstimate(m0, x, y, sigma)
    return PolyModel(m0.mexp, W)
end

# constructs a polynomial model from a function 'fun'
function PolyModel(ord::Integer, ndim::Integer, fun)
    m0 = PolyModel(Float64, ndim, ndim, ord)
	x = set_variables("x", numvars=ndim, order=ord)
    y = fun(x)
	for k=1:ndim
		for i = 1:size(m0.mexp,2)
	    	m0.W[k,i] = getcoeff(y[k], m0.mexp[:,i])
	     end
	end
    return m0
end

# returns the first index of mexp, which equals to iexp
function PolyFindIndex(mexp, iexp)
    return findfirst(dropdims(prod(mexp .== iexp, dims=1),dims=1))
end

function mulTable(oexp, in1exp, in2exp)
    res = []
    od = maximum(sum(oexp,dims=1))
    p1 = sum(in1exp,dims=1)
    p2 = sum(in2exp,dims=1)
    pexp = zero(in1exp[:,1])
    for k1=1:size(in1exp,2)
        for k2=1:size(in2exp,2)
            if p1[k1]+p2[k2] <= od
                pexp[:] = in1exp[:,k1] + in2exp[:,k2]
                idx = PolyFindIndex(oexp, pexp)
                push!(res, [k1,k2,idx])
            end
        end
    end
    out = zeros(typeof(od), length(res), 3)
    for k = 1:length(res)
        out[k,:] = res[k]
    end
    return out
end

# adds the result to the output
# the output must be different from the input, because of aliasing
function PolyMul!(out, in1, in2, multab)
    for k = 1:size(multab,1)
        out[multab[k,3]] += in1[multab[k,1]]*in2[multab[k,2]]
    end
    return nothing
end

# these are all vectors
# substitutes in2 into in1 and returns the coefficients corresponing to oexp
function PolySubs!(out_mexp, out_W::AbstractArray{T,2}, in1_mexp, in1_W::AbstractArray{T,2}, in2_mexp, in2_W::AbstractArray{T,2}, tab) where T
    to = zeros(T, size(out_mexp,2)) # temporary for d an k
    res = zeros(T, size(out_mexp,2)) # temporary for d an k
    # index of constant in the output
    out_W .= 0
    cfc = findfirst(dropdims(sum(out_mexp, dims=1),dims=1) .== 0)
    # substitute into all monomials
    for d = 1:size(in1_W,1) # all dimensions
        for k = 1:size(in1_mexp,2) # all monomials
            to .= 0
            to[cfc] = in1_W[d, k] # the constant coefficient
            # l select the variable in the monomial
            for l = 1:size(in1_mexp,1)
                # multiply the exponent times
                for p = 1:in1_mexp[l,k]
                    # should not add to the previous result
                    res .= 0
                    @views PolyMul!(res, to, in2_W[l,:], tab)
                    to[:] .= res
                end
            end
            out_W[d,:] .+= to
        end
    end
    return nothing
end

function PolySubs!(out::PolyModel{T}, in1::PolyModel{T}, in2::PolyModel{T}, tab) where T
    PolySubs!(out.mexp, out.W, in1.mexp, in1.W, in2.mexp, in2.W, tab)
    return nothing
end

# without tab
function PolySubs!(out::PolyModel{T}, in1::PolyModel{T}, in2::PolyModel{T}) where T
    tab = mulTable(out.mexp, out.mexp, in2.mexp)
    PolySubs!(out, in1, in2, tab)
    return nothing
end

# make a function with a similar table to poly multiplication.
# this saves finding the indices. Once.
function PolyDeriTab(oexp, iexp)
    outp = zero(iexp)
    for idx = 1:size(iexp,1)
        for k = 1:size(iexp,2)
            id = iexp[:,k] # this is a copy
            if id[idx] > 0
                id[idx] -= 1
                x = PolyFindIndex(oexp, id)
                outp[idx,k] = x
            end
        end
    end
    return outp
end

# takes the derivative with respect to idx
function PolyDeri!(oexp, outp, iexp, inp, deritab, idx)
    outp .= 0
    for k = 1:size(iexp,2)
        if iexp[idx,k] > 0
            outp[deritab[idx,k]] += iexp[idx,k]*inp[k]
        end
    end
    return nothing
end

# recreates the derivative table
# very inefficient
function PolyDeri!(out::PolyModel{T}, inp::PolyModel{T}, idx) where T
# function PolyDeri!(oexp, outp, iexp, inp, idx)
    deritab = PolyDeriTab(out.mexp, inp.mexp)
    if ndims(inp.W) == 1
        PolyDeri!(out.mexp, out.W, inp.mexp, inp.W, deritab, idx)
    elseif ndims(inp.W) == 2
        for k=1:size(inp.W,1)
            @views PolyDeri!(out.mexp, out.W[k,:], inp.mexp, inp.W[k,:], deritab, idx)
        end
    end
    return nothing
end

# outp 
function PolyIntegrateTab(oexp, iexp)
    outp = zero(iexp)
    for idx = 1:size(iexp,1)
        for k = 1:size(iexp,2)
            id = iexp[:,k] # this is a copy
            id[idx] += 1
            x = PolyFindIndex(oexp, id)
            if x != nothing
                outp[idx,k] = x
            else
                outp[idx,k] = -1
            end
        end
    end
    return outp
end

# integrates forward with respect to idx
# outp : output: scalar valued polynomial, same exponents as the input
# iexp : input : exponent of the input
# icp : input : initial conditions a scale valued polynomial
# inp : input : polynomial to be integrated, scalar valued
function PolyIntegrate!(outp, iexp, icp, inp, integratetab, idx)
    outp .= icp
    for k = 1:size(iexp,2)
        if integratetab[idx,k] >= 0
            outp[integratetab[idx,k]] += inp[k]/(iexp[idx,k] + 1)
        end
    end
    return nothing
end

function PolyIntegrate!(out::PolyModel{T}, icond::PolyModel{T}, inp::PolyModel{T}, idx) where T
# function PolyIntegrate!(outp, iexp, icp, inp, idx)
    integratetab = PolyIntegrateTab(inp.mexp, inp.mexp)
    if ndims(inp.W) == 1
        PolyIntegrate!(out.W, inp.mexp, icond.W, inp.W, integratetab, idx)
    elseif ndims(inp.W) == 2
        for k=1:size(inp.W,1)
            @views PolyIntegrate!(out.W[k,:], inp.mexp, icond.W[k,:], inp.W[k,:], integratetab, idx)
        end
    end
    return nothing
end

function PolyTestIntDeri(VF::PolyModel)
    toInt = PolyModel(VF.mexp, randn(size(VF.W)))
    IC = PolyModel(size(VF.mexp,1), size(VF.W,1), PolyOrder(VF.mexp)+1)
    IT = PolyZero(IC)
    PolyIntegrate!(IT, IC, toInt, size(VF.W,1))
    deri = PolyZero(IT)
    PolyDeri!(deri, IT, size(VF.W,1))
    @show toInt.W .- deri.W
    for k=1:size(toInt.W,1)
        @show sum(deri.mexp[:,findall(abs.(deri.W[k,:] .- toInt.W[k,:]) .> 1e-12)],dims=1)
    end
end

# the output has to be pre-allocated
# differentiate in1, and multiply with in2
#     multab = mulTable(out.mexp, in1.mexp, in1.mexp)
#     deritab = PolyDeriTab(out.mexp, in1.mexp)
function PolyDeriMul!(out::PolyModel{T}, in1::PolyModel{T}, in2::PolyModel{T}, multab, deritab) where T
    # temporary for the derivatives
    deri = zeros(T, size(out.mexp,2))
    # this is a matrix-vector multiplication
    for k = 1:size(in1.W,1) # number of rows in input
        for l = 1:size(in1.mexp,1) # number of variables in input, hence columns of derivative
            # out[k] = sum deri[k,l]*in2[l]
            # deri[k,l] is the derivative of in1[k] with respect to l
            @views PolyDeri!(in1.mexp, deri, in1.mexp, in1.W[k,:], deritab, l)
            # @views PolyMul!(out.mexp, out.W[k,:], in1.mexp, deri, in2.mexp, in2.W[l,:])
            @views PolyMul!(out.W[k,:], deri, in2.W[l,:], multab)
        end
    end
    return nothing
end

function PolyDeriMul!(out::PolyModel{T}, in1::PolyModel{T}, in2::PolyModel{T}) where T
    multab = mulTable(out.mexp, in1.mexp, in2.mexp)
    deritab = PolyDeriTab(out.mexp, in1.mexp)
    PolyDeriMul!(out, in1, in2, multab, deritab)
    return nothing
end

# mexp: exponents
# inp: the polynomial
# T: the linear transformation
function ModelLinearTransform!(out::PolyModel{T}, m::PolyModel{T}, Tran::AbstractArray{T,2}) where T
    e2 = one(ones(typeof(m.mexp[1]), size(Tran)))
    tr = PolyModel(e2, Tran)
    PolySubs!(out, m, tr)
    out.W .= Tran\out.W
    return nothing
end

function ModelLinearTransform!(out::PolyModel{T}, m::PolyModel{T}, Tran::AbstractArray{T,2}, TranBack::AbstractArray{T,2}) where T
    e2 = one(ones(typeof(m.mexp[1]), size(Tran)))
    tr = PolyModel(e2, Tran)
    PolySubs!(out, m, tr)
    out.W .= TranBack*out.W
    return nothing
end

function PolySetLinearPart!(m::PolyModel{T}, B::AbstractArray{T}) where T
    c = zero(m.mexp[:,1])
    for k=1:size(B,2)
        c[k] = 1
        p = findfirst(dropdims(prod(m.mexp .== c, dims=1),dims=1))
        c[k] = 0
        m.W[:,p] = B[:,k]
    end
    return nothing
end

function PolyGetLinearPart(m::PolyModel{T}) where T
    c = zero(m.mexp[:,1])
    B = zeros(T, size(m.W,1), size(m.mexp,1))
    for k=1:size(B,2)
        c[k] = 1
        p = findfirst(dropdims(prod(m.mexp .== c, dims=1),dims=1))
        c[k] = 0
        B[:,k] = m.W[:,p] 
    end
    return(B)
end

function PolyInverse(in::PolyModel)
    A = PolyGetLinearPart(in)
    PN = PolyCopy(in)
    PolySetLinearPart!(PN, zero(A))
    # - A^(-1) P_N
    mAiPN = PolyModel(PN.mexp, -A\PN.W)
    Q = PolyZero(in)
    PolySetLinearPart!(PN, inv(A))
    tmp = PolyZero(in)
    while true
        PolySubs!(tmp, mAiPN, Q)
        PolySetLinearPart!(tmp, inv(A))
        # @show maximum(abs.(tmp.W .- Q.W))
        if maximum(abs.(tmp.W .- Q.W)) < 1e-12
            break
        end
        Q.W .= tmp.W
    end
    return Q
end

# -------------------------------------------------------
# MATRIX VALUED POLY METHODS
# -------------------------------------------------------

function PolyMatrixTimesVector!(out::PolyModel{T}, in1::PolyMatrixModel{T}, in2::PolyModel{T}) where T
    mtab = mulTable(out.mexp, in1.mexp, in2.mexp)
    @assert size(in1.W, 2) == size(in2.W, 1)
    out.W .= 0
    for j = 1:size(in1.W, 1)
        for l = 1:size(in1.W, 2)
            @views PolyMul!(out.W[j,:], in1.W[j,l,:], in2.W[l,:], mtab)
        end
    end
    return nothing
end

function PolyMatrixTimesMatrix!(out::PolyMatrixModel{T}, in1::PolyMatrixModel{T}, in2::PolyMatrixModel{T}) where T
    mtab = mulTable(out.mexp, in1.mexp, in2.mexp)
    @assert size(in1.W, 2) == size(in2.W, 1)
    out.W .= 0
    for j = 1:size(in1.W, 1)
        for k = 1:size(in2.W, 2)
            for l = 1:size(in1.W, 2)
                @views PolyMul!(out.W[j,k,:], in1.W[j,l,:], in2.W[l,k,:], mtab)
            end
        end
    end
    return nothing
end

function PolyDMatrixTimesVector!(out::PolyMatrixModel{T}, in1::PolyMatrixModel{T}, in2::PolyModel{T}) where T
    mtab = mulTable(out.mexp, out.mexp, in2.mexp)
    dtab = PolyDeriTab(out.mexp, in1.mexp)
    outp = zeros(typeof(in1.W[1]), size(in1.mexp, 2))
    out.W .= 0
    for j = 1:size(in1.W, 1)
        for k = 1:size(in1.W, 2)
            for l = 1:size(in2.W, 1)
                @views PolyDeri!(out.mexp, outp, in1.mexp, in1.W[j,k,:], dtab, l)
                # PolyMul!(out, in1::AbstractArray{T}, in2::AbstractArray{T}, multab)
                @views PolyMul!(out.W[j,k,:], outp, in2.W[l,:], mtab)
            end
        end
    end
    return nothing
end

function PolyDVectorTimesMatrix!(out::PolyMatrixModel, in1::PolyModel, in2::PolyMatrixModel)
    mtab = mulTable(out.mexp, out.mexp, in2.mexp)
    dtab = PolyDeriTab(out.mexp, in2.mexp)
    outp = zeros(typeof(in1.W[1]), size(in1.mexp, 2))
    out.W .= 0
    for j = 1:size(in1.W, 1)
        for k = 1:size(in2.W, 2)
            for l = 1:size(in1.mexp, 1)
                @views PolyDeri!(out.mexp, outp, in1.mexp, in1.W[j,:], dtab, l)
                @views PolyMul!(out.W[j,k,:], outp, in2.W[l,k,:], mtab)
            end
        end
    end
    return nothing
end

function PolyMatrixTimesDVector!(out::PolyMatrixModel{T}, in1::PolyMatrixModel{T}, in2::PolyModel{T}) where T
    mtab = mulTable(out.mexp, in1.mexp, in2.mexp)
    dtab = PolyDeriTab(in2.mexp, in2.mexp)
    outp = zeros(typeof(in2.W[1]), size(in2.mexp, 2))
    @assert size(in1.W, 2) == size(in2.W, 1)
    out.W .= 0
    for j = 1:size(in1.W, 1)
        for k = 1:size(in2.mexp, 1)
            for l = 1:size(in2.W, 1)
                @views PolyDeri!(in2.mexp, outp, in2.mexp, in2.W[l,:], dtab, k)
                @views PolyMul!(out.W[j,k,:], in1.W[j,l,:], outp, mtab)
            end
        end
    end
    return nothing
end

function Jacobian!(out::PolyMatrixModel, in1::PolyModel)
    dtab = PolyDeriTab(out.mexp, in1.mexp)
#    println("in1.mexp = ", size(in1.mexp)) 
#    println("in1.W = ", size(in1.W)) 
    for j = 1:size(in1.W, 1) # number of outputs
        for k = 1:size(in1.mexp, 1) # number of variables
            @views PolyDeri!(out.mexp, out.W[j,k,:], in1.mexp, in1.W[j,:], dtab, k)
#            println("IN=", maximum(abs.(in1.W[j,:])))
#            println("OUT=", maximum(abs.(out.W[j,k,:])))
        end
    end
    return nothing
end

function PolyMatrixSubsVector!(out::PolyMatrixModel, mat::PolyMatrixModel, vec::PolyModel)
    mtab = mulTable(out.mexp, out.mexp, vec.mexp)
    for k=1:size(mat.W, 1)
        @views PolySubs!(out.mexp,  out.W[k,:,:], mat.mexp, mat.W[k,:,:], vec.mexp, vec.W, mtab)
    end
    return nothing
end

function PolyMatrixConstantPart!(out::PolyMatrixModel{T}, B::AbstractArray{T}) where T
    oz = PolyOrderIndices(out.mexp, 0)
    out.W[:,:,oz] = B
    return nothing
end

function PolyMatrixGetConstantPart(in::PolyMatrixModel)
    oz = PolyOrderIndices(in.mexp, 0)
    return dropdims(in.W[:,:,oz], dims=3)
end

function PolyMatrixLinearTransform!(out::PolyMatrixModel, mat::PolyMatrixModel, TranSub::AbstractArray{T,2}, TranLeft::AbstractArray{T,2}, TranRight::AbstractArray{T,2}) where T
    e2 = one(ones(typeof(mat.mexp[1]), size(TranSub)))
    mtab = mulTable(out.mexp, out.mexp, e2)
    for k=1:size(mat.W, 1)
        @views PolySubs!(out.mexp,  out.W[k,:,:], mat.mexp, mat.W[k,:,:], e2, TranSub, mtab)
    end
    for k=1:size(mat.W, 3)
        tmp = TranLeft*out.W[:,:,k]*TranRight
        out.W[:,:,k] = tmp
    end
    return nothing
end

function PolyMatrixInverse(in::PolyMatrixModel)
    A = PolyMatrixGetConstantPart(in)
    PN = PolyMatrixCopy(in)
    PolyMatrixConstantPart!(PN, zero(A))
    # - A^(-1) P_N
    mAiPN = PolyMatrixZero(PN)
    for k=1:size(PN.W,3) 
        mAiPN.W[:,:,k] = -A\PN.W[:,:,k]
    end
    Q = PolyMatrixZero(in)
    PolyMatrixConstantPart!(PN, inv(A))
    tmp = PolyMatrixZero(in)
    while true
        PolyMatrixTimesMatrix!(tmp, mAiPN, Q)
        PolyMatrixConstantPart!(tmp, inv(A))
        # @show maximum(abs.(tmp.W .- Q.W))
        if maximum(abs.(tmp.W .- Q.W)) < 1e-12
            break
        end
        Q.W .= tmp.W
    end
    return Q
end

# integrates the differential equation VF(Phi(y)) = d/dy(idx) Phi(y). Initial condition Phi(y) = y when y(idx) = 0
function PolyODESolve(VF::PolyModel, idx)
    # start the iteration with the identity
    IC = PolyModel(typeof(VF.W[1]), size(VF.mexp,1), size(VF.W,1), PolyOrder(VF.mexp)+1)
    IClin = one(PolyGetLinearPart(IC))
    IClin[idx,idx] = 0
    PolySetLinearPart!(IC, IClin)

    IT0 = PolyZero(IC) # PolyModel(size(VF.mexp,1), size(VF.W,1), PolyOrder(VF.mexp)+1) # being iterated
    PolySetLinearPart!(IT0, IClin)
    IT1 = PolyZero(IC) # PolyModel(size(VF.mexp,1), size(VF.W,1), PolyOrder(VF.mexp)+1)
    toInt = PolyZero(IC)
    
    while true
        PolySubs!(toInt, VF, IT0)
        PolyIntegrate!(IT1, IC, toInt, idx)
        res = IT0.W .- IT1.W
        # TESTING
            # t_res = PolyZero(IT0)
            # PolyDeri!(t_res, IT1, idx)
            # @show maximum(abs.(t_res.W .- toInt.W))
            # println("\n\nDISPLAY DIFFEERNCES")
            # for k=1:size(toInt.W,1)
            #     ids = findall(abs.(t_res.W[k,:] .- toInt.W[k,:]) .> 1e-12)
            #     if ids != nothing
            #         @show sum(t_res.mexp[:,ids],dims=1)
            #     end
            #     # @show t_res.mexp[:,findall(abs.(t_res.W[k,:] .- toInt.W[k,:]) .> 1e-12)]'
            #     # @show t_res.W[k,findall(abs.(t_res.W[k,:] .- toInt.W[k,:]) .> 1e-12)]
            #     # @show toInt.W[k,findall(abs.(t_res.W[k,:] .- toInt.W[k,:]) .> 1e-12)]
            #     println("\n---------------------")
            # end
            # println("\n\nRESIDUAL")
            # for k=1:size(toInt.W,1)
            #     ids = findall(abs.(res[k,:]) .> 1e-12)
            #     if ids != nothing
            #         @show sum(IT0.mexp[:,ids],dims=1)
            #     end
            #     # @show IT0.mexp[:,findall(abs.(res[k,:]) .> 1e-12)]'
            #     # @show IT0.W[k,findall(abs.(res[k,:]) .> 1e-12)]
            #     # @show IT1.W[k,findall(abs.(res[k,:]) .> 1e-12)]
            #     println("\n---------------------")
            # end
        # END TESTING
        if maximum(abs.(res)) < 1e-9
            break
        else
            IT0.W .= IT1.W
        end
    end
    return IT1
end

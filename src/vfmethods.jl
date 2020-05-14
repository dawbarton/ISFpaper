
include("polymethods.jl")

# -------------------------------------------------------
# THE SSM FUNCTIONS
# -------------------------------------------------------

# calculates the adjoint spectral submanifold for a map given by F0
# the eigenvalues are selected by 'intvar', hence many dimensions can be selected
# takes into account the near internal resonances
function ISFVFCalc(F0, vars, par)
    intvar = [vars; par]
    ndim = size(F0.W, 1)
    F01 = PolyGetLinearPart(F0)
    
    eigval, eigvec = eigen(F01)
    # rescale the parameter so that it remains unity
    eigvec[:,par] /= eigvec[par,par]

    # println("eigenvectors")
    # for k=1:size(eigvec,2)
    #     @show eigvec[:,k]
    # end

    # makes the linear part diagonal, so that the rest of the calculation is easy
    F0c = PolyModel(F0.mexp, Complex.(F0.W))
    F = PolyZero(F0c)
    ModelLinearTransform!(F, F0c, eigvec)
    
    # define the indices of internal variables
    # intvar = sel:sel+1
    # define the indices of external variables
    extvar = setdiff(1:ndim, intvar)
    
    # it can possibly be done for higher dimansional SSMs as well
    zdim = length(intvar)
    
    # from here the operations are scalar, because of the diagonal matrices
    F1 = Diagonal(eigval)
    R1 = Diagonal(eigval[intvar])
    W1 = zeros(typeof(F.W[1]), zdim, ndim)
    W1[:, intvar] = one(R1)
    
    # put it into the polynomials
    modelorder = PolyOrder(F.mexp)
    
    W = PolyModel(typeof(F.W[1]), ndim, zdim, modelorder)
    R = PolyModel(typeof(F.W[1]), zdim, zdim, modelorder)
    # set the linear parts R1 -> R, W1 -> W
    PolySetLinearPart!(R, R1)
    PolySetLinearPart!(W, W1)

    # @time multabDWF = mulTable(W.mexp, W.mexp, W.mexp)
    deritabDWF = PolyDeriTab(W.mexp, W.mexp)
    multabWW = mulTable(W.mexp, W.mexp, W.mexp)

    # recursively do the transformation by order
    for ord = 2:PolyOrder(W.mexp)
        id = PolyOrderIndices(W.mexp, ord)
        # the inhomogeneity: B = DW \dot F - R \circ W
        # the inhomogeneity: B = DW \dot R - F \circ W
        res0 = PolyZero(W)
        PolyDeriMul!(res0, W, F, multabWW, deritabDWF)
        res1 = PolyZero(W)
        PolySubs!(res1, R, W, multabWW)
        B = res0.W - res1.W
        # calculate for each monomial
        for x in id
            # the order of external variables
            extord = sum(W.mexp[extvar,x])
            if extord == 0
                # rx is the index of this monomial 'x' in R
                rx = PolyFindIndex(R.mexp, W.mexp[intvar,x]) # this is calculated only once per monomial, hence no need to optimise out
            else
                rx = 0
            end   
            # now calculate for each dimension, which is the number of interval variables
            for j=1:length(intvar)
                # k is the index of the internal variable (we need double indexing)
                k = intvar[j]
                # if there are no external variables involved, we can take the resonances into account
                if extord == 0
                    # internal monomials
                    # SOLVE: sum(eigval.*W.mexp[:,x]) * W.W[k,x] - eigval[k] * W.W[k,x] = R.W[j,rx] - B[k,x]
                    den = sum(eigval.*W.mexp[:,x]) - eigval[k]
                    # the threshold of 0.1 is arbitrary
                    # we probably should use fixed near resonances
                    if abs(den) < 0.1
                        R.W[j,rx] = B[j,x]
                        W.W[j,x] = 0
                        # println("Internal resonance: ", abs(den), " at dim=", k, " exp=", W.mexp[:,x])
                    else
                        R.W[j,rx] = 0
                        W.W[j,x] = -B[j,x]/den
                    end
                # here, external variables are involved, we cannot take the resonances into account
                else
                    # external and mixed monomials
                    # SOLVE: prod(eigval.^W.mexp[:,x])*W.W[k,x] - eigval[k]*W.W[k,x] = B[k,x]
                    den = sum(eigval.*W.mexp[:,x]) - eigval[k]
                    if abs(den) > 1e-12
                        W.W[j,x] = -B[j,x]/den
                    else
                        println("Fatal cross resonance, not calculating term: ", abs(den), " at dim=", k, " exp=", W.mexp[:,x])
                        W.W[j,x] = 0.0
                    end
                    if abs(den) < 0.1
                        println("Warning: near cross resonance: ", abs(den), " at dim=", k, " exp=", W.mexp[:,x])
                    end
                end
            end
        end
    end
    # transform result back
    trb = [1 1im; 1 -1im]
    trout = zeros(ComplexF64, zdim, zdim)
    for k=0:div(zdim,2)-1
        trout[1+2*k:2*(k+1),1+2*k:2*(k+1)] = trb
    end
    if mod(zdim,2) == 1
        trout[end,end] = 1
    end
    trin = zeros(ComplexF64, ndim, ndim)
    for k=0:div(ndim,2)-1
        trin[1+2*k:2*(k+1),1+2*k:2*(k+1)] = trb
    end
    if mod(zdim,2) == 1
        trin[end,end] = 1
    end
   
    
    Wout = PolyZero(W)
    Rout = PolyZero(R)
    # this only transforms back into real
    # ModelLinearTransform!(Wout, W, trin, inv(trout))
    # thi one properly transforms back
    ModelLinearTransform!(Wout, W, inv(eigvec), inv(trout))
    ModelLinearTransform!(Rout, R, trout)
    Wout = PolyModel(Wout.mexp, real(Wout.W))
    Rout = PolyModel(Rout.mexp, real(Rout.W))

    # Check the result
    res0 = PolyZero(Wout)
    PolyDeriMul!(res0, Wout, F0, multabWW, deritabDWF)
    res1 = PolyZero(Wout)
    PolySubs!(res1, Rout, Wout, multabWW)
    B = res0.W - res1.W
    if maximum(abs.(B)) > 1e-10
        println("High error in adjoint SSM calculation: ", maximum(abs.(B)))
    end
    
    return Wout, Rout, W, R
end

# calculates the spectral submanifold for a map given by F0
# the eigenvalues are selected by 'intvar', hence many dimensions can be selected
# takes into account the near internal resonances
function SSMVFCalc(F0, vars, par)
    intvar = [vars; par]
    ndim = size(F0.W, 1)
    F01 = PolyGetLinearPart(F0)
    
    eigval, eigvec = eigen(F01)
    # rescale the parameter so that it remains unity
    eigvec[:,par] /= eigvec[par,par]

    # makes the linear part diagonal, so that the rest of the calculation is easy
    F0c = PolyModel(F0.mexp, Complex.(F0.W))
    F = PolyModel(F0.mexp, zero(F0c.W))
    println("timing: linear model transformation:")
    @time ModelLinearTransform!(F, F0c, eigvec)
    
    # define the indices of internal variables
    # intvar = sel:sel+1
    # define the indices of external variables
    extvar = setdiff(1:ndim, intvar)
    
    # it can possibly be done for higher dimansional SSMs as well
    zdim = length(intvar)
    
    # from here the operations are scalar, because of the diagonal matrices
    # F1 = Diagonal(eigval)
    R1 = Diagonal(eigval[intvar])

    W1 = zeros(typeof(F.W[1]), ndim, zdim)
    W1[intvar,:] = one(R1)
    
    # put it into the polynomials
    modelorder = PolyOrder(F.mexp)
    
    W = PolyModel(typeof(F.W[1]), zdim, ndim, modelorder)
    R = PolyModel(typeof(F.W[1]), zdim, zdim, modelorder)
    # set the linear parts R1 -> R, W1 -> W
    PolySetLinearPart!(R, R1)
    PolySetLinearPart!(W, W1)

    # multabDWR = mulTable(W.mexp, W.mexp, W.mexp)
    deritabDWR = PolyDeriTab(W.mexp, W.mexp)
    multabWW = mulTable(W.mexp, W.mexp, W.mexp)

    # recursively do the transformation by order
    for ord = 2:PolyOrder(W.mexp)
        println("SSM order ", ord, " out of ", PolyOrder(W.mexp))
        id = PolyOrderIndices(W.mexp, ord)
        # the inhomogeneity: B = DW \dot R - F \circ W
        res0 = PolyZero(W)
        PolyDeriMul!(res0, W, R, multabWW, deritabDWR)
        res1 = PolyZero(W)
        PolySubs!(res1, F, W, multabWW)
        B = res0.W - res1.W
        for x in id
            # going through the internal diemsions
            for j=1:length(intvar)
                k = intvar[j]
                # SOLVE: eigval[k]*W.W[k,x] - prod(eigval.^W.mexp[:,id])*W.W[k,x] = R.W[sel,x] + B[k,x]
                den = eigval[k] - sum(eigval[intvar].*W.mexp[:,x])
                if abs(den) < 0.1
                    R.W[j,x] = -B[k,x]
                    W.W[k,x] = 0
                    println("Internal resonance: ", abs(den), " at dim=", k, " exp=", W.mexp[:,x])
                else
                    R.W[j,x] = 0
                    W.W[k,x] = B[k,x]/den
                end
            end
            # going through the external dimensions
            for k in extvar
                # SOLVE: eigval[k]*W.W[k,x] - prod(eigval.^W.mexp[:,id])*W.W[k,x] = B[k,x]
                den = eigval[k] - sum(eigval[intvar].*W.mexp[:,x])
                W.W[k,x] = B[k,x]/den
                if abs(den) < 0.1
                    println("Warning: near cross resonance: ", abs(den), " at dim=", k, " exp=", W.mexp[:,x])
                end
            end
        end
    end
    # this assumes a diagonal linear part of 'F'
    V, S = SSMVFBundleCalc(F, W, R, vars, par)
        
    # transform result back
    trb = [1 1im; 1 -1im]
    trout = zeros(ComplexF64, zdim, zdim)
    for k=0:div(zdim,2)-1
        trout[1+2*k:2*(k+1),1+2*k:2*(k+1)] = trb
    end
    if mod(zdim,2) == 1
        trout[end,end] = 1
    end
    
    # for the SSM
    Wout = PolyZero(W)
    Rout = PolyZero(R)
    ModelLinearTransform!(Wout, W, trout, eigvec)
    ModelLinearTransform!(Rout, R, trout)

    # for the invariant bundle
    Vout = PolyMatrixModel(typeof(eigval[1]), zdim, modelorder, zdim, ndim)
    Sout = PolyMatrixModel(typeof(eigval[1]), zdim, modelorder, zdim, zdim)
    PolyMatrixLinearTransform!(Vout, V, trout, inv(trout), inv(eigvec))
    PolyMatrixLinearTransform!(Sout, S, trout, inv(trout), trout)

    # Check the result
    # the inhomogeneity: B = DW \dot R - F \circ W
    res0 = PolyZero(W)
    PolyDeriMul!(res0, Wout, Rout, multabWW, deritabDWR)
    res1 = PolyZero(W)
    PolySubs!(res1, F0c, Wout, multabWW)
    B = res0.W - res1.W
    if maximum(abs.(B)) > 1e-12
        println("High error in SSM calculation: ", maximum(abs.(B)))
    end    
    
    # checking the result
    J0 = PolyMatrixModel(typeof(eigval[1]), ndim, modelorder, ndim, ndim)
    Jacobian!(J0, F0)
    # substitute the immersion into the Jacobian
    JAC = PolyMatrixModel(typeof(eigval[1]), zdim, modelorder, ndim, ndim)
    PolyMatrixSubsVector!(JAC, J0, Wout)
    
    res1 = PolyMatrixModel(Vout.mexp, zero(Vout.W))
    PolyDMatrixTimesVector!(res1, Vout, Rout)

    res2 = PolyMatrixModel(Vout.mexp, zero(Vout.W))
    PolyMatrixTimesMatrix!(res2, Vout, JAC)
    
    res3 = PolyMatrixModel(Vout.mexp, zero(Vout.W))
    PolyMatrixTimesMatrix!(res3, Sout, Vout)
    
    BV = res1.W + res2.W - res3.W
    if maximum(abs.(BV)) > 1e-12
        println("High error in invariant bundle (linearised ISF) calculation: ", maximum(abs.(BV)))
    end
    
    # do the transformation here
    v0 = zeros(ndim)
    v0[end] = 1
    v1 = eigvec\v0
    VTv = zeros(typeof(V.W[1]), size(V.W,1), size(V.W,3))
    for k=1:size(V.W,3)
        VTv[:,k] = V.W[:,:,k]*v1
    end
    # @show v1
    
    return Wout, Rout, Vout, Sout, R, VTv
    # return W, R
end


# as the input
# F has diagonal linear part
# W similarly has the identity as the linear part
function SSMVFBundleCalc(F, W, R, vars, par)
    ndim = size(F.W, 1)
    intvar = [vars; par]
    extvar = setdiff(1:ndim, intvar)
    zdim = length(intvar)
    # put it into the polynomials
    modelorder = PolyOrder(F.mexp)

    A = PolyGetLinearPart(F)
    eigval = diag(A)

    # calculate the Jacobian
    J0 = PolyMatrixModel(typeof(eigval[1]), ndim, modelorder, ndim, ndim)
    Jacobian!(J0, F)
    # substitute the immersion into the Jacobian
    JAC = PolyMatrixModel(typeof(eigval[1]), zdim, modelorder, ndim, ndim)
    println("timing: PolyMatrixSubsVector:")
    @time PolyMatrixSubsVector!(JAC, J0, W)

    # define the variables
    V = PolyMatrixModel(typeof(eigval[1]), zdim, modelorder, zdim, ndim)
    S = PolyMatrixModel(typeof(eigval[1]), zdim, modelorder, zdim, zdim)
    
    V0 = zeros(typeof(F.W[1]), zdim, ndim)
    S0 = Diagonal(eigval[intvar])
    V0[:, intvar] = one(S0)
    
    PolyMatrixConstantPart!(V, V0)
    PolyMatrixConstantPart!(S, S0)

    # recursively do the transformation by order
    for ord = 1:PolyOrder(W.mexp)
        println("SSM Bundle order ", ord, " out of ", PolyOrder(W.mexp))
        # the inhomogeneity: B = DV \dot R + V \dot DF -  DR \dot V
        res1 = PolyMatrixModel(V.mexp, zero(V.W))
        PolyDMatrixTimesVector!(res1, V, R)

        res2 = PolyMatrixModel(V.mexp, zero(V.W))
        PolyMatrixTimesMatrix!(res2, V, JAC)
        
        res3 = PolyMatrixModel(V.mexp, zero(V.W))
        PolyMatrixTimesMatrix!(res3, S, V)
        
        B = res1.W + res2.W - res3.W

        id = PolyOrderIndices(W.mexp, ord)
        for x in id
            for k=1:length(intvar)
                # k is the index of the internal variable (we need double indexing)
                q = intvar[k]
                for p=1:zdim
                    den = sum(eigval[intvar].*V.mexp[:,x]) + eigval[q] - eigval[intvar[p]]
                    if abs(den) > 0.1
                        V.W[p,q,x] = -B[p,q,x]/den
                        S.W[p,k,x] = 0.0
                        # R.W remains zero
                    else
                        # println("Internal resonance = ", V.mexp[:,x], " vs ", p, " : ", q)
                        V.W[p,q,x] = 0.0
                        S.W[p,k,x] = B[p,q,x]
                    end
                end
            end
            for k=1:length(extvar)
                # k is the index of the internal variable (we need double indexing)
                q = extvar[k]
                for p=1:zdim
                    den = sum(eigval[intvar].*V.mexp[:,x]) + eigval[q] - eigval[intvar[p]]
                    if abs(den) < 1e-6
                        println("Cross resonance = ", V.mexp[:,x], " vs ", p, " : ", q)
                    else
                        V.W[p,q,x] = -B[p,q,x]/den
                    end
                end
            end
        end
        res1 = PolyMatrixModel(V.mexp, zero(V.W))
        PolyDMatrixTimesVector!(res1, V, R)

        res2 = PolyMatrixModel(V.mexp, zero(V.W))
        PolyMatrixTimesMatrix!(res2, V, JAC)
        
        res3 = PolyMatrixModel(V.mexp, zero(V.W))
        PolyMatrixTimesMatrix!(res3, S, V)
        
        B = res1.W + res2.W - res3.W
        for x in id
            if maximum(abs.(B[:,:,x])) > 1e-6
                println("exps = ", V.mexp[:,x])
                @show findall(abs.(B[:,:,x]) .> 1e-7)
                @show B[findall(abs.(B[:,:,x]) .> 1e-7),x]
            end
        end
    end
    # the inhomogeneity: B = DV \dot R + V \dot DF -  DR \dot V
    res1 = PolyMatrixModel(V.mexp, zero(V.W))
    PolyDMatrixTimesVector!(res1, V, R)

    res2 = PolyMatrixModel(V.mexp, zero(V.W))
    PolyMatrixTimesMatrix!(res2, V, JAC)
    
    res3 = PolyMatrixModel(V.mexp, zero(V.W))
    PolyMatrixTimesMatrix!(res3, S, V)
    
    B = res1.W + res2.W - res3.W

    if maximum(abs.(B)) > 1e-12
        println("High error in invariant bundle (linearised ISF) calculation: ", maximum(abs.(B)))
    end
    
    return V, S
end

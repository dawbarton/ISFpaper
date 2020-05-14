include("polymethods.jl")

# -------------------------------------------------------
# THE SSM FUNCTIONS
# -------------------------------------------------------

function ISFTransform(W, R, eigvec)
    Tinv = eigvec
    T = inv(eigvec)    
    U = [1.0 1.0; -1.0im 1.0im]/2.0
    Uinv = [1.0 1.0im; 1.0 (-1.0im)]

    Wout = PolyZero(W)
    Rout = PolyZero(R)
    # F0 is with diagonal linear part
    # F = T^-1 F0(T x) is the original map
    # T = fwtran, U = trout
    # R( W(x) ) = W( F0(x) )
    # U*R( U^-1 [U W(T x)] ) = U*W( T [T^-1 F0(T x)] )
    # Ro( Wo(x) ) = Wo( F(x) )
    ModelLinearTransform!(Wout, W, T, U)    # Wo = U * W(T x)
    ModelLinearTransform!(Rout, R, Uinv, U) # Ro = U * R(U^-1 z)
    
    # These migh not vanish
    # @show maximum(abs.(imag.(Wout.W)))
    # @show maximum(abs.(imag.(Rout.W)))
    
    Wout = PolyModel(Wout.mexp, real(Wout.W))
    Rout = PolyModel(Rout.mexp, real(Rout.W))
    
    return Wout, Rout
end

function SSMTransform(W, R, eigvec)
    # transforms to original coordinates
    Tinv = eigvec
    T = inv(eigvec)    
    U = [1.0 1.0; -1.0im 1.0im]/2.0
    Uinv = [1.0 1.0im; 1.0 (-1.0im)]
    
    Wout = PolyZero(W)
    Rout = PolyZero(R)
    # F0 is with diagonal linear part
    # F = T^-1 F0(T x) is the original map
    # T = fwtran, U = trout
    # W( R(z) ) = F0( W(z) )
    # T^-1 W( R(z) ) = T^-1 F0( T [T^-1 W(z)] )
    # T^-1 W( U^-1 [U R(U^-1 z)] ) = T^-1 F0( T [T^-1 W(U^-1 z)] )
    # Wo( Ro(x) ) = F( Wo(x) )    
    ModelLinearTransform!(Wout, W, Uinv, Tinv) # Wo = T^-1 W(U^-1 x)
    ModelLinearTransform!(Rout, R, Uinv, U)    # Ro = U * R(U^-1 z)
    
    # These migh not vanish
#     @show maximum(abs.(imag.(Wout.W)))
#     @show maximum(abs.(imag.(Rout.W)))
    
    Wout = PolyModel(Wout.mexp, real(Wout.W))
    Rout = PolyModel(Rout.mexp, real(Rout.W))
    
    return Wout, Rout
end

# one can leave the parameter out as []
function ISFCalc(F0, vars, par)
    intvar = [vars; par]
    ndim = size(F0.W, 1)
    F01 = PolyGetLinearPart(F0)
    
    eigval, eigvec = eigen(F01)
    # rescale the parameter so that it remains unity
    if !isempty(par)
        eigvec[:,par] /= eigvec[par,par]
    end

    vals, adjvecs = eigen(collect(F01'))
    # assuming that adjvecs are complex conjugate pairs
    adjproj = vcat(real(adjvecs[:,1])', imag(adjvecs[:,1])', real(adjvecs[:,3])', imag(adjvecs[:,3])')

    # makes the linear part diagonal, so that the rest of the calculation is easy
    F0c = PolyModel(F0.mexp, Complex.(F0.W))
    F = PolyZero(F0c)
    ModelLinearTransform!(F, F0c, eigvec)
#     F0r = PolyZero(F0)
#     ModelLinearTransform!(F0r, F0, inv(adjproj))
    
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
    # mulTable(out.mexp, out.mexp, in2.mexp)
    multabWF = mulTable(W.mexp, W.mexp, W.mexp)
    multabRW = multabWF

    # recursively do the transformation by order
    for ord = 2:PolyOrder(W.mexp)
        id = PolyOrderIndices(W.mexp, ord)
        # the inhomogeneity: B = W \circ F - R \circ W
        
        res0 = PolyZero(W)
        PolySubs!(res0, W, F, multabWF)
        res1 = PolyZero(W)
        PolySubs!(res1, R, W, multabRW)
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
                    # SOLVE: prod(eigval.^W.mexp[:,x]) * W.W[k,x] - eigval[k] * W.W[k,x] = R.W[j,rx] - B[k,x]
                    den = prod(eigval.^W.mexp[:,x]) - eigval[k]
                    # the threshold of 0.1 is arbitrary
                    # we probably should use fixed near resonances
                    if abs(den) < 0.2
                        # Purely a graph over  
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
                    den = prod(eigval.^W.mexp[:,x]) - eigval[k]
                    if abs(den) > 1e-6
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
    Wout, Rout = ISFTransform(W, R, eigvec)

    res0 = PolyZero(Wout)
    PolySubs!(res0, Wout, F0, multabWF)
    res1 = PolyZero(Wout)
    PolySubs!(res1, Rout, Wout, multabRW)
    B = res0.W - res1.W
    if maximum(abs.(B)) > 1e-10
        println("High error in ISF calculation (REAL): ", maximum(abs.(B)))
    end
    
    res0 = PolyZero(W)
    PolySubs!(res0, W, F, multabWF)
    res1 = PolyZero(W)
    PolySubs!(res1, R, W, multabRW)
    B = res0.W - res1.W
    if maximum(abs.(B)) > 1e-10
        println("High error in ISF calculation (CPLX): ", maximum(abs.(B)))
        @show W.mexp[:,findall((abs.(B) .> 1e-12)[1,:])]
        @show abs.(B[1,findall((abs.(B) .> 1e-12)[1,:])])
    end
    
    return Wout, Rout, W, R
end

# calculates the 2-dim spectral submanifold for a map given by F0
# the eigenvalue is selected by 'sel'
# takes into account the near internal resonances
function SSMCalc(F0, intvar)
    # intvar: indices of internal variables
    ndim = size(F0.W, 1)
    zdim = length(intvar)
    # indices of external variables
    extvar = setdiff(1:ndim, intvar)

    F01 = PolyGetLinearPart(F0)

    eigval, eigvec = eigen(F01)

    F0c = PolyModel(F0.mexp, Complex.(F0.W))
    F = PolyZero(F0c)
    ModelLinearTransform!(F, F0c, eigvec)

    # from here the operations are scalar, because of the diagonal matrices
    F1 = Diagonal(eigval)
    R1 = Diagonal(eigval[intvar])
    W1 = zeros(typeof(F.W[1]), ndim, zdim)
    W1[intvar, :] = one(R1)

    # put it into the polynomials
    modelorder = PolyOrder(F.mexp)
    W = PolyModel(typeof(F.W[1]), zdim, ndim, modelorder)
    R = PolyModel(typeof(F.W[1]), zdim, zdim, modelorder)
    # set the linear parts
    PolySetLinearPart!(R, R1)
    PolySetLinearPart!(W, W1)

    for ord = 2:modelorder
        id = PolyOrderIndices(W.mexp, ord)

        # the inhomogeneity
        res0 = PolyZero(W)
        PolySubs!(res0, W, R)
        res1 = PolyZero(W)
        PolySubs!(res1, F, W)
        B = res0.W - res1.W

        for x in id
            # going through the internal diemsions
            for j=1:length(intvar)
                k = intvar[j]
                # SOLVE: eigval[k]*W.W[k,x] - prod(eigval.^W.mexp[:,id])*W.W[k,x] = R.W[sel,x] + B[k,x]
                den = eigval[k] - prod(eigval[intvar].^W.mexp[:,x])
                if abs(den) < 0.2
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
                den = eigval[k] - prod(eigval[intvar].^W.mexp[:,x])
                W.W[k,x] = B[k,x]/den
                if abs(den) < 0.1
                    println("Warning: near cross resonance: ", abs(den), " at dim=", k, " exp=", W.mexp[:,x])
                end
            end
        end
    end
    # Check the result
    res0 = PolyZero(W)
    PolySubs!(res0, W, R)
    res1 = PolyZero(W)
    PolySubs!(res1, F, W)
    B = res0.W - res1.W
    if maximum(abs.(B)) > 1e-10
        println("High error in SSM calculation (CPLX): ", maximum(abs.(B)))
    end

    Wout, Rout = SSMTransform(W, R, eigvec)
    # Check the transformed result
    res0 = PolyZero(Wout)
    PolySubs!(res0, Wout, Rout)
    res1 = PolyZero(Wout)
    PolySubs!(res1, F0, Wout)
    B = res0.W - res1.W
    if maximum(abs.(B)) > 1e-10
        println("High error in SSM calculation (REAL): ", maximum(abs.(B)))
    end

    return Wout, Rout
end

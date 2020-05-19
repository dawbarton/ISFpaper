using ForwardDiff

# creates a single map from a tuple of submersions
function makeUhat(Utup)
    lens = zeros(Int, length(Utup))
    begs = zeros(Int, length(Utup))
    ends = zeros(Int, length(Utup))
    begs[1] = 1
    lens[1] = size(Utup[1].W,1)
    ends[1] = begs[1] + lens[1] - 1
    for k=2:length(Utup)
        lens[k] = size(Utup[k].W,1)
        begs[k] = begs[k-1] + lens[k]
        ends[k] = begs[k] + lens[k] - 1
    end
    Uc = PolyModel(Utup[1].mexp, zeros(typeof(Utup[1].W[1]), ends[end], size(Utup[1].W,2)))
    for k=1:length(Utup)
        Uc.W[begs[k]:ends[k],:] = Utup[k].W
    end
    return Uc
end

# returns the inverse of (Utup) as a single polynomial of the same order
function Restore(Utup)
    Uc = makeUhat(Utup)
    UcA = PolyGetLinearPart(Uc)
    @show UcA
    UcN = PolyModel(Uc.mexp, - inv(UcA) * Uc.W)
    PolySetLinearPart!(UcN, zero(UcA))

    res0 = PolyModel(size(Uc.mexp,1), size(Uc.W,1), PolyOrder(U.mexp))
    PolySetLinearPart!(res0, inv(UcA))
    res1 = PolyCopy(res0)
    for k=1:PolyOrder(res0.mexp)+1
        PolySubs!(res1, UcN, res0)
        PolySetLinearPart!(res1, inv(UcA))
        err = maximum(abs.(res0.W .- res1.W))
        res0.W .= res1.W
        if err < 1e-12
            break
        end
    end
    return res0
end

# This makes a submersion into a set of graphs parametrised by z
function SubmersionToGraph(U)
    U = PolyModel(U.mexp, U.W)
    UL = PolyGetLinearPart(U)
    ULH = vcat(UL, zeros(typeof(UL[1]), size(UL,2) - size(UL,1), size(UL,2)))
    F = svd(ULH)
    Vpar = F.Vt[1:size(UL,1),:]'
    Vpar = Vpar*inv(UL*Vpar)
    Vperp = F.Vt[size(UL,1)+1:end,:]'
    
    # nonlinear part
    UN = PolyCopy(U)
    PolySetLinearPart!(UN, zero(UL))
    
#     order = max(PolyOrder(U.mexp)+2, 9)
    order = PolyOrder(U.mexp)
    # first variable is z, second variable is y
    # the identity in y 
    yID = PolyModel(size(UL,2), size(UL,2) - size(UL,1), order)
    yIDlin = zero(PolyGetLinearPart(yID))
    @show size(yIDlin)
    @show size(yIDlin[:,size(UL,1)+1:end])
    yIDlin[:,size(UL,1)+1:end] .= one(yIDlin[:,size(UL,1)+1:end])
    PolySetLinearPart!(yID, yIDlin)
    # the identity in z
    zID = PolyModel(size(UL,2), size(UL,1), order)
    zIDlin = zero(UL)
    zIDlin[:,1:size(UL,1)] .= one(zIDlin[:,1:size(UL,1)])
    PolySetLinearPart!(zID, zIDlin)
    
    # define the g function
    g = PolyModel(size(UL,2), size(UL,1), order)
    glin = zero(UL)
    glin[1:size(UL,1),1:size(UL,1)] = one(glin[1:size(UL,1),1:size(UL,1)])
    PolySetLinearPart!(g, glin)

    Wz = PolyModel(size(UL,2), size(UL,2), order)
    UNWz = PolyZero(g)
    gi = PolyZero(g)
    # out, out, in2
    multabUNWz = mulTable(UNWz.mexp, UNWz.mexp, Wz.mexp)
    # iteration starts
    for k=1:order+1
        Wz.W .= Vperp*yID.W + Vpar*g.W
        PolySubs!(UNWz, UN, Wz, multabUNWz)
        gi.W .= zID.W - UNWz.W
        @show maximum(abs.(gi.W .- g.W))
        g.W .= gi.W
    end
    return Wz
end

# calculates the parameterised leaves using Newton's method
function WzSVD(U, x0; init=nothing)
    # Linear part
    UL = ForwardDiff.jacobian(U, x0)
    # stack a zero to make it square 
    ULH = vcat(UL, zeros(typeof(UL[1]), size(UL,2) - size(UL,1), size(UL,2)))
    F = svd(ULH)
    Vpar = F.Vt[1:size(UL,1),:]'
    Vpar = Vpar*inv(UL*Vpar)
    Vperp = F.Vt[size(UL,1)+1:end,:]'

    z = x0[1:size(UL,1)]
    y = x0[1+size(UL,1):end]
    jac = xx -> ForwardDiff.jacobian(U, xx)

    if init == nothing
        gzy0 = copy(z)
    else
        gzy0 = init
    end
    deltagzy = zero(gzy0)
    it = 1
    while true
        x1 = Vperp*y + Vpar*gzy0
#             @show jac(x1)*Vpar
        deltagzy .= (jac(x1)*Vpar) \ (U(x1) .- z)
        gzy0 .-= deltagzy
        if maximum(abs.(deltagzy)) < 1e-8
            break
        end
        it += 1
        if it > 1200
            println("WzSVD: max iteration exceeded. Error = ", maximum(abs.(deltagzy)))
            x1 .= NaN
            gzy0 .= NaN
            break
        end
    end
    x1 = Vperp*y + Vpar*gzy0
    return x1, gzy0
end

# returns the solution z of x0 = Uc( z ) using Newton's method
function UcInv(Uc, x0; init=nothing)
    jac = xx -> ForwardDiff.jacobian(Uc, xx)
    A = PolyGetLinearPart(Uc)
    W0 = zero(x0)
    if init == nothing
        W0 .= A\x0
    else
        W0 .= init
    end
    it = 1
    while true
        deltaW = jac(W0)\(x0 - Uc(W0))
        W0 .+= deltaW
        if maximum(abs.(deltaW)) < 1e-8
            break
        end
        it += 1
        if it > 1200
            println("WzSSM: max iteration exceeded. Error = ", maximum(abs.(deltaW)))
            W0 .= NaN
            break
        end
    end
    return W0
end

# returns the solution z of x0 = Utup( z ) using Newton's method
function RestoreNewton(Utup, x0; init=nothing)
    Uc = makeUhat(Utup)
    return UcInv(Uc, x0; init=init)
end

# calculates the amplitude from the result of SubmersionToGraph(U)
function ISFGraphAmplitude(h, r, ndim)
    amps = zero(r)
    arg = zeros(ndim)
    for k=1:length(r)
        maxamp = 0.0
        for q in range(0, 2*pi, length=24)
            arg[1] = r[k]*cos(q)
            arg[2] = r[k]*sin(q)
            x = h(arg)
            nr = maximum(abs.(x))
            if nr > maxamp
                maxamp = nr
            end
        end
        amps[k] = maxamp
    end
    return amps
end

# it calculates the amplitude from the result of WzSVD
function ISFGraphNewtonAmplitude(U, r, ndim)
    amps = zero(r)
    x = zeros(ndim)
    gprev = zeros(2)
    arg = zero(x)
    thetas = range(0, 2*pi, length=24)
    for k=1:length(r)
        maxamp = 0.0
        for l=1:length(thetas)
            q = thetas[l]
            arg[1] = r[k]*cos(q)
            arg[2] = r[k]*sin(q)
            if any(isnan.(gprev))
                x, gprev = WzSVD(U, arg)
            else
                x, gprev = WzSVD(U, arg; init = gprev)
            end
            nr = maximum(abs.(x))
            if nr > maxamp
                maxamp = nr
            end
        end
        amps[k] = maxamp
    end
    return amps
end

# This resonstructs the SSM from the tuple of ISFs
# In particular, it calculates the amplitude from the result of UcInv
function ISFManifoldNewtonAmplitude(Utup, r)
    Uc = makeUhat(Utup)
    amps = zero(r)
    x = UcInv(Uc, zeros(size(Uc.W,1)))
    xprev = [zero(x) for k=1:24]
    arg = zero(x)
    thetas = range(0, 2*pi, length=24)
    for k=1:length(r)
        maxamp = 0.0
        for l=1:length(thetas)
            q = thetas[l]
            arg[1] = r[k]*cos(q)
            arg[2] = r[k]*sin(q)
            if any(isnan.(xprev[l]))
                x .= UcInv(Uc, arg)
            else
                x .= UcInv(Uc, arg; init = xprev[l])
            end
            xprev[l] .= x
            nr = maximum(abs.(x))
            if nr > maxamp
                maxamp = nr
            end
        end
        amps[k] = maxamp
    end
    return amps
end


function fRealImag(S, r)
    Sr = S([r, 0.0])
    return Sr[1]/r, -Sr[2]/r
end

# frequency, damping or the conjugate dynamics at parameter r
function freqdamp(S, r, T)
    fr, fi = fRealImag(S, r)
    omega = angle(fr+1im*fi)/T
    dm = -log(sqrt(fr^2 + fi^2))/T/abs(omega)
    return omega, dm
end

function VFfreqdamp(S, r)
    Sr = real(S([r, 0.0])/r)
    omega = abs(Sr[2])
    dm = -Sr[1]/omega
    return omega, dm
end

function ISFGraphBackbones(S, h, r, T)
    ndim = size(h.W,1)
    amps = ISFGraphAmplitude(h, r, ndim)
    freq = zero(r)
    damp = zero(r)
    for k=1:length(r)
        freq[k], damp[k] = freqdamp(S, r[k], T)
    end
    return amps, freq, damp
end

function ISFGraphNewtonBackbones(S, U, r, T)
    ndim = size(U.mexp,1)
    amps = ISFGraphNewtonAmplitude(U, r, ndim)
    freq = zero(r)
    damp = zero(r)
    for k=1:length(r)
        freq[k], damp[k] = freqdamp(S, r[k], T)
    end
    return amps, freq, damp
end

function ISFManifNewtonBackbones(S, Utup, r, T)
    amps = ISFManifoldNewtonAmplitude(Utup, r)
    freq = zero(r)
    damp = zero(r)
    for k=1:length(r)
        freq[k], damp[k] = freqdamp(S, r[k], T)
    end
    return amps, freq, damp
end

function VFISFGraphBackbones(S, h, r)
    ndim = size(h.W,1)
    amps = ISFGraphAmplitude(h, r, ndim)
    freq = zero(r)
    damp = zero(r)
    for k=1:length(r)
        freq[k], damp[k] = VFfreqdamp(S, r[k])
    end
    return amps, freq, damp
end

function VFISFGraphNewtonBackbones(S, U, r)
    ndim = size(U.mexp,1)
    amps = ISFGraphNewtonAmplitude(U, r, ndim)
    freq = zero(r)
    damp = zero(r)
    for k=1:length(r)
        freq[k], damp[k] = VFfreqdamp(S, r[k])
    end
    return amps, freq, damp
end

function VFISFManifNewtonBackbones(S, Utup, r)
    amps = ISFManifoldNewtonAmplitude(Utup, r)
    freq = zero(r)
    damp = zero(r)
    for k=1:length(r)
        freq[k], damp[k] = VFfreqdamp(S, r[k], T)
    end
    return amps, freq, damp
end

# amplitude
function SSMAmplitude(Wz, r)
    ndim = size(Wz.mexp,1)
    maxamp = 0.0
    for q in range(0, 2*pi, length=24)
        nr = maximum(abs.(Wz([r*cos(q), r*sin(q)])))
        if nr > maxamp
            maxamp = nr
        end
    end
    return maxamp
end

function SSMBackbones(S, Wz, r, T)
    amps = zero(r)
    freq = zero(r)
    damp = zero(r)
    for k=1:length(r)
        amps[k] = SSMAmplitude(Wz, r[k])
        freq[k], damp[k] = freqdamp(S, r[k], T)
    end
    return amps, freq, damp
end

function VFSSMBackbones(S, Wz, r)
    amps = zero(r)
    freq = zero(r)
    damp = zero(r)
    for k=1:length(r)
        amps[k] = SSMAmplitude(Wz, r[k])
        freq[k], damp[k] = VFfreqdamp(S, r[k])
    end
    return amps, freq, damp
end


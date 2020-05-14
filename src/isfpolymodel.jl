include("polymethods.jl")
# Here we define the polynomial ISF model
function ISFPolyExps(in::Integer, omin::Integer, omax::Integer)
    @polyvar x[1:in]
    mx0 = monomials(x, omin:omax)
    return hcat([exponents(m) for m in mx0]...)
end

struct ISFexps
    U::Array{Int64,2}
    f::Array{Int64,2}
end


function ISFexps(ndim::Integer, order::Integer)
    return ISFexps(ISFPolyExps(ndim, 1, order), ISFPolyExps(1, 0, div(order-1,2)))
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
function ISFconstruct(ndim, order, vsr, vsi, lr, li)
    mexp = ISFexps(ndim, order)
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

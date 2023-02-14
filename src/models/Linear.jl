import Distributions:Normal
export Linear
struct Linear <: Model
    nInput::Int
    nOutput::Int
    W::Matrix{Param}
    b::Union{Vector{Param},Nothing}
    function Linear(nin::Int,nout::Int;bias=true)
        xavier = Normal(0,√(2/(nin+nout)))
        W = Param.(rand(xavier,(nout,nin)))
        b = bias ? Param.(rand(xavier,nout)) : Nothing 
        return new(nin,nout,W,b)
    end
end

(model::Linear)(x) = model.W * x .+ model.b

function Base.show(io::IO,linear::Linear)
    Wstr = "$(linear.nInput) => $(linear.nOutput)"
    if linear.b isa Nothing
        print(io,"Linear($Wstr)")
    else
        print(io,"Linear($Wstr,$(linear.nOutput))")
    end
end
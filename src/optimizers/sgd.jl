export SGD

struct SGD <: Optimizer
    params::Vector{Parameter}
    η::Float64
    SGD(params;η=1e-4) = new(params,η)
end

function step!(sgd::SGD)
    for p in sgd.params
        p.val -= sgd.η * p.grad
    end
end
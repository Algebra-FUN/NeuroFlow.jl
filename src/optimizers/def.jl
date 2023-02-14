export Optimizer,zero_grad!

abstract type Optimizer end

function zero_grad!(optimizer::Optimizer)
    for p in optimizer.params
        p.grad = 0
    end
end

export sigmoid, relu, leakyrelu, softmax

sigmoid(x) = 1/(1+exp(-x))

@func_register sigmoid(x) begin
    @backward!(x,exp(x)/(1+exp(x))^2)
end

relu(x) = x > 0 ? x : zero(x)
leakyrelu(x;a=0.1) = x > 0 ? x : a*x

function softmax(x::Union{Vector,Matrix})
    ex = exp.(x)
    return ex./sum(ex,dims=1)
end
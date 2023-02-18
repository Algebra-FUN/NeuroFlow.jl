export Variable, Var, Param, value, backward!, AUTOGRAD, @no_grad

abstract type Variable end

Base.length(var::Variable) = 1
Base.iterate(var::Variable) = (var, nothing)
Base.iterate(::Variable, ::Nothing) = nothing

mutable struct Var <: Variable
    val::Float64
    func::Union{Function,Nothing}
    inputs::Union{Tuple,Nothing}
    Var(val) = new(val, nothing, nothing)
    Var(val, func, inputs) = new(val, func, inputs)
end

Base.show(io::IO, var::Var) = print(io, "Var($(var.val))")

mutable struct GradLock
    on::Bool
end

const AUTOGRAD = GradLock(true)

macro no_grad(expr)
    return quote
        AUTOGRAD.on = false
        local result = $(esc(expr))
        AUTOGRAD.on = true
        result
    end
end

mutable struct Param <: Variable
    val::Float64
    grad::Float64
    Param(val) = new(val, 0.0)
end

Base.show(io::IO, var::Param) = print(io, "Param($(var.val))")

value(x::Variable) = x.val
value(x::Real) = x

Base.getproperty(vararray::Array{<:Variable}, sym::Symbol) = getproperty.(vararray, sym)
Base.setproperty!(vararray::Array{<:Variable}, sym::Symbol, val::Union{Real,Array{<:Real}}) = setproperty!.(vararray, sym, val)

Base.zero(::Variable) = Var(0)

forward(f::Function, inputs...;diff=true) = diff && AUTOGRAD.on ? Var(f(value.(inputs)...), f, inputs) : f(value.(inputs)...)

function backward!(var::Variable, grad::Float64)
    if var isa Param
        var.grad += grad
    elseif var.inputs isa Tuple
        backward!(var.func, grad, var.inputs...)
    end
end

function backward!(var::Variable)
    if var.inputs isa Tuple
        @no_grad backward!(var.func, 1. , var.inputs...)
    end
end

macro backward!(var::Symbol, grad_expr::Any)
    return :($var isa Variable && backward!($var, grad * $grad_expr)) |> esc
end

macro func_register(func_expr::Expr, backward_exprs::Union{Expr,Nothing}=nothing)
    @assert func_expr.head == :call
    f = func_expr.args[1]
    params = func_expr.args[2:end]
    diff = !(backward_exprs isa Nothing)
    n = length(params)
    func_exprs = []
    all_types = collect(Iterators.product(collect(Iterators.repeated((:Variable, :Real), n))...))
    for types in all_types[1:end-1]
        pairs = [:($p::$type) for (p, type) in zip(params, types)]
        push!(func_exprs, :($f($(pairs...)) = forward($f, $(params...);diff=$diff)))
    end
    if diff
        return quote
            $(Expr(:block, func_exprs...))
            function backward!(::typeof($f), grad::Float64, $(params...))
                $backward_exprs
            end
        end |> esc
    else
        return Expr(:block, func_exprs...) |> esc
    end
end

@func_register Base.isless(x,y)

@func_register Base.:+(x, y) begin
    @backward!(x, 1)
    @backward!(y, 1)
end

@func_register Base.:-(x) begin
    @backward!(x,-1)
end

@func_register Base.:-(x, y) begin
    @backward!(x, 1)
    @backward!(y, -1)
end

@func_register Base.:*(x, y) begin
    @backward!(x, y)
    @backward!(y, x)
end

@func_register Base.:/(x, y) begin
    @backward!(x, 1 / y)
    @backward!(y, x / y^2)
end

@func_register Base.:^(x, y) begin
    @backward!(x, y * x^(y - 1))
    @backward!(y, log(x) * x^y)
end

@func_register Base.exp(x) begin
    @backward!(x, exp(x))
end

@func_register Base.log(x) begin
    @backward!(x, 1/x)
end
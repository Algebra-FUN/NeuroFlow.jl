# NeuroFlow

[![Build Status](https://github.com/Algebra-FUN/`NeuroFlow.jl`/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Algebra-FUN/NeuroFlow.jl/actions/workflows/CI.yml?query=branch%3Amain)

`NeuroFlow` is a experimential deep learning framework written in Julia.

It implements an atomic level dynamic computational graph in pure Julia and provides api just like `Pytorch` style. 

## Installation

```julia
import Pkg
Pkg.add("NeuroFlow")
```

## Quick Start

We start with a simple linear example:

```julia
using NeuroFlow
import Distributions: Uniform, Normal, mean
using Plots

# generate some fake data obeyed the linear model
N = 1000
x = rand(Uniform(-10, 10), N) |> sort
ϵ = rand(Normal(0, 1), N)
# parameters setting
a, b = 2.5, 1.5
y = a .* x .+ b .+ ϵ

# declare parameters which needs to be optimize
â,b̂ = Param(1.), Param(1.)
# define the linear model with parameters
lm(x) = â * x + b̂
# use SGD optimizer
optimizer = SGD([â;b̂]; η=1e-2)

loss_records = []

# train for 100 epochs
for epoch in 1:100
    ŷ = lm.(x)
    loss = mean((y.-ŷ).^2)

    # this three steps are just like pytorch
    zero_grad!(optimizer)
    backward!(loss)
    step!(optimizer)

    push!(loss_records, loss.val)
    if epoch % 5 == 0
        println("epoch=$epoch,loss=$(loss.val)")
    end
end
```

> More detail about this example can be seen in [examples/LinearRegression.jl](examples/LinearRegression.jl)

## Examples

More examples can be found in [`examples`](examples/)
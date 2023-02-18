@info "required packages loading..."
using NeuroFlow
import Distributions: Uniform, Normal
import Statistics: mean
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
    zero_grad!(optimizer)
    backward!(loss)
    step!(optimizer)
    push!(loss_records, loss.val)
    if epoch % 5 == 0
        println("epoch=$epoch,loss=$(loss.val)")
    end
end

plot(loss_records, label="loss", title="loss curve") |> display

@info "the loss curve should be shown, please checkout and see it"
println("parameters fit result: â=$(â.val),b̂=$(b̂.val)")

ŷ = @no_grad lm.(x)
display(begin
    scatter(x, y, label="data", title="linear regression example")
    plot!(x, ŷ, label="fit(â=$(â.val),b̂=$(b̂.val))")
end)

@info "the original data and fitted line should be shown, please checkout and see it"

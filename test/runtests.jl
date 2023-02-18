using NeuroFlow
using Test

@testset "NeuroFlow.jl" begin
    @test (@no_grad AUTOGRAD.on == false) 
end

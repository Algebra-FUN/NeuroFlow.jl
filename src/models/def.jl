export Model, Parameter, parameters

abstract type Model end

Parameter = Union{Param,Array{Param}}

function parameters(model::Model)
    c = Vector{Parameter}()
    for fieldsym in fieldnames(typeof(model))
        field = getproperty(model, fieldsym)
        if field isa Parameter
            push!(c, field)
        elseif field isa Model
            push!(c, parameters(field)...)
        end
    end
    return c
end
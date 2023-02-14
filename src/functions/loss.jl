export mseloss,cross_entropy

cross_entropy(pred::Union{Vector,Matrix},gt::Union{Vector,Matrix}) = -mean(sum(gt.*log.(pred),dims=1))
mseloss(pred::Union{Vector,Matrix},gt::Union{Vector,Matrix}) = mean(sum((pred.-gt).^2,dims=1))
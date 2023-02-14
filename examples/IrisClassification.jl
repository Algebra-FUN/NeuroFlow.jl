@info "required packages loading..."
using NeuroFlow
import Random, DataFrames
import MLDatasets:Iris
import Statistics:mean,std
import Plots:plot

@info "dataset loading..."
dataset = Iris(as_df=false)[:]
feats = dataset[:features]
labels = dataset[:targets]

@info "dataset preprocessing..."
X_mean = mean(feats,dims=2)
X_std = std(feats,dims=2);
X_n = replace((feats .- X_mean) ./ X_std, NaN => 0.);
y = float.(labels .== unique(labels))
N = 150;

# train_test_split
Random.seed!(123456)
train_mask = rand(N) .< .8;
X_train = X_n[:,train_mask]
y_train = y[:,train_mask]
X_test = X_n[:,.!train_mask]
y_test = y[:,.!train_mask];

# define model struct
struct DualLayerFC <: Model
    linear1::Linear
    linear2::Linear
    DualLayerFC(a,b,c) = new(Linear(a,b),Linear(b,c))
end

# define model functionality
function (m::DualLayerFC)(x)
    x = m.linear1(x)
    x = leakyrelu.(x)
    x = m.linear2(x)
    return softmax(x)
end

Random.seed!(123456)
model = DualLayerFC(4,16,3)
optimizer = SGD(parameters(model);η=1e-2);

loss_records = []

@info "training..."
for epoch in 1:1000
    pred = model(X_train)
    loss = mseloss(pred,y_train)
    zero_grad!(optimizer)
    backward!(loss)
    step!(optimizer)
    push!(loss_records,loss.val)
    epoch % 50 ==0 && println("epoch=$epoch,loss=$(loss.val)")
end

display(plot(loss_records,label="loss"))
@info "the loss curve should be shown, please checkout and see it"

poss2cls(poss) = getindex.(argmax(poss,dims=1),1)

pred_test = poss2cls(@no_grad model(X_test))
gt_test = poss2cls(y_test)
test_acc = mean(pred_test .== gt_test)
println("final result: the classification accuracy on test dataset of Iris is test_acc=$test_acc")

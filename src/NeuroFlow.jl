module NeuroFlow

import Base
import Statistics:mean

include("core.jl")

include("functions/activation.jl")
include("functions/loss.jl")

include("models/def.jl")
include("models/Linear.jl")

include("optimizers/def.jl")
export step!
include("optimizers/sgd.jl")


end

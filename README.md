# CPOMDPs.jl

This is a package build on [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl) to define constrained MDPs and POMDPs (CMDPs and CPOMDPs). 

The CPOMDPs require augmenting the POMDP definition with a `costs` function that acts similar to the `POMDPs.reward` function but instead returns a vector of single-step costs. The API also defines a `costs_budget` field for the total constraint budgets, and `n_costs` for the total number of cost objectives. Additionally, this repository overrides the POMDPs.jl generator interface to return costs and implements some helper tools to assist with simulation and with wrapping (similar to [POMDPTools](https://github.com/JuliaPOMDP/POMDPs.jl/tree/master/lib/POMDPTools).

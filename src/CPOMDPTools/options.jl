# Options Policy Interface
"""
Base type for a low-level policy (a map from every possible belief, or more abstract policy state, to an optimal or suboptimal action)
"""
abstract type LowLevelPolicy <: POMDPs.Policy end

"""
    terminate(p::LowLevelPolicy, x)
Returns a distribution over whether to terminate option `p` in state/belief `x`.
"""
function terminate end

"""
    POMDPs.action(p::LowLevelPolicy, x)
    POMDPTools.action_info(p::LowLevelPolicy, x)
Returns the low-level action (and possibly info) that LowLevelPolicy `p` takes in state/belief `x`. One of these must be defined for every LowLevelPolicy
"""
POMDPs.action(p::LowLevelPolicy, x) = first(POMDPTools.action_info(p, x))

"""
Base type for a high-level policy (a map from every possible belief, or more abstract policy state to an optimal or suboptimal action by calling a low-level policy)
"""
abstract type OptionsPolicy <: POMDPs.Policy end

"""
    rng(p::OptionsPolicy)
Returns the random number generator associated with the `OptionsPolicy`. Used in deciding to terminate the low-level option. Defaults to `Random.GLOBAL_RNG`     
"""
function rng end
rng(p::OptionsPolicy) = Random.GLOBAL_RNG

"""
    update!(p::OptionPolicy, x, option::LowLevelPolicy, new_option::Bool)
Update the option policy `p` given the start state/belief `x`, the running option, and whether the option is new. 
Optionally return the high-level info.
"""
function update! end

"""
    low_level(p::OptionPolicy)
Must return the `LowLevelPolicy` that the high-level `OptionPolicy` is currently executing or `nothing` in the first state.
"""
function low_level end

"""
    new_option(p::OptionPolicy, x)
Select a new option from state/belief `x`. Returns a `LowLevelPolicy`.
"""
function new_option end 

"""
    POMDPs.action(p::OptionPolicy, x)
    POMDPTools.action_info(p::OptionPolicy, x)
Returns the low-level action (and additionally a NamedTuple of the low-level and high-level info if calling `action_info`). This is done by:
1. Determining the current option thorugh `low_level(p)`.
2. Using `rng(p)` to sample whether to terminate that option from `terminate(option, x)`.
3. If so, choosing a new option through `new_option(p, x)`.
4. Generating the low-level action through `action(option, x)`.
5. Updating the internal state of the OptionPolicy through `update!(p, x, option, new_option)`.
"""
function POMDPTools.action_info(p::OptionPolicy, x)
    new_option = false
    option = low_level(p)
    if option===nothing || rand(rng(p), terminate(option, x))
        new_option = true
        option = new_option(p, x)
    end
    a, ll_info = POMDPTools.action_info(option, x)
    hl_info = update!(p, x, option, new_option)
    return a, (;low=ll_info, high=hl_info)
end
function POMDPs.action(p::OptionPolicy, x)
    new_option = false
    option = low_level(p)
    if option===nothing || rand(rng(p), terminate(option, x))
        new_option = true
        option = new_option(p, x)
    end
    update!(p, x, option, new_option)
    return action(option, x)
end

# Random Options Policy - given a set of options, choose the next one randomly after termination

mutable struct RandomOptionsPolicy <: OptionsPolicy
    options::Vector{LowLevelPolicy}
    running::Union{Nothing,LowLevelPolicy}
    rng::Random.AbstractRNG
    step_counter::Int
end
RandomOptionsPolicy(os::Vector{LowLevelPolicy};rng=Random.GLOBAL_RNG) = RandomOptionsPolicy(os, nothing, rng, 0)

rng(p::RandomOptionsPolicy) = p.rng
function update!(p::RandomOptionsPolicy, x, option::LowLevelPolicy, new_option::Bool) 
    p.running = option
    p.step_counter = new_option ? 0 : p.step_counter + 1
    return (;new_option = new_option, step_counter=p.step_counter)
end
low_level(p::RandomOptionsPolicy) = p.running
new_option(p::RandomOptionsPolicy, x) = rand(p.rng, p.options)

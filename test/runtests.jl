println("Testing...")
using Test
using POMDPs
using CPOMDPs
import CPOMDPs: costs, costs_budget, n_costs
using QuickPOMDPs
using POMDPTools
using Random

M = QuickPOMDP(
    states = ["left", "right"],
    actions = ["left", "right", "listen"],
    observations = ["left", "right"],
    initialstate = Uniform(["left", "right"]),
    discount = 0.95,

    transition = function (s, a)
        if a == "listen"
            return Deterministic(s) # tiger stays behind the same door
        else # a door is opened
            return Uniform(["left", "right"]) # reset
        end
    end,

    observation = function (s, a, sp)
        if a == "listen"
            if sp == "left"
                return SparseCat(["left", "right"], [0.85, 0.15]) # sparse categorical distribution
            else
                return SparseCat(["right", "left"], [0.85, 0.15])
            end
        else
            return Uniform(["left", "right"])
        end
    end,

    reward = function (s, a)
        if a == "listen"
            return -1.0
        elseif s == a # the tiger was found
            return -100.0
        else # the tiger was escaped
            return 10.0
        end
    end
)

# define simple wrapper class 
struct ConstrainedPOMDP{P,S,A,O} <: ConstrainPOMDPWrapper{P,S,A,O}
    pomdp::P
    budget::Vector{Float64}
    cost_func::Function # (s,a) -> c
end
ConstrainedPOMDP(pomdp::P, budget::Vector{Float64}, cost_func::Function
    ) where {P<:POMDP} = ConstrainedPOMDP{P, statetype(pomdp), actiontype(pomdp), obstype(pomdp)}(pomdp, budget, cost_func)
costs(p::ConstrainedPOMDP, s, a) = p.cost_func(s,a)
costs_limit(p::ConstrainedPOMDP) = p.budget
n_costs(p::ConstrainedPOMDP) = length(p.budget)

# make problem
cf(s,a) = (s==a) ? 0.1 : 0.0
CM = ConstrainedPOMDP(M, [1.], cf)

@testset "simple CPOMDP with wrapper" begin    
    @test n_costs(CM) == 1
    @test typeof(CM) <: CPOMDP
    @test typeof(CM) <: POMDP
    @test isapprox(costs_limit(CM)[1], 1.)
    @test isapprox(costs(CM, "left", "left"), 0.1)
    @test isapprox(costs(CM, "left", "right"), 0.0)      
    @test isapprox(reward(CM,"left", "left"),reward(M, "left", "left"))
    sp, o, r, c = @gen(:sp, :o, :r, :c)(CM, "left", "listen", Random.GLOBAL_RNG)
    @test sp in states(CM)
    @test o in observations(CM)
    @test isapprox(r, -1.0)
    @test length(c) == 1
    @test isapprox(c[1],0.)
end
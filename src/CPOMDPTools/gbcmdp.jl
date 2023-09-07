"""
    GenerativeBeliefCMDP(pomdp, updater)
Create a generative model of the belief CMDP corresponding to CPOMDP `pomdp` with belief updates performed by `updater`.
"""
struct GenerativeBeliefCMDP{P<:CPOMDP, U<:Updater, B, A} <: CMDP{B, A}
    cpomdp::P
    updater::U
    exact_rewards::Bool
end

function GenerativeBeliefCMDP(cpomdp::P, up::U; exact_rewards::Bool=false) where {P<:CPOMDP, U<:Updater}
    # XXX hack to determine belief type
    b0 = initialize_belief(up, initialstate(cpomdp))
    GenerativeBeliefCMDP{P, U, typeof(b0), actiontype(cpomdp)}(cpomdp, up, exact_rewards)
end

POMDPs.actions(bmdp::GenerativeBeliefCMDP{P,U,B,A}, b::B) where {P,U,B,A} = actions(bmdp.cpomdp, b)
POMDPs.actions(bmdp::GenerativeBeliefCMDP) = actions(bmdp.cpomdp)
POMDPs.isterminal(bmdp::GenerativeBeliefCMDP, b) = all(isterminal(bmdp.cpomdp, s) for s in support(b))
POMDPs.discount(bmdp::GenerativeBeliefCMDP) = discount(bmdp.cpomdp)
n_costs(bmdp::GenerativeBeliefCMDP) = n_costs(bmdp.cpomdp)
costs_limit(bmdp::GenerativeBeliefCMDP) = costs_limit(bmdp.cpomdp)

function POMDPs.reward(bmdp::GenerativeBeliefCMDP, b, a)
    r = 0.
    w_sum = 0.
    for (s,w) in weighted_particles(b)
        r += w * reward(bmdp.cpomdp, s, a)
        w_sum += w
    end
    return r / w_sum
end
function POMDPs.reward(bmdp::GenerativeBeliefCMDP, b, a, bp, o)
    r = 0.
    w_sum = 0.
    for (s,w) in weighted_particles(b)
        for (sp,wp) in weighted_particles(bp)
            r += w * wp * reward(bmdp.cpomdp, s, a, sp, o)
            w_sum += w * wp
        end
    end
    return r / w_sum
end
function costs(bmdp::GenerativeBeliefCMDP, b::Union{WeightedParticleBelief, ParticleCollection}, a)
    c = zeros(Float64, n_costs(bmdp.cpomdp))
    w_sum = 0.
    for (s,w) in weighted_particles(b)
        c += w * costs(bmdp.cpomdp, s, a)
        w_sum += w
    end
    return c / w_sum
end
function costs(bmdp::GenerativeBeliefCMDP, b::Union{WeightedParticleBelief, ParticleCollection}, 
    a, bp::Union{WeightedParticleBelief, ParticleCollection}, o)
    c = zeros(Float64, n_costs(bmdp.cpomdp))
    w_sum = 0.
    for (s,w) in weighted_particles(b)
        for (sp,wp) in weighted_particles(bp)
            c += w * wp * costs(bmdp.cpomdp, s, a, sp, o)
            w_sum += w * wp
        end
    end
    return c / w_sum
end
function POMDPs.gen(bmdp::GenerativeBeliefCMDP, b, a, rng::AbstractRNG)
    s = rand(rng, b)
    if isterminal(bmdp.cpomdp, s)
        return gbmdp_handle_terminal(bmdp, b, s, a, rng::AbstractRNG)
    end
    sp, o, r, c = @gen(:sp, :o, :r, :c)(bmdp.cpomdp, s, a, rng) # maybe this should have been generate_or?
    bp = update(bmdp.updater, b, a, o)
    if bmdp.exact_rewards
        r = reward(bmdp, b, a, bp, o)
        c = costs(bmdp, b, a, bp, o)
    end
    return (sp=bp, r=r, c=c)
end

# override this if you want to handle it in a special way
function gbmdp_handle_terminal(bmdp::GenerativeBeliefCMDP, b, s, a, rng)
    @warn("""
         Sampled a terminal state for a GenerativeBeliefCMDP transition - not sure how to proceed, but will try.
         See $(@__FILE__) and implement a new method of CPOMDPs.gbmdp_handle_terminal if you want special behavior in this case.
         """, maxlog=1)
    o =  @gen(:o)(bmdp.cpomdp, s, a, rng)
    bp = update(bmdp.updater, b, a, o)
    return (sp=bp, r=0.0, c=zeros(Float64, n_costs(bmdp.cpomdp)))
end

function POMDPs.initialstate(bmdp::GenerativeBeliefCMDP)
    return Deterministic(initialize_belief(bmdp.updater, initialstate(bmdp.cpomdp)))
end
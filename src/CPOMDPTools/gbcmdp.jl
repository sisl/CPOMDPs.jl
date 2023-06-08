"""
    GenerativeBeliefCMDP(pomdp, updater)
Create a generative model of the belief CMDP corresponding to CPOMDP `pomdp` with belief updates performed by `updater`.
"""
struct GenerativeBeliefCMDP{P<:CPOMDP, U<:Updater, B, A} <: CMDP{B, A}
    cpomdp::P
    updater::U
end

function GenerativeBeliefCMDP(cpomdp::P, up::U) where {P<:CPOMDP, U<:Updater}
    # XXX hack to determine belief type
    b0 = initialize_belief(up, initialstate(cpomdp))
    GenerativeBeliefCMDP{P, U, typeof(b0), actiontype(cpomdp)}(cpomdp, up)
end

POMDPs.actions(bmdp::GenerativeBeliefCMDP{P,U,B,A}, b::B) where {P,U,B,A} = actions(bmdp.cpomdp, b)
POMDPs.actions(bmdp::GenerativeBeliefCMDP) = actions(bmdp.cpomdp)
POMDPs.isterminal(bmdp::GenerativeBeliefCMDP, b) = all(isterminal(bmdp.cpomdp, s) for s in support(b))
POMDPs.discount(bmdp::GenerativeBeliefCMDP) = discount(bmdp.cpomdp)
n_costs(bmdp::GenerativeBeliefCMDP) = n_costs(bmdp.cpomdp)
costs_limit(bmdp::GenerativeBeliefCMDP) = costs_limit(bmdp.cpomdp)

function POMDPs.reward(bmdp::GenerativeBeliefCMDP, b::Union{WeightedParticleBelief, ParticleCollection}, a)
    r = 0.
    w_sum = 0.
    for (s,w) in weighted_particles(b)
        r += w * reward(bmdp.cpomdp, s, a)
        w_sum += w
    end
    if !(w_sum ≈ 1.)
        @warn("particle filter belief weights not summing to 1.")
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
    if !(w_sum ≈ 1.)
        @warn("particle filter belief weights not summing to 1.")
    end
    return c / w_sum
end

function POMDPs.gen(bmdp::GenerativeBeliefCMDP, b, a, rng::AbstractRNG)
    s = rand(rng, b)
    if isterminal(bmdp.cpomdp, s)
        bp = gbmdp_handle_terminal(bmdp.cpomdp, bmdp.updater, b, s, a, rng::AbstractRNG)::typeof(b)
        return (sp=bp, r=0.0, c=zeros(Float64, n_costs(bmdp.cpomdp)))
    end
    sp, o, r, c = @gen(:sp, :o, :r, :c)(bmdp.cpomdp, s, a, rng) # maybe this should have been generate_or?
    # why do this instead of generate r(b,a) and c(b,a)?? speed?? e.g.
    # r = reward(bmdp, b, a)
    # c = costs(bmdp, b, a)
    bp = update(bmdp.updater, b, a, o)
    return (sp=bp, r=r, c=c)
end

# override this if you want to handle it in a special way
function gbmdp_handle_terminal(pomdp::CPOMDP, updater::Updater, b, s, a, rng)
    @warn("""
         Sampled a terminal state for a GenerativeBeliefCMDP transition - not sure how to proceed, but will try.
         See $(@__FILE__) and implement a new method of CPOMDPs.gbmdp_handle_terminal if you want special behavior in this case.
         """, maxlog=1)
    sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a, rng)
    bp = update(updater, b, a, o)
    return bp
end

function POMDPs.initialstate(bmdp::GenerativeBeliefCMDP)
    return Deterministic(initialize_belief(bmdp.updater, initialstate(bmdp.cpomdp)))
end
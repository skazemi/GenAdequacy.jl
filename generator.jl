include("randomvar.jl")

module GeneratorModule
export Generator, available_capacity, power_trace, rv

using ..RandomVarModule
using StatsKit

#= 
struct that represents a single generator or set of identical generators as a linear Markov chain
 =#
mutable struct Generator
    unit_capacity::Float64
    unit_count::Int
    unit_availability::Float64
    unit_MTBF::Float64
    Generator(unit_capacity, unit_count, unit_availability, unit_mtbf) = new(unit_capacity, unit_count, unit_availability, unit_mtbf)
    up_count::Int
    unit_fail_rate::Float64
    unit_repair_rate::Float64
end

function Generator(; unit_capacity, unit_availability, unit_mtbf, unit_count=1)

    @assert (unit_capacity > 0) "Generator capacities must be larger than zero."
    @assert (0 <= unit_availability <= 1) "Generator availability must be between zero and one."
    @assert (unit_mtbf > 0) "Generator MTBF must be positive."
    @assert (unit_count >= 1) "Generating unit count must be positive."

    gen = Generator(unit_capacity, unit_count, unit_availability, unit_mtbf)

    # Compute the number of units in the 'up' state. This is used for sequential series generation.
    gen.up_count = rand(Binomial(unit_count, unit_availability))
    gen.unit_fail_rate = 1 / (unit_availability * unit_mtbf)
    gen.unit_repair_rate = 1 / ((1 - unit_availability) * unit_mtbf)
    return gen
end

#=
Return sum of capacities of available units.

:return: sum of capacities of available units.
=#
function available_capacity(gen::Generator)
    return gen.up_count * gen.unit_capacity
end

#=
Return time series of available capacities of the aggregate units.

:param num_steps: Number of time steps in the series
:param dt: Size of each time step
:param random_start: Option to create an independent sequence (true:default) or start from self.up_count (false)
:return: array of length num_steps and dtype float
=#
function power_trace(self::Generator, num_steps=1, dt=1.0, random_start=true)

    initial_up_count = \
    if random_start
        rand(Binomial(self.unit_count, self.unit_availability))
    else
        self.up_count
    end

    repair_prob = self.unit_repair_rate * dt
    fail_prob = self.unit_fail_rate * dt
    @assert repair_prob < 1.0
    @assert fail_prob < 1.0

    repair_factor = 1.0 / log(1.0 - repair_prob)
    fail_factor = 1.0 / log(1.0 - fail_prob)

    sum_factor = repair_factor + fail_factor

    adjust_trace = zeros(Int, num_steps + 1)
    adjust_trace[1] = initial_up_count

    for i in 1:self.unit_count
        if i <= initial_up_count
            state = 1
            change_factor = fail_factor
        else
            state = -1
            change_factor = repair_factor
        end
        t = 1
        while true # LEFTHERE
            t += ceil(Int, change_factor * log(1.0 - random.random()))
            if t - 1 > num_steps
                break
            end
            adjust_trace[t] -= state
            state = -state
            change_factor = sum_factor - change_factor
        end
    end
    trace = cumsum(adjust_trace)
    self.up_count = trace[end]
    return self.unit_capacity * Float64.(trace[1:end-1])# FIXME float type

end

#=
Returns a random variable representing the available capacity of the generator set.

:param resolution: desired resolution of the RandomVariable object
:return: RandomVariable object corresponding to the generator(s) output

This uses a simple iterative definition for multi-unit generators; this could easily be improved
(e.g. using a binomial distribution), but is rarely on the critical path.

If resolution is not a divisor of unit_capacity, the capacity is divided proportionally over
neighbouring bins, depending on resolution
=#
function rv(self::Generator, resolution=1)
    # Note that ceil_idx and floor_idx are either identical (if resolution matches) or one apart
    ceil_idx = ceil(Int, self.unit_capacity / resolution)
    floor_idx = floor(Int, self.unit_capacity / resolution)
    probability_array = zeros(Int(ceil_idx + 1))# FIXME Int is redundant

    probability_array[1] = 1.0 - self.unit_availability

    # allocate probability over two identical or neighbouring bins
    distance_to_end = ceil_idx - self.unit_capacity / resolution
    probability_array[floor_idx+1] += distance_to_end * self.unit_availability
    probability_array[ceil_idx+1] += (1 - distance_to_end) * self.unit_availability

    # log_array(probability_array)#debug
    # define a single unit rv
    # unit_rv = RandomVariable(resolution=resolution, base=0, probability_array=probability_array)
    unit_rv = RandomVariable(resolution, 0, probability_array)
    # log_array(unit_rv.cdf_array)#debug
    # iterative convolution to determine aggregate rv
    # result_rv = RandomVariable(resolution=resolution)
    result_rv = RandomVariable(resolution)
    # log_array(result_rv.probability_array)#debug
    # log_array(result_rv.cdf_array)#debug
    # println("unit_count: ", self.unit_count)#debug
    for i in 1:self.unit_count
        result_rv += unit_rv
    end
    # log_array(result_rv.probability_array)#debug
    # log_array(result_rv.cdf_array)#debug
    return result_rv
end

end

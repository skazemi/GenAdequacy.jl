#=
Discretised random variable functions
=#
module RandomVarModule

# export +,-,*
using LinearAlgebra: length
export RandomVariable, load_data, cdf_value, cdf_value_interpolate, quantile, quantile_interpolate, truncate!, log_array

using DSP
using Random
using StatsKit
using LinearAlgebra

import Base.:+
import Base.:-
import Base.:*
# import Base.:+=
# import Base.:-=
# import Base.:*=

function log_array(arr::Array{<:Number})
    outfile = "log-julia.txt"
    f = open(outfile, "a")

    println(f, length(arr))
    for i in arr
        println(f, i)
    end
    close(f)
end

mutable struct RandomVariable
    base::Float64
    step::Int
    probability_array::Array{Float64}
    cdf_array::Array{Float64}

    value_pool::Array{Float64}
    value_pool_left::Int
    value_pool_random_state::AbstractRNG
end

function RandomVariable(; resolution=1, base=0, probability_array=[1.0])
    randvar = RandomVariable(base, resolution, probability_array, cumsum(probability_array), [], 0, Random.GLOBAL_RNG)#FIXME: value_pool, value_pool_random_state
    return randvar
end

function RandomVariable(resolution=1, base=0, probability_array=[1.0])
    randvar = RandomVariable(base, resolution, probability_array, cumsum(probability_array), [], 0, Random.GLOBAL_RNG)#FIXME: value_pool, value_pool_random_state
    return randvar
end

#=
Initialise the random variable from an empirical data set.

This implementation creates a PMF array with bins that are aligned with zero.

:param data_array: empirical data set (1D array)
:return: none
=#
function load_data(self::RandomVariable, data_array)
    # determine lower and upper bounds for the bins, to compute array size and base
    # TODO: generalise this procedure to distribute probability mass over nearby points (interpolation)
    left_bound = (floor(Int, minimum(data_array) / self.step + 0.5) - 0.5) * self.step
    right_bound = (ceil(Int, maximum(data_array) / self.step - 0.5) + 0.5) * self.step
    num_bins = round(Int, (right_bound - left_bound) / self.step)
    self.base = left_bound + 0.5 * self.step
    # allocate that experimental data across the bins
    # prob_array, bins = np.histogram(
    #     data_array,
    #     bins=np.linspace(start=left_bound, stop=right_bound, num=(num_bins + 1)),
    #     density=True
    # )
    bins = LinRange(left_bound, right_bound + 1e-10, (num_bins + 1))# 1e-10 for right_bound inclision
    hist = fit(Histogram,
        data_array,
        bins
    )
    nhist = normalize(hist, mode=:pdf)
    prob_array = nhist.weights
    # enforce normalisation as probability mass function, and compute the CDF
    self.probability_array = prob_array * self.step
    self.cdf_array = cumsum(self.probability_array)
end

#=
Overloads the '+' operator.
Independence is assumed between the variables being added.
=#
function Base.:+(self::RandomVariable, other::RandomVariable)
    if self.step != other.step
        # raise NotImplementedError("Cannot add RandomVariable objects with different step sizes.")
        throw(DimensionMismatch("Cannot add RandomVariable objects with different step sizes."))
    end

    return RandomVariable(
        self.step,
        self.base + other.base,
        conv(self.probability_array, other.probability_array)
    )
end

#=
Overloads the '+' operator.
Independence is assumed between the variables being added.
=#
function Base.:+(self::RandomVariable, other::Float64)
    return RandomVariable(
        self.step,
        self.base + other,
        self.probability_array
    )
end

#=
Overloads the '-' operator.
Assumes independence between variables.
=#
function Base.:-(self::RandomVariable, other::RandomVariable)
    if self.step != other.step
        # raise NotImplementedError("Cannot subtract RandomVariable objects with different step sizes.")
        throw(DimensionMismatch("Cannot subtract RandomVariable objects with different step sizes."))
    end

    return RandomVariable(
        self.step,
        self.base - other.base - self.step * (length(other.probability_array) - 1),
        conv(self.probability_array, reverse(other.probability_array))
    )
end
#=
Overloads the '-' operator.
Assumes independence between variables.
=#
function Base.:-(self::RandomVariable, other::Float64)
    return RandomVariable(
        self.step,
        self.base - other,
        self.probability_array
    )
end

#=
Overloads the '*' operator.
Only valid for integer values of 'multiplier'.
=#
function Base.:*(self::RandomVariable, multiplier::Integer)
    new_base::Float64
    new_array::Array{Float64}
    if multiplier == 0
        return 0
    elseif multiplier > 0
        new_array = zeros(multiplier * (length(self.probability_array) - 1) + 1)
        # new_array[::multiplier] = self.probability_array
        new_array[1:multiplier:end] = self.probability_array
        new_base = self.base * multiplier
    else
        new_array = zeros(-multiplier * (length(self.probability_array) - 1) + 1)
        # new_array[::multiplier] = self.probability_array
        new_array[end:multiplier:1] = self.probability_array
        new_base = multiplier * (self.base + self.step * (length(self.probability_array) - 1))
    end
    return RandomVariable(self.step, new_base, new_array)#FIXME: order of params
end
# #=
# Overloads the '*' operator.
# Only valid for integer values of 'multiplier'.
# =#
# function *(self::RandomVariable,other::Float64)
# end


#=
Generate random values according to the object's distribution

:param number_of_items: number of random values to return
:param random_state: optional AbstractRNG object to use for the random stream
:param dither: optional dithering to generate a piecewise-linear PDF (default: true)
:return: ndarray of random values
=#
function random_value(self::RandomVariable, number_of_items::Int=1, random_state::AbstractRNG=Random.GLOBAL_RNG, dither=true)
    res = self.step

    res2 = sample(random_state, x_array(self), ProbabilityWeights(self.probability_array), number_of_items)
    if dither
        res2 += rand(random_state, TriangularDist(-res, res, 0), number_of_items)
    end
    return res2
end

#=
Efficient sampler implementation that returns a single value from a pool each time it is called

:param pool_size: pool size to use when new samples are required
:param random_state: optional AbstractRNG object to use for number generation
:param dither: optional dithering to generate a piecewise-linear PDF (default: true)
:param reset: force repopulation of the pool (default: false)
:return: random value

NOTE: the implementation is unsafe, in the sense that a manual reset is required to update the pool values
each time the system parameters change.
=#
function random_value_from_pool(self::RandomVariable, pool_size::Int, random_state::AbstractRNG=Random.GLOBAL_RNG, dither=true, reset=false)
    if reset || (self.value_pool_left <= 0) || random_state != self.value_pool_random_state
        self.value_pool = self.random_value(number_of_items=pool_size, random_state=random_state, dither=dither)
        self.value_pool_left = pool_size
        self.value_pool_random_state = random_state
    end
    self.value_pool_left = self.value_pool_left - 1
    # return self.value_pool[self.value_pool_left]
    return self.value_pool[self.value_pool_left+1]
end

#=Returns the minimum and maximum values for which probability masses are stored.=#
function min_max(self::RandomVariable)
    return self.base, self.base + self.step * (length(self.probability_array) - 1)
end

#=Returns an array of x-values for which probability masses are stored.=#
function x_array(self::RandomVariable)
    return LinRange(min_max(self)..., length(self.probability_array))
end

#=Returns the expectation cdf_value of the probability distribution.=#
function mean(self::RandomVariable)
    return dot(x_array(self), self.probability_array)
end

#=Truncates the probability mass function.=#
function truncate!(self::RandomVariable, tolerance=1e-10)
    sum_array = cumsum(self.probability_array)
    boundary_indices = searchsortedfirst.(Ref(sum_array), [tolerance, 1.0 - tolerance])

    log_array(sum_array)
    new_array = copy(self.probability_array[boundary_indices[1]:boundary_indices[2]])#FIXME +1, its ok(inclusive)
    if boundary_indices[1] > 1
        new_array[1] += sum_array[boundary_indices[1]-1]
    end
    # if boundary_indices[1] < len(new_array) - 1
    #     new_array[-1] += 1.0 - sum_array[boundary_indices[1]+2]
    # end
    if boundary_indices[2] < length(new_array)
        new_array[end] += 1.0 - sum_array[boundary_indices[2]+2]
    end

    self.base += (boundary_indices[1] - 1) * self.step #FIXME: FIXED
    self.probability_array = new_array         # NOTE: copy can be avoided, but this is clearer.
    self.cdf_array = cumsum(self.probability_array)
end

#=
Returns the value associated with the quantile q in [0,1]

:param q: single cdf_value or list of quantiles to return
:param side: rounding down ('left') or up ('right').
:return: returns the type of q
=#
function quantile(self::RandomVariable, q::Array{Float64}, side="left")
    if side == "left"
        indices = searchsortedfirst.(Ref(self.cdf_array), q)
    else
        indices = searchsortedlast.(Ref(self.cdf_array), q)#FIXME check equivalence to numpy
    end
    return self.base + indices * self.step
end

#=
Returns the value associated with the quantile q in [0,1]

:param q: single cdf_value or list of quantiles to return
:return: returns the type of q
=#
function quantile_interpolate(self::RandomVariable, q::Float64)
    # TODO: create and use interpolated function object.
    # TODO: use numpy.digitize() to perform operation on array indices in one go
    index = searchsortedfirst(self.cdf_array, q)  # default 'left' side
    if index != 1
        delta = (q - self.cdf_array[index-1]) / (self.cdf_array[index] - self.cdf_array[index-1])
    else
        delta = q / self.cdf_array[1]
    end
    return self.base + self.step * ((index - 1) - 1 + delta + 0.5)#FIXME adjust index
end
# function quantile_interpolate(self::RandomVariable, q::Array{Float64})
#     # TODO: create and use interpolated function object.
#     # TODO: use numpy.digitize() to perform operation on array indices in one go
#     # index = searchsorted(self.cdf_array, q)  # default 'left' side
#     index = searchsortedfirst.(Ref(self.cdf_array), q)  # default 'left' side
#     if index != 1
#         delta = (q - self.cdf_array[index - 1]) / (self.cdf_array[index] - self.cdf_array[index - 1])
#     else
#         delta = q / self.cdf_array[1]
#     end
#     return self.base + self.step * (index - 1 + delta + 0.5)#FIXME adjust index
# end


#=
Returns the CDF value [0,1] for a given x

:param x:
:return: CDF value [0,1]
=#
function cdf_value(self::RandomVariable, x::Float64)
    # TODO: vectorise this in x
    if x < self.base
        return 0.0
    elseif x > self.base + self.step * (length(self.cdf_array) - 1)
        return 1.0
    else
        return self.cdf_array[floor(Int, (x - self.base) / self.step)]
    end
end

#=
Returns the CDF value [0,1] for a given x

:param x:
:return: CDF value [0,1]
=#
function cdf_value_interpolate(self::RandomVariable, x::Float64)
    # TODO: vectorise this in x
    x_unit = (x - self.base) / self.step + 0.5
    idx = floor(Int, x_unit)
    println("x:$(x) base:$(self.base) step:$(self.step) x_unit:$(x_unit) idx:$(idx) len:$(length(self.cdf_array))")#debuginfo
    if idx < 0
        return 0.0
    end
    if idx >= length(self.cdf_array) #FIXME
        return 1.0
    else
        return self.cdf_array[idx+1] - (idx + 1 - x_unit) * self.probability_array[idx+1] #FIXME FIXED
    end
end

function change_resolution(self::RandomVariable, new_resolution::Int)
    #=Returns a RandomVariable with new resolution, and distributes the PMF across its bins

    :param new_resolution: new resolution
    =#

    # identify lower and upper bounds, and number of bins
    new_base = floor(Int, self.base / new_resolution) * new_resolution
    current_pmf_right = self.base + (length(self.probability_array) - 1) * self.step
    new_pmf_right = (ceil(Int, current_pmf_right / new_resolution)) * new_resolution
    new_num_bins = 1 + round(Int, (new_pmf_right - new_base) / new_resolution)

    # construct probability array
    new_probability_array = zeros(new_num_bins)
    for (idx, x_value) in enumerate(x_array(self))
        x_unit = (x_value - new_base) / new_resolution
        left_idx = floor(Int, x_unit)
        right_idx = ceil(Int, x_unit)
        distance_from_end = right_idx - x_unit
        new_probability_array[left_idx] += distance_from_end * self.probability_array[idx]
        new_probability_array[right_idx] += (1 - distance_from_end) * self.probability_array[idx]
    end
    return RandomVariable(
        resolution=new_resolution,
        base=new_base,
        probability_array=new_probability_array
    )
end

end
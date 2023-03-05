#=
struct and supporting function to instantiate a single node generation adequacy problem.
=#

include("generator.jl")

module SystemModule

export SingleNodeSystem, lolp, lole, epns, compute_load_offset

using ..RandomVarModule
using ..GeneratorModule

using Formatting: printfmt


#=struct representing generic single bus system with two-state generators, and profiles for wind power and load.=#
mutable struct SingleNodeSystem
    #=Initialises the System object with a supplied list of generators and a load profile.

    :param gen_list: list of Generator objects
    :param load_profile: array with load profile
    :param wind_profile: array with wind profile. Assumed independent from load.
    :param resolution: resolution for random variables
    :param load_offset: structural load offset

    It is assumed that a system is instantiated only once and is not modified afterwards.
    =#
    resolution::Int
    gen_list::Array{Generator}
    load_profile::Array{Float64}# FIXME:Matrix
    load_offset::Float64

    # NOTE: wind_profile may be None
    wind_profile::Array{Float64}
    SingleNodeSystem(gen_list::Array{Generator}, load_profile::Array{Float64}, resolution::Int) = new(resolution, gen_list, load_profile)
    SingleNodeSystem(; gen_list::Array{Generator}, load_profile::Array{Float64}, resolution::Int, load_offset::Float64) = new(resolution, gen_list, load_profile, load_offset)
    SingleNodeSystem() = new()
    # SingleNodeSystem(args...) = new(args...)
    _gen_rv::RandomVariable
    _ld_rv::RandomVariable
    _w_rv::RandomVariable
    _m_rv::RandomVariable
end

function Base.getproperty(self::SingleNodeSystem, sym::Symbol)
    if sym === :generation_rv
        #=
        Return a random variable representing the available generation of 'areas' RTS areas.
        :return: RandomVariable object that represents the generating capacity
        =#
        if !isdefined(self, :_gen_rv)
            self._gen_rv = RandomVariable(self.resolution)
            for gen in self.gen_list
                self._gen_rv += rv(gen, self.resolution)
            end
        end
        return self._gen_rv
    elseif sym === :load_rv
        #=
        Return a random variable representing the load duration curve of 'areas' RTS areas.
        :return: RandomVariable object that represents the system load
        =#
        if !isdefined(self, :_ld_rv)
            self._ld_rv = RandomVariable(self.resolution)
            load_data(self._ld_rv, self.load_profile)
            if isdefined(self, :load_offset)
                self._ld_rv += self.load_offset
            end
        end
        return self._ld_rv
    elseif sym === :wind_rv
        #=
        Return a random variable representing the gen duration curve of wind power.
        :return: RandomVariable object that represents the system load
        =#
        if !isdefined(self, :wind_profile)
            return nothing
        end
        if !isdefined(self, :_w_rv)
            self._w_rv = RandomVariable(self.resolution)
            load_data(self._w_rv, self.wind_profile)
        end
        return self._w_rv
    elseif sym === :margin_rv
        #=
        Return a random variable representing the generation margin.
        :return: RandomVariable object that represents the system margin
        =#
        if !isdefined(self, :_m_rv)
            if !isdefined(self, :wind_profile)
                self._m_rv = self.generation_rv - self.load_rv
            else
                self._m_rv = self.generation_rv + self.wind_rv - self.load_rv
            end
        end
        return self._m_rv
    else # fallback to getfield
        return getfield(self, sym)
    end
end

function Base.propertynames(obj::SingleNodeSystem)
    basic_properties = invoke(propertynames, Tuple{Any}, obj)
    # dcm_keys = keys(dcm.meta)
    pnames = Symbol[basic_properties...]
    # for (k, v) in DICOM.fieldname_dict
    #     if v in dcm_keys
    #         push!(pnames, k)
    #     end
    # end
    push!(pnames, :generation_rv, :load_rv, :wind_rv, :margin_rv)
    return tuple(pnames...)
end


#=
Compute system LOLP (loss of load probability).

:param load_offset: Additional load added to the system prior to computing LOLP
:param interpolation: Specify whether linear interpolation of probability mass is used.
:return: LOLP value in range [0,1]
=#
function lolp(self::SingleNodeSystem; load_offset=0.0, interpolation=true)
    if interpolation
        return cdf_value_interpolate(self.margin_rv, load_offset)
    else
        return cdf_value(self.margin_rv, load_offset - 1E-10)
    end
end

#=
`lole` for `SingleNodeSystem`
Compute system LOLE (loss of load expectation).

:param load_offset: Additional load added to the system prior to computing LOLE
:return: LOLE value

The LOLE is computed by summing loss-of-load probabilities for each load level in self.load_profile. If an
hourly annual load profile is used, this results in LOLE (hr/year); when a profile of daily peak loads is used,
it returns LOLE (days/year).

Note that LOLE is approx. LOLP * len(load_profile), but there is a small numerical difference due to the fact
that LOLP is based on the margin distribution, which uses load levels that have been discretised onto a grid.
As a result, the lole() result should be used for comparison with LOLE values in the literature.
=#
function lole(self::SingleNodeSystem, load_offset=0)
    total_load_offset = (
        if !isdefined(self, :load_offset)
            0
        else
            self.load_offset
        end
    ) + load_offset
    if self.wind_rv == nothing
        gw_rv = self.generation_rv - total_load_offset
    else
        gw_rv = self.generation_rv + self.wind_rv - total_load_offset
    end
    return sum([cdf_value(gw_rv, (load_level - 1E-10)) for load_level in self.load_profile])
end


#=
Compute EPNS (expected power not supplied).

:param load_offset: Additional load added to the system prior to computing EPNS
:param interpolation: Specify whether linear interpolation of probability mass is used.
:return: EPNS value
=#
function epns(self::SingleNodeSystem; load_offset=0, interpolation=true)
    # TODO: implement interpolation
    mrv = self.margin_rv
    # determine index corresponding to target offset (on left)
    # ceil_idx = max(0, min(ceil(Int, (load_offset - mrv.base) / mrv.step), length(mrv.cdf_array)))
    ceil_idx = max(1, min(ceil(Int, (load_offset - mrv.base) / mrv.step), length(mrv.cdf_array)))# FIXME

    # return mrv.step * mrv.cdf_array[0:ceil_idx].sum() + \
    return mrv.step * sum(mrv.cdf_array[1:ceil_idx]) + # FIXME
           mrv.cdf_array[ceil_idx+1] * (load_offset - mrv.base - ceil_idx * mrv.step)
end

#=
Determine additional load that must be added to the system to achieve the target LOLP.

:param lolp_target: LOLP target in [0,1]
:return: additional load level

The load addition is implemented as a constant addition (no profile). It is effectively the inverse of
lolp_target = lolp(load, interpolation=true). Interpolation is used to ensure a unique solution.
=#
function compute_load_offset(self::SingleNodeSystem; lolp_target)
    return quantile_interpolate(self.margin_rv, lolp_target)
end

#=
Generate random samples from the margin distribution.

:param number_of_values: number of samples
:return: single value or array of values
=#
function margin_sample(self::SingleNodeSystem, number_of_values=1)
    mrv = self.margin_rv
    return random_value(mrv, number_of_items=number_of_values)
end

#    @jit
#=
Sample a random generation trace for all generators in the system.

:param num_steps: Number of steps
:param dt: Time step
:param random_start: Whether to create an independent sequence (true; default) or start from previous state (false)
:return: Sequence of available capacity values
=#
function generation_trace(self::SingleNodeSystem, num_steps=0, dt=1.0, random_start=true)

    # infer length of trace
    if num_steps == 0
        generate_steps = length(self.load_profile)
        if isdefined(self, :wind_profile)
            generate_steps = max(generate_steps, length(self.wind_profile))
        end
    else
        generate_steps = num_steps
    end
    trace = zeros(generate_steps)
    for gen in self.gen_list
        trace += power_trace(gen, num_steps=generate_steps, dt=dt, random_start=random_start)
    end
    return trace
end

# TODO: pass on most arguments to make less fragile
#=Initialises the generators and load curve of the RTS object and calls the System initialiser.

:param load_profile: optional ndarray load profile of arbitrary length
:param peak_load: if specified, peak load of the system, in same units as generator capacities
:param wind_profile: optinoal ndarray wind profile of arbitrary length. Assumed independent from load.
:param resolution: resolution for the random variables
:param gen_availability: availability of each generator
:param MTBF: generator MTBF (not required for steady state analysis)
:param LOLH: system risk standard (expected loss-of-load hours)
:param base_unit: size of smallest unit in the system
:param max_unit: upper bound for maximum unit size. Actual unit sizes are generated in powers of two from base_unit
:param gen_set: array of generator sizes
:param num_sets: number of generator sets to use ('None' for automatic optimisation)
:param apply_load_offset: adjust load profile to hit LOLH target exactly
=#
function autogen_system(;
    load_profile=nothing, peak_load=NaN, wind_profile=nothing, resolution=10,
    gen_availability=0.90, MTBF=2000, LOLH=3, base_unit=NaN, max_unit=NaN, gen_set=nothing, num_sets=NaN,
    apply_load_offset=false)

    @assert (max_unit === NaN && base_unit === NaN) || (gen_set === nothing)

    if gen_set !== nothing
        gen_types = gen_set
    else
        @assert (max_unit >= base_unit) "autogen_system: max_unit should exceed base_unit"

        # Create a 'basic' generator set, starting from base_unit (in MW), increasing by factors of two until 'max_unit'
        gen_types = []
        current_size = base_unit
        while current_size <= max_unit
            append!(gen_types, current_size)
            current_size *= 2
        end
    end
    println("Generator set:", gen_types)


    # Check whether a load_profile has been supplied. If not, use a constant peak_load
    if load_profile === nothing
        @assert peak_load !== NaN
        load_profile = [1.0]
    else
        @assert load_profile isa Vector{<:Real} || load_profile isa Array{<:Real}
    end
    # Check whether peak_load has been supplied. If so, rescale load profile to target peak load
    if peak_load !== NaN
        load_profile *= peak_load / maximum(load_profile)
    end
    # Check whether the number of generator sets must be optimised.
    if num_sets === NaN

        println("Optimising number of generator sets to LOLE target of $LOLH hours")

        # initialise generation portfolio
        # println("Generator types:", gen_types)# debug
        gen_list = [
            Generator(unit_count=1, unit_capacity=power, unit_availability=gen_availability, unit_mtbf=MTBF)
            for power in gen_types]
        # println("Generator list:", gen_list)# debug
        num_sets = 0

        # create temporary load and gen rv's for iteration
        net_ld_rv = RandomVariable(resolution=resolution)
        # println("\nHi\n", typeof(net_ld_rv), "    ", typeof(load_profile), "\n", methods(load_data), "\nHi\n", )# debug
        load_data(net_ld_rv, load_profile)

        if wind_profile !== nothing
            wind_rv = RandomVariable(resolution=resolution)
            load_data(wind_rv, wind_profile)
            net_ld_rv -= wind_rv
        end
        gen_rv = RandomVariable(resolution=resolution)

        # println("Generator:", gen_list)#debug
        # add sets of generators one by one, until the risk is less than the threshold
        while true
            for generator in gen_list
                # println("Generator:", typeof(gen_list))#debug
                gen_rv += rv(generator, resolution)
            end
            truncate!(gen_rv)
            num_sets += 1

            if cdf_value((gen_rv - net_ld_rv), -0.00001) * 8760 < LOLH
                break
            end
        end
        # Report number of generator sets and LOLH value
        println("$num_sets generator sets. Base LOLE=$(cdf_value_interpolate(gen_rv - net_ld_rv, 0.0) * 8760) h; determined load offset of $(quantile_interpolate(gen_rv - net_ld_rv, LOLH / 8760)) MW to reach LOLE target.\n")

        if apply_load_offset
            load_offset = quantile_interpolate((gen_rv - net_ld_rv), LOLH / 8760)
        else
            load_offset = NaN #FIXME
        end
    else
        load_offset = NaN #FIXME
    end
    # Generate generator list
    gen_list = [
        Generator(unit_count=num_sets, unit_capacity=power, unit_availability=gen_availability, unit_mtbf=MTBF)
        for (idx, power) in enumerate(gen_types)]

    # initialise parent object
    return SingleNodeSystem(gen_list=gen_list, load_profile=load_profile,
        resolution=resolution,# wind_profile=wind_profile,
        load_offset=load_offset)
end

function main()
    println("Starting direct execution\n")

    system = autogen_system(load_profile=nothing, peak_load=10000, wind_profile=nothing, resolution=10,
        gen_availability=0.90, MTBF=2000, LOLH=3, base_unit=10, max_unit=1000, gen_set=nothing,
        apply_load_offset=true)

    println("Properties after adjustment:")
    printfmt("LOLE: {:.4f} h\n", 8760 * lolp(system))
    printfmt("EPNS: {:.4f} h\n", epns(system))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
main()

end

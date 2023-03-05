#= IEEE RTS Generation System =#

include("system.jl")

module ieeerts

using ..GeneratorModule
using ..SystemModule
using ..RandomVarModule

using Formatting: printfmt

#=
Initialises the generators and load curve of the RTS object and calls the GenLoadSystem initialiser.

:param resolution: desired resolution for any random variables (generation or load)
:param areas: number of RTS areas
=#
function ieee_rts(; resolution=1, areas=1)

    # Initialise generation portfolio
    gen_list = [
        Generator(unit_count=5 * areas, unit_capacity=12, unit_availability=0.98, unit_mtbf=3000),
        Generator(unit_count=4 * areas, unit_capacity=20, unit_availability=0.9, unit_mtbf=500),
        Generator(unit_count=6 * areas, unit_capacity=50, unit_availability=0.99, unit_mtbf=2000),
        Generator(unit_count=4 * areas, unit_capacity=76, unit_availability=0.98, unit_mtbf=2000),
        Generator(unit_count=3 * areas, unit_capacity=100, unit_availability=0.96, unit_mtbf=1250),
        Generator(unit_count=4 * areas, unit_capacity=155, unit_availability=0.96, unit_mtbf=1000),
        Generator(unit_count=3 * areas, unit_capacity=197, unit_availability=0.95, unit_mtbf=1000),
        Generator(unit_count=1 * areas, unit_capacity=350, unit_availability=0.92, unit_mtbf=1250),
        Generator(unit_count=2 * areas, unit_capacity=400, unit_availability=0.88, unit_mtbf=1250)
    ]

    weekly_load_factors = [86.2, 90.0, 87.8, 83.4, 88.0, 84.1, 83.2, 80.6, 74.0, 73.7,
        71.5, 72.7, 70.4, 75.0, 72.1, 80.0, 75.4, 83.7, 87.0, 88.0,
        85.6, 81.1, 90.0, 88.7, 89.6, 86.1, 75.5, 81.6, 80.1, 88.0,
        72.2, 77.6, 80.0, 72.9, 72.6, 70.5, 78.0, 69.5, 72.4, 72.4,
        74.3, 74.4, 80.0, 88.1, 88.5, 90.9, 94.0, 89.0, 94.2, 97.0,
        100.0, 95.2] / 100
    daily_load_factors = [93, 100, 98, 96, 94, 77, 75] / 100
    winter_weekday_hourly = [67, 63, 60, 59, 59, 60,
        74, 86, 95, 96, 96, 95,
        95, 95, 93, 94, 99, 100,
        100, 96, 91, 83, 73, 63] / 100
    winter_weekend_hourly = [78, 72, 68, 66, 64, 65,
        66, 70, 80, 88, 90, 91,
        90, 88, 87, 87, 91, 100,
        99, 97, 94, 92, 87, 81] / 100
    summer_weekday_hourly = [64, 60, 58, 56, 56, 58,
        64, 76, 87, 95, 99, 100,
        99, 100, 100, 97, 96, 96,
        93, 92, 92, 93, 87, 72] / 100
    summer_weekend_hourly = [74, 70, 66, 65, 64, 62,
        62, 66, 81, 86, 91, 93,
        93, 92, 91, 91, 92, 94,
        95, 95, 100, 93, 88, 80] / 100
    spring_fall_weekday_hourly = [63, 62, 60, 58, 59, 65,
        72, 85, 95, 99, 100, 99,
        93, 92, 90, 88, 90, 92,
        96, 98, 96, 90, 80, 70] / 100
    spring_fall_weekend_hourly = [75, 73, 69, 66, 65, 65,
        68, 74, 83, 89, 92, 94,
        91, 90, 90, 86, 85, 88,
        92, 100, 97, 95, 90, 85] / 100

    winter_week = vcat(repeat(winter_weekday_hourly, 5), repeat(winter_weekend_hourly, 2)) .*
                  repeat(daily_load_factors, inner=24)
    summer_week = vcat(repeat(summer_weekday_hourly, 5), repeat(summer_weekend_hourly, 2)) .*
                  repeat(daily_load_factors, inner=24)
    spring_fall_week = vcat(repeat(spring_fall_weekday_hourly, 5),
        repeat(spring_fall_weekend_hourly, 2)) .*
                       repeat(daily_load_factors, inner=24)

    load_profile = areas * 2850 * repeat(weekly_load_factors, inner=7 * 24) .* vcat(
        repeat(winter_week, 8),
        repeat(spring_fall_week, 9),
        repeat(summer_week, 13),
        repeat(spring_fall_week, 13),
        repeat(winter_week, 9)
    )

    println(typeof(load_profile))# debug
    # initialise parent object
    println(length(gen_list), " ", length(load_profile))# debug
    # log_array(load_profile)#debug
    return SingleNodeSystem(gen_list=gen_list, load_profile=load_profile, resolution=resolution, load_offset=0.0)#FIXME load_offset
end

# function log_array(arr::Array{Float64})
#     outfile = "log-julia.txt"
#     f = open(outfile, "a")

#     for i in arr
#         println(f, i)
#     end
#     close(f)
# end
function main()
    println("Starting direct execution")

    rts1 = ieee_rts(areas=1, resolution=1)
    println(rts1.resolution, ' ', rts1.load_offset, ' ', rts1.margin_rv.base,
        ' ', rts1.margin_rv.step, " margin ", length(rts1.margin_rv.cdf_array),
        ' ', length(rts1.load_rv.cdf_array), ' ', length(rts1.generation_rv.cdf_array)) #debug

    log_array(rts1.margin_rv.cdf_array) #debug

    rts3 = ieee_rts(areas=3)

    println("\nIEEE RTS 1 Area:")
    printfmt("LOLP [discrete levels]       : {:.6f}\n", (lolp(rts1, interpolation=false)))
    printfmt("LOLP [linear interpolation]  : {:.6f}\n", (lolp(rts1, interpolation=true)))
    printfmt("EENS [discrete levels]       : {:.6f} MWh\n", (epns(rts1, interpolation=false) * 8736))
    printfmt("EENS [linear interpolation]  : {:.6f} MWh\n", (epns(rts1, interpolation=true) * 8736))
    printfmt("LOLE                         : {:.6f} h\n", (lole(rts1)))
    printfmt("Load offset for LOLP==3/8736 : {:.6f} MW\n", (compute_load_offset(rts1, lolp_target=3 / 8736)))

    println("\nIEEE RTS 3 Area (single node representation):")
    printfmt("LOLP                         : {:.6f}\n", (lolp(rts3)))
    printfmt("EENS                         : {:.6f} MWh\n", (epns(rts3) * 8736))
    printfmt("LOLE                         : {:.6f} h\n", (lole(rts3)))
    printfmt("Load offset for LOLP==3/8736 : {:.6f} MW\n", (compute_load_offset(rts3, lolp_target=3 / 8736)))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
main()
end
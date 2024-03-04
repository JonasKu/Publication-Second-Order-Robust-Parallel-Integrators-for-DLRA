__precompile__

mutable struct Settings
    # grid settings
    # number spatial interfaces
    Nx::Int64
    Ny::Int64
    # number spatial cells
    NCellsX::Int64
    NCellsY::Int64
    # start and end point
    a::Float64
    b::Float64
    c::Float64
    d::Float64
    # grid cell width
    Δx::Float64
    Δy::Float64

    # time settings
    # end time
    tₑ::Float64
    # time increment
    Δt::Float64
    # CFL number 
    cfl::Float64

    # degree PN
    nₚₙ::Int64

    # spatial grid
    x::Vector{Float64}
    xMid::Vector{Float64}
    y::Vector{Float64}
    yMid::Vector{Float64}

    # problem definitions
    problem::String

    # physical parameters
    σₐ::Array{Float64,1}
    σₛ::Array{Float64,1}
    Q::Array{Float64,1}

    # rank
    r::Int
    ϑ::Float64
    rMax::Int
    rMin::Int
    ϑIndex::Int

    # rejection step
    cη::Float64

    function Settings(Nx::Int=302, Ny::Int=302, nₚₙ::Int=21, r::Int=20, problem::String="LineSource")

        # spatial grid setting
        NCellsX = Nx - 1
        NCellsY = Ny - 1

        if problem == "LineSource"
            a = -1.5 # left boundary
            b = 1.5 # right boundary
            c = -1.5 # lower boundary
            d = 1.5 # upper boundary
            tₑ = 1 #0.005
            BC = "none"
            cfl = 0.7 # CFL condition
        elseif problem == "Lattice"
            a = 0.0 # left boundary
            b = 7.0 # right boundary
            c = 0.0 # lower boundary
            d = 7.0 # upper boundary
            tₑ = 3.2
            BC = "Dirichlet"
            cfl = 0.5 # CFL condition
        else
            println("ERROR: Problem ", problem, " undefined.")
        end

        # spatial grid
        x = collect(range(a, stop=b, length=NCellsX))
        Δx = x[2] - x[1]
        x = [x[1] - Δx; x] # add ghost cells so that boundary cell centers lie on a and b
        x = x .+ Δx / 2
        xMid = x[1:(end-1)] .+ 0.5 * Δx
        y = collect(range(c, stop=d, length=NCellsY))
        Δy = y[2] - y[1]
        y = [y[1] - Δy; y] # add ghost cells so that boundary cell centers lie on a and b
        y = y .+ Δy / 2
        yMid = y[1:(end-1)] .+ 0.5 * Δy

        nx = NCellsX
        ny = NCellsY
        Q = zeros(nx * ny)
        # physical parameters
        if problem == "LineSource"
            σₛ = ones(nx * ny)
            σₐ = zeros(nx * ny)
        elseif problem == "Lattice"
            σₛ = ones(nx * ny)
            σₐ = zeros(nx * ny)
            for i = 1:nx
                for j = 1:ny
                    if (xMid[i] <= 2.0 && xMid[i] >= 1.0) || (xMid[i] <= 6.0 && xMid[i] >= 5.0)
                        if (yMid[j] <= 2.0 && yMid[j] >= 1.0) || (yMid[j] <= 4.0 && yMid[j] >= 3.0) || (yMid[j] <= 6.0 && yMid[j] >= 5.0)
                            σₛ[vectorIndex(ny, i, j)] = 0.0
                            σₐ[vectorIndex(ny, i, j)] = 10.0
                        end
                    end
                    if (xMid[i] <= 3.0 && xMid[i] >= 2.0) || (xMid[i] <= 5.0 && xMid[i] >= 4.0)
                        if (yMid[j] <= 3.0 && yMid[j] >= 2.0) || (yMid[j] <= 5.0 && yMid[j] >= 4.0)
                            σₛ[vectorIndex(ny, i, j)] = 0.0
                            σₐ[vectorIndex(ny, i, j)] = 10.0
                        end
                    end
                    if xMid[i] <= 4.0 && xMid[i] >= 3.0
                        if yMid[j] <= 6.0 && yMid[j] >= 5.0
                            σₛ[vectorIndex(ny, i, j)] = 0.0
                            σₐ[vectorIndex(ny, i, j)] = 10.0
                        elseif yMid[j] <= 4.0 && yMid[j] >= 3.0
                            σₛ[vectorIndex(ny, i, j)] = 0.0
                            σₐ[vectorIndex(ny, i, j)] = 10.0
                            Q[vectorIndex(ny, i, j)] = 1.0
                        end
                    end
                end
            end
        end

        # time settings

        Δt = cfl * Δx

        ϑ = 1.5e-2
        ϑIndex = 0
        rMax = Int(floor(min(nₚₙ^2, nx * ny) / 4))
        rMin = 2
        cη = 1

        # build class
        new(Nx, Ny, NCellsX, NCellsY, a, b, c, d, Δx, Δy, tₑ, Δt, cfl, nₚₙ, x, xMid, y, yMid, problem, σₐ, σₛ, Q, r, ϑ, rMax, rMin, ϑIndex, cη)
    end
end

function IC(obj::Settings, x, y)
    if obj.problem == "LineSource"
        x0 = 0.0
        y0 = 0.0
        out = zeros(length(x), length(y))
        σ² = 0.03^2
        floor = 1e-4
        for j in eachindex(x)
            for i in eachindex(y)
                out[j, i] = max(floor, 1.0 / (4.0 * π * σ²) * exp(-((x[j] - x0) * (x[j] - x0) + (y[i] - y0) * (y[i] - y0)) / 4.0 / σ²)) / 4.0 / π
            end
        end
    elseif obj.problem == "Lattice"
        out = 1e-9 * ones(length(x), length(y))
    end
    return out
end
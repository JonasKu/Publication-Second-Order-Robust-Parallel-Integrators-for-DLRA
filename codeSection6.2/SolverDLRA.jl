__precompile__

using ProgressMeter
using LinearAlgebra
using LegendrePolynomials
using QuadGK
using SparseArrays
using SphericalHarmonicExpansions, SphericalHarmonics, TypedPolynomials, GSL
using MultivariatePolynomials
using Einsum
using PyCall
using PyCall
np = pyimport("numpy")

include("PNSystem.jl")
include("utils.jl")

struct SolverDLRA
    # spatial grid of cell interfaces
    x::Array{Float64}
    y::Array{Float64}

    # Solver settings
    settings::Settings

    # squared L2 norms of Legendre coeffs
    γ::Array{Float64,1}
    # Roe matrix
    AbsAx::Array{Float64,2}
    AbsAz::Array{Float64,2}

    # functionalities of the PN system
    pn::PNSystem

    Dxx::SparseMatrixCSC{Float64,Int64}
    Dyy::SparseMatrixCSC{Float64,Int64}
    Dx::SparseMatrixCSC{Float64,Int64}
    Dy::SparseMatrixCSC{Float64,Int64}

    # constructor
    function SolverDLRA(settings)
        x = settings.x
        y = settings.y

        # setup flux matrix
        γ = zeros(settings.nₚₙ + 1)
        for i = 1:settings.nₚₙ+1
            n = i - 1
            γ[i] = 2 / (2 * n + 1)
        end

        # construct PN system matrices
        pn = PNSystem(settings)
        SetupSystemMatrices(pn)

        # setup Roe matrix
        S = eigvals(pn.Ax)
        V = eigvecs(pn.Ax)
        AbsAx = V * abs.(diagm(S)) * inv(V)

        S = eigvals(pn.Az)
        V = eigvecs(pn.Az)
        AbsAz = V * abs.(diagm(S)) * inv(V)

        # setupt stencil matrix
        nx = settings.NCellsX
        ny = settings.NCellsY
        Dxx = spzeros(nx * ny, nx * ny)
        Dyy = spzeros(nx * ny, nx * ny)
        Dx = spzeros(nx * ny, nx * ny)
        Dy = spzeros(nx * ny, nx * ny)

        # setup index arrays and values for allocation of stencil matrices
        II = zeros(3 * (nx - 2) * (ny - 2))
        J = zeros(3 * (nx - 2) * (ny - 2))
        vals = zeros(3 * (nx - 2) * (ny - 2))
        counter = -2

        for i = 2:nx-1
            for j = 2:ny-1
                counter = counter + 3
                # x part
                index = vectorIndex(nx, i, j)
                indexPlus = vectorIndex(nx, i + 1, j)
                indexMinus = vectorIndex(nx, i - 1, j)

                II[counter+1] = index
                J[counter+1] = index
                vals[counter+1] = 2.0 / 2 / settings.Δx
                if i > 1
                    II[counter] = index
                    J[counter] = indexMinus
                    vals[counter] = -1 / 2 / settings.Δx
                end
                if i < nx
                    II[counter+2] = index
                    J[counter+2] = indexPlus
                    vals[counter+2] = -1 / 2 / settings.Δx
                end
            end
        end
        Dxx = sparse(II, J, vals, nx * ny, nx * ny)

        II .= zeros(3 * (nx - 2) * (ny - 2))
        J .= zeros(3 * (nx - 2) * (ny - 2))
        vals .= zeros(3 * (nx - 2) * (ny - 2))
        counter = -2

        for i = 2:nx-1
            for j = 2:ny-1
                counter = counter + 3
                # y part
                index = vectorIndex(nx, i, j)
                indexPlus = vectorIndex(nx, i, j + 1)
                indexMinus = vectorIndex(nx, i, j - 1)

                II[counter+1] = index
                J[counter+1] = index
                vals[counter+1] = 2.0 / 2 / settings.Δy

                if j > 1
                    II[counter] = index
                    J[counter] = indexMinus
                    vals[counter] = -1 / 2 / settings.Δy
                end
                if j < ny
                    II[counter+2] = index
                    J[counter+2] = indexPlus
                    vals[counter+2] = -1 / 2 / settings.Δy
                end
            end
        end
        Dyy = sparse(II, J, vals, nx * ny, nx * ny)

        II = zeros(2 * (nx - 2) * (ny - 2))
        J = zeros(2 * (nx - 2) * (ny - 2))
        vals = zeros(2 * (nx - 2) * (ny - 2))
        counter = -1

        for i = 2:nx-1
            for j = 2:ny-1
                counter = counter + 2
                # x part
                index = vectorIndex(nx, i, j)
                indexPlus = vectorIndex(nx, i + 1, j)
                indexMinus = vectorIndex(nx, i - 1, j)

                if i > 1
                    II[counter] = index
                    J[counter] = indexMinus
                    vals[counter] = -1 / 2 / settings.Δx
                end
                if i < nx
                    II[counter+1] = index
                    J[counter+1] = indexPlus
                    vals[counter+1] = 1 / 2 / settings.Δx
                end
            end
        end
        Dx = sparse(II, J, vals, nx * ny, nx * ny)

        II .= zeros(2 * (nx - 2) * (ny - 2))
        J .= zeros(2 * (nx - 2) * (ny - 2))
        vals .= zeros(2 * (nx - 2) * (ny - 2))
        counter = -1

        for i = 2:nx-1
            for j = 2:ny-1
                counter = counter + 2
                # y part
                index = vectorIndex(nx, i, j)
                indexPlus = vectorIndex(nx, i, j + 1)
                indexMinus = vectorIndex(nx, i, j - 1)

                if j > 1
                    II[counter] = index
                    J[counter] = indexMinus
                    vals[counter] = -1 / 2 / settings.Δy
                end
                if j < ny
                    II[counter+1] = index
                    J[counter+1] = indexPlus
                    vals[counter+1] = 1 / 2 / settings.Δy
                end
            end
        end
        Dy = sparse(II, J, vals, nx * ny, nx * ny)

        new(x, y, settings, γ, AbsAx, AbsAz, pn, Dxx, Dyy, Dx, Dy)
    end
end

py"""
import numpy
def qr(A):
    return numpy.linalg.qr(A)
"""

function SetupIC(obj::SolverDLRA)
    u = zeros(obj.settings.NCellsX, obj.settings.NCellsY, obj.pn.nTotalEntries)
    u[:, :, 1] = IC(obj.settings, obj.settings.xMid, obj.settings.yMid)
    return u
end

function Solve(obj::SolverDLRA)
    # Get rank
    s = obj.settings
    # Set up initial condition and store as matrix
    v = SetupIC(obj)
    nx = obj.settings.NCellsX
    ny = obj.settings.NCellsY
    N = obj.pn.nTotalEntries
    u = zeros(nx * ny, N)
    for k = 1:N
        u[:, k] = vec(v[:, :, k])
    end
    # free memory
    v = 0

    nT = Int(ceil(s.tₑ / s.Δt))
    Δt = s.Δt

    e₁ = zeros(N)
    e₁[1] = 1.0
    E₁ = sparse([1], [1], [1.0], N, N)

    σₛ = Diagonal(obj.settings.σₛ)
    σₜ = Diagonal(obj.settings.σₐ .+ obj.settings.σₛ)
    Q = obj.settings.Q

    prog = Progress(nT, 1)
    t = 0.0

    for n in 1:nT

        u .= u .- Δt * (obj.Dx * u * obj.pn.Ax .+ obj.Dy * u * obj.pn.Az .+ obj.Dxx * u * obj.AbsAx .+ obj.Dyy * u * obj.AbsAz .+ σₜ * u .- σₛ * u * E₁ .- Q * e₁')

        t += Δt
        next!(prog) # update progress bar
    end

    # return end time and solution
    return 0.5 * sqrt(obj.γ[1]) * u[:, 1]

end

function SolveBUG(obj::SolverDLRA)
    # Get rank
    r = obj.settings.rMax
    s = obj.settings
    # Set up initial condition and store as matrix
    v = SetupIC(obj)
    nx = obj.settings.NCellsX
    ny = obj.settings.NCellsY
    N = obj.pn.nTotalEntries
    u = zeros(nx * ny, N)
    for k = 1:N
        u[:, k] = vec(v[:, :, k])
    end
    # free memory
    v = 0

    nT = Int(ceil(s.tₑ / s.Δt))
    Δt = s.Δt

    # Low-rank approx of init data:
    X, S, W = svd(u)
    # free memory
    u = 0

    # rank-r truncation:
    X = X[:, 1:r]
    W = W[:, 1:r]
    S = Diagonal(S)
    S = S[1:r, 1:r]

    e₁ = zeros(N)
    e₁[1] = 1.0
    E₁ = sparse([1], [1], [1.0], N, N)

    rankInTime = zeros(2, nT)
    NormInTime = zeros(2, nT)
    MassInTime = zeros(2, nT)

    σₛ = Diagonal(obj.settings.σₛ)
    σₜ = Diagonal(obj.settings.σₐ .+ obj.settings.σₛ)
    Q = obj.settings.Q

    prog = Progress(nT, 1)
    t = 0.0

    for n = 1:nT
        rankInTime[1, n] = t
        rankInTime[2, n] = r
        NormInTime[1, n] = t
        NormInTime[2, n] = norm(S, 2)
        MassInTime[1, n] = t
        for j = 1:size(X, 1)
            MassInTime[2, n] += X[j, :]' * S * W[1, :]
        end

        ################## K-step ##################

        WᵀAₓ₂W = W' * obj.pn.Az * W
        WᵀRₓ₂W = W' * obj.AbsAz * W
        WᵀRₓ₁W = W' * obj.AbsAx * W
        WᵀAₓ₁W = W' * obj.pn.Ax * W
        WᵀE₁W = W' * E₁ * W

        rhsK = k -> begin
            return -(obj.Dx * k * WᵀAₓ₁W .+ obj.Dy * k * WᵀAₓ₂W .+ obj.Dxx * k * WᵀRₓ₁W .+ obj.Dyy * k * WᵀRₓ₂W .+ σₜ * k .- σₛ * k * WᵀE₁W .- Q * (e₁' * W))
        end

        K = X * S
        K = rk(rhsK, K, Δt)

        K = [K X]
        X₁, _ = qr!(K)
        X₁ = Matrix(X₁)
        X₁ = X₁[:, 1:2*r]

        Mᵤ = X₁' * X

        ################## L-step ##################
        XᵀDₓ₁X = X' * obj.Dx * X
        XᵀDₓ₂X = X' * obj.Dy * X
        XᵀDxxX = X' * obj.Dxx * X
        XᵀDyyX = X' * obj.Dyy * X
        XᵀσₛX = X' * σₛ * X
        XᵀσₜX = X' * σₜ * X

        rhsL = l -> begin
            return -(obj.pn.Ax * l * XᵀDₓ₁X' .+ obj.pn.Az * l * XᵀDₓ₂X' .+ obj.AbsAx * l * XᵀDxxX' .+ obj.AbsAz * l * XᵀDyyX' .+ l * XᵀσₜX .- E₁ * l * XᵀσₛX .- e₁ * (Q' * X))
        end

        L = W * S'
        L = rk(rhsL, L, Δt)

        L = [L W]
        W₁, _ = qr(L)
        W₁ = Matrix(W₁)
        W₁ = W₁[:, 1:2*r]

        Nᵤ = W₁' * W
        W = W₁
        X = X₁

        ################## S-step ##################
        XᵀDₓ₁X = X' * obj.Dx * X
        XᵀDₓ₂X = X' * obj.Dy * X
        XᵀDxxX = X' * obj.Dxx * X
        XᵀDyyX = X' * obj.Dyy * X

        WᵀAₓ₂W = W' * obj.pn.Az' * W
        WᵀRₓ₂W = W' * obj.AbsAz' * W
        WᵀRₓ₁W = W' * obj.AbsAx' * W
        WᵀAₓ₁W = W' * obj.pn.Ax' * W
        WᵀE₁W = W' * E₁ * W
        XᵀσₛX = X' * σₛ * X
        XᵀσₜX = X' * σₜ * X

        rhsS = s -> begin
            return -(XᵀDₓ₁X * s * WᵀAₓ₁W .+ XᵀDₓ₂X * s * WᵀAₓ₂W .+ XᵀDxxX * s * WᵀRₓ₁W .+ XᵀDyyX * s * WᵀRₓ₂W .+ (XᵀσₜX * s .- XᵀσₛX * s * WᵀE₁W .- (X' * Q) * (e₁' * W)))
        end

        S = Mᵤ * S * (Nᵤ')
        S = rk(rhsS, S, Δt)

        ################## truncate ##################

        X, S, W = truncateFixed!(obj, X, S, W)

        # update rank
        r = size(S, 1)

        t += Δt
        next!(prog) # update progress bar
    end

    # return end time and solution
    return 0.5 * sqrt(obj.γ[1]) * X * S * W[1, :], rankInTime, MassInTime

end

function SolveBUG2nd(obj::SolverDLRA)
    # Get rank
    r = obj.settings.rMax
    s = obj.settings
    # Set up initial condition and store as matrix
    v = SetupIC(obj)
    nx = obj.settings.NCellsX
    ny = obj.settings.NCellsY
    N = obj.pn.nTotalEntries
    u = zeros(nx * ny, N)
    for k = 1:N
        u[:, k] = vec(v[:, :, k])
    end
    # free memory
    v = 0

    nT = Int(ceil(s.tₑ / s.Δt))
    Δt = s.Δt

    # Low-rank approx of init data:
    X, S, W = svd(u)
    # free memory
    u = 0

    # rank-r truncation:
    X = X[:, 1:r]
    W = W[:, 1:r]
    S = Diagonal(S)
    S = S[1:r, 1:r]

    e₁ = zeros(N)
    e₁[1] = 1.0
    E₁ = sparse([1], [1], [1.0], N, N)

    rankInTime = zeros(2, nT)
    NormInTime = zeros(2, nT)
    MassInTime = zeros(2, nT)

    σₛ = Diagonal(obj.settings.σₛ)
    σₜ = Diagonal(obj.settings.σₐ .+ obj.settings.σₛ)
    Q = obj.settings.Q

    prog = Progress(nT, 1)
    t = 0.0

    for n = 1:nT
        rankInTime[1, n] = t
        rankInTime[2, n] = r
        NormInTime[1, n] = t
        NormInTime[2, n] = norm(S, 2)
        MassInTime[1, n] = t
        for j = 1:size(X, 1)
            MassInTime[2, n] += X[j, :]' * S * W[1, :]
        end

        X0 = Base.deepcopy(X)
        W0 = Base.deepcopy(W)
        S0 = Base.deepcopy(S)

        ################## K-step ##################

        WᵀAₓ₂W = W' * obj.pn.Az * W
        WᵀRₓ₂W = W' * obj.AbsAz * W
        WᵀRₓ₁W = W' * obj.AbsAx * W
        WᵀAₓ₁W = W' * obj.pn.Ax * W
        WᵀE₁W = W' * E₁ * W

        rhsK = k -> begin
            return -(obj.Dx * k * WᵀAₓ₁W .+ obj.Dy * k * WᵀAₓ₂W .+ obj.Dxx * k * WᵀRₓ₁W .+ obj.Dyy * k * WᵀRₓ₂W .+ σₜ * k .- σₛ * k * WᵀE₁W .- Q * (e₁' * W))
        end

        K = X * S
        K = rk(rhsK, K, 0.5*Δt)

        K = [K X]
        X₁, _ = qr!(K)
        X₁ = Matrix(X₁)
        X₁ = X₁[:, 1:2*r]

        Mᵤ = X₁' * X

        ################## L-step ##################
        XᵀDₓ₁X = X' * obj.Dx * X
        XᵀDₓ₂X = X' * obj.Dy * X
        XᵀDxxX = X' * obj.Dxx * X
        XᵀDyyX = X' * obj.Dyy * X
        XᵀσₛX = X' * σₛ * X
        XᵀσₜX = X' * σₜ * X

        rhsL = l -> begin
            return -(obj.pn.Ax * l * XᵀDₓ₁X' .+ obj.pn.Az * l * XᵀDₓ₂X' .+ obj.AbsAx * l * XᵀDxxX' .+ obj.AbsAz * l * XᵀDyyX' .+ l * XᵀσₜX .- E₁ * l * XᵀσₛX .- e₁ * (Q' * X))
        end

        L = W * S'
        L = rk(rhsL, L, 0.5*Δt)

        L = [L W]
        W₁, _ = qr(L)
        W₁ = Matrix(W₁)
        W₁ = W₁[:, 1:2*r]

        Nᵤ = W₁' * W
        W = W₁
        X = X₁

        ################## S-step ##################
        XᵀDₓ₁X = X' * obj.Dx * X
        XᵀDₓ₂X = X' * obj.Dy * X
        XᵀDxxX = X' * obj.Dxx * X
        XᵀDyyX = X' * obj.Dyy * X

        WᵀAₓ₂W = W' * obj.pn.Az' * W
        WᵀRₓ₂W = W' * obj.AbsAz' * W
        WᵀRₓ₁W = W' * obj.AbsAx' * W
        WᵀAₓ₁W = W' * obj.pn.Ax' * W
        WᵀE₁W = W' * E₁ * W
        XᵀσₛX = X' * σₛ * X
        XᵀσₜX = X' * σₜ * X

        rhsS = s -> begin
            return -(XᵀDₓ₁X * s * WᵀAₓ₁W .+ XᵀDₓ₂X * s * WᵀAₓ₂W .+ XᵀDxxX * s * WᵀRₓ₁W .+ XᵀDyyX * s * WᵀRₓ₂W .+ (XᵀσₜX * s .- XᵀσₛX * s * WᵀE₁W .- (X' * Q) * (e₁' * W)))
        end

        S = Mᵤ * S * (Nᵤ')
        S = rk(rhsS, S, 0.5*Δt)

        ################## augment ##################

        XᵀDₓ₁X = X' * obj.Dx * X
        XᵀDₓ₂X = X' * obj.Dy * X
        XᵀDxxX = X' * obj.Dxx * X
        XᵀDyyX = X' * obj.Dyy * X
        XᵀσₛX = X' * σₛ * X
        XᵀσₜX = X' * σₜ * X
        WᵀAₓ₂W = W' * obj.pn.Az * W
        WᵀRₓ₂W = W' * obj.AbsAz * W
        WᵀRₓ₁W = W' * obj.AbsAx * W
        WᵀAₓ₁W = W' * obj.pn.Ax * W
        WᵀE₁W = W' * E₁ * W

        rhsK = k -> begin
            return -(obj.Dx * k * WᵀAₓ₁W .+ obj.Dy * k * WᵀAₓ₂W .+ obj.Dxx * k * WᵀRₓ₁W .+ obj.Dyy * k * WᵀRₓ₂W .+ σₜ * k .- σₛ * k * WᵀE₁W .- Q * (e₁' * W))
        end

        rhsL = l -> begin
            return -(obj.pn.Ax * l * XᵀDₓ₁X' .+ obj.pn.Az * l * XᵀDₓ₂X' .+ obj.AbsAx * l * XᵀDxxX' .+ obj.AbsAz * l * XᵀDyyX' .+ l * XᵀσₜX .- E₁ * l * XᵀσₛX .- e₁ * (Q' * X))
        end

        L = [L W]
        W₁, _ = qr(L)
        W₁ = Matrix(W₁)
        W = W₁[:, 1:4*r]

        K = [K X]
        X₁, _ = qr!(K)
        X₁ = Matrix(X₁)
        X = X₁[:, 1:4*r]

        Mᵤ = X'*X0
        Nᵤ = W'*W0

        ################## S-step ##################
        XᵀDₓ₁X = X' * obj.Dx * X
        XᵀDₓ₂X = X' * obj.Dy * X
        XᵀDxxX = X' * obj.Dxx * X
        XᵀDyyX = X' * obj.Dyy * X

        WᵀAₓ₂W = W' * obj.pn.Az' * W
        WᵀRₓ₂W = W' * obj.AbsAz' * W
        WᵀRₓ₁W = W' * obj.AbsAx' * W
        WᵀAₓ₁W = W' * obj.pn.Ax' * W
        WᵀE₁W = W' * E₁ * W
        XᵀσₛX = X' * σₛ * X
        XᵀσₜX = X' * σₜ * X

        rhsS = s -> begin
            return -(XᵀDₓ₁X * s * WᵀAₓ₁W .+ XᵀDₓ₂X * s * WᵀAₓ₂W .+ XᵀDxxX * s * WᵀRₓ₁W .+ XᵀDyyX * s * WᵀRₓ₂W .+ (XᵀσₜX * s .- XᵀσₛX * s * WᵀE₁W .- (X' * Q) * (e₁' * W)))
        end

        S = Mᵤ * S0 * (Nᵤ')
        S = rk(rhsS, S, Δt)

        ################## truncate ##################

        X, S, W = truncateFixed!(obj, X, S, W)

        # update rank
        r = size(S, 1)

        t += Δt
        next!(prog) # update progress bar
    end

    # return end time and solution
    return 0.5 * sqrt(obj.γ[1]) * X * S * W[1, :], rankInTime, MassInTime

end

function SolveParallel(obj::SolverDLRA)
    # Get rank
    r = obj.settings.rMax
    s = obj.settings
    # Set up initial condition and store as matrix
    v = SetupIC(obj)
    nx = obj.settings.NCellsX
    ny = obj.settings.NCellsY
    N = obj.pn.nTotalEntries
    u = zeros(nx * ny, N)
    for k = 1:N
        u[:, k] = vec(v[:, :, k])
    end
    # free memory
    v = 0

    nT = Int(ceil(s.tₑ / s.Δt))
    Δt = s.Δt

    # Low-rank approx of init data:
    X, S, W = svd(u)

    # free memory
    u = 0

    # rank-r truncation:
    X = X[:, 1:r]
    W = W[:, 1:r]
    S = Diagonal(S)
    S = S[1:r, 1:r]

    e₁ = zeros(N)
    e₁[1] = 1.0
    E₁ = sparse([1], [1], [1.0], N, N)

    rankInTime = zeros(2, nT)
    NormInTime = zeros(2, nT)

    σₛ = Diagonal(obj.settings.σₛ)
    σₜ = Diagonal(obj.settings.σₐ .+ obj.settings.σₛ)
    Q = obj.settings.Q

    prog = Progress(nT, 1)
    t = 0.0

    for n = 1:nT
        rankInTime[1, n] = t
        rankInTime[2, n] = r
        NormInTime[1, n] = t
        NormInTime[2, n] = norm(S, 2)

        XᵀDₓ₁X = X' * obj.Dx * X
        XᵀDₓ₂X = X' * obj.Dy * X
        XᵀDxxX = X' * obj.Dxx * X
        XᵀDyyX = X' * obj.Dyy * X

        WᵀAₓ₂W = W' * obj.pn.Az' * W
        WᵀRₓ₂W = W' * obj.AbsAz' * W
        WᵀRₓ₁W = W' * obj.AbsAx' * W
        WᵀAₓ₁W = W' * obj.pn.Ax' * W
        WᵀE₁W = W' * E₁ * W
        XᵀσₛX = X' * σₛ * X
        XᵀσₜX = X' * σₜ * X

        ################## K-step ##################

        rhsK = k -> begin
            return -(obj.Dx * k * WᵀAₓ₁W .+ obj.Dy * k * WᵀAₓ₂W .+ obj.Dxx * k * WᵀRₓ₁W .+ obj.Dyy * k * WᵀRₓ₂W .+ σₜ * k .- σₛ * k * WᵀE₁W .- Q * (e₁' * W))
        end

        K = X * S
        K = rk(rhsK, K, Δt)

        X₁, _ = qr([X K])
        X₁ = Matrix(X₁)
        tildeX₁ = X₁[:, (r+1):(2*r)]        

        ################## L-step ##################

        rhsL = l -> begin
            return -(obj.pn.Ax * l * XᵀDₓ₁X' .+ obj.pn.Az * l * XᵀDₓ₂X' .+ obj.AbsAx * l * XᵀDxxX' .+ obj.AbsAz * l * XᵀDyyX' .+ l * XᵀσₜX .- E₁ * l * XᵀσₛX .- e₁ * (Q' * X))
        end

        L = W * S'
        L = rk(rhsL, L, Δt)

        W₁, _ = qr([W L])
        W₁ = Matrix(W₁)
        tildeW₁ = W₁[:, (r+1):(2*r)]

        ################## S-step ##################

        rhsS = s -> begin
            return -(XᵀDₓ₁X * s * WᵀAₓ₁W .+ XᵀDₓ₂X * s * WᵀAₓ₂W .+ XᵀDxxX * s * WᵀRₓ₁W .+ XᵀDyyX * s * WᵀRₓ₂W .+ (XᵀσₜX * s .- XᵀσₛX * s * WᵀE₁W .- (X' * Q) * (e₁' * W)))
        end

        S = rk(rhsS, S, Δt)

        ################## truncate ##################

        SNew = zeros(2 * r, 2 * r)

        SNew[1:r, 1:r] = S
        SNew[(r+1):end, 1:r] = tildeX₁' * K
        SNew[1:r, (r+1):(2*r)] = L' * tildeW₁

        # truncate
        X, S, W = truncateFixed!(obj, [X tildeX₁], SNew, [W tildeW₁])

        # update rank
        r = size(S, 1)

        t += Δt
        next!(prog) # update progress bar
    end

    # return end time and solution
    return 0.5 * sqrt(obj.γ[1]) * X * S * W[1, :], rankInTime

end

function SolveParallel2nd(obj::SolverDLRA)
    # Get rank
    r = obj.settings.rMax
    s = obj.settings
    # Set up initial condition and store as matrix
    v = SetupIC(obj)
    nx = obj.settings.NCellsX
    ny = obj.settings.NCellsY
    N = obj.pn.nTotalEntries
    u = zeros(nx * ny, N)
    for k = 1:N
        u[:, k] = vec(v[:, :, k])
    end
    # free memory
    v = 0

    nT = Int(ceil(s.tₑ / s.Δt))
    Δt = s.Δt

    # Low-rank approx of init data:
    X, S, W = svd(u)

    # free memory
    u = 0

    # rank-r truncation:
    X = X[:, 1:r]
    W = W[:, 1:r]
    S = Diagonal(S)
    S = S[1:r, 1:r]

    e₁ = zeros(N)
    e₁[1] = 1.0
    E₁ = sparse([1], [1], [1.0], N, N)

    rankInTime = zeros(2, nT)
    NormInTime = zeros(2, nT)

    σₛ = Diagonal(obj.settings.σₛ)
    σₜ = Diagonal(obj.settings.σₐ .+ obj.settings.σₛ)
    Q = obj.settings.Q

    prog = Progress(nT, 1)
    t = 0.0

    for n = 1:nT
        rankInTime[1, n] = t
        rankInTime[2, n] = r
        NormInTime[1, n] = t
        NormInTime[2, n] = norm(S, 2)

        XᵀDₓ₁X = X' * obj.Dx * X
        XᵀDₓ₂X = X' * obj.Dy * X
        XᵀDxxX = X' * obj.Dxx * X
        XᵀDyyX = X' * obj.Dyy * X

        WᵀAₓ₂W = W' * obj.pn.Az' * W
        WᵀRₓ₂W = W' * obj.AbsAz' * W
        WᵀRₓ₁W = W' * obj.AbsAx' * W
        WᵀAₓ₁W = W' * obj.pn.Ax' * W
        WᵀE₁W = W' * E₁ * W
        XᵀσₛX = X' * σₛ * X
        XᵀσₜX = X' * σₜ * X

        rhsK = k -> begin
            return -(obj.Dx * k * WᵀAₓ₁W .+ obj.Dy * k * WᵀAₓ₂W .+ obj.Dxx * k * WᵀRₓ₁W .+ obj.Dyy * k * WᵀRₓ₂W .+ σₜ * k .- σₛ * k * WᵀE₁W .- Q * (e₁' * W))
        end

        rhsL = l -> begin
            return -(obj.pn.Ax * l * XᵀDₓ₁X' .+ obj.pn.Az * l * XᵀDₓ₂X' .+ obj.AbsAx * l * XᵀDxxX' .+ obj.AbsAz * l * XᵀDyyX' .+ l * XᵀσₜX .- E₁ * l * XᵀσₛX .- e₁ * (Q' * X))
        end

        X0 = deepcopy(X)
        W0 = deepcopy(W)
        K0 = X * S
        L0 = W * S'
        hatX, _ = np.linalg.qr([X rhsK(K0)], mode="reduced");
        hatW, _ = np.linalg.qr([W rhsL(L0)], mode="reduced");
        hatX[:, 1:r] = X0 # this is very important!!!
        hatW[:, 1:r] = W0 # this is very important!!!
        X = hatX
        W = hatW

        XᵀDₓ₁X = X' * obj.Dx * X
        XᵀDₓ₂X = X' * obj.Dy * X
        XᵀDxxX = X' * obj.Dxx * X
        XᵀDyyX = X' * obj.Dyy * X

        WᵀAₓ₂W = W' * obj.pn.Az' * W
        WᵀRₓ₂W = W' * obj.AbsAz' * W
        WᵀRₓ₁W = W' * obj.AbsAx' * W
        WᵀAₓ₁W = W' * obj.pn.Ax' * W
        WᵀE₁W = W' * E₁ * W
        XᵀσₛX = X' * σₛ * X
        XᵀσₜX = X' * σₜ * X

        ################## K-step ##################

        rhsK = k -> begin
            return -(obj.Dx * k * WᵀAₓ₁W .+ obj.Dy * k * WᵀAₓ₂W .+ obj.Dxx * k * WᵀRₓ₁W .+ obj.Dyy * k * WᵀRₓ₂W .+ σₜ * k .- σₛ * k * WᵀE₁W .- Q * (e₁' * W))
        end

        K = [K0 zeros(size(K0))]
        K = rk(rhsK, K, Δt)

        X₁, _ = qr([X K])
        X₁ = Matrix(X₁)
        tildeX₁ = X₁[:,(2*r+1):4*r]

        ################## L-step ##################

        rhsL = l -> begin
            return -(obj.pn.Ax * l * XᵀDₓ₁X' .+ obj.pn.Az * l * XᵀDₓ₂X' .+ obj.AbsAx * l * XᵀDxxX' .+ obj.AbsAz * l * XᵀDyyX' .+ l * XᵀσₜX .- E₁ * l * XᵀσₛX .- e₁ * (Q' * X))
        end

        L = [L0 zeros(size(L0))]
        L = rk(rhsL, L, Δt)

        W₁, _ = qr([W L])
        W₁ = Matrix(W₁)
        tildeW₁ = W₁[:,(2*r+1):4*r]

        ################## S-step ##################

        rhsS = s -> begin
            return -(XᵀDₓ₁X * s * WᵀAₓ₁W .+ XᵀDₓ₂X * s * WᵀAₓ₂W .+ XᵀDxxX * s * WᵀRₓ₁W .+ XᵀDyyX * s * WᵀRₓ₂W .+ (XᵀσₜX * s .- XᵀσₛX * s * WᵀE₁W .- (X' * Q) * (e₁' * W)))
        end

        S = [S zeros(size(S)); zeros(size(S)) zeros(size(S))]
        S = rk(rhsS, S, Δt)

        ################## truncate ##################

        SNew = zeros(2 * r, 2 * r)

        SNew = zeros(4*r, 4*r)
        SNew[1:2*r,1:2*r] = S
        SNew[1:2*r,(2*r + 1):4*r] = L' * tildeW₁
        SNew[(2*r + 1):4*r,1:2*r] = tildeX₁' * K

        # truncate
        X, S, W = truncateFixed!(obj, [X tildeX₁], SNew, [W tildeW₁])

        # update rank
        r = size(S, 1)

        t += Δt
        next!(prog) # update progress bar
    end

    # return end time and solution
    return 0.5 * sqrt(obj.γ[1]) * X * S * W[1, :], rankInTime

end

function truncate!(obj::SolverDLRA, X::Array{Float64,2}, S::Array{Float64,2}, W::Array{Float64,2})
    # Compute singular values of S and decide how to truncate:
    U, D, V = svd(S)
    rmax = -1
    rMaxTotal = obj.settings.rMax
    rMinTotal = obj.settings.rMin
    S .= zeros(size(S))

    tmp = 0.0
    ϑ = obj.settings.ϑ * norm(D)#obj.settings.ϑ * max(1e-7, norm(D)^obj.settings.ϑIndex)

    rmax = Int(floor(size(D, 1) / 2))

    for j = 1:2*rmax
        tmp = sqrt(sum(D[j:2*rmax]) .^ 2)
        if tmp < ϑ
            rmax = j
            break
        end
    end

    # if 2*r was actually not enough move to highest possible rank
    if rmax == -1
        rmax = rMaxTotal
    end

    rmax = min(rmax, rMaxTotal)
    rmax = max(rmax, rMinTotal)

    # return rank
    return X * U[:, 1:rmax], diagm(D[1:rmax]), W * V[:, 1:rmax]
end


function truncateFixed!(obj::SolverDLRA, X::Array{Float64,2}, S::Array{Float64,2}, W::Array{Float64,2})
    # Compute singular values of S and decide how to truncate:
    U, D, V = svd(S)
    
    # return rank
    return X * U[:, 1:obj.settings.r], diagm(D[1:obj.settings.r]), W * V[:, 1:obj.settings.r]
end

function truncateConservative!(obj::SolverDLRA, X::Array{Float64,2}, S::Array{Float64,2}, W::Array{Float64,2})
    r0 = size(S, 1)
    rMaxTotal = obj.settings.rMax

    # ensure that e1 is first column in W matrix, most likely not needed since conservative basis is preserved. Ensure cons basis is in front
    e1 = [1.0; zeros(size(W, 1) - 1)]
    W1, _ = py"qr"([e1 W])
    S = S * (W' * W1)
    W = W1
    K = X * S

    # split solution in conservative and remainder
    Kcons = K[:, 1]
    Krem = K[:, 2:end]
    Wcons = W[:, 1]
    Wrem = W[:, 2:end]
    Xcons = Kcons ./ norm(Kcons)
    Scons = norm(Kcons)
    Xrem, Srem = py"qr"(Krem)

    # truncate remainder part and leave conservative part as is
    U, Sigma, V = svd(Srem)
    rmax = -1

    tmp = 0.0
    tol = obj.settings.ϑ * norm(Sigma)

    rmax = Int(floor(size(Sigma, 1) / 2))

    for j = 1:2*rmax
        tmp = sqrt(sum(Sigma[j:2*rmax]) .^ 2)
        if (tmp < tol)
            rmax = j
            break
        end
    end

    rmax = min(rmax, rMaxTotal)
    r1 = max(rmax, 2)

    # if 2*r was actually not enough move to highest possible rank
    if rmax == -1
        rmax = rMaxTotal
    end

    Srem = Diagonal(Sigma[1:r1])
    Xrem = Xrem * U[:, 1:r1]
    Wrem = Wrem * V[:, 1:r1]
    What = [e1 Wrem]
    Xhat = [Xcons Xrem]
    Xnew, R1 = py"qr"(Xhat)
    Wnew, R2 = py"qr"(What)
    Snew = R1 * [Scons zeros(1, r1); zeros(r1, 1) Srem] * R2'
    return Xnew, Snew, Wnew, r1
end
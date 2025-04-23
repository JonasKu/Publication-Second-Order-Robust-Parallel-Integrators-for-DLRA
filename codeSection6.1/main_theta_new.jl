using Pkg
Pkg.activate(".")
Pkg.instantiate()

using Random

# Set a seed and store it
seed = 1234
Random.seed!(seed)

using PyPlot
using LinearAlgebra
using Suppressor
using SparseArrays
include("methods.jl")

# Cleaning and set parameters
close("all")

function setupD(N::Int)
    # Construction of the RHS-side.
    global D
    problem = "BCSchr"
    if problem == "standardSchr"
        D = 1/2 *  Tridiagonal(-ones(ComplexF64,N-1), 2*ones(ComplexF64,N), -ones(ComplexF64,N-1)) #non-stiff FD Laplacian.
    elseif problem == "BCSchr"
        idxI = zeros(3*N); idxJ = zeros(3*N); vals = zeros(3*N);
        counter = 0;

        for i = 1:N
            if i > 1
                counter = counter + 1
                idxI[counter] = i;
                idxJ[counter] = i-1;
                vals[counter] = -1/2
            end

            counter = counter + 1
            idxI[counter] = i;
            idxJ[counter] = i;
            vals[counter] = 1

            if i < N
                counter = counter + 1
                idxI[counter] = i;
                idxJ[counter] = i+1;
                vals[counter] = -1/2
            end
        end
        idxI[counter+1] = 1;
        idxJ[counter+1] = N;
        vals[counter+1] = -1/2
        idxI[counter+2] = N;
        idxJ[counter+2] = 1;
        vals[counter+2] = -1/2

        D = sparse(idxI,idxJ,vals .+ 0.0*1im,N,N);
    elseif problem == "Adv"
        idxI = zeros(2*N); idxJ = zeros(2*N); vals = zeros(2*N);
        counter = 0;

        for i = 1:N
            if i > 1
                counter = counter + 1
                idxI[counter] = i;
                idxJ[counter] = i-1;
                vals[counter] = -1
            end

            counter = counter + 1
            idxI[counter] = i;
            idxJ[counter] = i;
            vals[counter] = 1
        end
        idxI[counter+1] = 1;
        idxJ[counter+1] = N;
        vals[counter+1] = -1

        D = sparse(idxI,idxJ,vals .+ 0.0*1im,N,N);
    end
    return D
end

# Definition of the tolerance for ODE45
global tol
tol = 1e-10;

N= 500; #size problem.
T = 0.5;#final time.

# Definitions
D = setupD(N)

global V_cos
DD = -N/2 : N/2-1;
dx = (2*pi*N^-1);
x = dx.*DD;

V_cos = Diagonal(1 .- cos.(x));

global fun
global funK
global funS
Q = randn(N,5);
Q = Q ./ norm(Q);

fun = X -> begin 
    return -(D*X + X*D') / 1im + V_cos*X*V_cos / 1im
end # + V_cos*X*V_cos; # this is the Schroedinger RHS

funK = (K, V) -> begin 
    return -(D*K + K*(V'*D'*V)) / 1im + V_cos*K*(V'*V_cos*V) / 1im
end

funK_pre = (K, VᵀDᵀV, VᵀVcosV) -> begin 
    return -(D*K + K*VᵀDᵀV) / 1im + V_cos*K*(VᵀVcosV) / 1im
end

funL = (L, U) -> begin 
    return -(L*(U'*D*U)' + D*L) / 1im + V_cos*L*(U'*V_cos*U) / 1im
end

funL_pre = (L, UᵀDU, UᵀVcosU) -> begin 
    return -(L*UᵀDU' + D*L) / 1im + V_cos*L*(UᵀVcosU) / 1im
end

funS = (U, S, V) -> begin 
    return -((U'*D*U)*S + S*(V'*D'*V)) / 1im + (U'*V_cos*U)*S*(V'*V_cos*V) / 1im
end

funS_pre = (S, UᵀDU, VᵀDᵀV, UᵀVcosU, VᵀVcosV) -> begin 
    return -(UᵀDU*S + S*VᵀDᵀV) / 1im + (UᵀVcosU*S*VᵀVcosV) / 1im
end

funKMain = (K, V) -> -(D*K + K*(V'*D'*V)) / 1im # + V_cos*X*V_cos; # this is the Schroedinger RHS
funLMain = (L, U) -> -(L*(U'*D*U)' + D*L) / 1im
#fun = X -> D*X + X*D' + Q*Q'; # this is the source RHS
#fun = X -> D*X + X*D' +Q*Q' + 0.1* ( D*X.^2 + X.^2*D' );

function runMethod(Y, T, h, stepFunction, ϑ)
    Max = round(T/h);
    if Max*h < T
        Max = Max+1
    end
    ranks = [size(Y[2], 1)]
    for i=1:Max
        ti = i*h

        if ti > T
            ti = T
        end
        Y = stepFunction(Y, (i-1)*h, ti, ϑ);
        push!(ranks, size(Y[3], 2))
    end
    return Y, ranks
end

function run(tt, r, ϑVec)

    # Initial Data:
    U0,_ = qr(rand(N,N)); U0 = Matrix(U0) .+ 0 * 1im
    V0,_ = qr(rand(N,N)); V0 = Matrix(V0) .+ 0 * 1im

    S0 = zeros(Complex,N,N);

    for i=1:N
        S0[i,i] = 10.0^-i;
    end

    norm¹ = norm(S0)
    norm² = norm(S0)^2

    # refSolerence solution:
    sol::Matrix{ComplexF64} = refSol(T, U0, V0, S0);

    #print("Norm error is ", abs(norm(sol) - norm(S0)))

    Error_1 = zeros(length(ϑVec),length(tt));
    Error_2 = zeros(length(ϑVec),length(tt));

    NormError = zeros(4,length(ϑVec),length(tt));
    t = zeros(length(tt))
    runTimes = zeros(2,length(ϑVec),length(tt));

    ranks = [ Int64[] for i in 1:length(ϑVec), j in 1:length(tt) ];

    index_h=1;
    for (i, h) in enumerate(tt)

        index_r=1;
        for (k, ϑ) in enumerate(ϑVec)

            UU0 = U0[:,1:r];
            VV0 = V0[:,1:r];
            SS0 = S0[1:r, 1:r];

            Y0 =[UU0, VV0, SS0];
            println("ϑ = ", ϑ, "; h = ", h)
            out, runTimes[1,k,i], memory_alloc, num_allocs = @timed runMethod(deepcopy(Y0), T, h, ParallelIntegrator_2nd_eff_precompute_adaptive, h^3 * ϑ)
            Y1_1 = out[1]
            ranks[k,i] = out[2]
            #push!(ranks_1, out[2])
            println("ranks: ", out[2])
            println("runtime parallel: ", runTimes[1,k,i])

            Error_1[index_r,index_h] = norm(sol .- buildMat(Y1_1)) ./ norm(sol) ;

            index_r = index_r+1;
        end

        t[index_h] = h;
        index_h = index_h+1;
    end
    return Error_1, runTimes, ranks
end

#idx = collect(range(0,-2,10))
idx = collect(range(0.1,-2.0,10))
tt = T.*10.0.^idx#[1 0.75 0.5 0.25 0.1 0.075 0.05 0.025 10^-2 10^-3];
ϑVec = [1, 1e-1, 1e-2];

Error_1, runTimes, ranks = run(tt, 15, ϑVec)

# Plotting:
nanidx1 = isnan.(Error_1)
idx1 = findall(x -> x < 10, Error_1)

ymax = 1.5 * maximum( [maximum(maximum(Error_1[idx1])) maximum(maximum(Error_1[idx1]))] );
ymin = 0.2 * minimum( [minimum(minimum(Error_1[idx1])) minimum(minimum(Error_1[idx1]))] );
ymin0 = 0.99 * minimum( [minimum(minimum(Error_1[idx1])) minimum(minimum(Error_1[idx1]))] );

#### Comparison RK4, aug. BUG ####
fig, ax4 = plt.subplots(1, 1,figsize=(10,10),dpi=100)
lt = ["k>:", "k*--", "ko-", "k^-."]
la = [L"par. 2nd, ϑ = h^3", L"par. 2nd, ϑ = h^3\cdot 10^{-1}", L"par. 2nd, ϑ = h^3\cdot 10^{-2}"]
for i = 1: size(Error_1,1)
    ax4.loglog(tt', Error_1[i,:], lt[i], label=la[i], markersize=10)
end

tt1 = tt./T
tt2 = tt1.^2
tt3 = tt1.^3
tt4 = tt1.^4

#ax4.loglog(tt',tt1, "k-", alpha=0.3)
ax4.loglog(tt',tt2, "r-", alpha=0.3)
#ax4.loglog(tt',tt3, "k-", alpha=0.3)
#ax4.loglog(tt',tt4, "k-", alpha=0.3)
#ax4.set_title("Runge-Kutta 4", fontsize=20)
ax4.tick_params("both",labelsize=20) 
ylim([ymin,ymax])
ax4.set_ylim([ymin,ymax])
ax4.set_xlabel("h", fontsize=20)
tight_layout()
legend(fontsize=20, loc="lower right")
plt.show()
savefig("errorParBUGvsMidBUG.pdf")


#### Comparison RK4, aug. BUG ####
nt = length(ranks[1,end])
fig, ax4 = plt.subplots(1, 1,figsize=(10,10),dpi=100)
lt = ["k:", "k--", "k-", "k-."]
la = [L"par. 2nd, ϑ = h^3, h = 5\cdot 10^{-2}", L"par. 2nd, ϑ = h^3\cdot 10^{-1}, h = 5\cdot 10^{-2}", L"par. 2nd, ϑ = h^3\cdot 10^{-2}, h = 5\cdot 10^{-2}"]
for i = 1: size(ranks,1)
    ax4.plot((2:nt)*0.005, ranks[i,end][2:end], lt[i], label=la[i], markersize=10)
end

ax4.tick_params("both",labelsize=20) 
#ylim([ymin,ymax])
#ax4.set_ylim([ymin,ymax])
ax4.set_ylabel("rank", fontsize=20)
ax4.set_xlabel("t", fontsize=20)
tight_layout()
legend(fontsize=20, loc="lower right")
plt.show()
savefig("rankParBUGvsMidBUG.pdf")
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

N= 100; #size problem.
T = 1.0;#final time.

# Definitions
D = setupD(N)

global V_cos
DD = -N/2 : N/2-1;
dx = (2*pi*N^-1);
x = dx.*DD;

V_cos = Diagonal(1 .- cos.(x));

global fun
Q = randn(N,5);
Q = Q ./ norm(Q);

fun = X -> begin 
    return -(D*X + X*D') / 1im + V_cos*X*V_cos / 1im
end # + V_cos*X*V_cos; # this is the Schroedinger RHS
funKMain = (K, V) -> -(D*K + K*(V'*D'*V)) / 1im # + V_cos*X*V_cos; # this is the Schroedinger RHS
funLMain = (L, U) -> -(L*(U'*D*U)' + D*L) / 1im
#fun = X -> D*X + X*D' + Q*Q'; # this is the source RHS
#fun = X -> D*X + X*D' +Q*Q' + 0.1* ( D*X.^2 + X.^2*D' );

function run(tt, rVec)

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

    Error_BUG = zeros(length(rVec),length(tt));
    Error_unconv_adapt = zeros(length(rVec),length(tt));
    Error_new = zeros(length(rVec),length(tt));
    Error_3rd = zeros(length(rVec),length(tt));

    NormError = zeros(4,length(rVec),length(tt));
    t = zeros(length(tt))

    index_h=1;
    for h = tt

        index_r=1;
        for r in rVec

            UU0 = U0[:,1:r];
            VV0 = V0[:,1:r];
            SS0 = S0[1:r, 1:r];

            Y0 =[UU0, VV0, SS0];

            Y1_BUG = deepcopy(Y0);
            Y1_new = deepcopy(Y0);
            Y1_unconv_adpt = deepcopy(Y0);
            Y1_3rd = deepcopy(Y0);

            Max = round(T/h);
            if Max*h < T
                Max = Max+1
            end
            for i=1:Max
                ti = i*h

                if ti > T
                    ti = T
                end
                Y1_new = ParallelIntegrator(Y1_new, (i-1)*h, ti);
                Y1_BUG = ParallelIntegrator_2nd_3r(Y1_BUG, (i-1)*h, ti);
                #Y1_unconv_adpt .= Y1_BUG
                #Y1_new .= Y1_BUG
                Y1_unconv_adpt = MidpointBUG4r(Y1_unconv_adpt, (i-1)*h, ti);

                Y1_3rd = ParallelIntegrator_2nd(Y1_3rd, (i-1)*h, ti);
                #Y1_3rd = ParallelIntegrator_2nd_midpoint(Y1_3rd, (i-1)*h, ti);
            end

            Error_BUG[index_r,index_h] = norm(sol .- buildMat(Y1_BUG)) ./ norm(sol) ;

            Error_unconv_adapt[index_r,index_h] = norm(sol .- buildMat(Y1_unconv_adpt)) ./ norm(sol) ;

            Error_new[index_r,index_h] = norm(sol .- buildMat(Y1_new)) ./ norm(sol) ;

            Error_3rd[index_r,index_h] = norm(sol .- buildMat(Y1_3rd)) ./ norm(sol) ;

            NormError[1,index_r,index_h] = abs(norm(buildMat(Y1_new)) - norm¹)
            NormError[2,index_r,index_h] = abs(norm(buildMat(Y1_BUG)) - norm¹)
            NormError[3,index_r,index_h] = abs(norm(buildMat(Y1_3rd)) - norm¹)
            NormError[4,index_r,index_h] = abs(norm(buildMat(Y1_unconv_adpt)) - norm¹)

            index_r = index_r+1;
        end

        t[index_h] = h;
        index_h = index_h+1;
    end
    return Error_BUG, Error_unconv_adapt, Error_new, Error_3rd, NormError
end

#idx = collect(range(0,-2,10))
idx = collect(range(0,-2,10))
tt = T.*10.0.^idx#[1 0.75 0.5 0.25 0.1 0.075 0.05 0.025 10^-2 10^-3];
rVec = [5 10 15];

Error_BUG, Error_unconv_adapt, Error_new, Error_3rd, NormError = run(tt, rVec)

# Plotting:
ymax = 1.5 * maximum( [maximum(maximum(Error_BUG)) maximum(maximum(Error_new)) maximum(maximum(Error_unconv_adapt)) maximum(maximum(Error_3rd)) ] );
ymin = 0.9 * minimum( [minimum(minimum(Error_BUG)) minimum(minimum(Error_new)) minimum(minimum(Error_unconv_adapt)) minimum(minimum(Error_3rd)) ] );


fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4,figsize=(15,10),dpi=100)
lt = ["k>:", "k*--", "ko-", "k^-."]
for i = 1: size(Error_BUG,1)
    ax1.loglog(tt', Error_new[i,:], lt[i], markersize=10)
    ax2.loglog(tt', Error_BUG[i,:], lt[i], markersize=10)
    ax3.loglog(tt', Error_3rd[i,:], lt[i], markersize=10)
    ax4.loglog(tt', Error_unconv_adapt[i,:], lt[i], markersize=10)
end
tt1 = tt./T
tt2 = tt1.^2
tt3 = tt1.^3

ax1.loglog(tt',tt1, "g-")
ax1.loglog(tt',tt2, "r-")
ax1.loglog(tt',tt3, "b-")
ax2.loglog(tt',tt1, "g-")
ax2.loglog(tt',tt2, "r-")
ax2.loglog(tt',tt3, "b-")
ax3.loglog(tt',tt1, "g-")
ax3.loglog(tt',tt2, "r-")
ax3.loglog(tt',tt3, "b-")
ax4.loglog(tt',tt1, "g-")
ax4.loglog(tt',tt2, "r-")
ax4.loglog(tt',tt3, "b-")
ax1.set_title(L"parallel, $1^{st}$", fontsize=20)
ax2.set_title(L"parallel, $2^{nd}$, version 1", fontsize=20)
ax3.set_title(L"parallel, $2^{nd}$, version 2", fontsize=20)
ax4.set_title(L"midpoint BUG, $4r$", fontsize=20)
ax2.set_yticks([])
ax3.set_yticks([])
ax4.set_yticks([])
ax1.tick_params("both",labelsize=15) 
ax2.tick_params("both",labelsize=15) 
ax3.tick_params("both",labelsize=15) 
ax4.tick_params("both",labelsize=15) 
ylim([ymin,ymax])
ax1.set_ylim([ymin,ymax])
ax2.set_ylim([ymin,ymax])
ax3.set_ylim([ymin,ymax])
ax4.set_ylim([ymin,ymax])
ax1.set_xlabel("h", fontsize=15)
ax2.set_xlabel("h", fontsize=15)
ax3.set_xlabel("h", fontsize=15)
ax4.set_xlabel("h", fontsize=15)
tight_layout()
plt.show()
savefig("error.pdf")

ymax = 4.0 * maximum( NormError );
ymin = 0.9 * minimum( NormError );


fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4,figsize=(15,10),dpi=100)
lt = ["k>:", "k*--", "ko-", "k^-."]
for i = 1: size(Error_BUG,1)
    ax1.loglog(tt', NormError[1,i,:], lt[i], markersize=10)
    ax2.loglog(tt', NormError[2,i,:], lt[i], markersize=10)
    ax3.loglog(tt', NormError[3,i,:], lt[i], markersize=10)
    ax4.loglog(tt', NormError[4,i,:], lt[i], markersize=10)
end
tt1 = 0.09*tt./T
tt2 = tt1.^2
tt3 = tt1.^3
tt4 = tt1.^4

ax1.loglog(tt',tt1, "g-")
ax1.loglog(tt',tt2, "r-")
ax1.loglog(tt',tt3, "b-")
ax1.loglog(tt',tt4, "m-")
ax2.loglog(tt',tt1, "g-")
ax2.loglog(tt',tt2, "r-")
ax2.loglog(tt',tt3, "b-")
ax2.loglog(tt',tt4, "m-")
ax3.loglog(tt',tt1, "g-")
ax3.loglog(tt',tt2, "r-")
ax3.loglog(tt',tt3, "b-")
ax3.loglog(tt',tt4, "m-")
ax4.loglog(tt',tt1, "g-")
ax4.loglog(tt',tt2, "r-")
ax4.loglog(tt',tt3, "b-")
ax4.loglog(tt',tt4, "m-")
ax1.set_title(L"parallel, $1^{st}$", fontsize=20)
ax2.set_title(L"parallel, $2^{nd}$, version 1", fontsize=20)
ax3.set_title(L"parallel, $2^{nd}$, version 2", fontsize=20)
ax4.set_title(L"midpoint BUG, $4r$", fontsize=20)
ax2.set_yticks([])
ax3.set_yticks([])
ax4.set_yticks([])
ax1.tick_params("both",labelsize=15) 
ax2.tick_params("both",labelsize=15) 
ax3.tick_params("both",labelsize=15) 
ax4.tick_params("both",labelsize=15) 
ylim([ymin,ymax])
ax1.set_ylim([ymin,ymax])
ax2.set_ylim([ymin,ymax])
ax3.set_ylim([ymin,ymax])
ax4.set_ylim([ymin,ymax])
ax1.set_xlabel("h", fontsize=15)
ax2.set_xlabel("h", fontsize=15)
ax3.set_xlabel("h", fontsize=15)
ax4.set_xlabel("h", fontsize=15)
tight_layout()
plt.show()
savefig("normPreservation.pdf")



using ODE
using PyCall
np = pyimport("numpy")

function ParallelIntegrator(Y0, t0, t1)

    global fun 
    global tol

    # Initial values
    U_0 = Y0[1];
    V_0 = Y0[2];
    S_0 = Y0[3];

    r = size(U_0,2);
    h = t1-t0;

    K_0::Matrix{ComplexF64} = U_0*S_0;

    # K-step:
    funK = K -> 
    begin 
        Y_0::Matrix{ComplexF64} = K*V_0';
        out::Matrix{ComplexF64} = fun(Y_0)*V_0 
        return out
    end;
    K_1 = odeSolver(funK, K_0, h);
    U_1,_ = np.linalg.qr([U_0 K_1], mode="reduced");
    UTilde = U_1[:, (r+1):end];
    U_1 = Matrix([U_0 UTilde]);

    # L-step:
    funL = L -> 
    begin 
        Y_0::Matrix{ComplexF64} = U_0*L';
        out::Matrix{ComplexF64} = fun(Y_0)'*U_0 
        return out
    end;
    L_1 = odeSolver(funL, V_0*S_0', h);
    V_1, _ = np.linalg.qr([V_0 L_1], mode="reduced");
    VTilde = V_1[:, (r+1):end];
    V_1 = Matrix([V_0 VTilde]);

    # S-step:
    funS = S -> 
    begin 
        Y_0::Matrix{ComplexF64} = U_0*S*V_0';
        out::Matrix{ComplexF64} = U_0'*fun(Y_0)*V_0;
        return out
    end;
    S_1 = odeSolver(funS, S_0, h);

    hatS = zeros(ComplexF64, 2 * r, 2 * r);

    hatS[1:r,1:r] .= ComplexF64.(S_1);
    hatS[(r+1):end,1:r] .= ComplexF64.(UTilde'*K_1);
    hatS[1:r,(r+1):end] .= ComplexF64.(L_1' * VTilde);

    return truncMat([ComplexF64.(U_1),ComplexF64.(V_1),ComplexF64.(hatS)],r);
end

function ParallelIntegrator_2nd(Y0, t0, t1)

    global fun 
    global tol

    # Initial values
    U_0 = Y0[1];
    V_0 = Y0[2];
    S_0 = Y0[3];

    Y_0::Matrix{ComplexF64} = U_0*S_0*V_0'

    # parameters
    m = size(U_0,1);
    n = size(V_0,1);
    h = t1-t0;
    r = size(U_0,2);

    hatU_0, _ = np.linalg.qr([U_0 fun(Y_0)*V_0], mode="reduced");
    hatV_0, _ = np.linalg.qr([V_0 fun(Y_0)'*U_0], mode="reduced");
    hatU_0[:, 1:r] = U_0 # this is very important!!!
    hatV_0[:, 1:r] = V_0 # this is very important!!!
    
    # K-step
    K_0 = [U_0*S_0 zeros(m, r)]
    funK = K -> fun(K*hatV_0')*hatV_0
    K_1 = odeSolver(funK, K_0, h)
    hatU_1,_ = np.linalg.qr([hatU_0 K_1], mode="reduced");
    hatU_1 = [hatU_0 hatU_1[:,(2*r+1):4*r]];
    tildeU2 = hatU_1[:,(2*r+1):4*r]

    # L-step
    L_0 = [V_0*S_0' zeros(n, r)]
    funL = L -> fun(hatU_0*L')'*hatU_0
    L_1 = odeSolver(funL, L_0, h)
    hatV_1,_ = np.linalg.qr([hatV_0 L_1], mode="reduced");
    hatV_1 = [hatV_0 hatV_1[:,(2*r+1):4*r]];
    tildeV2 = hatV_1[:,(2*r+1):4*r]

    # Sbar-step:
    Sbar_0 = (hatU_0'* U_0)* S_0 * (hatV_0'*V_0)';
    funSbar = S -> hatU_0'* fun(hatU_0*S*hatV_0')*hatV_0;
    Sbar_1 = odeSolver(funSbar, Sbar_0, h);

    #S_1 = Sbar_1
    #hatU_1 = hatU_0
    #hatV_1 = hatV_0
    S_1 = zeros(ComplexF64, 4*r, 4*r)
    S_1[1:2*r,1:2*r] .= Sbar_1
    S_1[1:2*r,(2*r + 1):4*r] .= L_1'*tildeV2
    S_1[(2*r + 1):4*r,1:2*r] .= tildeU2'*K_1

    return truncMat([hatU_1,hatV_1,S_1],r);
end

# the 3r variant of the 2nd order parallel integrator
function ParallelIntegrator_2nd_3r(Y0, t0, t1)

    global fun 
    global tol

    # Initial values
    U_0 = Y0[1];
    V_0 = Y0[2];
    S_0 = Y0[3];

    Y_0::Matrix{ComplexF64} = U_0*S_0*V_0'

    # parameters
    m = size(U_0,1);
    n = size(V_0,1);
    h = t1-t0;
    r = size(U_0,2);

    hatU_0, _ = np.linalg.qr([U_0 fun(Y_0)*V_0], mode="reduced");
    hatV_0, _ = np.linalg.qr([V_0 fun(Y_0)'*U_0], mode="reduced");
    hatU_0[:, 1:r] = U_0 # this is very important!!!
    hatV_0[:, 1:r] = V_0 # this is very important!!!
    
    # K-step
    V_star, _ = np.linalg.qr(V_0 * S_0' + 0.5*h* fun(Y_0)'*U_0, mode="reduced");
    K_0 = [U_0*S_0 zeros(m, r)]#*(V_0'*hatV_0)
    #K_1 = K_0 + h * fun(Y_0 + h*fun(Y_0 + 0.5*h*fun(Y_0)*hatV_0*hatV_0')*hatV_0*hatV_0')*hatV_0
    funK = K -> fun(K*hatV_0')*hatV_0
    K_1 = odeSolver(funK, K_0, h)
    hatU_1,_ = np.linalg.qr([hatU_0 K_1*hatV_0'*V_star], mode="reduced");
    hatU_1 = [hatU_0 hatU_1[:,(2*r+1):3*r]];
    tildeU2 = hatU_1[:,(2*r+1):3*r]

    # L-step
    U_star, _ = np.linalg.qr(U_0 * S_0 + 0.5*h* fun(Y_0)*V_0, mode="reduced");
    L_0 = [V_0*S_0' zeros(n, r)]#*(U_0'*hatU_0)
    #L_1 = L_0 + h * fun(Y_0 + h*hatU_0*hatU_0'*fun(Y_0 + 0.5*h*hatU_0*hatU_0'*fun(Y_0)))'*hatU_0
    funL = L -> fun(hatU_0*L')'*hatU_0
    L_1 = odeSolver(funL, L_0, h)
    hatV_1,_ = np.linalg.qr([hatV_0 L_1*hatU_0'*U_star], mode="reduced");
    hatV_1 = [hatV_0 hatV_1[:,(2*r+1):3*r]];
    tildeV2 = hatV_1[:,(2*r+1):3*r]

    # Sbar-step:
    Sbar_0 = (hatU_0'* U_0)* S_0 * (hatV_0'*V_0)';
    funSbar = S -> hatU_0'* fun(hatU_0*S*hatV_0')*hatV_0;
    Sbar_1 = odeSolver(funSbar, Sbar_0, h);

    S_1 = zeros(ComplexF64, 3*r, 3*r)
    S_1[1:2*r,1:2*r] .= Sbar_1
    S_1[1:2*r,(2*r + 1):3*r] .= L_1'*tildeV2
    S_1[(2*r + 1):3*r,1:2*r] .= tildeU2'*K_1

    return truncMat([hatU_1,hatV_1,S_1],r);
end

function MidpointBUG4r(Y0, t0, t1, augment=true)

    global fun
    global tol

    # Initial values
    U_0 = Y0[1];
    V_0 = Y0[2];
    S_0 = Y0[3];

    # parameters
    h = t1-t0;
    r = size(U_0,2);

    # Midpoint
    Y12 = Method_augmented_BUG_ode(Y0, t0, t0 + 0.5*h, augment)
    U12 = Y12[1];
    V12 = Y12[2];
    S12 = Y12[3];

    # basis augmentation
    if augment
        hatU_1,_ = np.linalg.qr([U12 fun(U12*S12*V12')*V12], mode="reduced"); #println("augment ", size(hatU_1), " vs ", 4*r)#hatU_1 = Matrix(hatU_1[:,1:4*r])
        hatV_1,_ = np.linalg.qr([V12 fun(U12*S12*V12')'*U12], mode="reduced"); #hatV_1 = Matrix(hatV_1[:,1:4*r])
    else
        hatU_1,_ = np.linalg.qr([U_0 U12 fun(U12*S12*V12')*V12], mode="reduced"); #println("no augment ", size(hatU_1), " vs ", 3*r)# hatU_1 = Matrix(hatU_1[:,1:3*r])
        hatV_1,_ = np.linalg.qr([V_0 V12 fun(U12*S12*V12')'*U12], mode="reduced"); #hatV_1 = Matrix(hatV_1[:,1:3*r])
    end

    # S-step:
    S_0 = (hatU_1'* U_0)* S_0 * (hatV_1'*V_0)';
    funS = S -> hatU_1'* fun(hatU_1*S*hatV_1')*hatV_1;
    S_1 = odeSolver(funS, S_0, h);

    #Y1 = roundMat({hatU_1,hatV_1,S_1},tol);
    return truncMat([hatU_1,hatV_1,S_1],r);
end

## Extra functions
function rk(f, Y, h, order=1)
    if order == 1
        return Y + h*f(Y)
    elseif order == 2
        return Y + h* f(Y + 0.5 * h * f(Y));
    else
        print("rk not implemented")
    end
end

function odeSolver(f, Y, h)
    global tol
    N = size(Y);
    F(y,p,t) = f(y)
    tspan = [0,h]
    Yin::Matrix{ComplexF64} = Y
    prob = ODEProblem(F,Yin,tspan)
    yout = []
    @suppress begin
        tout, yout = solve(prob,ode45(),reltol=tol,abstol=tol)
    end

    return Matrix(yout[:,:,end]);
end

function calcSlopes(t, y)

    for i=1:size(y,1)

        slopeTab = diff(log10.(y[i,:])) ./ diff(log10.(t));
        slopes[i] = maximum(slopeTab);

        #slope = polyfit(log10(t), log10(y(i,:)), 1);
        #slopes(i) = slope(1);
    end
    return slopes
end

function buildMat(X)
    U = X[1];
    V = X[2];
    S = X[3];
    Y::Matrix{ComplexF64} = U*S*V'

    return Y;
end

function roundMat(Y,tol)

    global maxRank

    U_1 = Y[1];
    V_1 = Y[2];
    S = Y[end];


    U,S,V = svd(S);

    tol = tol*norm(S);


    sg = diag(S);
    rmax = size(sg,1);

    for j=1:rmax
        tmp = sqrt(sum(sg(j:rmax)).^2);
        if(tmp<tol)
            break;
        end
    end

    rmax = j;
    rmax = min(rmax,maxRank);

    # Truncation:
    U_1 = U_1*U;
    V_1 = V_1*V;

    S_1 = S[1:rmax,1:rmax];
    U_1 = U_1[:,1:rmax];
    V_1 = V_1[:,1:rmax];

    return [U_1,V_1,S_1];
end

function truncMat(Y,r)

    U_1 = Y[1];
    V_1 = Y[2];
    S = Y[3];

    U,S,V = svd(S);

    rmax = r;

    # Truncation:
    U_1 = U_1*U;
    V_1 = V_1*V;

    S_1 = diagm(S[1:rmax]);
    U_1 = U_1[:,1:rmax];
    V_1 = V_1[:,1:rmax];

    return [U_1,V_1,S_1];
end


## Reference solution

function refSol(t,U0, V0, S0)

    global fun

    Y0 = U0*S0*V0';
    return refSolver(fun, Y0, 0, t);
end

function refSolver(fun, y0, t0, t1)
    global tol
    N = size(y0);
    f(y,p,t) = F(t,y, fun, N)
    tspan = (t0,t1)
    prob = ODEProblem(f,y0,tspan)
    yout = []
    @suppress begin
        tout, yout = solve(prob,ode45(),reltol=tol,abstol=tol)
    end

    return yout[:,:,end];
end


function F(t,y, fun, N)

    X = reshape(y, N);
    tmp = fun(X);
    return tmp[:];
end

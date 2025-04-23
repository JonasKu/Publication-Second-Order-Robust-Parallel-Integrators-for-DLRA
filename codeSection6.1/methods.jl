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


function ParallelIntegrator_2nd_eff(Y0, t0, t1)
    global funK 
    global funL 
    global funS
    global tol

    # Initial values
    U_0 = ComplexF64.(Y0[1]);
    V_0 = ComplexF64.(Y0[2]);
    S_0 = ComplexF64.(Y0[3]);

    K_0::Matrix{ComplexF64} = U_0*S_0
    L_0::Matrix{ComplexF64} = V_0*S_0'

    # parameters
    m = size(U_0,1);
    n = size(V_0,1);
    h = t1-t0;
    r = size(U_0,2);

    F₀V₀ = funK(K_0, V_0)
    F₀ᵀU₀::Matrix{ComplexF64} = funL(L_0, U_0)
    hatU_0, _ = np.linalg.qr([U_0 F₀V₀], mode="reduced");
    hatV_0, _ = np.linalg.qr([V_0 F₀ᵀU₀], mode="reduced");
    hatU_0[:, 1:r] = U_0 # this is very important!!!
    hatV_0[:, 1:r] = V_0 # this is very important!!!
    
    # K-step
    K_0 = [U_0*S_0 zeros(m, r)]
    fK = K -> funK(K, hatV_0)
    K_1 = odeSolver(fK, K_0, h)
    hatU_1,_ = np.linalg.qr([hatU_0 K_1], mode="reduced");
    hatU_1 = [hatU_0 hatU_1[:,(2*r+1):4*r]];
    tildeU2 = hatU_1[:,(2*r+1):4*r]

    # L-step
    L_0 = [V_0*S_0' zeros(n, r)]#*(U_0'*hatU_0)
    fL = L -> funL(L, hatU_0)
    L_1 = odeSolver(fL, L_0, h)
    hatV_1,_ = np.linalg.qr([hatV_0 L_1], mode="reduced");
    hatV_1 = [hatV_0 hatV_1[:,(2*r+1):4*r]];
    tildeV2 = hatV_1[:,(2*r+1):4*r]

    # Sbar-step:
    Sbar_0 = (hatU_0'* U_0)* S_0 * (hatV_0'*V_0)';
    funSbar = S -> funS(hatU_0, S, hatV_0);
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

function ParallelIntegrator_2nd_eff_precompute(Y0, t0, t1)
    global funK 
    global funL 
    global funS
    global tol

    # Initial values
    U_0 = ComplexF64.(Y0[1]);
    V_0 = ComplexF64.(Y0[2]);
    S_0 = ComplexF64.(Y0[3]);

    K_0::Matrix{ComplexF64} = U_0*S_0
    L_0::Matrix{ComplexF64} = V_0*S_0'

    # parameters
    m = size(U_0,1);
    n = size(V_0,1);
    h = t1-t0;
    r = size(U_0,2);

    F₀V₀ = funK(K_0, V_0)
    F₀ᵀU₀::Matrix{ComplexF64} = funL(L_0, U_0)
    hatU_0, _ = np.linalg.qr([U_0 F₀V₀], mode="reduced");
    hatV_0, _ = np.linalg.qr([V_0 F₀ᵀU₀], mode="reduced");
    hatU_0[:, 1:r] = U_0 # this is very important!!!
    hatV_0[:, 1:r] = V_0 # this is very important!!!

    # precompute projections
    UᵀDU = hatU_0'*D*hatU_0
    VᵀDᵀV = hatV_0'*D'*hatV_0
    UᵀVcosU = hatU_0'*V_cos*hatU_0
    VᵀVcosV = hatV_0'*V_cos*hatV_0
    
    # K-step
    K_0 = [U_0*S_0 zeros(m, r)]
    fK = K -> funK_pre(K, VᵀDᵀV, VᵀVcosV)
    K_1 = rk(fK, K_0, h)
    hatU_1,_ = np.linalg.qr([hatU_0 K_1], mode="reduced");
    hatU_1 = [hatU_0 hatU_1[:,(2*r+1):4*r]];
    tildeU2 = hatU_1[:,(2*r+1):4*r]

    # L-step
    L_0 = [V_0*S_0' zeros(n, r)]#*(U_0'*hatU_0)
    fL = L -> funL_pre(L, UᵀDU, UᵀVcosU)
    L_1 = rk(fL, L_0, h)
    hatV_1,_ = np.linalg.qr([hatV_0 L_1], mode="reduced");
    hatV_1 = [hatV_0 hatV_1[:,(2*r+1):4*r]];
    tildeV2 = hatV_1[:,(2*r+1):4*r]

    # Sbar-step:
    Sbar_0 = (hatU_0'* U_0)* S_0 * (hatV_0'*V_0)';
    funSbar = S -> funS_pre(S, UᵀDU, VᵀDᵀV, UᵀVcosU, VᵀVcosV);
    Sbar_1 = rk(funSbar, Sbar_0, h);

    S_1 = zeros(ComplexF64, 4*r, 4*r)
    S_1[1:2*r,1:2*r] .= Sbar_1
    S_1[1:2*r,(2*r + 1):4*r] .= L_1'*tildeV2
    S_1[(2*r + 1):4*r,1:2*r] .= tildeU2'*K_1

    return truncMat([hatU_1,hatV_1,S_1],r);
end

function ParallelIntegrator_2nd_eff_precompute_adaptive(Y0, t0, t1, ϑ=0.00001)
    global funK 
    global funL 
    global funS
    global tol

    # Initial values
    U_0 = ComplexF64.(Y0[1]);
    V_0 = ComplexF64.(Y0[2]);
    S_0 = ComplexF64.(Y0[3]);

    r = size(S_0, 1)

    K_0::Matrix{ComplexF64} = U_0*S_0
    L_0::Matrix{ComplexF64} = V_0*S_0'

    # parameters
    m = size(U_0,1);
    n = size(V_0,1);
    h = t1-t0;
    r = size(U_0,2);

    F₀V₀ = funK(K_0, V_0)
    F₀ᵀU₀::Matrix{ComplexF64} = funL(L_0, U_0)
    hatU_0, _ = np.linalg.qr([U_0 F₀V₀], mode="reduced");
    hatV_0, _ = np.linalg.qr([V_0 F₀ᵀU₀], mode="reduced");
    hatU_0[:, 1:r] = U_0 # this is very important!!!
    hatV_0[:, 1:r] = V_0 # this is very important!!!

    # precompute projections
    UᵀDU = hatU_0'*D*hatU_0
    VᵀDᵀV = hatV_0'*D'*hatV_0
    UᵀVcosU = hatU_0'*V_cos*hatU_0
    VᵀVcosV = hatV_0'*V_cos*hatV_0
    
    # K-step
    K_0 = [U_0*S_0 zeros(m, r)]
    fK = K -> funK_pre(K, VᵀDᵀV, VᵀVcosV)
    K_1 = rk(fK, K_0, h)
    hatU_1,_ = np.linalg.qr([hatU_0 K_1], mode="reduced");
    hatU_1 = [hatU_0 hatU_1[:,(2*r+1):4*r]];
    tildeU2 = hatU_1[:,(2*r+1):4*r]

    # L-step
    L_0 = [V_0*S_0' zeros(n, r)]#*(U_0'*hatU_0)
    fL = L -> funL_pre(L, UᵀDU, UᵀVcosU)
    L_1 = rk(fL, L_0, h)
    hatV_1,_ = np.linalg.qr([hatV_0 L_1], mode="reduced");
    hatV_1 = [hatV_0 hatV_1[:,(2*r+1):4*r]];
    tildeV2 = hatV_1[:,(2*r+1):4*r]

    # Sbar-step:
    Sbar_0 = (hatU_0'* U_0)* S_0 * (hatV_0'*V_0)';
    funSbar = S -> funS_pre(S, UᵀDU, VᵀDᵀV, UᵀVcosU, VᵀVcosV);
    Sbar_1 = rk(funSbar, Sbar_0, h);

    S_1 = zeros(ComplexF64, 4*r, 4*r)
    S_1[1:2*r,1:2*r] .= Sbar_1
    S_1[1:2*r,(2*r + 1):4*r] .= L_1'*tildeV2
    S_1[(2*r + 1):4*r,1:2*r] .= tildeU2'*K_1

    return truncMat([hatU_1,hatV_1,S_1],r, ϑ);
end

# the 3r variant of the 2nd order parallel integrator
function ParallelIntegrator_2nd_3r(Y0, t0, t1)

    global fun 
    global funK 
    global funL 
    global funS
    global tol

    # Initial values
    U_0 = ComplexF64.(Y0[1]);
    V_0 = ComplexF64.(Y0[2]);
    S_0 = ComplexF64.(Y0[3]);

    #Y_0::Matrix{ComplexF64} = U_0*S_0*V_0'
    K_0::Matrix{ComplexF64} = U_0*S_0
    L_0::Matrix{ComplexF64} = V_0*S_0'

    # parameters
    m = size(U_0,1);
    n = size(V_0,1);
    h = t1-t0;
    r = size(U_0,2);

    F₀V₀::Matrix{ComplexF64} = funK(K_0, V_0)
    F₀ᵀU₀::Matrix{ComplexF64} = funL(L_0, U_0)
    hatU_0, _ = np.linalg.qr([U_0 F₀V₀], mode="reduced");
    hatV_0, _ = np.linalg.qr([V_0 F₀ᵀU₀], mode="reduced");
    hatU_0[:, 1:r] = U_0 # this is very important!!!
    hatV_0[:, 1:r] = V_0 # this is very important!!!
    
    # K-step
    V_star, _ = np.linalg.qr(V_0 * S_0' + 0.5*h* F₀ᵀU₀, mode="reduced");
    K_0 = [U_0*S_0 zeros(m, r)]#*(V_0'*hatV_0)
    #K_1 = K_0 + h * fun(Y_0 + h*fun(Y_0 + 0.5*h*fun(Y_0)*hatV_0*hatV_0')*hatV_0*hatV_0')*hatV_0
    #funK = K -> fun(K*hatV_0')*hatV_0
    fK = K -> funK(K, hatV_0)
    K_1 = odeSolver(fK, K_0, h)
    hatU_1,_ = np.linalg.qr([hatU_0 K_1*hatV_0'*V_star], mode="reduced");
    hatU_1 = [hatU_0 hatU_1[:,(2*r+1):3*r]];
    tildeU2 = hatU_1[:,(2*r+1):3*r]

    # L-step
    U_star, _ = np.linalg.qr(U_0 * S_0 + 0.5*h* F₀V₀, mode="reduced");
    L_0 = [V_0*S_0' zeros(n, r)]#*(U_0'*hatU_0)
    #L_1 = L_0 + h * fun(Y_0 + h*hatU_0*hatU_0'*fun(Y_0 + 0.5*h*hatU_0*hatU_0'*fun(Y_0)))'*hatU_0
    #funL = L -> fun(hatU_0*L')'*hatU_0
    fL = L -> funL(L, hatU_0)
    L_1 = odeSolver(fL, L_0, h)
    hatV_1,_ = np.linalg.qr([hatV_0 L_1*hatU_0'*U_star], mode="reduced");
    hatV_1 = [hatV_0 hatV_1[:,(2*r+1):3*r]];
    tildeV2 = hatV_1[:,(2*r+1):3*r]

    # Sbar-step:
    Sbar_0 = (hatU_0'* U_0)* S_0 * (hatV_0'*V_0)';
    funSbar = S -> funS(hatU_0, S, hatV_0);
    Sbar_1 = odeSolver(funSbar, Sbar_0, h);

    S_1 = zeros(ComplexF64, 3*r, 3*r)
    S_1[1:2*r,1:2*r] .= Sbar_1
    S_1[1:2*r,(2*r + 1):3*r] .= L_1'*tildeV2
    S_1[(2*r + 1):3*r,1:2*r] .= tildeU2'*K_1

    return truncMat([hatU_1,hatV_1,S_1],r);
end


## Rank-adaptive unconventional Integrator
function Method_augmented_BUG_ode(Y0, t0, t1, augment=false)

    global fun
    global tol

    # Initial values
    U_0 = Y0[1];
    V_0 = Y0[2];
    S_0 = Y0[3];

    r = size(U_0,2);
    h = t1-t0;

    # K-step:
    funK = K ->
    begin
        Y_0::Matrix{ComplexF64} = K*V_0';
        out::Matrix{ComplexF64} = fun(Y_0)*V_0
        return out
    end;

    K_0::Matrix{ComplexF64} = U_0*S_0;
    K_1 = odeSolver(funK, K_0 .+ 0.0*1im, h);


    K_1 = [K_1 U_0];
    U_1,_ = qr(K_1);
    U_1 = Matrix(U_1[:,1:2*r])

    # L-step:
    funL = L ->
    begin
        Y_0::Matrix{ComplexF64} = U_0*L';
        out::Matrix{ComplexF64} = fun(Y_0)'*U_0
        return out
    end;

    K_0 = V_0*S_0';
    K_1 = odeSolver(funL, K_0, h);

    K_1 = [K_1 V_0];
    V_1, _ = qr(K_1);
    V_1 = Matrix(V_1[:,1:2*r])


    # S-step:
    funS = S ->
    begin
        Y_0::Matrix{ComplexF64} = U_1*S*V_1';
        out::Matrix{ComplexF64} = U_1'*fun(Y_0)*V_1;
        return out
    end;

    S_0 = (U_1'*U_0) * S_0 * (V_1'*V_0)';
    S_1 = odeSolver(funS, S_0, h);

    #Y1 = roundMat({U_1, V_1, S_1}, tol);
    if augment
        return [U_1,V_1,S_1];
    else
        return truncMat([U_1,V_1,S_1],r);
    end

end

## Rank-adaptive unconventional Integrator
function Method_augmented_BUG_ode_eff(Y0, t0, t1, augment=false)

    global fun
    global tol
    global funK 
    global funL 
    global funS

    # Initial values
    U_0 = ComplexF64.(Y0[1]);
    V_0 = ComplexF64.(Y0[2]);
    S_0 = ComplexF64.(Y0[3]);

    K_0::Matrix{ComplexF64} = U_0*S_0
    L_0::Matrix{ComplexF64} = V_0*S_0'

    r = size(U_0,2);
    h = t1-t0;

    # K-step:
    fK = K -> funK(K, V_0)
    K_1 = odeSolver(fK, K_0, h)
    K_1 = [K_1 U_0];
    U_1,_ = qr(K_1);
    U_1 = Matrix(U_1[:,1:2*r])

    # L-step:
    fL = L -> funL(L, U_0)
    L_1 = odeSolver(fL, L_0, h)
    L_1 = [L_1 V_0];
    V_1, _ = qr(L_1);
    V_1 = Matrix(V_1[:,1:2*r])


    # S-step:
    fS = S -> funS(U_1, S, V_1);
    S_0 = (U_1'*U_0) * S_0 * (V_1'*V_0)';
    S_1 = odeSolver(fS, S_0, h);

    if augment
        return [U_1,V_1,S_1];
    else
        return truncMat([U_1,V_1,S_1],r);
    end

end

function Method_augmented_BUG_ode_eff_precompute(Y0, t0, t1, augment=false)

    global fun
    global tol
    global funK 
    global funL 
    global funS

    # Initial values
    U_0 = ComplexF64.(Y0[1]);
    V_0 = ComplexF64.(Y0[2]);
    S_0 = ComplexF64.(Y0[3]);

    K_0::Matrix{ComplexF64} = U_0*S_0
    L_0::Matrix{ComplexF64} = V_0*S_0'

    r = size(U_0,2);
    h = t1-t0;

    # precompute projections
    UᵀDU = U_0'*D*U_0
    VᵀDᵀV = V_0'*D'*V_0
    UᵀVcosU = U_0'*V_cos*U_0
    VᵀVcosV = V_0'*V_cos*V_0

    # K-step:
    fK = K -> funK_pre(K, VᵀDᵀV, VᵀVcosV)
    K_1 = rk(fK, K_0, h)
    K_1 = [K_1 U_0];
    U_1,_ = qr(K_1);
    U_1 = Matrix(U_1[:,1:2*r])

    # L-step:
    fL = L -> funL_pre(L, UᵀDU, UᵀVcosU)
    L_1 = rk(fL, L_0, h)
    L_1 = [L_1 V_0];
    V_1, _ = qr(L_1);
    V_1 = Matrix(V_1[:,1:2*r])

    # precompute projections
    UᵀDU = U_1'*D*U_1
    VᵀDᵀV = V_1'*D'*V_1
    UᵀVcosU = U_1'*V_cos*U_1
    VᵀVcosV = V_1'*V_cos*V_1

    # S-step:
    fS = S -> funS_pre(S, UᵀDU, VᵀDᵀV, UᵀVcosU, VᵀVcosV);
    S_0 = (U_1'*U_0) * S_0 * (V_1'*V_0)';
    S_1 = rk(fS, S_0, h);

    if augment
        return [U_1,V_1,S_1];
    else
        return truncMat([U_1,V_1,S_1],r);
    end
end

function Method_naive(Y0, t0, t1)

    global fun
    global tol
    global funK 
    global funL 
    global funS

    # Initial values
    U_0 = ComplexF64.(Y0[1]);
    V_0 = ComplexF64.(Y0[2]);
    S_0 = ComplexF64.(Y0[3]);

    K_0::Matrix{ComplexF64} = U_0*S_0
    L_0::Matrix{ComplexF64} = V_0*S_0'

    r = size(U_0,2);
    h = t1-t0;

    # K-step:
    funY = Y ->
    begin
        U = ComplexF64.(Y[1]);
        V = ComplexF64.(Y[2]);
        S = ComplexF64.(Y[3]);
        Y_0::Matrix{ComplexF64} = U*S*V';
        outU::Matrix{ComplexF64} = (I-U*U')*fun(Y_0)*V*inv(S)
        outV::Matrix{ComplexF64} = (I-V*V')*fun(Y_0)'*U*inv(S')
        outS::Matrix{ComplexF64} = U'*fun(Y_0)*V
        return [outU, outV, outS]
    end;

    k1 = funY(Y0)
    k2 = funY(Y0 .+ (h/2) .* k1)
    k3 = funY(Y0 .+ (h/2) .* k2)
    k4 = funY(Y0 .+ h .* k3)
    Y1 = Y0 .+ (h/6) .* (k1 .+ 2 .*k2 .+ 2 .*k3 .+ k4)

    return Y1;
end

## Rank-adaptive unconventional Integrator
function Method_fixed_rank_BUG_ode(Y0, t0, t1)

    global fun
    global tol

    # Initial values
    U_0 = Y0[1];
    V_0 = Y0[2];
    S_0 = Y0[3];

    r = size(U_0,2);
    h = t1-t0;

    # K-step:
    funK = K ->
    begin
        Y_0::Matrix{ComplexF64} = K*V_0';
        out::Matrix{ComplexF64} = fun(Y_0)*V_0
        return out
    end;

    K_0::Matrix{ComplexF64} = U_0*S_0;
    K_1 = odeSolver(funK, K_0 .+ 0.0*1im, h);


    K_1 = ComplexF64.(K_1);
    U_1,_ = qr(K_1);
    U_1 = Matrix(U_1[:,1:r])

    # L-step:
    funL = L ->
    begin
        Y_0::Matrix{ComplexF64} = U_0*L';
        out::Matrix{ComplexF64} = fun(Y_0)'*U_0
        return out
    end;

    K_0 = V_0*S_0';
    K_1 = odeSolver(funL, K_0, h);

    K_1 = ComplexF64.(K_1);
    V_1, _ = qr(K_1);
    V_1 = Matrix(V_1[:,1:r])


    # S-step:
    funS = S ->
    begin
        Y_0::Matrix{ComplexF64} = U_1*S*V_1';
        out::Matrix{ComplexF64} = U_1'*fun(Y_0)*V_1;
        return out
    end;

    S_0 = (U_1'*U_0) * S_0 * (V_1'*V_0)';
    S_1 = odeSolver(funS, S_0, h);

    return [U_1,V_1,S_1];
end

function Method_PSI_ode(Y0, t0, t1)

    global fun
    global tol

    # Initial values
    U_0 = Y0[1];
    V_0 = Y0[2];
    S_0 = Y0[3];

    r = size(U_0,2);
    h = t1-t0;

    # K-step:
    funK = K ->
    begin
        Y_0::Matrix{ComplexF64} = K*V_0';
        out::Matrix{ComplexF64} = fun(Y_0)*V_0
        return out
    end;

    K_0::Matrix{ComplexF64} = U_0*S_0;
    K_1 = odeSolver(funK, K_0 .+ 0.0*1im, h);

    K_1 = ComplexF64.(K_1);
    U_1, SK = qr(K_1);
    U_1 = Matrix(U_1[:,1:r])
    SK = Matrix(SK)

    # S-step:
    funS = S ->
    begin
        Y_0::Matrix{ComplexF64} = U_1*S*V_0';
        out::Matrix{ComplexF64} = -U_1'*fun(Y_0)*V_0;
        return out
    end;

    S_0 = SK;
    S_1 = odeSolver(funS, S_0, h);

    # L-step:
    funL = L ->
    begin
        Y_0::Matrix{ComplexF64} = U_1*L';
        out::Matrix{ComplexF64} = fun(Y_0)'*U_1
        return out
    end;

    L_0 = V_0*S_1';
    K_1 = odeSolver(funL, L_0, h);

    K_1 = ComplexF64.(K_1);
    V_1, S_1 = qr(K_1);
    V_1 = Matrix(V_1[:, 1:r]);
    S_1 = Matrix(S_1)';

    return [U_1,V_1,S_1];
end

function Method_fixed_rank_BUG_eff(Y0, t0, t1)

    global fun
    global tol
    global funK 
    global funL 
    global funS

    # Initial values
    U_0 = ComplexF64.(Y0[1]);
    V_0 = ComplexF64.(Y0[2]);
    S_0 = ComplexF64.(Y0[3]);

    K_0::Matrix{ComplexF64} = U_0*S_0
    L_0::Matrix{ComplexF64} = V_0*S_0'

    r = size(U_0,2);
    h = t1-t0;

    # K-step:
    fK = K -> funK(K, V_0)
    K_1 = odeSolver(fK, K_0, h)
    U_1,_ = qr(K_1);
    U_1 = Matrix(U_1[:,1:r])

    # L-step:
    fL = L -> funL(L, U_0)
    L_1 = odeSolver(fL, L_0, h)
    V_1, _ = qr(L_1);
    V_1 = Matrix(V_1[:,1:r])

    # S-step:
    fS = S -> funS(U_1, S, V_1);
    S_0 = (U_1'*U_0) * S_0 * (V_1'*V_0)';
    S_1 = odeSolver(fS, S_0, h);

    return [U_1,V_1,S_1];
end

function Method_fixed_rank_BUG_eff_precompute(Y0, t0, t1)

    global fun
    global tol
    global funK 
    global funL 
    global funS

    # Initial values
    U_0 = ComplexF64.(Y0[1]);
    V_0 = ComplexF64.(Y0[2]);
    S_0 = ComplexF64.(Y0[3]);

    K_0::Matrix{ComplexF64} = U_0*S_0
    L_0::Matrix{ComplexF64} = V_0*S_0'

    r = size(U_0,2);
    h = t1-t0;

    # precompute projections
    UᵀDU = U_0'*D*U_0
    VᵀDᵀV = V_0'*D'*V_0
    UᵀVcosU = U_0'*V_cos*U_0
    VᵀVcosV = V_0'*V_cos*V_0

    # K-step:
    fK = K -> funK_pre(K, VᵀDᵀV, VᵀVcosV)
    K_1 = rk(fK, K_0, h)
    U_1,_ = qr(K_1);
    U_1 = Matrix(U_1[:,1:r])

    # L-step:
    fL = L -> funL_pre(L, UᵀDU, UᵀVcosU)
    L_1 = rk(fL, L_0, h)
    V_1, _ = qr(L_1);
    V_1 = Matrix(V_1[:,1:r])

    # precompute projections
    UᵀDU = U_1'*D*U_1
    VᵀDᵀV = V_1'*D'*V_1
    UᵀVcosU = U_1'*V_cos*U_1
    VᵀVcosV = V_1'*V_cos*V_1

    # S-step:
    fS = S -> funS_pre(S, UᵀDU, VᵀDᵀV, UᵀVcosU, VᵀVcosV);
    S_0 = (U_1'*U_0) * S_0 * (V_1'*V_0)';
    S_1 = rk(fS, S_0, h);

    return [U_1,V_1,S_1];
end

function MidpointBUG4r(Y0, t0, t1, augment=true)

    global fun
    global tol
    global funS

    # Initial values
    U_0 = Y0[1];
    V_0 = Y0[2];
    S_0 = Y0[3];

    # parameters
    h = t1-t0;
    r = size(U_0,2);

    # Midpoint
    Y12 = Method_augmented_BUG_ode_eff(Y0, t0, t0 + 0.5*h, augment)
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
    fS = S -> funS(hatU_1, S, hatV_1); #S -> hatU_1'* fun(hatU_1*S*hatV_1')*hatV_1;
    S_1 = odeSolver(fS, S_0, h);

    #Y1 = roundMat({hatU_1,hatV_1,S_1},tol);
    return truncMat([hatU_1,hatV_1,S_1],r);
end

function MidpointBUG4r_precompute(Y0, t0, t1, augment=true)

    global fun
    global tol
    global funS

    # Initial values
    U_0 = Y0[1];
    V_0 = Y0[2];
    S_0 = Y0[3];

    # parameters
    h = t1-t0;
    r = size(U_0,2);

    # Midpoint
    Y12 = Method_augmented_BUG_ode_eff_precompute(Y0, t0, t0 + 0.5*h, augment)
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

    # precompute projections
    UᵀDU = hatU_1'*D*hatU_1
    VᵀDᵀV = hatV_1'*D'*hatV_1
    UᵀVcosU = hatU_1'*V_cos*hatU_1
    VᵀVcosV = hatV_1'*V_cos*hatV_1

    # S-step:
    S_0 = (hatU_1'* U_0)* S_0 * (hatV_1'*V_0)';
    fS = S -> funS_pre(S, UᵀDU, VᵀDᵀV, UᵀVcosU, VᵀVcosV);
    S_1 = rk(fS, S_0, h);

    #Y1 = roundMat({hatU_1,hatV_1,S_1},tol);
    return truncMat([hatU_1,hatV_1,S_1],r);
end

function MidpointBUG4r_precompute_adaptive(Y0, t0, t1, augment=true, ϑ=0.01)

    global fun
    global tol
    global funS

    # Initial values
    U_0 = Y0[1];
    V_0 = Y0[2];
    S_0 = Y0[3];

    # parameters
    h = t1-t0;
    r = size(U_0,2);

    # Midpoint
    Y12 = Method_augmented_BUG_ode_eff_precompute(Y0, t0, t0 + 0.5*h, augment)
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

    # precompute projections
    UᵀDU = hatU_1'*D*hatU_1
    VᵀDᵀV = hatV_1'*D'*hatV_1
    UᵀVcosU = hatU_1'*V_cos*hatU_1
    VᵀVcosV = hatV_1'*V_cos*hatV_1

    # S-step:
    S_0 = (hatU_1'* U_0)* S_0 * (hatV_1'*V_0)';
    fS = S -> funS_pre(S, UᵀDU, VᵀDᵀV, UᵀVcosU, VᵀVcosV);
    S_1 = rk(fS, S_0, h);

    #Y1 = roundMat({hatU_1,hatV_1,S_1},tol);
    return truncMat([hatU_1,hatV_1,S_1],r);
end

## Extra functions
function rk(f, Y, h, order=4)
    if order == 1
        return Y + h*f(Y)
    elseif order == 2
        return Y + h* f(Y + 0.5 * h * f(Y));
    elseif order == 4
        Y = ComplexF64.(Y)
        k1::Matrix{ComplexF64} = f(Y)
        k2::Matrix{ComplexF64} = f(Y .+ (h/2) * k1)
        k3::Matrix{ComplexF64} = f(Y .+ (h/2) * k2)
        k4::Matrix{ComplexF64} = f(Y .+ h * k3)
        Y = Y .+ (h/6) * (k1 .+ 2*k2 .+ 2*k3 .+ k4)
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

function truncMat(Y, r, ϑ)

    U_1 = Y[1];
    V_1 = Y[2];
    S = Y[3];

    U,D,V = svd(S);

    tmp = 0.0
    ϑ = ϑ# * norm(D)

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

    rmax = min(rmax, 100)
    rmax = max(rmax, 2)

    # Truncation:
    U_1 = U_1*U[:, 1:rmax];
    V_1 = V_1*V[:, 1:rmax];
    S_1 = diagm(D[1:rmax]);

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
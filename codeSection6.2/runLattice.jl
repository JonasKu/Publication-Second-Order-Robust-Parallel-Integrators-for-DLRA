using Base: Float64
include("utils.jl")
include("settings.jl")
include("SolverDLRA.jl")

using PyPlot
using DelimitedFiles

close("all")

nₚₙ = 21;
ϑₚ = 1e-2;      # parallel tolerance
ϑᵤ = 5e-2;    # (unconventional) BUG tolerance # BUG 0.05, parrallel 0.02 looks okay
ϑₚₕ = 1e-2;      # parallel tolerance
ϑᵤₕ = 3e-2;    # (unconventional) BUG tolerance

s = Settings(251,251, nₚₙ, 20,"Lattice"); # create settings class with 351 x 351 spatial cells and a rank of 50

################################################################
######################### execute code #########################
################################################################

##################### classical checkerboard #####################

################### run full method ###################
solver = SolverDLRA(s);
@time rhoFull = Solve(solver);
rhoFull = Vec2Mat(s.NCellsX,s.NCellsY,rhoFull)

##################### low tolerance #####################

################### run BUG adaptive ###################
s.ϑ = ϑᵤ;
solver = SolverDLRA(s);
@time rhoDLRA,rankInTime = SolveBUG2nd(solver);
rhoDLRA = Vec2Mat(s.NCellsX,s.NCellsY,rhoDLRA);

################### run parallel ###################
s.ϑ = ϑₚ;
solver = SolverDLRA(s);
@time rhoDLRAp,rankInTimep = SolveParallel(solver);
rhoDLRAp = Vec2Mat(s.NCellsX,s.NCellsY,rhoDLRAp);

################### run parallel 2nd ###################
s.ϑ = ϑₚ;
solver = SolverDLRA(s);
@time rhoDLRAp2,rankInTimep2 = SolveParallel2nd(solver);
rhoDLRAp2 = Vec2Mat(s.NCellsX,s.NCellsY,rhoDLRAp2);

############################################################
######################### plotting #########################
############################################################

X = (s.xMid[2:end-1]'.*ones(size(s.xMid[2:end-1])));
Y = (s.yMid[2:end-1]'.*ones(size(s.yMid[2:end-1])))';

## full
maxV = maximum(rhoFull[2:end-1,2:end-1])
minV = max(1e-7,minimum(4.0*pi*sqrt(2)*rhoFull[2:end-1,2:end-1]))
idxNeg = findall((rhoFull.<=0.0))
rhoFull[idxNeg] .= NaN;
fig = figure("full, log",figsize=(10,10),dpi=100)
ax = gca()
pc = pcolormesh(X,Y,4.0*pi*sqrt(2)*rhoFull[2:(end-1),(end-1):-1:2]',norm=matplotlib.colors.LogNorm(vmin=minV, vmax=maxV))
ax.tick_params("both",labelsize=20) 
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title(L"$\Phi$, P$_{21}$", fontsize=25)
cbar = plt.colorbar(pc)
cbar.ax.tick_params(labelsize=20)
tight_layout()
show()
savefig("results/scalar_flux_PN_$(s.problem)_nx$(s.NCellsX)_N$(s.nₚₙ).eps")

## DLRA BUG adaptive
idxNeg = findall((rhoDLRA.<=0.0))
rhoDLRA[idxNeg] .= NaN;
fig = figure("midpoint BUG, log, ϑ coarse",figsize=(9,10),dpi=100)
ax = gca()
pcolormesh(X,Y,4.0*pi*sqrt(2)*rhoDLRA[2:(end-1),(end-1):-1:2]',norm=matplotlib.colors.LogNorm(vmin=minV, vmax=maxV))
ax.tick_params("both",labelsize=20) 
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title(L"$\Phi$, midpoint BUG, $r = $ "*LaTeXString(string(s.r)), fontsize=25)
tight_layout()
show()
savefig("results/scalar_flux_DLRA_adBUG_$(s.problem)_rank$(s.r)_nx$(s.NCellsX)_N$(s.nₚₙ).png")

## DLRA parallel
idxNeg = findall((rhoDLRAp.<=0.0))
rhoDLRAp[idxNeg] .= NaN;
fig = figure("parallel, log, ϑ coarse",figsize=(9,10),dpi=100)
ax = gca()
pcolormesh(X,Y,4.0*pi*sqrt(2)*rhoDLRAp[2:(end-1),(end-1):-1:2]',norm=matplotlib.colors.LogNorm(vmin=minV, vmax=maxV))
ax.tick_params("both",labelsize=20) 
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title(L"$\Phi$, parallel, $r =$ "*LaTeXString(string(s.r)), fontsize=25)
tight_layout()
show()
savefig("results/scalar_flux_DLRA_$(s.problem)_theta$(ϑₚ)_nx$(s.NCellsX)_N$(s.nₚₙ).png")


## DLRA parallel 2nd order
idxNeg = findall((rhoDLRAp2.<=0.0))
rhoDLRAp2[idxNeg] .= NaN;
fig = figure("parallel 2nd, log, ϑ coarse",figsize=(9,10),dpi=100)
ax = gca()
pcolormesh(X,Y,4.0*pi*sqrt(2)*rhoDLRAp2[2:(end-1),(end-1):-1:2]',norm=matplotlib.colors.LogNorm(vmin=minV, vmax=maxV))
ax.tick_params("both",labelsize=20) 
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title(L"$\Phi$, parallel $2^{nd}$ order, $r = $ "*LaTeXString(string(s.r)), fontsize=25)
tight_layout()
show()
savefig("results/scalar_flux_parallel2nd_$(s.problem)_rank$(s.r)_nx$(s.NCellsX)_N$(s.nₚₙ).png")

fig = figure("setup",figsize=(9,10),dpi=100)
ax = fig.add_subplot(111)
rect1 = matplotlib.patches.Rectangle((1,5), 1.0, 1.0, color="cornflowerblue")
ax.add_patch(rect1)
rect1 = matplotlib.patches.Rectangle((5,5), 1.0, 1.0, color="cornflowerblue")
ax.add_patch(rect1)
rect1 = matplotlib.patches.Rectangle((4,4), 1.0, 1.0, color="cornflowerblue")
ax.add_patch(rect1)
rect1 = matplotlib.patches.Rectangle((2,4), 1.0, 1.0, color="cornflowerblue")
ax.add_patch(rect1)
rect1 = matplotlib.patches.Rectangle((3,3), 1.0, 1.0, color="lightcoral")
ax.add_patch(rect1)
rect1 = matplotlib.patches.Rectangle((1,3), 1.0, 1.0, color="cornflowerblue")
ax.add_patch(rect1)
rect1 = matplotlib.patches.Rectangle((5,3), 1.0, 1.0, color="cornflowerblue")
ax.add_patch(rect1)
rect1 = matplotlib.patches.Rectangle((4,2), 1.0, 1.0, color="cornflowerblue")
ax.add_patch(rect1)
rect1 = matplotlib.patches.Rectangle((2,2), 1.0, 1.0, color="cornflowerblue")
ax.add_patch(rect1)
rect1 = matplotlib.patches.Rectangle((3,1), 1.0, 1.0, color="cornflowerblue")
ax.add_patch(rect1)
rect1 = matplotlib.patches.Rectangle((1,1), 1.0, 1.0, color="cornflowerblue")
ax.add_patch(rect1)
rect1 = matplotlib.patches.Rectangle((5,1), 1.0, 1.0, color="cornflowerblue")
ax.add_patch(rect1)
ax.grid()
plt.xlim([0, 7])
plt.ylim([0, 7])
ax.tick_params("both",labelsize=20) 
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title("lattice setup", fontsize=25)
tight_layout()
plt.show()
savefig("results/setup_lattice_testcase.png")


println("main finished")

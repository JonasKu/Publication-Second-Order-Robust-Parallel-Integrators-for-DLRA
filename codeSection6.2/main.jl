using Pkg
Pkg.activate(".")
Pkg.instantiate()

using Base: Float64
include("utils.jl")
include("settings.jl")
include("SolverDLRA.jl")

using PyPlot
using DelimitedFiles

close("all")

nₚₙ = 21;

################################################################
######################### execute code #########################
################################################################

##################### classical checkerboard #####################
s = Settings(251,251, nₚₙ, 10,"Lattice"); # create settings class with 251 x 251 spatial cells and a rank of 10

################### run full method ###################
solver = SolverDLRA(s);
@time rhoFull = Solve(solver);
rhoFull = Vec2Mat(s.NCellsX,s.NCellsY,rhoFull)

##################### low rank #####################

################### run midpoint BUG ###################
solver = SolverDLRA(s);
@time rhoDLRA,rankInTime = SolveBUG2nd(solver);
rhoDLRA = Vec2Mat(s.NCellsX,s.NCellsY,rhoDLRA);

################### run parallel 2nd ###################
solver = SolverDLRA(s);
@time rhoDLRAp2,rankInTimep2 = SolveParallel2nd(solver);
rhoDLRAp2 = Vec2Mat(s.NCellsX,s.NCellsY,rhoDLRAp2);

##################### higher rank #####################
s = Settings(251,251, nₚₙ, 20,"Lattice"); # create settings class with 251 x 251 spatial cells and a rank of 20

################### run midpoint BUG ###################
solver = SolverDLRA(s);
@time rhoDLRAh,rankInTime = SolveBUG2nd(solver);
rhoDLRAh = Vec2Mat(s.NCellsX,s.NCellsY,rhoDLRAh);

################### run parallel 2nd ###################
solver = SolverDLRA(s);
@time rhoDLRAp2h,rankInTimep2 = SolveParallel2nd(solver);
rhoDLRAp2h = Vec2Mat(s.NCellsX,s.NCellsY,rhoDLRAp2h);

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
savefig("scalar_flux_PN_$(s.problem)_nx$(s.NCellsX)_N$(s.nₚₙ).eps")

## DLRA BUG adaptive
idxNeg = findall((rhoDLRA.<=0.0))
rhoDLRA[idxNeg] .= NaN;
fig = figure("midpoint BUG, log, lower rank",figsize=(9,10),dpi=100)
ax = gca()
pcolormesh(X,Y,4.0*pi*sqrt(2)*rhoDLRA[2:(end-1),(end-1):-1:2]',norm=matplotlib.colors.LogNorm(vmin=minV, vmax=maxV))
ax.tick_params("both",labelsize=20) 
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title(L"$\Phi$, midpoint BUG, $r = $ "*LaTeXString(string(10)), fontsize=25)
tight_layout()
show()
savefig("scalar_flux_DLRA_adBUG_$(s.problem)_rank10_nx$(s.NCellsX)_N$(s.nₚₙ).png")

## DLRA parallel 2nd order
idxNeg = findall((rhoDLRAp2.<=0.0))
rhoDLRAp2[idxNeg] .= NaN;
fig = figure("parallel 2nd, log, lower rank",figsize=(9,10),dpi=100)
ax = gca()
pcolormesh(X,Y,4.0*pi*sqrt(2)*rhoDLRAp2[2:(end-1),(end-1):-1:2]',norm=matplotlib.colors.LogNorm(vmin=minV, vmax=maxV))
ax.tick_params("both",labelsize=20) 
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title(L"$\Phi$, parallel $2^{nd}$ order, $r = $ "*LaTeXString(string(10)), fontsize=25)
tight_layout()
show()
savefig("scalar_flux_parallel2nd_$(s.problem)_rank10_nx$(s.NCellsX)_N$(s.nₚₙ).png")

## DLRA BUG adaptive higher rank
idxNeg = findall((rhoDLRAh.<=0.0))
rhoDLRAh[idxNeg] .= NaN;
fig = figure("midpoint BUG, log, higher rank",figsize=(9,10),dpi=100)
ax = gca()
pcolormesh(X,Y,4.0*pi*sqrt(2)*rhoDLRAh[2:(end-1),(end-1):-1:2]',norm=matplotlib.colors.LogNorm(vmin=minV, vmax=maxV))
ax.tick_params("both",labelsize=20) 
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title(L"$\Phi$, midpoint BUG, $r = $ "*LaTeXString(string(s.r)), fontsize=25)
tight_layout()
show()
savefig("scalar_flux_DLRA_adBUG_$(s.problem)_rank20_nx$(s.NCellsX)_N$(s.nₚₙ).png")

## DLRA parallel 2nd order higher rank
idxNeg = findall((rhoDLRAp2h.<=0.0))
rhoDLRAp2h[idxNeg] .= NaN;
fig = figure("parallel 2nd, log, higher rank",figsize=(9,10),dpi=100)
ax = gca()
pcolormesh(X,Y,4.0*pi*sqrt(2)*rhoDLRAp2h[2:(end-1),(end-1):-1:2]',norm=matplotlib.colors.LogNorm(vmin=minV, vmax=maxV))
ax.tick_params("both",labelsize=20) 
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title(L"$\Phi$, parallel $2^{nd}$ order, $r = $ "*LaTeXString(string(s.r)), fontsize=25)
tight_layout()
show()
savefig("scalar_flux_parallel2nd_$(s.problem)_rank20_nx$(s.NCellsX)_N$(s.nₚₙ).png")

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
savefig("setup_lattice_testcase.png")


println("main finished")

##
using QuasiMonteCarlo: LowDiscrepancySample, sample
using Plots

lb = [0.0, -0.5]  # Lower bounds
ub = [4.0, 3.5]   # Upper bounds
n = 100           # Number of samples

halton = LowDiscrepancySample([3, 5])  # Halton with prime number sequence
xtrain = sample(n, lb, ub, halton)

scatter(xtrain[1,:], xtrain[2,:])
##

# f(x) = cos.(x[1,:]).*sin.(x[2,:])
# ytrain = f(xtrain)

# #scatter3d(xtrain[1,:], xtrain[2,:], ytrain, marker_z=ytrain)
# #savefig("test.png")

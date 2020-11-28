##
using QuasiMonteCarlo: LowDiscrepancySample, sample
using Plots

function kern_sqexp(x)
    exp(-x^2/2)
end

function cov!(K, kernel, x1, x2, l)
    for (k1, xv1) in enumerate(x1)
        for (k2, xv2) in enumerate(x2)
        end
    end
end

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

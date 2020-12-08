using BenchmarkTools
using Distances

##
function nu_L2!(nu, xa, xb, l)
  nu .= 0.0
  for ka = axes(xa, 2)
    @simd for kd = axes(xa, 1)
      @inbounds nu[ka] = nu[ka] + 0.5*(((xa[kd, ka] - xb[kd])/l[kd])^2)
    end
  end
end

function kern_sqexp!(out, nu)
  out[:] = exp.(-nu)
end

function build_K!(K, xa, xb, l, kern!, nufunc!)
  # Build a kernel matrix using a function `kern` to construct columns/rows
  nu = Array{Float64}(undef, size(xa, 2))
  for kb = axes(xb, 2)
    @views nufunc!(nu, xa, xb[:, kb], l)
    @views kern!(K[:, kb], nu)
  end
end

function kern_sqexp_elem(nu)
  exp(-nu)
end

@inline function nu_L2_elem(xa, xb, l)
  # mapreduce(k -> ((xa[k] - xb[k])/l[k])^2, +, axes(xa, 1)) / 2
  out = 0.0
  @inbounds @simd for k in axes(xa, 1)
    out += ((xa[k] - xb[k])/l[k])^2
  end
  out = 0.5*out
end


@inline function nu_L2_index(xa, xb, ka, kb, l)
  out = 0.0
  @inbounds @simd for k in axes(xa, 1)
    out += ((xa[k, ka] - xb[k, kb])/l[k])^2
  end
  out = 0.5*out
end

function build_K_elem!(K, xa, xb, l, kern::Function, nufunc::Function)
  # Build a kernel matrix using a function `kern` to construct elements
  @inbounds for kb = axes(xb, 2)
    for ka = axes(xa, 2)
      #@views nu = sqeuclidean(xa[:, ka], xb[:, kb])
      #@views nu = nu_L2_index(xa, xb, ka, kb, l)
      @views nu = nufunc(xa[:, ka], xb[:, kb], l)
      K[ka, kb] = kern(nu)
      #@views nu = nu_L2_elem(xa[:, ka], xb[:, kb], l)
      #K[ka, kb] = kern_sqexp_elem(nu)
    end
  end
end

nd = 4
na = 4096
nb = na

l = ones(nd)
xa = rand(nd, na)
xb = xa
K = zeros((na, nb))
##
@time build_K_elem!(K, xa, xb, l, kern_sqexp_elem, nu_L2_elem)
#@code_native build_K_elem!(K, xa, xb, l, kern_sqexp_elem, nu_L2_elem)
##
@time build_K!(K, xa, xb, l, kern_sqexp!, nu_L2!)
##
# @btime @views K = [kern_sqexp_elem(nu_L2_index(xa, xb, ka, kb, l)) for ka in axes(xa, 2), kb in axes(xb, 2)]
# print()
##
# using KernelFunctions

# k₁ = SqExponentialKernel()
# ##
# @time kernelmatrix!(K₁,k₁,xa,obsdim=2)
# print()

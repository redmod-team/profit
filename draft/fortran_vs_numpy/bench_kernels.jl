##
function nu_L2!(nu, xa, xb, l)
  for ka = 1:size(xa, 2)
    nu[ka] = 0.5*sum(((xa[:, ka] - xb)/l).^2)
  end
end

function kern_sqexp!(out, nu)
    out[:] = exp.(-nu)
end

function build_K!(K, xa, xb, l, kern!)
  # Build a kernel matrix using a function `kern` to construct columns/rows
  nu = Array{Float64}(undef, size(xa, 2))
  for kb = 1:size(xb, 2)
    nu_L2!(nu, xa, xb[:, kb], l)
    kern!(K[:, kb], nu)
  end
end

function kern_sqexp_elem(nu)
  exp(-nu)
end

function nu_L2_elem(xa, xb, l)
  out = 0.0
  for k in size(xa, 1)
    out += ((xa[k] - xb[k])/l[k])^2
  end
  out = 0.5*out
end

function build_K_elem!(K, xa, xb, l, kern)
  # Build a kernel matrix using a function `kern` to construct elements
  for kb = 1:size(xb, 2)
    for ka = 1:size(xa, 2)
      nu = nu_L2_elem(xa[:, ka], xb[:, kb], l)
      K[ka, kb] = kern(nu)
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

@time build_K_elem!(K, xa, xb, l, kern_sqexp_elem)
##

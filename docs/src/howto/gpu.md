# Keep data on a GPU

!!! warning "Untested"
    This path has not been run on GPU hardware. It is a work in progress;
    expect problems and check results against a CPU run.

The parameters stay on the CPU and the data lives on a CUDA device, so the
model's matrix products and observe statements run on the GPU. Build the data
as `CuArray`s before creating the model:

```julia
using PracticalBayes, Distributions, LinearAlgebra
import CUDA
CUDA.allowscalar(false)

@model function gpu_reg(X, y)
    k = size(X, 2)
    beta ~ MvNormal(zeros(paramtype(__mode__), k), 1f0 * I)
    sigma ~ Exponential(1f0)
    y ~ MvNormal(X * beta, sigma^2 * I)
end

model = gpu_reg(CUDA.CuArray(Float32.(X)), CUDA.CuArray(Float32.(y)))
```

Write the likelihood as whole-array operations, as above: one matrix product
and one array-valued observe. A loop over elements indexes the GPU array one
scalar at a time; `CUDA.allowscalar(false)` turns that into an error so you
find it early. PracticalBayes itself does not index your data element by
element.

Sample as on the CPU; see [Run a model in Float32](@ref) for Float32
parameters.

## Check that the GPU is usable

`CUDA.functional()` must return `true` before the data can live on the device.
If it returns `false`, check that the node's NVIDIA driver (`nvidia-smi`) is
recent enough for the installed CUDA.jl.

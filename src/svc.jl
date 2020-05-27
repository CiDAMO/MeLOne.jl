export SVC

using LinearAlgebra, Logging, NLPModels, NLPModelsIpopt

mutable struct SVC <: MeLOneModel
  α
  F

  options :: Dict{Symbol,Any}
end

const svc_option_list = [(:C, 1.0), (:kernel, :rbf), (:degree, 3), (:gamma, :scale), (:coef0, 0.0)]

function SVC(; kwargs...)
  options = Dict{Symbol,Any}(k => get(kwargs, k, v) for (k,v) in svc_option_list)
  for k in keys(kwargs)
    if !(k in getfield.(svc_option_list, 1))
      @warn "Keyword argument $k ignored"
    end
  end

  return SVC([], x -> 0.0, options)
end

function fit!(model :: SVC,
              X :: Matrix, y :: Vector)

  kernel = model.options[:kernel]
  n, p = size(X)
  Ker = if kernel == :rbf
    γ = model.options[:gamma]
    if !(γ isa Number)
      γ = (γ == :scale ? 1 / n / p : 1 / n)
    end
    (x1, x2) -> exp(-γ * norm(x1 - x2)^2)
  elseif kernel == :poly
    d = model.options[:degree]
    c₀ = model.options[:coef0]
    (x1, x2) -> (c₀ + dot(x1, x2))^d
  elseif kernel == :sigmoid
    γ = model.options[:gamma]
    if !(γ isa Number)
      γ = (γ == :scale ? 1 / n / p : 1 / n)
    end
    c₀ = model.options[:coef0]
    (x1, x2) -> tanh(c₀ + γ * dot(x1, x2))
  elseif kernel == :linear
    (x1, x2) -> dot(x1, x2)
  end
  Kmat = [Ker(X[i,:], X[j,:]) for i = 1:n, j = 1:n]

  C = model.options[:C]
  f(α) = 0.5 * dot(α .* y, Kmat * (α .* y)) - sum(α)
  nlp = ADNLPModel(f, zeros(n), lvar=zeros(n), uvar=C * ones(n), c=α->[sum(α .* y)], lcon=[0.0], ucon=[0.0])

  output = with_logger(NullLogger()) do
    ipopt(nlp, print_level=0)
  end

  α = model.α = output.solution
  ϵ = 1e-8 * C
  I = findall(ϵ .< α .< C - ϵ)
  J = findall(α .> ϵ)
  k = J[1]
  b = y[k] - sum(α[i] * y[i] * Kmat[i,k] for i in J)
  model.F = x -> sum(α[i] * y[i] * Ker(X[i,:], x) for i in J) + b

  return model
end

function predict(model :: SVC,
                 X :: Matrix; prob = false)

  n, p = size(X)
  return [model.F(X[i,:]) > 0 ? 1 : -1 for i = 1:n]
end

predict_proba(model :: SVC, X :: Matrix) = predict(model, X, prob=true)

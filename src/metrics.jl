export accuracy_score, rss_score, r2_score

using Statistics

"""
    accuracy_score(y_real, y_pred)

Returns the amount of times `y_pred` is equal to `y_real` as a percentage.
"""
function accuracy_score(y_real :: AbstractArray, y_pred :: AbstractArray)
  return sum(y_real .== y_pred) / length(y_real)
end

"""
    rss_score(y_real, y_pred)

Computes the residual sum of squares.
"""
function rss_score(y_real, y_pred)
  return sum((y_real - y_pred).^2)
end

"""
    r2_score(y_real, y_pred)

Computes the R² score given by ``1 - ∑(yᵢ - ŷᵢ)² / ∑(yᵢ - ȳ)²``, where
`y` is `y_real`, `ŷ` is `y_pred` and `ȳ` is the mean of `y_real`.
"""
function r2_score(y_real, y_pred)
  μ = mean(y_real)
  return 1 - rss_score(y_real, y_pred) / sum((y_real .- μ).^2)
end

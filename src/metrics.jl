export accuracy_score

function accuracy_score(y :: AbstractArray, y_pred :: AbstractArray)
  return sum(y .== y_pred) / length(y)
end

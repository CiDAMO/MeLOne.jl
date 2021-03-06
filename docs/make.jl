using Documenter, MeLOne

makedocs(
  modules = [MeLOne],
  doctest = true,
  # linkcheck = true,
  strict = true,
  format = Documenter.HTML(assets = ["assets/style.css"], prettyurls = get(ENV, "CI", nothing) == "true"),
  sitename = "MeLOne.jl",
  pages = Any["Home" => "index.md",
              "Reference" => "reference.md"]
)

deploydocs(repo = "github.com/CiDAMO/MeLOne.jl.git")

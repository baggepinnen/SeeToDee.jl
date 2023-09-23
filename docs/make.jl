ENV["GKSwstype"] = 322 # workaround for gr segfault on GH actions
# ENV["GKS_WSTYPE"]=100 # try this if above does not work
using Documenter, SeeToDee

makedocs(
      sitename = "SeeToDee Documentation",
      doctest = false,
      modules = [SeeToDee],
      # warnonly = [],
      pages = [
            "Home" => "index.md",
            "API" => "api.md",
      ],
      format = Documenter.HTML(prettyurls = haskey(ENV, "CI")),
)

deploydocs(
      repo = "github.com/baggepinnen/SeeToDee.jl.git",
)

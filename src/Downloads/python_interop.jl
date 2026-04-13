# ===========================================================================
# Python interop — subprocess-based CDS/MARS API calls
#
# The codebase uses Python subprocess (not PyCall.jl) for CDS/MARS access.
# This is the established pattern (see download_era5.jl, download_gfas_fire.jl).
# ===========================================================================

"""
    detect_python_env(python_path="python3") -> PythonEnvironment

Probe the Python installation for available packages and API credentials.
Called once during module initialization.
"""
function detect_python_env(python_path::String="python3")
    check_pkg = pkg -> begin
        try
            success(`$python_path -c "import $pkg"`)
        catch
            false
        end
    end

    has_cdsapi    = check_pkg("cdsapi")
    has_ecmwfapi  = check_pkg("ecmwfapi")
    has_cfgrib    = check_pkg("cfgrib")
    has_xarray    = check_pkg("xarray")

    cds_creds  = isfile(expanduser("~/.cdsapirc"))
    mars_creds = isfile(expanduser("~/.ecmwfapirc"))

    return PythonEnvironment(python_path, has_cdsapi, has_ecmwfapi,
                             has_cfgrib, has_xarray, cds_creds, mars_creds)
end

"""
    preferred_era5_api(env::PythonEnvironment) -> Symbol

Returns :mars if MARS is available (faster), :cds otherwise.
"""
function preferred_era5_api(env::PythonEnvironment)
    env.mars_credentials && env.has_ecmwfapi && return :mars
    env.cds_credentials && env.has_cdsapi && return :cds
    error("No ECMWF API credentials found. Configure ~/.ecmwfapirc (MARS) " *
          "or ~/.cdsapirc (CDS)")
end

"""
    run_python(script::String, env::PythonEnvironment; label="")

Write `script` to a temporary file, execute with the configured Python
interpreter, and clean up. Logs timing.
"""
function run_python(script::String, env::PythonEnvironment; label::String="")
    tmp = tempname() * ".py"
    write(tmp, script)
    try
        t0 = time()
        run(`$(env.python_path) $tmp`)
        elapsed = round(time() - t0; digits=1)
        !isempty(label) && @info "  Python ($label): $(elapsed)s"
    finally
        rm(tmp; force=true)
    end
end

"""
    dict_to_python(d::Dict) -> String

Convert a Julia Dict to a Python dict literal string.
"""
function dict_to_python(d::Dict)
    pairs = String[]
    for (k, v) in d
        push!(pairs, "    $(repr(string(k))): $(value_to_python(v))")
    end
    return "{\n" * join(pairs, ",\n") * "\n}"
end

function value_to_python(v::String)
    repr(v)
end

function value_to_python(v::Number)
    string(v)
end

function value_to_python(v::Vector)
    "[" * join(value_to_python.(v), ", ") * "]"
end

function value_to_python(v::Dict)
    dict_to_python(v)
end

function value_to_python(v::Bool)
    v ? "True" : "False"
end

"""
    build_cds_retrieve_script(dataset, request, outfile) -> String

Generate a Python script that calls cdsapi.Client().retrieve(...).
"""
function build_cds_retrieve_script(dataset::String,
                                    request::Dict{String, Any},
                                    outfile::String)
    req_py = dict_to_python(request)
    return """
import cdsapi
import os

os.makedirs(os.path.dirname("$outfile"), exist_ok=True)
c = cdsapi.Client()
c.retrieve("$dataset", $req_py, "$outfile")
sz = os.path.getsize("$outfile") / 1e6
print(f"  OK: {sz:.1f} MB -> $outfile")
"""
end

"""
    build_mars_retrieve_script(request, outfile) -> String

Generate a Python script that calls ecmwfapi.ECMWFService("mars").execute(...).
"""
function build_mars_retrieve_script(request::Dict{String, Any},
                                     outfile::String)
    req_py = dict_to_python(request)
    return """
import ecmwfapi
import os

os.makedirs(os.path.dirname("$outfile"), exist_ok=True)
server = ecmwfapi.ECMWFService("mars")
server.execute($req_py, "$outfile")
sz = os.path.getsize("$outfile") / 1e6
print(f"  OK: {sz:.1f} MB -> $outfile")
"""
end

# ---------------------------------------------------------------------------
# DownloadUtils — shared download utility with Content-Length verification
#
# All GEOS download scripts include this module for consistent integrity
# checking. The WashU archive returns Content-Length headers, so we compare
# local filesize against the server's reported size after each download.
#
# Usage in download scripts:
#   include(joinpath(@__DIR__, "download_utils.jl"))
#   using .DownloadUtils
#   ok = verified_download(url, dest)
# ---------------------------------------------------------------------------

module DownloadUtils

using Downloads
using Dates

export verified_download, verify_downloads

"""
    verified_download(url, dest; max_retries=3) → Bool

Download `url` to `dest` with Content-Length integrity checking.

1. HTTP HEAD request to get Content-Length before downloading
2. If file already exists with matching size → skip (verified)
3. If file exists but wrong size → delete and re-download
4. After download: compare local size against Content-Length
5. If mismatch: delete corrupt file and retry

Falls back to `filesize > 1 MB` heuristic if server doesn't provide Content-Length.
"""
function verified_download(url::String, dest::String; max_retries::Int=3)
    remote_size = _get_content_length(url)

    # Check existing file against remote Content-Length
    if isfile(dest)
        local_size = filesize(dest)
        if remote_size > 0 && local_size == remote_size
            @info "  Verified: $(basename(dest)) ($(local_size ÷ 1_000_000) MB)"
            return true
        elseif remote_size > 0
            @warn "  Size mismatch: $(basename(dest)) " *
                  "local=$(local_size) expected=$(remote_size) — re-downloading"
            rm(dest; force=true)
        elseif local_size > 1_000_000
            # No Content-Length available, use old heuristic
            @info "  Exists (no remote size check): $(basename(dest)) " *
                  "($(local_size ÷ 1_000_000) MB)"
            return true
        end
    end

    mkpath(dirname(dest))
    for attempt in 1:max_retries
        try
            @info "  Downloading $(basename(dest)) (attempt $attempt)..."
            Downloads.download(url, dest)
            local_size = filesize(dest)
            @info "    → $(local_size ÷ 1_000_000) MB"

            # Verify against Content-Length
            if remote_size > 0 && local_size != remote_size
                @warn "    Truncated: got $(local_size) bytes, " *
                      "expected $(remote_size) — retrying"
                rm(dest; force=true)
                attempt < max_retries && sleep(5 * attempt)
                continue
            end

            return true
        catch e
            @warn "  Attempt $attempt failed: $e"
            isfile(dest) && rm(dest; force=true)
            attempt < max_retries && sleep(5 * attempt)
        end
    end
    @error "  Failed after $max_retries attempts: $(basename(dest))"
    return false
end

"""
    _get_content_length(url) → Int

Get remote file size via HTTP HEAD request. Returns 0 on failure.
"""
function _get_content_length(url::String)
    try
        resp = Downloads.request(url; method="HEAD")
        for (k, v) in resp.headers
            if lowercase(k) == "content-length"
                return parse(Int, v)
            end
        end
    catch
    end
    return 0
end

"""
    verify_downloads(dir, dates, file_pattern; url_builder=nothing) → NamedTuple

Scan a download directory for missing or corrupt files.
`file_pattern(datestr)` returns the expected filename for a date string.
`url_builder(datestr)` builds the URL to check Content-Length (optional).

Returns `(; ok, corrupt, missing)` — vectors of file paths.
"""
function verify_downloads(dir::String, dates,
                          file_pattern::Function;
                          url_builder::Union{Nothing, Function}=nothing)
    ok = String[]
    corrupt = String[]
    missing_files = String[]

    for date in dates
        datestr = Dates.format(date, "yyyymmdd")
        fname = file_pattern(datestr)
        # Check in date subdirectory first, then flat directory
        path = joinpath(dir, datestr, fname)
        if !isfile(path)
            path = joinpath(dir, fname)
        end

        if !isfile(path)
            push!(missing_files, path)
        elseif url_builder !== nothing
            url = url_builder(datestr)
            expected = _get_content_length(url)
            if expected > 0 && filesize(path) != expected
                push!(corrupt, path)
            else
                push!(ok, path)
            end
        else
            push!(ok, path)
        end
    end

    return (; ok, corrupt, missing=missing_files)
end

end # module

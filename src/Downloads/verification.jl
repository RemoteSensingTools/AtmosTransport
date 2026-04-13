# ===========================================================================
# Download verification — ported from scripts/downloads/download_utils.jl
#
# Content-Length integrity checking for HTTP-based downloads.
# ===========================================================================

using Downloads: Downloads as DL

"""
    verified_download(url, dest; max_retries=3) -> Bool

Download `url` to `dest` with Content-Length integrity checking.

1. HTTP HEAD → Content-Length
2. If file exists with matching size → skip (verified)
3. If exists but wrong size → delete and re-download
4. After download: verify local size matches Content-Length
5. If mismatch: delete corrupt file and retry
"""
function verified_download(url::String, dest::String; max_retries::Int=3)
    remote_size = _get_content_length(url)

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
            @info "  Exists (no remote size check): $(basename(dest)) " *
                  "($(local_size ÷ 1_000_000) MB)"
            return true
        end
    end

    mkpath(dirname(dest))
    for attempt in 1:max_retries
        try
            @info "  Downloading $(basename(dest)) (attempt $attempt)..."
            DL.download(url, dest)
            local_size = filesize(dest)
            @info "    → $(local_size ÷ 1_000_000) MB"

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
    _get_content_length(url) -> Int

Get remote file size via HTTP HEAD request. Returns 0 on failure.
"""
function _get_content_length(url::String)
    try
        resp = DL.request(url; method="HEAD")
        for (k, v) in resp.headers
            lowercase(k) == "content-length" && return parse(Int, v)
        end
    catch e
        @debug "HEAD request failed for $url: $e"
    end
    return 0
end

"""
    verify_downloads(dir, dates, file_pattern; url_builder=nothing) -> NamedTuple

Scan a download directory for missing or corrupt files.
Returns `(; ok, corrupt, missing)`.
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

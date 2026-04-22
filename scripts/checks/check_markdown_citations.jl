#!/usr/bin/env julia
# Markdown citation checker — enforce principle 8 of plan 23.
#
# For every `[text](path)` or `[text](path#Lnn)` link in a given
# markdown file, assert:
#
# - The target path resolves (relative to the markdown file's dir,
#   or absolute if the path starts with `/`).
# - If the link includes a `#Lnn` or `#Lnn-Lmm` anchor, the line
#   number (or range) is within the file's line count.
#
# External links (`http://`, `https://`, `mailto:`) are ignored.
# In-page anchors (`#section`, `#some-header`) are ignored — only
# `#Lnn[-Lmm]` style line anchors are checked.
#
# Usage:
#
#   julia --project=. scripts/checks/check_markdown_citations.jl <file.md> [file2.md ...]
#
# Exit code 0 on all-resolve, 1 on any broken citation.

const LINK_PATTERN = r"\[([^\]]*)\]\(([^)\s]+)\)"
const LINE_ANCHOR = r"^([^#]+)(?:#L(\d+)(?:-L(\d+))?)?$"

struct Broken
    file::String
    line::Int
    text::String
    target::String
    reason::String
end

function repo_root(start::String)
    dir = abspath(dirname(start))
    while dir != "/"
        isfile(joinpath(dir, "Project.toml")) && return dir
        dir = dirname(dir)
    end
    return nothing
end

function resolve_path(md_file::AbstractString, target::AbstractString)
    # `/absolute/path` is literal filesystem; everything else is
    # resolved relative to the markdown file's directory.
    path_part = match(LINE_ANCHOR, target)
    path_part === nothing && return (nothing, nothing, nothing)
    raw_path, lo_str, hi_str = path_part.captures

    lo = lo_str === nothing ? nothing : parse(Int, lo_str)
    hi = hi_str === nothing ? nothing : parse(Int, hi_str)

    resolved = startswith(raw_path, '/') ? String(raw_path) :
               abspath(joinpath(dirname(md_file), raw_path))

    return (resolved, lo, hi)
end

function is_external(target::AbstractString)
    return startswith(target, "http://") ||
           startswith(target, "https://") ||
           startswith(target, "mailto:") ||
           startswith(target, "#")
end

function check_file(md_file::String)
    broken = Broken[]
    isfile(md_file) || return [Broken(md_file, 0, "", md_file,
                                       "markdown file itself does not exist")]

    for (lineno, line) in enumerate(eachline(md_file))
        for m in eachmatch(LINK_PATTERN, line)
            text = m.captures[1]
            target = m.captures[2]
            is_external(target) && continue

            resolved, lo, hi = resolve_path(md_file, target)
            if resolved === nothing
                push!(broken, Broken(md_file, lineno, text, target,
                                     "malformed link target"))
                continue
            end
            if !isfile(resolved)
                # Tolerate directories: if the target is a directory,
                # pass. Actual broken paths error.
                if !isdir(resolved)
                    push!(broken, Broken(md_file, lineno, text, target,
                                         "path does not exist: $resolved"))
                end
                continue
            end

            if lo !== nothing
                nlines = countlines(resolved)
                if lo > nlines
                    push!(broken, Broken(md_file, lineno, text, target,
                                         "line anchor L$lo exceeds file length $nlines"))
                elseif hi !== nothing && hi > nlines
                    push!(broken, Broken(md_file, lineno, text, target,
                                         "line range L$lo-L$hi exceeds file length $nlines"))
                elseif hi !== nothing && hi < lo
                    push!(broken, Broken(md_file, lineno, text, target,
                                         "inverted line range L$lo-L$hi"))
                end
            end
        end
    end

    return broken
end

function main(files::Vector{String})
    all_broken = Broken[]
    for f in files
        append!(all_broken, check_file(f))
    end

    if isempty(all_broken)
        println("All ", length(files), " markdown file(s) have resolving citations.")
        return 0
    end

    println("Broken citations:\n")
    for b in all_broken
        println("  $(b.file):$(b.line)")
        println("    text:   $(b.text)")
        println("    target: $(b.target)")
        println("    reason: $(b.reason)")
        println()
    end
    println("Total broken: $(length(all_broken))")
    return 1
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) == 0
        println(stderr, "Usage: $(basename(@__FILE__)) <file.md> [file2.md ...]")
        exit(2)
    end
    exit(main(collect(String, ARGS)))
end

# Resource ownership for the driven file/window loop. Numerical stepping does
# not own external driver handles; this runner closes what it opens.

function _with_run_resource(f, resource)
    result = try
        f()
    catch run_error
        try
            close(resource)
        catch cleanup_error
            # A cleanup failure must not hide the original run failure.
            throw(CompositeException(Any[run_error, cleanup_error]))
        end
        rethrow()
    end
    close(resource)
    return result
end

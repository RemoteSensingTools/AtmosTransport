# ---------------------------------------------------------------------------
# Buffering strategies for met-data I/O ↔ compute overlap
#
# SingleBuffer: load met window, then compute (sequential)
# DoubleBuffer: ping-pong — load N+1 while computing on N (overlapped)
# ---------------------------------------------------------------------------

"""
$(TYPEDEF)

Supertype for met-data buffering strategies.
"""
abstract type AbstractBufferingStrategy end

"""
$(TYPEDEF)

Sequential single-buffer strategy. Each met window is loaded, transferred
to GPU, and then advection runs. No overlap between I/O and compute.
"""
struct SingleBuffer <: AbstractBufferingStrategy end

"""
$(TYPEDEF)

Double-buffer (ping-pong) strategy. Two sets of met buffers alternate
so that CPU I/O for window N+1 overlaps with GPU compute on window N.
"""
struct DoubleBuffer <: AbstractBufferingStrategy end

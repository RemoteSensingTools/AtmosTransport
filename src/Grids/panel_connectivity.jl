# ---------------------------------------------------------------------------
# Cubed-sphere panel connectivity
#
# GEOS-FP native cubed-sphere panel numbering (from nf dimension):
#   Panel 1: equatorial, centered ~350°E  (Africa/Atlantic)
#   Panel 2: equatorial, centered ~80°E   (Asia/Indian Ocean)
#   Panel 3: north polar cap              (lat > 35°N)
#   Panel 4: equatorial, centered ~170°E  (Pacific, rotated 90° CW)
#   Panel 5: equatorial, centered ~260°E  (Americas, rotated 90° CW)
#   Panel 6: south polar cap              (lat < 35°S)
#
# Panels 1 & 2 have local axes X=east, Y=north (standard orientation).
# Panels 4 & 5 have local axes X=south, Y=east (rotated 90° CW).
# Panels 3 & 6 are polar caps with curvilinear local axes.
#
# Each panel has 4 edges: north (highY), south (lowY), east (highX), west (lowX).
# `orientation` encodes the along-edge direction when transferring halo data:
#   0 = aligned (s on dst maps to s on src)
#   2 = reversed (s on dst maps to Nc+1-s on src)
#
# Derived from GEOS-FP C720 file corner coordinates and verified against
# the `contacts` and `anchor` variables in the NetCDF metadata.
#
# References:
#   Putman & Lin (2007) — FV3 cubed-sphere grid
#   Martin et al. (2022, GMD) — GCHP v13
# ---------------------------------------------------------------------------

"""
$(SIGNATURES)

Return the GEOS-FP native cubed-sphere panel connectivity.

Panel numbering follows the GEOS-FP file convention (nf=1..6), which differs
from the textbook convention. Connectivity and orientations are verified
against the corner coordinate data in the native C720 NetCDF files.

Edge-to-edge connections (Panel p edge → Panel q edge):
  P1 north→P3 west(rev)   P1 south→P6 north(aln)   P1 east→P2 west(aln)   P1 west→P5 north(rev)
  P2 north→P3 south(aln)  P2 south→P6 east(rev)    P2 east→P4 south(rev)  P2 west→P1 east(aln)
  P3 north→P5 west(rev)   P3 south→P2 north(aln)   P3 east→P4 west(aln)   P3 west→P1 north(rev)
  P4 north→P5 south(aln)  P4 south→P2 east(rev)    P4 east→P6 south(rev)  P4 west→P3 east(aln)
  P5 north→P1 west(rev)   P5 south→P4 north(aln)   P5 east→P6 west(aln)   P5 west→P3 north(rev)
  P6 north→P1 south(aln)  P6 south→P4 east(rev)    P6 east→P2 south(rev)  P6 west→P5 east(aln)
"""
function default_panel_connectivity()
    # (north, south, east, west) for each panel
    # Each entry: (panel=neighbor_panel, orientation=along_edge_code)
    return PanelConnectivity((
        # Panel 1 (equatorial ~350°E)
        ((panel=3, orientation=2), (panel=6, orientation=0),
         (panel=2, orientation=0), (panel=5, orientation=2)),
        # Panel 2 (equatorial ~80°E)
        ((panel=3, orientation=0), (panel=6, orientation=2),
         (panel=4, orientation=2), (panel=1, orientation=0)),
        # Panel 3 (north polar cap)
        ((panel=5, orientation=2), (panel=2, orientation=0),
         (panel=4, orientation=0), (panel=1, orientation=2)),
        # Panel 4 (equatorial ~170°E, rotated)
        ((panel=5, orientation=0), (panel=2, orientation=2),
         (panel=6, orientation=2), (panel=3, orientation=0)),
        # Panel 5 (equatorial ~260°E, rotated)
        ((panel=1, orientation=2), (panel=4, orientation=0),
         (panel=6, orientation=0), (panel=3, orientation=2)),
        # Panel 6 (south polar cap)
        ((panel=1, orientation=0), (panel=4, orientation=2),
         (panel=2, orientation=2), (panel=5, orientation=0)),
    ))
end

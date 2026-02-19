# ---------------------------------------------------------------------------
# Cubed-sphere panel connectivity
#
# Standard gnomonic cubed-sphere panel numbering (GEOS convention):
#   Panel 1: front  (0°E centered)
#   Panel 2: east   (90°E centered)
#   Panel 3: back   (180°E centered)
#   Panel 4: west   (90°W centered)
#   Panel 5: north pole
#   Panel 6: south pole
#
# Each panel has 4 edges: north, south, east, west.
# `orientation` encodes the rotation needed when transferring halo data
# across a panel boundary: 0 = aligned, 1 = 90° CW, 2 = 180°, 3 = 90° CCW.
# ---------------------------------------------------------------------------

"""
$(SIGNATURES)

Return the standard GEOS-convention panel connectivity for a gnomonic
cubed-sphere with 6 panels.
"""
function default_panel_connectivity()
    # (north, south, east, west) for each panel
    # Each entry: (panel=neighbor_panel, orientation=rotation_code)
    return PanelConnectivity((
        # Panel 1 (front)
        ((panel=5, orientation=0), (panel=6, orientation=0),
         (panel=2, orientation=0), (panel=4, orientation=0)),
        # Panel 2 (east)
        ((panel=5, orientation=1), (panel=6, orientation=3),
         (panel=3, orientation=0), (panel=1, orientation=0)),
        # Panel 3 (back)
        ((panel=5, orientation=2), (panel=6, orientation=2),
         (panel=4, orientation=0), (panel=2, orientation=0)),
        # Panel 4 (west)
        ((panel=5, orientation=3), (panel=6, orientation=1),
         (panel=1, orientation=0), (panel=3, orientation=0)),
        # Panel 5 (north pole)
        ((panel=3, orientation=2), (panel=1, orientation=0),
         (panel=2, orientation=3), (panel=4, orientation=1)),
        # Panel 6 (south pole)
        ((panel=1, orientation=0), (panel=3, orientation=2),
         (panel=2, orientation=1), (panel=4, orientation=3)),
    ))
end

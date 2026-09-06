using AtmosTransport
s=join(readlines("test/core/test_cubed_sphere_advection.jl")[1:167],"\n")
s=replace(s,"panels_am, panels_bm, panels_cm = make_mirrored_cs_horizontal_fluxes(mesh, Nz)"=>"panels_am, panels_bm, panels_cm = make_mirrored_cs_horizontal_fluxes(mesh, Nz)\n    panels_am=map(a->a.*GAIN[],panels_am);panels_bm=map(a->a.*GAIN[],panels_bm)")
const GAIN=Ref(1.)
include_string(Main,s)
for g in (1.,10.,100.), scheme in (UpwindScheme(),PPMScheme(),LinRoodPPMScheme(7))
    GAIN[]=g
    println("SEAM ",g," ",typeof(scheme)," ",run_mirrored_seam_advection_conservation(scheme));flush(stdout)
end

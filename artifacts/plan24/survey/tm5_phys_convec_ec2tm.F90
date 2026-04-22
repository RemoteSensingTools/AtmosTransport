!###############################################################################
!
! Convert ecmwf convective fields to TM convective fields.
!
! Original code by Dirk Olivie, KNMI.
!
! Research notes by DO :
!
!  sign:
!   mfup : positive
!   dtup : positive
!   mfdo : negative
!   dtdo : positive
!
!  input mass fluxes are contaminated by profiles :
!  the same for every grid point, different for each time step
!   mfup : not contaminated
!   dtup : between  0.      and -0.99e-19
!   mfdo : between -0.99e-6 and  0.99e-6
!   dtdo : betweeb  0.      and -0.99e-20
!
!  minumum values:
!   mfup : 1.e-6
!   dtup : 1.e-10
!   mfdo : 1.e-6
!   dtdo : 1.e-10
!
!  remarks:
!
!   1. start updraft is not always the lowest level : sometimes second or third
!
!   2. end downdraft is not always lowest level : sometimes 6th level
!
!   3. in the updraft cloud : delta_updraft = 1.e-4
!                             epsilon_updraft = 1.e-5
!      this automatically/naturally decreases the mass flux : GOOD GOOD GOOD
!
!   4. there is a part incloud where the mass flus is constant;
!      later over many layer (order of 10) the mass flux decreases
!
!   5. detrainment updraft 2 or 3 levels higher (sometimes 15 levels : i = 17565)
!      then the updraft mass flux
!
!   6. only detrainment downdraft lowest level different from zero
!
!   7. the cloudbase of updraft and downdraft correspond sometimes quite well
!
!###############################################################################
#include "macros.h"
!###############################################################################


module phys_convec_ec2tm

  use GO, only : gol, goPr, goErr

  implicit none

  ! --- in/out -----------------------------

  private

  public  ::  ECconv_to_TMconv
  public  ::  convec_mfldet_to_entdet

  ! --- const ---------------------------------------

  character(len=*), parameter  ::  mname = 'phys_convec_ec2tm'


contains


  ! =============================================================

  !
  ! o Level order is always top->down (ecmwf order)
  !
  ! o ECMWF mass fluxes mflu_ec and mfld_ec are defined for half levels.
  !   According to the original code, the mass fluxes read from the grib
  !   files are valid for the upper half level of a cell.
  !   The bottom values mflu_ec(lm) and mfld_ec(lm) should have been
  !   set to zero.
  !


  subroutine ECconv_to_TMconv( lm, zh_ec, &
                               mflu_ec, detu_ec, mfld_ec, detd_ec, &
                               entu   , detu   , entd   , detd   , &
                               status )

    ! --- in/out -------------------------------

    integer, intent(in)   ::  lm
    real, intent(in)      ::    zh_ec(0:lm)   ! geopot. height       m        (half lev)
    real, intent(in)      ::  mflu_ec(0:lm)   !   updr. mass flux    kg/m2/s  (half lev)
    real, intent(in)      ::  detu_ec(1:lm)   !   updr. detr. rate   kg/m3/s  (full lev)
    real, intent(in)      ::  mfld_ec(0:lm)   ! downdr. mass flux    kg/m2/s  (half lev)
    real, intent(in)      ::  detd_ec(1:lm)   ! downdr. detr. rate   kg/m3/s  (full lev)

    real, intent(out)     ::  entu(1:lm)      !   updr. entr.        kg/m2/s  (full lev)
    real, intent(out)     ::  detu(1:lm)      ! downdr. detr.        kg/m2/s  (full lev)
    real, intent(out)     ::  entd(1:lm)      !   updr. entr.        kg/m2/s  (full lev)
    real, intent(out)     ::  detd(1:lm)      ! downdr. detr.        kg/m2/s  (full lev)

    integer, intent(out)  ::  status

    ! --- const -----------------------------------------

    character(len=*), parameter  ::  rname = mname//'/ECconv_to_TMconv'

    ! --- local ------------------------------

    integer               :: l           ! level number

    real                  ::  mflu(0:lm)   !   updr. mass flux    kg/m2/s (half lev)
    real                  ::  mfld(0:lm)   ! downdr. mass flux    kg/m2/s (half lev)

    ! layer thickness
    real                  :: dz          ! m

    ! updraft top
    integer               :: uptop       ! layer
    logical               :: uptop_found

    ! downdraft top
    integer               :: dotop       ! layer
    logical               :: dotop_found

    ! --- begin --------------------------------

    ! copy ecmwf arrays into local arrays
    mflu = mflu_ec
    detu = detu_ec
    entu = 0.0
    mfld = mfld_ec
    detd = detd_ec
    entd = 0.0

    ! removing small values due to gribbing
    where ( mflu <  1.e-6  ) mflu = 0.0
    where ( detu <  1.e-10 ) detu = 0.0
    where ( mfld > -1.e-6  ) mfld = 0.0
    where ( detd <  1.e-10 ) detd = 0.0

    ! height integral of detrainment rates
    do l = 1, lm
      dz = zh_ec(l-1) - zh_ec(l)
      detu(l) = detu(l)*dz    ! kg/m3/s  ->  kg/m2/s
      detd(l) = detd(l)*dz    ! kg/m3/s  ->  kg/m2/s
    enddo

    ! find updraftop
    uptop = 0
    uptop_found = .false.
    do l = 1, lm
      if ( mflu(l) > 0.0 ) then
        uptop       = l
        uptop_found = .true.
        exit
      end if
    end do

    ! find downdrafttop
    dotop = 0
    dotop_found = .false.
    do l = 1, lm
      if ( mfld(l) < 0.0 ) then
        dotop       = l
        dotop_found = .true.
        exit
      end if
    end do

    ! updraft
    if ( uptop_found ) then
       ! set updr. entr/detr to zero above uptop
       do l = 1, uptop-1
          entu(l) = 0.0
          detu(l) = 0.0
       enddo
       ! loop from uptop to bot:
       do l = uptop, lm
          ! entr  = out through top - in through bot + out by detr
          entu(l) =    mflu(l-1)    -     mflu(l)    +    detu(l)
       enddo
    else
       ! no updr. at all
       mflu(:) = 0.0
       detu(:) = 0.0
       entu(:) = 0.0
    endif

    ! downdraft
    if ( dotop_found ) then
       ! set updr. entr/detr to zero above dotop
       !AJS: BUG ? do l = 1, dotop+1
       do l = 1, dotop-1
          detd(l) = 0.0
          entd(l) = 0.0
       enddo
       ! loop from dotop to bot:
       do l = dotop, lm
          ! entr  = out through top - in through bot + out by detr
          entd(l) =    mfld(l-1)    -     mfld(l)    +   detd(l)
       enddo
    else
       ! no downdr. at all
       mfld(:) = 0.0
       detd(:) = 0.0
       entd(:) = 0.0
    endif

    ! check on negative values:
    do l = 1, lm
       if ( entu(l) < 0.0 ) then
          detu(l) = detu(l) - entu(l)
          entu(l) = 0.0
       endif
       if ( detu(l) < 0.0 ) then
          entu(l) = entu(l) - detu(l)
          detu(l) = 0.0
       endif
       if ( entd(l) < 0.0 ) then
          detd(l) = detd(l) - entd(l)
          entd(l) = 0.0
       endif
       if ( detd(l) < 0.0 ) then
          entd(l) = entd(l) - detd(l)
          detd(l) = 0.0
       endif
    enddo

    ! ok
    status = 0

  end subroutine ECconv_to_TMconv


  ! =============================================================

  !
  ! Generalized version of ECconv_to_TMconv .
  !
  ! o Level order is either 'u' upwards (tm order) or 'd' downwards (ecmwf order)
  !
  ! o ECMWF mass fluxes mflu_ec and mfld_ec are defined for half levels.
  !   According to the original code, the mass fluxes read from the grib
  !   files are valid for the upper half level of a cell.
  !   The bottom values mflu_ec(lm) and mfld_ec(lm) should have been
  !   set to zero.
  !


  subroutine convec_mfldet_to_entdet( updo, lm, zh_ec, &
                                      mflu_ec, detu_ec, mfld_ec, detd_ec, &
                                      entu   , detu   , entd   , detd   , &
                                      status )

    ! --- in/out -------------------------------

    character(len=1)      ::  updo
    integer, intent(in)   ::  lm
    real, intent(in)      ::    zh_ec(0:lm)   ! geopot. height       m        (half lev)
    real, intent(in)      ::  mflu_ec(0:lm)   !   updr. mass flux    kg/m2/s  (half lev)
    real, intent(in)      ::  detu_ec(1:lm)   !   updr. detr. rate   kg/m3/s  (full lev)
    real, intent(in)      ::  mfld_ec(0:lm)   ! downdr. mass flux    kg/m2/s  (half lev)
    real, intent(in)      ::  detd_ec(1:lm)   ! downdr. detr. rate   kg/m3/s  (full lev)

    real, intent(out)     ::  entu(1:lm)      !   updr. entr.        kg/m2/s  (full lev)
    real, intent(out)     ::  detu(1:lm)      ! downdr. detr.        kg/m2/s  (full lev)
    real, intent(out)     ::  entd(1:lm)      !   updr. entr.        kg/m2/s  (full lev)
    real, intent(out)     ::  detd(1:lm)      ! downdr. detr.        kg/m2/s  (full lev)

    integer, intent(out)  ::  status

    ! --- const -----------------------------------------

    character(len=*), parameter  ::  rname = mname//'/convec_mfldet_to_entdet'

    ! --- local ------------------------------

    integer               :: l           ! level number

    real                  ::  mflu(0:lm)   !   updr. mass flux    kg/m2/s (half lev)
    real                  ::  mfld(0:lm)   ! downdr. mass flux    kg/m2/s (half lev)

    ! layer thickness
    real                  :: dz          ! m

    ! updraft top
    integer               :: uptop       ! layer
    logical               :: uptop_found

    ! downdraft top
    integer               :: dotop       ! layer
    logical               :: dotop_found

    ! --- begin --------------------------------

    ! copy ecmwf arrays into local arrays
    mflu = mflu_ec
    detu = detu_ec
    entu = 0.0
    mfld = mfld_ec
    detd = detd_ec
    entd = 0.0

    ! removing small values due to gribbing
    where ( mflu <  1.e-6  ) mflu = 0.0
    where ( detu <  1.e-10 ) detu = 0.0
    where ( mfld > -1.e-6  ) mfld = 0.0
    where ( detd <  1.e-10 ) detd = 0.0

    ! height integral of detrainment rates
    do l = 1, lm
      dz = abs( zh_ec(l) - zh_ec(l-1) )  ! m
      detu(l) = detu(l)*dz    ! kg/m3/s  ->  kg/m2/s
      detd(l) = detd(l)*dz    ! kg/m3/s  ->  kg/m2/s
    end do

    ! levels upwards (TM) or downwards (EC)
    select case ( updo )

      ! ***

      case ( 'u', 'U' )

        ! find updraftop
        uptop = 0
        uptop_found = .false.
        ! loop over layers from top to bot:
        do l = lm, 1, -1
          ! flux through bottom ? then this is the top level
          if ( mflu(l-1) > 0.0 ) then
            uptop       = l
            uptop_found = .true.
            exit
          end if
        end do

        ! compute entrainments of updraught
        if ( uptop_found ) then
          ! loop from bot to uptop:
          do l = 1, uptop
             ! entr  = out through top - in through bot + out by detr
             entu(l) =     mflu(l)     -     mflu(l-1)  +    detu(l)
          enddo
          ! set updr. entr/detr to zero above uptop
          do l = uptop+1, lm
             entu(l) = 0.0
             detu(l) = 0.0
          enddo
        else
          ! no updr. at all
          detu(:) = 0.0
          entu(:) = 0.0
        endif

        ! find downdrafttop
        dotop = 0
        dotop_found = .false.
        ! loop over layers from top to bot:
        do l = lm, 1, -1
          ! flux through bottom ? then this is the top level
          if ( mfld(l-1) < 0.0 ) then
            dotop       = l
            dotop_found = .true.
            exit
          end if
        end do

        ! compute entrainments of downdraught
        if ( dotop_found ) then
          ! loop from bot to dotop:
          do l = 1, dotop
             ! entr  = out through top - in through bot + out by detr
             entd(l) =    mfld(l)      -     mfld(l-1)  +   detd(l)
          enddo
          ! set updr. entr/detr to zero above dotop
          do l = dotop+1, lm
             detd(l) = 0.0
             entd(l) = 0.0
          enddo
        else
          ! no downdr. at all
          detd(:) = 0.0
          entd(:) = 0.0
        endif

      ! ***

      case ( 'd', 'D' )

        ! find updraftop
        uptop = 0
        uptop_found = .false.
        ! loop over layers from top to bot:
        do l = 1, lm
          ! flux through bottom ? then this is the top level
          if ( mflu(l) > 0.0 ) then
            uptop       = l
            uptop_found = .true.
            exit
          end if
        end do

        ! compute entrainments of updraught
        if ( uptop_found ) then
           ! set updr. entr/detr to zero above uptop
           do l = 1, uptop-1
              entu(l) = 0.0
              detu(l) = 0.0
           enddo
           ! loop from uptop to bot:
           do l = uptop, lm
              ! entr  = out through top - in through bot + out by detr
              entu(l) =    mflu(l-1)    -     mflu(l)    +    detu(l)
           enddo
        else
           ! no updr. at all
           detu(:) = 0.0
           entu(:) = 0.0
        endif

        ! find downdrafttop
        dotop = 0
        dotop_found = .false.
        ! loop over layers from top to bot:
        do l = 1, lm
          ! flux through bottom ? then this is the top level
          if ( mfld(l) < 0.0 ) then
            dotop       = l
            dotop_found = .true.
            exit
          end if
        end do

        ! compute entrainments of downdraught
        if ( dotop_found ) then
           ! set updr. entr/detr to zero above dotop
           do l = 1, dotop-1
              detd(l) = 0.0
              entd(l) = 0.0
           enddo
           ! loop from dotop to bot:
           do l = dotop, lm
              ! entr  = out through top - in through bot + out by detr
              entd(l) =    mfld(l-1)    -     mfld(l)    +   detd(l)
           enddo
        else
           ! no downdr. at all
           detd(:) = 0.0
           entd(:) = 0.0
        endif

      ! ***

      case default

        write (gol,'("unsupported updo : ",a)') updo; call goErr
        write (gol,'("in ",a)') rname; call goErr; status=1; return

    end select


    ! check on negative values:
    do l = 1, lm
       if ( entu(l) < 0.0 ) then
          detu(l) = detu(l) - entu(l)
          entu(l) = 0.0
       endif
       if ( detu(l) < 0.0 ) then
          entu(l) = entu(l) - detu(l)
          detu(l) = 0.0
       endif
       if ( entd(l) < 0.0 ) then
          detd(l) = detd(l) - entd(l)
          entd(l) = 0.0
       endif
       if ( detd(l) < 0.0 ) then
          entd(l) = entd(l) - detd(l)
          detd(l) = 0.0
       endif
    enddo


    ! ok
    status = 0

  end subroutine convec_mfldet_to_entdet


end module phys_convec_ec2tm

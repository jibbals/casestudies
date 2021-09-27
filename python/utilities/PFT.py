#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 10:00:09 2019

    Kevin's PFT code ported to python
    
    HISTORY
        25/10/2019: working/tested against fortran.
        20/1/2020: Updated: added Wmax=10 parameter 
            to limit effects stretching atmosphere on PFT

@author: jesse
"""

###
## IMPORTS
###
import numpy


###
## GLOBALS
###

__DEBUG__=False   # Set this to True to print lots of info

Cpd=1005.7           # Specific heat of dry air (J/kg/K)
Rd=287.04            # Gas constant for dry air (J/kg/K)
kappa=Rd/Cpd         # Ratio of gas constant to specific heat
LV=2.501e6           # Latent heat of vapourisation (J/kg)
P0=1.0e5             # Standard pressure (Pa)
grav=9.8             # Acceleration due to gravity (m/s^2)
epsln=0.622          # Ratio of gas constants of dry air and water
pi=3.1415926536      # pi


###
## METHODS
###

def PFT(TT0,qq0,uu0,vv0,ww0,th0,pr0,zsfc,psfc,Tsfc,
        phi=6e-5,DbSP=.001,ni=1001,nj=20,zp=1.0,beta_e=0.4,Pmin=35000,
        betaSPmin=.002,Wmin=.05,Umin=1.0,Prcntg=6,Wmax=10.0):
    """Calculate required energy to form a pyroCB.
    
    Port of Kevin Tory fortran code to calculate PFT.
    pd argument removed (array lengths of inputs used instead)
    default arguments added for phi, DbSP, ....
    some loops removed to reduce computation time (comments from fortran code remain)
    pressure level based arrays are lengthened by 1 to pretend they're base-1 like fortran
    
    
    ARGUMENTS
    ---------
    real, intent(in), dimension(:) :: TT       ! Column temperature (K)
    real, intent(in), dimension(:) :: qq       ! Column specific humidity (kg/kg)
    real, intent(in), dimension(:) :: uu       ! Column zonal wind (m/s)
    real, intent(in), dimension(:) :: vv       ! Column meridional wind (m/s)
    real, intent(in), dimension(:) :: ww       ! Column vertical wind (m/s)
    real, intent(in), dimension(:) :: th       ! Column potential temperature (K)
    real, intent(in), dimension(:) :: pr       ! Column pressure (Pa)
       
    real, intent(in) :: zsfc        ! Surface height (m)
    real, intent(in) :: psfc        ! Surface pressure (Pa)
    real, intent(in) :: Tsfc        ! Surface temperature (K)
    
    real, intent(in) :: phi         ! Ratio of fire moisture to heat (kg/kg/K)
    real, intent(in) :: DbSP        ! Beta increment along the saturation point curve
    integer, intent(in) :: ni       ! Number of increments along the SP curve
    integer, intent(in) :: nj       ! Number of iterations for solving plume centre-line height
    real, intent(in) :: zp          ! The power to which zz is raised in calculating wt. (2/3 for MTT, 1.0 for Briggs)
    real, intent(in) :: beta_e      ! Briggs entrainment paramter (Briggs uses 0.6, we may use up to 0.9)
    real, intent(in) :: Pmin        ! The minimum pressure the moist plume needs to rise to for 
                                    !   pyroCb to be considered.  Probably choose 300 to 250 hPa, 
                                    !   just below the tropopause.
    real, intent(in) :: betaSPmin   ! A minimum value of beta to be added to betaSP, to account 
                                    !   for buoyancy losses from entrainment etc. above the condensation level
    real, intent(in) :: Wmin        ! Minimum value of mixed-layer wind speed to be considered to influence zb
    real, intent(in) :: Umin        ! Minimum value of mixed-layer horizontal wind speed
    integer, intent(in) :: Prcntg   ! Index for specifying percentage of plume required to reach zFC
    real, parameter :: Wmax=10.0    ! Upper limit to WML (added 2020-01-15 Quick fix)
    
    
    OUTPUTS
    ------
    UML        ! Mixed-layer horizontal velocity magnitude
    Um         ! Mixed-layer zonal wind
    Vm         ! Mixed-layer meridional wind
    betaFC     ! Free convection beta parameter
    DthFC      ! Free-convection plume excess potential temperature (K)
    zFC        ! Free convection height above ground level (m)
    pFC        ! Free convection pressure (Pa)
    Bflux      ! Net buoyancy flux required for the plume to reach zFC (m^4.s^-3)
    Hflux      ! Net heat flux required for the plume to reach zFC (J.s^-1 = W)
    
    HISTORY
    -------
    23/10/2019
        Ported by Jesse (bboyjesse@gmail.com)
    """

    # pressure dimension
    pd = len(TT0)
    
    # to simulate fortran arrays, I set all the arrays here to length+1
    # then all the arrays are treated as if the starting index is 1
    # except for zz, which has an extra surface level
    # this should make it easier to keep consistent with the fortran version
    
    ## Input arrays are lengthened by 1, and the zero index is not used
    NAN = numpy.NaN # not a number
    TT = numpy.append([NAN], TT0)
    qq = numpy.append([NAN], qq0)
    uu = numpy.append([NAN], uu0)
    vv = numpy.append([NAN], vv0)
    ww = numpy.append([NAN], ww0)
    th = numpy.append([NAN], th0)
    pr = numpy.append([NAN], pr0)
    
    
    
    ## Arrays:
    #    ! Declare local vertical profile arrays
    #   real, dimension(0:pd) :: zz     ! Height array at each pressure level with zsfc in the k=0 position
    #   real, dimension(pd)   :: wt     ! Weighting factor used to calculate a weighted mean th and qq 
    #                                   !   representing the relative fractions of th and qq entrained into a plume
    zz = numpy.zeros(pd+1)
    wt = numpy.zeros(pd+1)
    
    #! Arrays for plume circle cuts
    #   real, dimension(21) ::cosphi    ! Fraction of plume radius between the plume circle origin and zFC
    #   real :: bcosphi                 ! beta_e*cosphi(Prcntg)
    #
    cosphi = [1.0, 0.8054, 0.6870, 0.5851, 0.4919, 0.4040, 0.3197, 0.2379, 0.1577, 0.0786, 0.0,
              -0.0786, -0.1577, -0.2379, -0.3197, -0.4040, -0.4919, -0.5851, -0.6870, -0.8054, -1.0]
    
    #! Declare local arrays along the SP curve
    #   real, dimension(ni) :: sSP      ! Saturation specifc entropy. (Column s* is temporarily stored in sSP(1))
    #   real, dimension(ni) :: bSP      ! Buoyancy parameter
    #   real, dimension(ni) :: pSP      ! Pressure
    #   real, dimension(ni) :: TSP      ! Temperature
    #   real, dimension(ni) :: thpl     ! Plume potential temperature 
    #   real, dimension(ni) :: qqpl     ! Plume specific humidity
    # initializing arrays filled with zeros
    sSP = numpy.zeros(ni+1)
    bSP = numpy.zeros(ni+1)
    thpl = numpy.zeros(ni+1)
    qqpl = numpy.zeros(ni+1)
    pSP = numpy.zeros(ni+1)
    TSP = numpy.zeros(ni+1)
    
    #! Declare local iteration arrays
    #   real, dimension(nj) :: zb       ! Effective plume centreline height (m)
    #   real, dimension(nj) :: f        ! Newton's method f(j)=0
    #   real, dimension(nj) ::fdash     ! Newton's Method first derivative of f(j) (f'(j) )
    zb = numpy.zeros(nj)
    f = numpy.zeros(nj)
    fdash = numpy.zeros(nj)
    
    # Find the lowest pressure level above the land surface
    ksfc = numpy.sum(pr0 > psfc) # how many pressures are greater than psfc
    debug("debug: zsfc, psfc, pr[1], ksfc",zsfc, psfc, pr[1], ksfc)
    
    # Find the height index above the mixed-layer LCL (assumes pressure indices begin
    # with 1 at the lowest model level)
    #  Need to force the code to consider more than one pressure level so that a fog
    #  layer does not register as the LCL.  In the future this might need to be set
    #  to a minimum height (e.g., 250 m) when there are more frequent pressure
    #  levels.  Option added 13-09-2018.
    
    ## Note, looping over k in range(1,5) loops over k=1,2,3, and 4. need to add one extra
    # skip first 2 levels in case of fog issue
    kLCL = None
    for k in range(ksfc+3,pd+1):
         recipk = 1.0/float(k)
         thmean = numpy.sum(th[1:k+1])*recipk
         qmean  = numpy.sum(qq[1:k+1])*recipk
         PLCL, TLCL = LCL(thmean,qmean)
         
         if (PLCL > pr[k]):
             kLCL = k
             break
    assert kLCL is not None, "perhaps profile is not all there? could not find kLCL" + str(TT)
        
    debug("debug: kLCL, pr[kLCL], TT[kLCL]",kLCL, pr[kLCL], TT[kLCL])
    
    ## Calculate the weighting factors for a weighted mixed-layer LCL
    # zz array is altitude with an extra level at the surface
    zz[0] = zsfc
    plower = psfc
    Tlower = Tsfc
    for k in range(1,pd+1):
        if(pr[k] > psfc):
            zz[k] = zsfc
        else:
            zz[k] = zz[k-1] + numpy.log(plower/pr[k])*Rd*(TT[k]+Tlower)/(2*grav)
            plower = pr[k]  # Reset for next iteration
            Tlower = TT[k]
    debug("debug: zz[:5],zz[-5:]",zz[:5],zz[-5:])
    
    #   ! Now all sub-ground height levels are set to the topographic height
    #   ! and ksfc is the k-index of the pressure level just below the sfc
    #   !
    #   weight the profile boundary layer using pressure differences
    for k in range(ksfc+1,kLCL+1):
        zh = (zz[k]+zz[k-1])*0.5
        if (k == 1):
            wt[k] = (zh**zp)*(psfc-pr[k])
        else:
            wt[k] = (zh**zp)*(pr[k-1]-pr[k])
    
    # Express the weighting factor as a fraction of the total wt
    wt = wt/numpy.sum(wt)
    debug("debug: wt",wt)
    # Here the weighting factor wt(k) represents the layer between k and k-1
    
    #
    #! Calculate the weighted mixed-layer potential temperature and specific
    #! humidity. 
    #!  These calculations ignore the surface/near-surface values.
    
    thWML = 0.0
    qqWML = 0.0
    for k in range(ksfc+1,kLCL+1):
        if k == 1:
            thWML = th[k]*wt[k]
            qqWML = qq[k]*wt[k]
        else:
            thWML = thWML + wt[k]*(th[k]+th[k-1])*0.5 
            qqWML = qqWML + wt[k]*(qq[k]+qq[k-1])*0.5
        #debug("debug: k, thWML",k, thWML)

    debug("debug: thWML,qqWML",thWML,qqWML)
    
    #! Calculate the  mixed-layer horizontal (UML) and vertical (WML) winds.
    #! Here UML is the magnitude of the average vector wind (not the average wind
    #! speed).  Thus components on different levels can cancel.
    #!  These calculations ignore the surface/near-surface values.
    # in python, slicing an array arr[m:n] does not include the nth element
    # so here we go to kLCL+1
    recipk = 1.0/float(kLCL-ksfc)
    WML = sum(ww[ksfc+1:kLCL+1])*recipk
    Um = sum(uu[ksfc+1:kLCL+1])*recipk
    Vm = sum(vv[ksfc+1:kLCL+1])*recipk
    UML = numpy.sqrt(Um**2 + Vm**2)
    UML = max(UML,Umin)
    debug("debug: UML,WML",UML,WML)
    debug("debug: uu[ksfc+1:kLCL+1]",uu[ksfc+1:kLCL+1])

    #! Determine the maximum value of saturation specific entropy (s*) 
    #! on the pressure levels above the LCL up to the specified upper 
    #! pressure level(Pmin).  
    #! 
    #!---------------------------------------------------------------------
    #!  14-08-2018
    #! Dean Sgarbossa suggested using the pressure level where the 
    #! plume parcel temperature cools to -20 C (an electrification limit)
    #! to define the upper pressure level.
    #!
    #! Using the pressure level at which the environment T first
    #! drops below -20 C should work.  Need to pass a variable called
    #! Telec with value 253.15 K (electrification temperature limit). 
    #!
    #!  28-03-2019
    #! An alternative is to calculate the pressure at which the environment
    #! temperature is -20 C and pass this value as Pmin. Adjustments to the code have
    #! been added below so that Pmin is tested for maximum saturation specific
    #! entropy.
    #!---------------------------------------------------------------------
    #!
    #! Begin by finding the intersection of the SP curve with the 
    #! environment temperature trace.  s* at this point should be the 
    #! minimum possible s*_max.  To do this we need to identify the
    #! pressure levels above and below the intersection, and fill an 
    #! array of P and theta at very small increments between between
    #! these two levels.  By progressing up the SP curve increment by
    #! increment until both Pinc > PSP, and th_inc < theta_pl, then we
    #! have arrived at the first increment slightly beyond the intersection.
    #! Here the first s* should be calculated, and the increment index 
    #! recorded, since the looping along the SP curve below can start 
    #! from here.

    kl = find_k_index_below_intersection(th,qq,TT,pr,thWML,qqWML,phi)
    debug("debug: kl,pr[kl],TT[kl]",kl,pr[kl],TT[kl])
    debug("debug: pd,thWML,qqWML,phi",pd,thWML,qqWML,phi)

    #! Calculate linear function thenv = AA*pr + BB
    AA = (th[kl+1] - th[kl])/(pr[kl+1] - pr[kl])
    BB = th[kl] - AA*pr[kl]

    #! Proceed along the SP curve in small increments.  Determine the pressure
    #! at each step, and find the environment theta at that pressure level. 
    #! When the theta on the SP curve first exceeds the environment, the 
    #! intersecton has been exceeded.  Assume this is close enough to the 
    #! true intersection position.  Store the SP curve increment in 'imax'.
    #! Calculate s* at this position and set it to s_star_max.

    for i in range(ni):
        bSP[i] = (i)*DbSP
        thpl[i] = (1 + bSP[i])*thWML
        qqpl[i] = qqWML + bSP[i]*phi*thWML
        pSP[i],TSP[i] = LCL(thpl[i],qqpl[i])
        thenv = AA*pSP[i] + BB 
        if(thpl[i] > thenv):
            imax = i
            s_star_max = sat_spec_entr(thpl[i],qqpl[i],pSP[i])
            break
        # crash if ni is not enough
        assert i < ni-1, "WARNING: variable 'ni'=%d, is too small"%ni
    
    debug("debug: s_star_max",s_star_max)
    
    #! Calculate s* at all the levels above kl, and search for the maximum
    #! value below Pmin.
    #! Store s* in the first position of the sSP array. 

    for k in range(kl+1,pd+1):
        if(pr[k] > Pmin):
            sSP[0] = sat_spec_entr(th[k],qq[k],pr[k])

            if(sSP[0] > s_star_max):
                s_star_max = sSP[0]
        else:
            kup = k
            break
    debug("debug: kup",kup)
    debug("debug: Old s_star_max = ",s_star_max)

    #! Calculate s* at Pmin and test for s_star_max (Added 2019-03-28)
    #   ! Calculate theta and q at Pmin and store in thenv and qenv
    alpha = (Pmin - pr[kup])/(pr[kup-1] - pr[kup])
    thenv = alpha*( th[kup-1] - th[kup] ) + th[kup]
    qenv  = alpha*( qq[kup-1] - qq[kup] ) + qq[kup]
    sSP[0] = sat_spec_entr(thenv,qenv,Pmin)
    if (sSP[0] > s_star_max):
        s_star_max = sSP[0]

    debug("debug: new s_star_max",s_star_max)

    #! Loop through increments along the SP curve starting at the WML-LCL
    sSP[:] = 0.0   # Reset sSP array
    #debug("debug: sSP[i],thpl[i],qqpl[i]*1000,pSP[i],bSP[i],i")
    #jflag=0
    for i in range(imax,ni):
        bSP[i] = (i)*DbSP
        thpl[i] = (1 + bSP[i])*thWML
        qqpl[i] = qqWML + bSP[i]*phi*thWML
        pSP[i],TSP[i] = LCL(thpl[i],qqpl[i])
        sSP[i] = sat_spec_entr(thpl[i],qqpl[i],pSP[i])
        #if jflag < 40:
        #    debug("debug: ",sSP[i],thpl[i],qqpl[i]*1000,pSP[i],bSP[i],i)
        #    jflag += 1
        if (sSP[i] >= s_star_max):  # Maximum i-index found
            imax = i
            break

    #! Use linear interpolation to find specific values of betaSP, PrSP that
    #! more closely match where sSP = s_star_max
    if(imax > 0):
        alpha = (s_star_max - sSP[imax-1])/(sSP[imax] - sSP[imax-1])
        betaSP = bSP[imax-1] + alpha*(bSP[imax] - bSP[imax-1])
    else:
        betaSP = bSP[0]
    debug("debug: s_star_max,betaSP,imax",s_star_max,betaSP,imax)
    debug("debug: bSP, sSP",bSP,sSP)
    
    #! Add the minimum buoyancy to betaSP to get betaFC (free convection
    #! beta).  This is to account for some dilution of the rising cloud.  It
    #! could be defined by how dry the air is above the LCL (a future
    #! project perhaps).
    betaFC = betaSP + betaSPmin
    
    #! Calculate free convection pressure (pFC) and density (rhoFC)
    thFC = (betaFC+1)*thWML
    qqFC = qqWML + betaFC*phi*thWML
    pFC, TFC = LCL(thFC,qqFC)
    #!   rhoFC = pFC/(Rd*TFC)  # Commented out on 2019-10-15
    DthFC = betaFC*thWML # Added 2019-10-15
    
    #
    #! Use linear interpolation to get the free convection height (zFC)
    #! First find the pressure level above the free convection level
    #! Start the search below kLCL to account for cases of bFC < 0
    #! Subtract zfc to get height above ground level (added 29-01-2019)
    for k in range(ksfc+1,pd+1):
        if(pr[k] < pFC): # pressure level above pFC found
            kFC = k
            break
    ## TODO: probably this will work instead of loop
    #if numpy.any(pr[ksfc+1:pd]<pFC): # if there's a pressure level above pFC
    #    kFC = ksfc+1 + numpy.argmax(pr[ksfc+1:pd]<pFC) # index of first occurrence
    
    
    debug("debug: kFC,thFC,qqFC",kFC,thFC,qqFC)
    alpha = (pFC - pr[kFC-1])/(pr[kFC] - pr[kFC-1])
    zFC = zz[kFC-1] + alpha*(zz[kFC] - zz[kFC-1]) - zsfc
    
    #!   ACCESS-C has been generating unrealistically large vertical winds, which cause 
    #!   the iterative process below to crash.  It will liekly cause unrealistic
    #!   plume stretching too.  A quick fix is to introduce an upper limit to the vertical
    #!   wind speed: Wmax
    if (WML > Wmax): 
        WML = Wmax
    if (WML < -Wmax): 
        WML = -Wmax
    
    #! Begin the net Buoyancy flux calculation (Bflux = pi*BB)
    #! Assume Briggs model for now.  May consider a hybrid Briggs/MTT model
    #! in the future.  If the mean mixed-layer vertical motion (WML) is zero
    #! the calculation is simple and direct.  If WML is non-zero an iterative
    #! method is needed to solve for zb, an adjusted height that takes into account
    #! the difference in height between the plume centreline and the plume top and
    #! the impact of vertical motion on the plume.
    #
    bcosphi = beta_e*cosphi[Prcntg]
    zb[0] = zFC/(1+bcosphi)
    if (numpy.abs(WML) > Wmin): # W is significant
        jconv = 0
        for j in range(nj):
            f[j] = zFC - (1+bcosphi)*zb[j] - numpy.sqrt( (2.0*zb[j])/(3.0*grav*betaFC) )*WML
            fdash[j] = - (1+bcosphi) - 0.5*numpy.sqrt( (2.0)/(3.0*grav*betaFC*zb[j]) )*WML
            zb[j+1] = zb[j] - f[j]/fdash[j]
            diff = numpy.abs( (zb[j+1]-zb[j])/zb[j] )
            
            if (diff < 0.002): # Convergence achieved. Place converged value at start of zb
                zb[0] = zb[j+1]
                jconv = j 
                break
        if (jconv == 0):
            print("WARNING: zb did not converge after ",nj,"iterations")
            
    #! Calculate BB, Bflux and Hflux
    #! It's not clear what value of rho is most appropriate.  Is it the density of
    #! the combustion gases, or the plume at some elevated level?  This might depend
    #! on whether the heat flux or buoyancy flux is conserved in the plume.  It may
    #! be that the density does not change much (pressure and temperature decrease at
    #! a similar rate) in which case both BB and HH are conserved, and we can choose
    #! any value of rho that we have suitable data for.  
    #!    2019-10-15  Much of the above uncertainty has been sorted out.
    #! The buoyancy flux is conserved, which means the heatflux divided by pressure
    #! is conserved.  Here the plume centreline pressure (pC) is used to represent
    #! the pressure for the entire plume cross-section. For simplicity, assume the
    #! ratio of zFC to the plume centreline height (zb) is the same as the ratios of
    #! free-conveection pressure and plume centreline pressure above the surface.
    #!   Calculate pC
    pC = psfc + (pFC - psfc)/( zFC/zb[0] )
    #
    Bflux = pi*grav*betaFC*UML*(beta_e*zb[0])**2
    #!   Hflux = (rhoFC*Cpd*thWML/grav)*Bflux        ! Commented out on 2019-10-15
    Hflux = pi/kappa*(pC/thFC)*(beta_e*zb[0])**2*UML*DthFC #Added 2019-10-15
    
    return [UML,Um,Vm,betaFC,DthFC,zFC,pFC,Bflux,Hflux]
    
    
def LCL(th,qq):
    """
    ! Use Bolton's (1980) formula to calculate the lifting condensation level (LCL)
    ! temperature and pressure.
    
    real, intent(in) :: th     ! potential temperature
    real, intent(in) :: qq     ! specific humidity

    real, intent(out):: pLCL   ! pressure at the LCL
    real, intent(out):: TLCL   ! temperature at the LCL
    """
    
    #! Local variables
    #   real :: ev                 ! vapour pressure (Pa)
    #   real :: rr                 ! mixing ratio (kg/kg)

    # Calculate the vapour pressure valid at pressure = P0
    ev = P0/( (1-epsln) + epsln/qq)

    # Calculate the mixing ratio
    rr = qq/(1 - qq)

    # Calculate the LCL temperature.  Here we can use theta in place of the air
    # parcel temperature because the vapour pressure was evaluated at p = p0.
    # Note, the vapour pressure is multiplied by 0.01 to convert to units of hPa.
    TLCL = 2840.0/( 3.5*numpy.log(th) - numpy.log(ev*0.01) - 4.805 ) + 55.0

    # Calculate the LCL pressure
    pLCL = P0*(TLCL/th)**( Cpd/(Rd*(1.0-0.24*rr)) )

    return pLCL, TLCL

def find_k_index_below_intersection(th,qq,TT,pr,thWML,qqWML,phi):
    """
    real, intent(in), dimension(pd) :: TT       ! Column temperature (K)
    real, intent(in), dimension(pd) :: qq       ! Column specific humidity (kg/kg)
    real, intent(in), dimension(pd) :: th       ! Column potential temperature (K)
    real, intent(in), dimension(pd) :: pr       ! Column pressure (Pa)
    
    real, intent(in) :: thWML     ! Weighted mixed-layer potential temperature (K)
    real, intent(in) :: qqWML     ! Weighted mixed-layer specific humidity (kg/kg)
    real, intent(in) :: phi       ! Fire moisture to heat ratio (kg/kg/K)
       
    integer, intent(out) :: kl    ! k-index below the SP curve, temperature trace intersection
    """
    pd = len(TT) # how many pressure levels
    # Fill beta array
    beta = th/thWML - 1
    
    # first element may be a dummy NaN, let's set it to -500
    if numpy.isnan(beta[0]):
        # this removes the warning when comparing nan in the next step
        beta[0] = -500 
    
    # Find the first positive beta, this is first guess of kl
    kl = numpy.argwhere(beta > 0)[0][0]

    # Find the first lower k-index where Tbeta is greater than the temperature on the upper k-index.  
    for k in range(kl, pd):
        qbeta = qqWML + beta[k+1]*phi*thWML
        Pbeta,Tbeta = LCL(th[k+1],qbeta)
        if(Tbeta > TT[k+1]):
            kl = k
            break
    return kl

def sat_spec_entr(th,qq,pr):
    """
    real, intent(in) :: th     ! potential temperature
    real, intent(in) :: qq     ! specific humidity
    real, intent(in) :: pr     ! pressure
    real, intent(out):: s_star ! saturation specific entropy
    """
    # e_star             ! saturation vapour pressure (Pa)
    # pd_star            ! saturation partial pressure of dry air (Pa)
    # r_star             ! saturation mixing ratio (kg/kg)
    # TT                 ! temperature (K)

    # Calculate mixing ratio and store in r_star temporarily
    r_star = qq/(1 - qq)

    # Calculate e_star
    TT = th*(pr/P0)**(kappa*(1 - 0.24*r_star))
    e_star = 611.2*numpy.exp( (17.67*TT- 4826.56)/(TT - 29.65) )
   
    # Calculate pd_star
    pd_star = pr - e_star

    # Calculate r_star
    r_star = epsln*e_star/pd_star

    # Calculate s_star
    s_star = Cpd*numpy.log(TT) + LV*r_star/TT - Rd*numpy.log(pd_star)

    return s_star

def debug(*args):
    """ This method only prints if __DEBUG__ is set to True """
    if __DEBUG__:
        print(*args)

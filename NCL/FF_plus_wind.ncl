
  begin
    
    
    
    it     = (/301,661,1021/)         ; reading 3 timesteps
    diri   = "/g/data/en0/sza565/ACCESS-fire/yanchep/2021-01-20/20191210T1500Z/0p3/fire/"
    f1     = addfile(diri+"firefront.20191210T1500Z.nc","r")
    ff     = doubletofloat(f1->firefront(it,:,:))
    lon    = doubletofloat(f1->lon)
    lat    = doubletofloat(f1->lat)
    ff!0   = "time"
    ff!1   = "lon"
    ff&lon = lon
    ff&lon@units="degrees_east"
    ff!2   = "lat"
    ff&lat = lat
    ff&lat@units="degrees_north"
    delete([/lon,lat/])
    f2     = addfile(diri+"10m_uwind.20191210T1500Z.nc","r")
    u      = doubletofloat(f2->UWIND_2(it,:,:))
    f3     = addfile(diri+"10m_vwind.20191210T1500Z.nc","r")
    v      = doubletofloat(f3->VWIND_2(it,:,:))
    copy_VarMeta(ff,u)
    copy_VarMeta(ff,v)

;***********************************************************************    
;  PLOT    
;***********************************************************************    

    wks    = gsn_open_wks("png","Yanchep.Ffrnt.wind") ; send graphics to PNG file
    gsn_define_colormap(wks,"amwg")

    res                      = True               ; plot mods desired
    res@cnFillOn             = True               ; turn on color for contours
    res@cnLinesOn            = False              ; turn off contour lines
    res@cnLineLabelsOn       = False              ; turn off contour line labels
    res@gsnScalarContour     = True               ; contours desired
    res@gsnAddCyclic         = False
    res@mpMinLonF            = 115.               ; select a subregion
    res@mpMaxLonF            = 116.
    res@mpMinLatF            = -32.5
    res@mpMaxLatF            = -30.5
    res@mpFillDrawOrder      = "Predraw"
    res@mpDataBaseVersion    = "MediumRes"
    res@pmTickMarkDisplayMode= "Always"
    res@cnLevelSelectionMode = "ManualLevels"     ; set manual contour levels
    res@cnMinLevelValF       = -0.03              ; set min contour level
    res@cnMaxLevelValF       = 0.005              ; set max contour level
    res@cnLevelSpacingF      = 0.005              ; set contour spacing
    res@lbLabelBarOn         = False
    res@vcRefMagnitudeF      = 4.0                ; define vector ref mag
    res@vcRefLengthF         = 0.045              ; define length of vec ref
    res@vcMinDistanceF       = 0.035
    res@vcRefAnnoOrthogonalPosF   = -1.0            ; move ref vector
    res@vcRefAnnoArrowLineColor   = "black"         ; change ref vector color
    res@vcRefAnnoArrowUseVecColor = False
    res@vcGlyphStyle         = "CurlyVector"     ; turn on curly vectors
    res@vcLineArrowColor     = "black"           ; change vector color
    res@vcLineArrowThicknessF= 2.0               ; change vector thickness
    res@vcVectorDrawOrder    = "PostDraw"        ; draw vectors last

    plot = gsn_csm_vector_scalar_map_ce(wks,u(time|0,lat|:,lon|:),\
                 v(time|0,lat|:,lon|:),ff(time|0,lat|:,lon|:),res)
    
  end

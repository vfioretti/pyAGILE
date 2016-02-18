"""
 AGILE.py  -  the Python library for the scientific analysis of AGILE data
 ---------------------------------------------------------------------------------
 A collection of tools, functions, and algorithms for the analysis 
 for the ASI AGILE Gamma-ray mission
 ---------------------------------------------------------------------------------
 Dependencies:
 - python 2.7
 - numpy
 - astropy
 - matplotlib
 ---------------------------------------------------------------------------------
 Example:
 import AGILE as agile
 agile.plgal()
 ---------------------------------------------------------------------------------
 Created by V. Fioretti (INAF/IASF Bologna)
 - 2016/02/18: creation date
 
"""



import string
import os
import astropy.io.fits as pyfits
from astropy import units as u
from astropy.coordinates import SkyCoord as sky
from numpy import * 
import sys


import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.cm as cmx
import matplotlib.colors as cmc
import numpy as np
from math import pi


def plgal(title='', source_list='', square_list = '', ring_list='', vip_sources=0):
	"""
	 plgal()  -  description
	 ---------------------------------------------------------------------------------
	 Function to plot the galaxy in galactic coordinates and AITOFF projection.
	 It is possible to plot a set of rings or points as additional feature.
	 ---------------------------------------------------------------------------------
	 copyright            : (C) 2014 Valentina Fioretti
	 email                : fioretti@iasfbo.inaf.it
	 ---------------------------------------------------------------------------------
	 Parameters (default = None):
	 - title: title of the plot
	 - source_list: ASCII file (+ path) with the galactic coordinates, in degrees, of the sources				                
	 - ring_list: ASCII file (+ path) with the galactic coordinates, in degrees, of the rings center plus the radius
	 - vip_sources: flag to load the most famous Gamma-ray sources (Crab, Vela, Geminga, 3C 454.3)
	 ---------------------------------------------------------------------------------
	 Required data input format: ASCII file (one entry each row)
	 Optional parameters: color keyword c_1 (integer) for points and fill keyword (0/1) plus fill color 
	 (integer) for the rings (f_1 and fc_1)
	 - Sources:
	 l_1 b_1 (c_1)
	 - Rings:
	 l_1 b_1 r_1 (f_1 fc_1)
	 ---------------------------------------------------------------------------------
	 Caveats:
	 None
	 ---------------------------------------------------------------------------------
	 Modification history:
	 - 2014/08/20: creation date
	"""

	deg2rad = np.pi/180.
	cmap = cmx.jet  ##I set colomap to 'jet' 
	norm = cmc.Normalize(vmin=0, vmax=10) 

	# converting latitude to be plotted in the 180, 90, 0, 270, 180 axis
	def convert_l(l_in):
			if ((l_in >= 0.) & (l_in <= 180.)):
				l_in = l_in*(-1.) 
			if (l_in > 180.):
				l_in = 360. - l_in
			return l_in

	fig = plt.figure(1,figsize=[10,5])
	ax = fig.add_subplot(111, projection='aitoff')

	tick_labels = np.array([150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210])
	ax.set_xticklabels(tick_labels)

	# Loading the sources
	if source_list:
		l_source = []
		b_source = []
		c_source = []
		f_read = open(source_list, 'r')
		for line in f_read:
			line = line.strip()
			columns = line.split()
			n_cols = len(columns)
			columns[0] = float(columns[0])  # converting from string to float
			columns[1] = float(columns[1])
			if (n_cols > 2):
				columns[2] = int(columns[2])
				c_source.append(columns[2])
			else:
				c_source.append(0)
			l_in = convert_l(columns[0])*deg2rad
			b_in = columns[1]*deg2rad
			l_source.append(l_in) 
			b_source.append(b_in)
		ax.scatter(l_source, b_source, s=5, c=c_source)

	# Loading the rings
	if ring_list:
		f_read = open(ring_list, 'r')
		for line in f_read:
			f_ring = 'False'
			fc_ring = 'none'
			line = line.strip()
			columns = line.split()
			n_cols = len(columns)
			columns[0] = float(columns[0])  # converting from string to float
			columns[1] = float(columns[1])
			columns[2] = float(columns[2])
			if (n_cols > 3):
				columns[3] = int(columns[3])
				if (columns[3] == 1):
					f_ring = 'True'
				else:
					f_ring = 'False'
					fc_ring = 'none'
			if (n_cols > 4):
				columns[4] = int(columns[4])
				if (columns[3] == 1): 
					fc_ring = cmap(norm(columns[4])) 
				else: 
					fc_ring = 'none'
			l_in = convert_l(columns[0])
			b_in = columns[1]
			r_in = columns[2]
			ax.add_artist(Circle(xy=((l_in)*deg2rad, (b_in)*deg2rad), facecolor=fc_ring, fill=f_ring, edgecolor='k', radius=r_in*deg2rad, zorder = 1)) 

	# Loading the vip sources	
	if vip_sources:
		crab_coord = [convert_l(184.557593)*deg2rad, -5.784197*deg2rad]
		vela_coord = [convert_l(263.552021)*deg2rad, -2.787006*deg2rad]
		geminga_coord = [convert_l(195.13428)*deg2rad, 4.26608*deg2rad]
		diamond_coord = [convert_l(86.1110374)*deg2rad, -38.1837815*deg2rad]
		lcoord = [crab_coord[0], vela_coord[0], geminga_coord[0], diamond_coord[0]]
		bcoord = [crab_coord[1], vela_coord[1], geminga_coord[1], diamond_coord[1]]
		ax.scatter(lcoord, bcoord, color='r', marker='*', s=50, zorder=2)
		
	# Make-up
	plt.grid(True)
	plt.title(title)
	plt.text(0,-(90+15)*deg2rad,'Galactic longitude (degrees)',
      ha='center', va='center')
	plt.ylabel('Galactic latitude (degrees)')

	plt.show()


def grb_pipe(evt_file='', log_file='', par_file='', GRB_ra=0., GRB_dec=0., t1s=0., t2s=0., t1b=0., t2b=0.):
    """
     grb_pipe() -  description
     ---------------------------------------------------------------------------------
     Search for significant signal in gamma-ray follow-ups
     ---------------------------------------------------------------------------------
     copyright            : (C) 2016 S. Cutini (ASDC) and A. Giuliani (IASF Milano)
     ----------------------------------------------
     - evt_file = event file'
     - log_file = log file'
     - par_file = parameter file for follow-up search'
     - GRB_ra = RA of the source in deg.'
     - GRB_dec = DEC of the source in deg.'
     - (optional)
        - t1s = start time of N_on (GRB time reference frame)
        - t2s = stop time of N_on (GRB time reference frame)
        - t1b = start time of N_off (GRB time reference frame)
        - t2b = stop time of N_off (GRB time reference frame)
     ---------------------------------------------------------------------------------
     Caveats:
     None
     ---------------------------------------------------------------------------------
     Modification history:
     - 2016/02/18: V. Fioretti (INAF/IASF Bologna) - including code in AGILE library

    """

    parfile=open(par_file,"r")

    # transformations

    dreq =  3.141592/180.0 # from deg to rad
    dreq1 = 1./(3.141592/180.0) # from rad to deg.

 
    # reading par file
    
    GRB_time = float(parfile.readline())  # TT time (s)
    raggio=float(parfile.readline()) # radius of the ring in degrees
    fov=float(parfile.readline())     # FOVRADMAX
    ea_th=float(parfile.readline())    # ALBEDORAD

    if (t2s - t1s) == 0.:
    	t1s=float(parfile.readline())     # T1 for the source
    	t2s=float(parfile.readline())     # T2 for the source

    if (t2b - t1b) == 0.:    
    	t1b=float(parfile.readline())     # T1 for the background
    	t2b=float(parfile.readline())     # T2 for the background


    print ''
    print 'GRB T0 :',GRB_time
    print 'GRB (Ra, Dec) :',GRB_ra, GRB_dec
    print 'Ricerca eventi (Raggio):',raggio
    print 'Ricerca eventi (Tmin, Tmax):',t1s, t2s
    print 'Background (Tmin, Tmax):',t1b, t2b
    print 'F.O.V.  :',fov
    print 'Albedo cut  :',ea_th
    print ''

    # reading log file

    hdulist_log = pyfits.open(log_file)
    tbdata_log = hdulist_log[1].data
    TIME_log = tbdata_log.field('TIME')  # sec
    RA_earth= tbdata_log.field('EARTH_RA') # deg.
    DEC_earth= tbdata_log.field('EARTH_DEC') # deg.
    livetime= tbdata_log.field('LIVETIME')   # ms
    ra_punt=tbdata_log.field('ATTITUDE_RA_Y')   # deg.
    dec_punt=tbdata_log.field('ATTITUDE_DEC_Y')   # deg.
    phase_log=tbdata_log.field('PHASE')

    TIMEnew_log=TIME_log-GRB_time  # time starting point is the source T0.


    # reading evt file

    hdulist = pyfits.open(evt_file)
    tbdata = hdulist[1].data
    RAcol = tbdata.field('RA')  # deg.
    DECcol = tbdata.field('DEC')   # deg.
    TIMEcol = tbdata.field('TIME')    # sec.
    PH_col = tbdata.field('PH_EARTH')  # deg.
    EVcol = tbdata.field('EVSTATUS')  # Event Classification Flag
    Enecol = tbdata.field('ENERGY')    # MeV
    THETAcol = tbdata.field('THETA')   # deg., coordinates in the P/L Ref Sys.
    PHASEcol = tbdata.field('PHASE')    # deg.
    
    TIMEnew=TIMEcol-GRB_time   # time starting point is the source T0.
    
    dec = DECcol*dreq    # in rad, photon ra
    ra = RAcol*dreq   # in rad, photon dec.
    dec_ea = DEC_earth*dreq   # in rad, ra of the earth
    ra_ea = RA_earth*dreq   # in rad, dec. of the earth


    Vx_ea=zeros(len(dec_ea),float)
    Vy_ea=zeros(len(dec_ea),float)
    Vz_ea=zeros(len(dec_ea),float)

    argx_ea=zeros(len(dec_ea),float)
    argy_ea=zeros(len(dec_ea),float)
    argz_ea=zeros(len(dec_ea),float)
    arg_ea=zeros(len(dec_ea),float)

    DELTA_ea=zeros(len(dec_ea),float)

    expo=zeros(len(dec_ea),float)
    expo1=zeros(len(dec_ea),float)
    expo2=zeros(len(dec_ea),float)
    exposure=zeros(len(dec_ea),float)
    
    t1b=max(min(TIMEnew),t1b)
    t2b=min(max(TIMEnew),t2b)

    t1b=max(min(TIME_log-GRB_time),t1b)   # changing the range of the background time if outside the evt file time
    t2b=min(max(TIME_log-GRB_time),t2b)   # changing the range of the background time if outside the evt file time
    print 'Background T1 [sec]: ', t1b
    print 'Background T2 [sec]: ', t2b

    # GRB in coord. cartesiane

    Vxgrb = cos(GRB_dec*dreq)*cos(GRB_ra*dreq)
    Vygrb = cos(GRB_dec*dreq)*sin(GRB_ra*dreq)
    Vzgrb = sin(GRB_dec*dreq)

    # Distanza angolare grb-eventi

    Vx = cos(dec)*cos(ra)
    Vy = cos(dec)*sin(ra)
    Vz = sin(dec)

    argx = Vxgrb*Vx
    argy = Vygrb*Vy
    argz = Vzgrb*Vz
    arg= argx+argy+argz


    DELTA = arccos(arg)*dreq1


    # Off-axis angle del GRB (with respect to the telescope attitude)

    Vx_punt = cos(dec_punt*dreq)*cos(ra_punt*dreq)
    Vy_punt = cos(dec_punt*dreq)*sin(ra_punt*dreq)
    Vz_punt = sin(dec_punt*dreq)

    argx_punt = Vxgrb*Vx_punt
    argy_punt = Vygrb*Vy_punt
    argz_punt = Vzgrb*Vz_punt
    arg_punt= argx_punt+argy_punt+argz_punt

    OFF = arccos(arg_punt)*dreq1


    # Calcolo segnale e background

    source=0
    bkg=0


    for k in range(len(dec)):    

        
        if DELTA[k] <raggio:  # if the event is within the ring
            if Enecol[k] > 0.:        # if the event energy is > 0
             if PH_col[k] >ea_th:        # removing the events from albedo
                if PHASEcol[k] != 1:      # requiring PHASE not equal 1     ??????????
                  if THETAcol[k]<fov:       # theta of the event in the P/L Sys. Ref. within FOVRADMAX
                    if TIMEnew[k] > t1s:
                        if TIMEnew[k] < t2s: # if the event time is within the T1-T2 of the source
                            source=source+1   # counting the events that pass the selection
                            print '      trovato (t-T0) : ', TIMEnew[k]
                    if (TIMEnew[k] > t1b) and (TIMEnew[k] <t1s) or (TIMEnew[k] > t2s) and (TIMEnew[k] <t2b): #??????

                            bkg=bkg+1

    print ''
    print "Source :", source
    print "Bkg :", bkg
    print ''

    # calcolo delle significativita con il metodo di Li&Ma
    summ_src=0
    ntot_src=0
    summ_bkg=0
    ntot_bkg=0


    for i in range(len(dec_ea)):

        Vx_ea[i] = cos(dec_ea[i])*cos(ra_ea[i])
        Vy_ea[i] = cos(dec_ea[i])*sin(ra_ea[i])
        Vz_ea[i] = sin(dec_ea[i])

        argx_ea[i] = Vxgrb*Vx_ea[i]
        argy_ea[i] = Vygrb*Vy_ea[i]
        argz_ea[i] = Vzgrb*Vz_ea[i]
        arg_ea[i] = argx_ea[i]+argy_ea[i]+argz_ea[i]


        DELTA_ea[i] = arccos(arg_ea[i])*dreq1  # angular distance between Earth and Source
        if (DELTA_ea[i]-ea_th+raggio)/(2.*raggio) < 0.:
            expo[i]=0.   # ring witihin Earth region
        elif (DELTA_ea[i]-ea_th+raggio)/(2.*raggio) > 1.:
            expo[i]=1.   # ring outside Earth region

        else:
            expo[i]=(DELTA_ea[i]-ea_th+raggio)/(2*raggio)   # is this true?

        if (fov-OFF[i]+raggio)/(2.*raggio) < 0.:
            expo2[i]=0. # the ring is outside the fov
        elif (fov-OFF[i]+raggio)/(2.*raggio) > 1.:
            expo2[i]=1. # the ring is within the fov
        else:
            expo2[i]=(fov-OFF[i]+raggio)/(2*raggio) # is it true?

        if (((ra_punt[i] < 0) == 0)*((ra_punt[i]>0) == 0) ) :  # ??????????????
            expo2[i]=0        

        if (livetime[i] != 0) and (phase_log[i] != 1):
            expo1[i]=livetime[i]/100.     # ??????????????????
        else:
            expo1[i]=0.

        exposure[i] = expo[i]*expo1[i]*expo2[i]

        if (TIMEnew_log[i] >t1s) and  (TIMEnew_log[i] < t2s):
            summ_src=exposure[i]+summ_src
            ntot_src=ntot_src+1  

        if (TIMEnew_log[i] > t1b) and (TIMEnew_log[i] <t1s) or (TIMEnew_log[i] > t2s) and (TIMEnew_log[i] <t2b):
            summ_bkg=exposure[i]+summ_bkg
            ntot_bkg=ntot_bkg+1 

    print "mean src occulted ", summ_src/ntot_src
    print "mean bkg occulted", summ_bkg/ntot_bkg
    mean_src=summ_src/ntot_src
    mean_bkg=summ_bkg/ntot_bkg
    
    if (t1s >= t2b) or (t2s <= t1b):
      tback=t2b-t1b
    else:
      if (t1s >= t1b) and (t2s <= t2b):
	tback=t2b-t1b - (t2s-t1s)
      else: 
        print "   !!!! Scegli un altro intervallo di background !!!!"
        tback=0
    
    print tback
    
    #alp = ((t2s-t1s)*mean_src)/((t2b-t1b-(t2s-t1s))*mean_bkg)
    alp = ((t2s-t1s)*mean_src)/(tback*mean_bkg)
    alp1=alp/(1+alp)
    alp2=alp+1
    print "source", source
    print "bkg", bkg/((tback)*mean_bkg)*((t2s-t1s)*mean_src), "(", bkg, ")"
    source1=float(source)
    bkg1=float(bkg)

    if source>0 :
       L1 = math.pow(((source1+bkg1)/source1)*alp1,source1)
       L2 = math.pow(((bkg1+source1)/bkg1)/alp2,bkg1)
       L=L1*L2
       #print "L", alp2
       S=math.sqrt(-2.*math.log(L))
       print "Li&Ma sigma", S


    hdulist.close()
    
    
if __name__ == '__main__':
    plgal()
    grb_pipe()
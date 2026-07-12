import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pyrap.tables import table

def plot_bandpass_table(gain_table, plt_scale=6, plt_dpi=600, plot_file=None):
#Reads in the gain table; makes plots of gain amplitude/phase with 
#time/frequency for all the antennas.

#Read in the table
   plot_file = plot_file or gain_table+'.png'
   G_tab = table(gain_table)
   print(('Reading gain table:', gain_table, '\n'))
   G_tab_names = table(gain_table+"::ANTENNA")
   ant_names = G_tab_names.getcol("NAME")
   print(("Antennas present in the table:", ant_names))

#Get number of antennas (needed for plotting)       
   N_ants = len(ant_names)
   print(('Number of Antennas to plot:', N_ants))    


#Read in the Frequency information
   G_tab_freq = table(gain_table+"::SPECTRAL_WINDOW")
   FRQ = G_tab_freq.getcol('CHAN_FREQ')
   FRQ_GHZ = np.round(FRQ[0,:]/10**9,4)
   
#Read in the flags
   flags = G_tab.getcol("FLAG")
   
#Read in the solutions
   Gsols = G_tab.getcol("CPARAM")
   nchans = Gsols.shape[1]
   nsols  = Gsols.shape[0]//N_ants
   npol = Gsols.shape[2]        
#Read in the error. Additions of the parameter errors in a later release.
   Gsols_err = G_tab.getcol("PARAMERR")

#Read in the timestamps. 
 
#   Gsols_time = G_tab.getcol("TIME")

#Prepare for plotting; store gain amp and phases.
 
   Gsols_HH = Gsols[:,:,0]
   Gsols_VV = Gsols[:,:,1]
#   print "Shape of solutions:", np.shape(Gsols_HH), np.shape(Gsols_VV)
   Gsols_HH_flag = np.ma.masked_array(Gsols_HH, mask=flags[:,:,0])
   Gsols_VV_flag = np.ma.masked_array(Gsols_VV, mask=flags[:,:,1])
   Gsols_HH_amp = abs(Gsols_HH_flag)
   Gsols_VV_amp = abs(Gsols_VV_flag)
   Gsols_HH_ph = (180.0/np.pi)*np.arctan2(Gsols_HH_flag.imag,Gsols_HH_flag.real) #Because np.angle does not respect masked arrays.
   Gsols_VV_ph = (180.0/np.pi)*np.arctan2(Gsols_VV_flag.imag,Gsols_VV_flag.real)
#   Gsols_HH_ph = np.angle(Gsols_HH_flag,deg=True)
#   Gsols_VV_ph = np.angle(Gsols_VV_flag,deg=True)

#Plotting
#Plot in a more or less square grid.
   nplts = int(np.sqrt(N_ants))+1       #(if non zero remainder, add one)

#Set matplotlib options
   matplotlib.rcParams['lines.markersize'] = 4.0
   matplotlib.rcParams['xtick.major.size'] = 5.0
   matplotlib.rcParams['ytick.major.size'] = 5.0
   matplotlib.rcParams['ytick.direction'] = 'out'
   matplotlib.rcParams['xtick.direction'] = 'out'
   matplotlib.rcParams['font.size'] = 15.0
   matplotlib.rcParams['axes.linewidth'] = 3.0
   matplotlib.rcParams['legend.framealpha'] = 0.5
   matplotlib.rcParams['font.style'] = 'italic'
#   matplotlib.rcParams[]
   print("Starting plotting process")
   #plt_dpi=800
   f, axarr = plt.subplots(nplts, nplts, dpi=plt_dpi, figsize=(nplts*plt_scale,nplts*plt_scale))
   f.text(0.5,0.94,"Bandpass Plot",ha='center',fontsize=40)
   f.text(0.5, 0.04, 'Frequency (GHz)', ha='center',fontsize=30)
   f.text(0.04, 0.5, 'Gain Amp', va='center', rotation='vertical',fontsize=30)
   print("Defining plots completed")
#Plot amplitudes first
   globmax = np.round(np.max(np.maximum(Gsols_VV_amp, Gsols_HH_amp)),1)
   globmin = np.round(np.min(np.minimum(Gsols_VV_amp, Gsols_HH_amp)),1)
   for ant in range(N_ants):
       for sol in range(nsols):
           axarr[ant // nplts, ant % nplts].plot(FRQ_GHZ,Gsols_HH_amp[ant+N_ants*sol],color='#4169E1', marker = '.', label='CORR0')
           axarr[ant // nplts, ant % nplts].plot(FRQ_GHZ,Gsols_VV_amp[ant+N_ants*sol],color='#FF681F', marker = '.', label='CORR1')
           if sol==0:
              axarr[ant // nplts, ant % nplts].legend()
       axarr[ant // nplts, ant % nplts].set_title('Antenna '+str(ant_names[ant]))
       axarr[ant // nplts, ant % nplts].set_ylim(globmin-0.05,globmax+0.05)
   for i in range(nplts):
       for j in range(nplts):
           if ( ((i*nplts)+(j+1))> N_ants):
              axarr[i,j].set_visible(False)
  
   print("iterating finished...")
   plot_f = plot_file+"-bandpass-amp.png"
   f.savefig(plot_f) 
   print("plotting finished")
   plt.close(f)
   print("cleaning up")
   f, axarr = plt.subplots(nplts, nplts, dpi=plt_dpi, figsize=(nplts*plt_scale,nplts*plt_scale))
   f.text(0.5,0.94,"Bandpass Plot",ha='center',fontsize=40)
   f.text(0.5, 0.04, 'Frequency(GHz)', ha='center',fontsize=30)
   f.text(0.04, 0.5, 'Gain Phase', va='center', rotation='vertical',fontsize=30)

#Plot phases now
   globmaxph = np.round(np.max(np.maximum(Gsols_VV_ph, Gsols_HH_ph)),1)
   globminph = np.round(np.min(np.minimum(Gsols_VV_ph, Gsols_HH_ph)),1)

   for ant in range(N_ants):
       for sol in range(nsols):
           axarr[ant // nplts, ant % nplts].plot(FRQ_GHZ,Gsols_HH_ph[ant+N_ants*sol],color='#4169E1', marker = '.', label='CORR0')
           axarr[ant // nplts, ant % nplts].plot(FRQ_GHZ,Gsols_VV_ph[ant+N_ants*sol],color='#FF681F', marker = '.', label='CORR1')
           if sol==0:
              axarr[ant // nplts, ant % nplts].legend()
       axarr[ant // nplts, ant % nplts].set_title('Antenna '+str(ant_names[ant]))
       axarr[ant // nplts, ant % nplts].set_ylim(globminph-5.0,globmaxph+5.0)
   for i in range(nplts):
       for j in range(nplts):
           if ( ((i*nplts)+(j+1))> N_ants):
              axarr[i,j].set_visible(False)
 

   plot_f = plot_file+"-bandpass-phase.png"
   f.savefig(plot_f)
   plt.close(f)
   G_tab.close()
   G_tab_names.close()
   plt.clf()
   print("Plotting and cleaning over")

def plot_gain_table(gain_table, plt_scale=6, plt_dpi=600, plot_file=None):
#Reads in the gain table; makes plots of gain amplitude/phase with 
#time/frequency for all the antennas.

#Read in the table
   plot_file = plot_file or gain_table+'.png'
   G_tab = table(gain_table)
   print(('Reading gain table:', gain_table, '\n'))
   G_tab_names = table(gain_table+"::ANTENNA")
   ant_names = G_tab_names.getcol("NAME")
   print(("Antennas present in the table:", ant_names))

#Get number of antennas (needed for plotting)       
   Ant_n1 = G_tab.getcol("ANTENNA1")
   Ant_n2 = G_tab.getcol("ANTENNA2")
   Ant_list = list(set(np.append(Ant_n1,Ant_n2)))
   N_ants = len(ant_names)
   print(('Number of Antennas to plot:', N_ants))

#Read in the flags
   flags = G_tab.getcol("FLAG")

#Read in the solutions
   Gsols = G_tab.getcol("CPARAM")
   nchans = Gsols.shape[1]
   nsols  = Gsols.shape[0]//N_ants
   print(("Number of solution per antenna:", nsols))
   npol = Gsols.shape[2]
#Read in the error.
   Gsols_err = G_tab.getcol("PARAMERR")

#Read in the timestamps

   Gsols_time = G_tab.getcol("TIME")

#Prepare for plotting; store gain amp and phases.

   Gsols_HH = Gsols[:,:,0]
   Gsols_VV = Gsols[:,:,1]
   print(("Shape of solutions:", np.shape(Gsols_HH), np.shape(Gsols_VV)))
   Gsols_HH_flag = np.ma.masked_array(Gsols_HH, mask=flags[:,:,0])
   Gsols_VV_flag = np.ma.masked_array(Gsols_VV, mask=flags[:,:,1])
   Gsols_HH_amp = abs(Gsols_HH_flag)
   Gsols_VV_amp = abs(Gsols_VV_flag)
   Gsols_HH_ph = (180.0/np.pi)*np.arctan2(Gsols_HH_flag.imag,Gsols_HH_flag.real) #Because np.angle does not respect masked arrays
   Gsols_VV_ph = (180.0/np.pi)*np.arctan2(Gsols_VV_flag.imag,Gsols_VV_flag.real)
   #Gsols_HH_ph = np.angle(Gsols_HH_flag,deg=True)
   #Gsols_VV_ph = np.angle(Gsols_VV_flag,deg=True)

   
#Plotting
#Plot in a more or less square grid.
   nplts = int(np.sqrt(N_ants))+1       #(if non zero remainder, add one)


#Set matplotlib options
   matplotlib.rcParams['lines.markersize'] = 15.0
   matplotlib.rcParams['xtick.major.size'] = 5.0
   matplotlib.rcParams['ytick.major.size'] = 5.0
   matplotlib.rcParams['ytick.direction'] = 'out'
   matplotlib.rcParams['xtick.direction'] = 'out'
   matplotlib.rcParams['font.size'] = 15.0
   matplotlib.rcParams['axes.linewidth'] = 3.0
   matplotlib.rcParams['legend.framealpha'] = 0.5
  # matplotlib.rcParams['font.family'] = 'sans-serif'
  # matplotlib.rcParams['font.sans-serif'] = 'Verdana'
   matplotlib.rcParams['font.style'] = 'italic'
#Make a single plot for gain amplitude and phase, colourize different fields?


   obs_time = G_tab.getcol("TIME")

#Scale to start time and scale to hours
   obs_time = obs_time-obs_time[0]

   obs_time = obs_time/3600.0

   f, axarr = plt.subplots(nplts, nplts, dpi=plt_dpi, figsize=(nplts*plt_scale,nplts*plt_scale))
#Plot amplitudes first
   globmax = np.round(np.max(np.maximum(Gsols_VV_amp, Gsols_HH_amp)),1)
   globmin = np.round(np.min(np.minimum(Gsols_VV_amp, Gsols_HH_amp)),1)
   f.text(0.5,0.94,"Gain Amplitude Plot",ha='center',fontsize=40)
   f.text(0.5, 0.04, 'Time(Hours)', ha='center',fontsize=30)
   f.text(0.04, 0.5, 'Gain Amp', va='center', rotation='vertical',fontsize=30)
   for ant in range(N_ants):
       for sol in range(nsols):
           axarr[ant // nplts, ant % nplts].plot(obs_time[ant+N_ants*sol], Gsols_HH_amp[ant+N_ants*sol],color='#4169E1', marker = '.', label='CORR0')
           axarr[ant // nplts, ant % nplts].plot(obs_time[ant+N_ants*sol], Gsols_VV_amp[ant+N_ants*sol],color='#FF681F', marker = '.',label='CORR1')
           if sol==0:
               axarr[ant // nplts, ant % nplts].legend()
           axarr[ant // nplts, ant % nplts].set_title('Antenna '+str(ant_names[ant]))
      #     axarr[ant // nplts, ant % nplts].set_xlabel('Time (Hours)')
      #     axarr[ant // nplts, ant % nplts].set_ylabel('Gain Amplitude')
      #     axarr[ant // nplts, ant % nplts].set_ylim(0,20)
           axarr[ant // nplts, ant % nplts].set_ylim(globmin-0.5,globmax+0.5)
   for i in range(nplts):
       for j in range(nplts):
           if ( ((i*nplts)+(j+1))> N_ants):
              axarr[i,j].set_visible(False)

   plot_f = plot_file+"-gain-amp.png"
   f.savefig(plot_f)
   plt.close(f)

   f, axarr = plt.subplots(nplts, nplts, dpi=plt_dpi, figsize=(nplts*plt_scale,nplts*plt_scale))
   #Plot phases 
   globmaxph = np.round(np.max(np.maximum(Gsols_VV_ph, Gsols_HH_ph)),1)
   globminph = np.round(np.min(np.minimum(Gsols_VV_ph, Gsols_HH_ph)),1)

   f.text(0.5,0.94,"Gain Phase Plot",ha='center',fontsize=40)
   f.text(0.5, 0.04, 'Time(Hours)', ha='center',fontsize=30)
   f.text(0.04, 0.5, 'Gain Phase(Degrees)', va='center', rotation='vertical',fontsize=30)
   for ant in range(N_ants):
       for sol in range(nsols):
           axarr[ant // nplts, ant % nplts].plot(obs_time[ant+N_ants*sol], Gsols_HH_ph[ant+N_ants*sol],color='#4169E1', marker = '.', label='CORR0')
           axarr[ant // nplts, ant % nplts].plot(obs_time[ant+N_ants*sol], Gsols_VV_ph[ant+N_ants*sol],color='#FF681F', marker = '.', label='CORR1')
           if sol==0:
               axarr[ant // nplts, ant % nplts].legend()
           axarr[ant // nplts, ant % nplts].set_title('Antenna '+str(ant_names[ant]))
      #     axarr[ant // nplts, ant % nplts].set_xlabel('Time (Hours)')
      #     axarr[ant // nplts, ant % nplts].set_ylabel('Gain Amplitude')
      #     axarr[ant // nplts, ant % nplts].set_ylim(0,20)
           axarr[ant // nplts, ant % nplts].set_ylim(globminph-5.0,globmaxph+5.0)
   for i in range(nplts):
       for j in range(nplts):
           if ( ((i*nplts)+(j+1))> N_ants):
              axarr[i,j].set_visible(False)

   plot_f = plot_file+"-gain-ph.png"
   f.savefig(plot_f)
   plt.close(f)
   G_tab.close()
   G_tab_names.close()
   plt.close('all')

def plot_delay_table(gain_table, plt_scale=6, plt_dpi=600, plot_file=None):
#Reads in the gain table; makes plots of gain amplitude/phase with 
#time/frequency for all the antennas.

#Read in the table
   plot_file = plot_file or gain_table+'.png'
   G_tab = table(gain_table)
   print(('Reading gain table:', gain_table, '\n'))
   G_tab_names = table(gain_table+"::ANTENNA")
   ant_names = G_tab_names.getcol("NAME")
   print(("Antennas present in the table:", ant_names))

#Get number of antennas (needed for plotting)       
   Ant_n1 = G_tab.getcol("ANTENNA1")
   Ant_n2 = G_tab.getcol("ANTENNA2")
  # Ant_list = list(set(np.append(Ant_n1,Ant_n2)))
   N_ants = len(ant_names)
   print(('Number of Antennas to plot:', N_ants))

#Read in the flags
   flags = G_tab.getcol("FLAG")

#Read in the solutions
   Gsols = G_tab.getcol("FPARAM")
   nchans = Gsols.shape[1]
   nsols  = Gsols.shape[0]//N_ants   
   print(("Number of solutions per antenna:", nsols))
   npol = Gsols.shape[2]
#Read in the error.
   Gsols_err = G_tab.getcol("PARAMERR")

#Read in the timestamps

   Gsols_time = G_tab.getcol("TIME")

#Prepare for plotting; store gain amp and phases.

   Gsols_HH = Gsols[:,:,0]
   Gsols_VV = Gsols[:,:,1]
   print(("Shape of solutions:", np.shape(Gsols_HH), np.shape(Gsols_VV)))
   Gsols_HH_plt = np.ma.masked_array(Gsols_HH, mask=flags[:,:,0])
   Gsols_VV_plt = np.ma.masked_array(Gsols_VV, mask=flags[:,:,1])
   #Plotting
#Plot in a more or less square grid.
   nplts = int(np.sqrt(N_ants))+1       #(if non zero remainder, add one)
   print(("Number of plots:", nplts*nplts))

#Set Global matplotlib options
   matplotlib.rcParams['lines.markersize'] = 15.0
   matplotlib.rcParams['xtick.major.size'] = 5.0
   matplotlib.rcParams['ytick.major.size'] = 5.0
   matplotlib.rcParams['ytick.direction'] = 'out'
   matplotlib.rcParams['xtick.direction'] = 'out'
   matplotlib.rcParams['font.size'] = 15.0
   matplotlib.rcParams['axes.linewidth'] = 3.0
   matplotlib.rcParams['legend.framealpha'] = 0.5
   matplotlib.rcParams['font.style'] = 'italic'
#   matplotlib.rcParams['legend.frameon']='false'
#Make a single plot for gain amplitude and phase, colourize different fields?

   obs_time = G_tab.getcol("TIME")

#Scale to start time and scale to hours
   obs_time = obs_time-obs_time[0]

   obs_time = obs_time/3600.0
   
   f, axarr = plt.subplots(nplts, nplts, dpi=plt_dpi, figsize=(nplts*plt_scale,nplts*plt_scale)) 
   
   f.text(0.5,0.94,"Delay Calibration Plot",ha='center',fontsize=40)
   f.text(0.5, 0.04, 'Time (Hours)', ha='center',fontsize=30)
   f.text(0.04, 0.5, 'Delay (ns)', va='center', rotation='vertical',fontsize=30)
   globmax = np.round(np.max(np.maximum(Gsols_VV, Gsols_HH)),1)
   globmin = np.round(np.min(np.minimum(Gsols_VV, Gsols_HH)),1)

   for ant in range(N_ants):
       for sol in range(nsols):
           axarr[ant // nplts, ant % nplts].plot(obs_time[ant+N_ants*sol], Gsols_HH_plt[ant+N_ants*sol],color='#4169E1', marker = '.', label='CORR0')
           axarr[ant // nplts, ant % nplts].plot(obs_time[ant+N_ants*sol], Gsols_VV_plt[ant+N_ants*sol],color='#FF681F', marker = '.', label='CORR1')
           if sol==0:
               axarr[ant // nplts, ant % nplts].legend()
           axarr[ant // nplts, ant % nplts].set_title('Antenna '+str(ant_names[ant]))
      #     axarr[ant // nplts, ant % nplts].set_xlabel('Time (Hours)')
      #     axarr[ant // nplts, ant % nplts].set_ylabel('Gain Amplitude')
      #     axarr[ant // nplts, ant % nplts].set_ylim(0,20)
           axarr[ant // nplts, ant % nplts].set_ylim(globmin-0.5,globmax+0.5)
   for i in range(nplts):
       for j in range(nplts):
           if ( ((i*nplts)+(j+1))> N_ants):
              axarr[i,j].set_visible(False)

   plot_f = plot_file+"-delay.png"
   f.savefig(plot_f)
   plt.close(f)
   plt.close('all')
   G_tab.close()
   G_tab_names.close()

def gain_plotter(caltable,typ,outfile,plt_scale=6,plt_dpi=600):
#"Plots gaintables. Types should be 'delay', 'gain' or 'bandpass'. Scale is the plot scale, dpi is the plot dpi"
    if (typ=='delay'):
        plot_delay_table(caltable,plt_scale,plt_dpi,outfile)
    elif (typ=='gain'):
        plot_gain_table(caltable,plt_scale,plt_dpi,outfile)
    elif (typ=='bandpass'):
        plot_bandpass_table(caltable,plt_scale,plt_dpi,outfile)
    else:
        raise RuntimeError('Gain table type "{}" not recognised'.format(typ))

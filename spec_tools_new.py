import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

def open_file(star_id):
    with fits.open(star_id+'.fits') as dat:
        if len(dat[0].data.shape) > 1:
            fl = dat[0].data[0]
        else:
            fl = dat[0].data
        wv = np.linspace(dat[0].header['CRVAL1'],dat[0].header['CRVAL1']+dat[0].header['CDELT1']*dat[0].header['NAXIS1'],dat[0].header['NAXIS1'])
        if wv[0] < 100:
            wv = 10**wv
    return wv,fl

def measure_EW(w_,f_,cont):
    '''Measures the equivalent width of feature.  Requires wavelength and flux array and a tuple with the continuum bounds.
    EW = integral(f(lamdba)-1 .dlambda)'''
    lowerbound,upperbound = cont
    #range to conduct EW measurement
    indrange = (w_ > lowerbound) & (w_ < upperbound)
    ew = 1. - f_[indrange]
    ew = ew[:-1] * np.diff(w_[indrange])
    return ew.sum()

def normalise_spectrum(w_, f_, cont):
    '''Normalize flux of the spectrum around a line feature.  Requires wavelength and flux array and a tuple with the continuum bounds.
    Parameters
    ----------
    w_ : 1 dim np.ndarray
    array of wavelengths
    flux : np.ndarray of flux of spectrum
    array of flux values for different spectra in the series
    cont : list of lists
    wavelengths for continuum normalization [[low1,up1],[low2, up2]]
    that describe two areas on both sides of the line
    '''
    #index is true in the region where we fit the polynomial
    lowerbound_window1,upperbound_window1 = cont[0]
    lowerbound_window2,upperbound_window2 = cont[1]
    indcont = ((w_ > lowerbound_window1) & (w_ < upperbound_window1)) |((w_ > lowerbound_window2) & (w_ < upperbound_window2))
    #index of the region we want to return
    indrange = (w_ > lowerbound_window1) & (w_ < upperbound_window2)
    # fit polynomial of second order to the continuum region
    linecoeff = np.polyfit(w_[indcont],f_[indcont],2)
    # divide the flux by the polynomial and put the result in our
    # new flux array
    f_norm = f_[indrange]/np.polyval(linecoeff, w_[indrange])
    
    fig,ax = plt.subplots()
    ax.plot(w_[indrange],f_[indrange],alpha=0.3)
    ax.plot(w_[indrange],np.polyval(linecoeff, w_[indrange]),ls='--')
    ax.plot(w_[indrange],f_norm)
    ax.axvline(lowerbound_window1,ls='--',alpha=0.5,c='k')
    ax.axvline(upperbound_window1,ls='--',alpha=0.5,c='k')
    ax.axvline(lowerbound_window2,ls='--',alpha=0.5,c='k')
    ax.axvline(upperbound_window2,ls='--',alpha=0.5,c='k')
    return w_[indrange], f_norm

def read_n_display(f_in):
    '''Read input file from argument filename and display contents.  Requires filename argument in quotation marks.'''
    file = open(f_in)
    f_contents = file.read()
    print(f_contents)
    file.close()

def plot_n_measure_EW_spectrum(w_, f_, cont):
    '''Plot spectrum and measure the equivalent width.  Requires wavelength and flux array and a tuple with the continuum bounds.
    Parameters
    ----------
    w_ : 1 dim np.ndarray
    array of wavelengths
    f_ : 1 dim np.ndarray
    flux values for spectrum
    cont : 1 dim np.darray
    continuum bounds around a feature
    '''
    lowerbound,upperbound = cont
    #range to conduct EW measurement
    indrange = (w_ > lowerbound) & (w_ < upperbound)
    #extended range of plot to show feature on spectrum
    plotrange = (w_ > lowerbound-(upperbound-lowerbound)/3) & (w_ < upperbound+(upperbound-lowerbound)/3)
    #plot spectrum and bounds over which EW is measured
    fig,ax = plt.subplots()
    ax.plot(w_[plotrange], f_[plotrange])
    ax.plot(w_[indrange], f_[indrange])
    ax.axvline(lowerbound,ls='--',alpha=0.5,c='k')
    ax.axvline(upperbound,ls='--',alpha=0.5,c='k')
    #display EW
    print("Equivalent Width = %.1f mA"%measure_EW(w_,f_,cont))
    return

def measure_EW(w_,f_,cont):
    '''Measures the equivalent width of feature.  Requires wavelength and flux array and a tuple with the continuum bounds.
    EW = integral(f(lamdba)-1 .dlambda)'''
    lowerbound,upperbound = cont
    #range to conduct EW measurement
    indrange = (w_ > lowerbound) & (w_ < upperbound)
    ew = 1. - f_[indrange]
    ew = ew[:-1] * np.diff(w_[indrange])
    return ew.sum() * 1000

def gen_ew_list(star_id,elem,wavelength,ew_val):
    '''Generate a lineist input file for MOOG for a range of lines.  Requires element, wavelength and equivalent width (single values or tuples).
    Parameters
    ----------
    star_id : name of star for which you are generate input line list
    elem : element for which the line feature is present: e.g. 11.0 is Na, 26.1 is FeI
    wavelength : wavelength in angstroms of line feature: e.g. 3912.513 A
    ew_val : equivalent width of feature in milliangstroms: e.g. 153.2 mA'''
    if type(wavelength) == list and type(elem) != list:
        elem = [elem] * len(wavelength)

    dat = np.genfromtxt('example.ew',dtype='str',skip_header=True)
    file = open(star_id+'.ew','w')
    file.write(star_id+'\n')
    if type(elem)==list:
        for z in range(len(elem)):
            selec = ((dat[:,1]==str(elem[z])) & (dat[:,0]==str(wavelength[z])))
            ew_ = str(float(ew_val[z]))
            for l in dat[selec]:
                file.write("%s%s%s%s%s\n"%(l[0].rjust(10),l[1].rjust(10),l[2].rjust(10),l[3].rjust(10),ew_.rjust(30)))
    else:
        selec = ((dat[:,1]==str(elem)) & (dat[:,0]==str(wavelength)))
        ew_ = str(float(ew_val))
        for l in dat[selec]:
            file.write("%s%s%s%s%s\n"%(l[0].rjust(10),l[1].rjust(10),l[2].rjust(10),l[3].rjust(10),ew_.rjust(30)))
    file.close()
    return
def write_batch_file(star_id):
    file = open('batch.par','w')
    file.write("abfind\n")
    file.write("standard_out     \'%s.out1\'\n"%star_id)
    file.write("summary_out      \'%s.out2\'\n"%star_id)
    file.write("smoothed_out     \'%s.out3\'\n"%star_id)
    file.write("model_in         \'%s.atm\'\n"%star_id)
    file.write("lines_in         \'%s.ew\'\n"%star_id)
    file.write("atmosphere    1\n")
    file.write("units         0\n")
    file.write("damping       0\n")
    file.write("trudamp       0\n")
    file.write("lines         1\n")
    file.write("flux/int      0\n")
    file.write("obspectrum    0\n")
    file.close
    return

def find_line(file_in,line):
    f_contents = np.genfromtxt(file_in)
    lines = np.array([int(i) for i in f_contents[:,0]])
    vel_lim = 600 #km/s
    wv_tol = vel_lim/3e5 * line
    mask = (lines < line+wv_tol) & (lines > line-wv_tol)
    if not np.any(mask):
        print("No line available at this wavelength")
    else:
        for l_ in f_contents[mask]:
            print("{}  \t{}\t{}\t{}\t\tXXX.X".format(*l_))
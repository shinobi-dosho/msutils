import matplotlib
matplotlib.use('Agg')
import numpy
import numpy.ma as ma
import pylab
from scipy import interpolate
from . import _ms as msutils
from ._log import create_logger
from casacore.tables import table
import matplotlib.cm as cm

LOGGER = create_logger(__name__)

MEERKAT_SEFD = numpy.array([
 [ 856e6, 580.], 
 [ 900e6, 578.],
 [ 950e6, 559.],
 [1000e6, 540.],
 [1050e6, 492.],
 [1100e6, 443.],
 [1150e6, 443.],
 [1200e6, 443.],
 [1250e6, 443.],
 [1300e6, 453.],
 [1350e6, 443.],
 [1400e6, 424.],
 [1450e6, 415.],
 [1500e6, 405.],
 [1550e6, 405.],
 [1600e6, 405.],
 [1650e6, 424.],
 [1711e6, 421.]], dtype=numpy.float32)


class MSNoise(object):
    """
        Estimates visibility noise statistics as a function of frequency given a measurement set (MS).
        This statistics can be used to generate weights which can be saved in the MS.
    """
    def __init__(self, ms):
        """
        Args:
            ms (Directory, CASA Table):
                CASA measurement set
        """

        self.ms = ms
        # First get some basic info about the Ms
        self.msinfo = msutils.summary(self.ms, display=False)
        self.nrows = self.msinfo['NROW']
        self.ncor = self.msinfo['NCOR']
        self.spw = {
            "freqs" : self.msinfo['SPW']['CHAN_FREQ'],
            "nchan" : self.msinfo['SPW']['NUM_CHAN'],
        }
        self.nspw = len(self.spw['freqs'])

    def estimate_weights(self, mode='specs',
                stats_data=None, normalise=True, 
                smooth='polyn', fit_order=9,
                plot_stats=True):
        """
        Args:
            mode (str, optional):
                Mode for estimating noise statistics. These are the options:
                - specs : This is a file or array of values that are proportional to the sensitivity as a function of
                  frequency. For example SEFD values. Column one should be the frequency, and column two should be the sensitivity.
                - calc : Calculate noise internally. The calculations estimates the noise by taking differences between
                  adjacent channels. 
            noise_data (file, list, numpy.ndarray):
                File or array containing information about sensitivity as a function of frequency (in Hz)
            smooth (str, optional):
                Generate a smooth version of the data. This version is used for further calculations. Options are:
                - polyn  : Smooth with a polynomial. This is the dafualt
                - spline : Smooth with a spline
            fit_order (int, optional):
                Oder for function used to smooth the data. Default is 9

        """

        # TODO(sphe): support a 'calc' mode that estimates noise internally.
        # For now, only the 'specs' mode is implemented.
        mode = 'specs'
        if isinstance(stats_data, str):
            __data = numpy.load(stats_data)
        else:
            __data = numpy.array(stats_data, dtype=numpy.float32)

        x,y = __data[:,0], __data[:,1]

        if normalise:
            y /= y.max()

        # lets work in MHz
        x = x*1e-6
        if smooth=='polyn':
            fit_parms = numpy.polyfit(x, y, fit_order)
            fit_func = lambda freqs: numpy.poly1d(fit_parms)(freqs)
        elif smooth=='spline':
            fit_parms = interpolate.splrep(x, y, s=fit_order)
            fit_func = lambda freqs: interpolate.splev(freqs, fit_parms, der=0)

        # Get noise from the parameterised functions for each spectral window
        fig, ax1 = pylab.subplots(figsize=(12,9))
        ax2 = ax1.twinx()
        color = iter(cm.rainbow(numpy.linspace(0,1,self.nspw)))
        noise = []
        weights = []
        for i in range(self.nspw):
            freqs = numpy.array(self.spw['freqs'][i], dtype=numpy.float32)*1e-6
            _noise = fit_func(freqs)
            _weights = 1.0/_noise**2

            if plot_stats:
                # Use a differnet color to mark a new SPW
                ax1.axvspan(freqs[0]/1e3, freqs[-1]/1e3, facecolor=next(color), alpha=0.25)
                # Plot noise/weights
                l1, = ax1.plot(x/1e3, y, 'rx')
                l2, = ax1.plot(freqs/1e3, _noise, 'k-')
                ax1.set_xlabel('Freq [GHz]')
                ax1.set_ylabel('Norm Noise')
                l3, = ax2.plot(freqs/1e3, _weights, 'g-')
                ax2.set_ylabel('Weight')
            noise.append(_noise)
            weights.append(_weights)
        # Set limits based on non-smooth noise
        ylims = 1/y**2
        ax2.set_ylim(ylims.min()*0.9, ylims.max()*1.1)
        pylab.legend([l1,l2,l3], 
            ['Norm. Noise', 'Polynomial fit: n={0:d}'.format(fit_order), 'Weights'], loc=1)
        if isinstance(plot_stats, str):
            pylab.savefig(plot_stats)
        else:
            pylab.savefig(self.ms + '-noise_weights.png')
        pylab.clf()

        return noise, weights


    def write_toms(self, data, 
                  columns=['WEIGHT', 'WEIGHT_SPECTRUM'], 
                  stat='sum', rowchunk=None, multiply_old_weights=False):
        """
          Write noise or weights into an MS.

          Args:
            columns (list):
                columns to write weights/noise and spectral counterparts into. Default is
                columns = ['WEIGHT', 'WEIGHT_SPECTRUM']
            stat (str):
                Statistic to compute when combining data along frequency axis. For example,
                used the sum along Frequency axis of WEIGHT_SPECTRUM as weight for the WEIGHT column
        """

        # Initialise relavant columns. It will exit with zero status if the column alredy exists
        for i, column in enumerate(columns):
            msutils.addcol(self.ms, colname=column,
                       valuetype='float',
                       clone='WEIGHT' if i==0 else 'DATA',

                       )

        for spw in range(self.nspw):
            tab = table(self.ms, readonly=False)
            # Write data into MS in chunks
            rowchunk = rowchunk or max(self.nrows//10, 1)
            for row0 in range(0, self.nrows, rowchunk):
                nr = min(rowchunk, self.nrows-row0)
                # Shape for this chunk
                dshape = [nr, self.spw['nchan'][spw], self.ncor]
                __data = numpy.ones(dshape, dtype=numpy.float32) * data[spw][numpy.newaxis,:,numpy.newaxis]
                # Consider old weights if user wants to
                if multiply_old_weights:
                    old_weight = tab.getcol('WEIGHT', row0, nr)
                    LOGGER.info("Multiplying old weights into WEIGHT_SPECTRUM")
                    __data *= old_weight[:,numpy.newaxis,:]
                # make a masked array to compute stats using unflagged data
                flags = tab.getcol('FLAG', row0, nr)
                mdata = ma.masked_array(__data, mask=flags)

                LOGGER.info("Populating %s column (rows %d to %d)", columns[1], row0, row0+nr-1)
                tab.putcol(columns[1], __data, row0, nr)

                LOGGER.info("Populating %s column (rows %d to %d)", columns[0], row0, row0+nr-1)
                if stat=="stddev":
                    tab.putcol(columns[0], mdata.std(axis=1).data, row0, nr)
                elif stat=="sum":
                    tab.putcol(columns[0], mdata.sum(axis=1).data, row0, nr)

        # Done
        tab.close()

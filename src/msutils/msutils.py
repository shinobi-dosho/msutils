from pyrap.tables import table
import pyrap.tables
import pyrap.measures
import numpy
import traceback
from pyrap.tables import maketabdesc
from pyrap.tables import makearrcoldesc
from pyrap.tables import makescacoldesc
from distutils import spawn
import sys
import subprocess
import math
import json
import codecs
import os

dm = pyrap.measures.measures()

STOKES_TYPES = { 
    0  : "Undefined", 
    1  : "I", 
    2  : "Q", 
    3  : "U", 
    4  : "V", 
    5  : "RR", 
    6  : "RL", 
    7  : "LR", 
    8  : "LL", 
    9  : "XX", 
    10 : "XY", 
    11 : "YX", 
    12 : "YY", 
    13 : "RX", 
    14 : "RY", 
    15 : "LX", 
    16 : "LY", 
    17 : "XR", 
    18 : "XL", 
    19 : "YR", 
    20 : "YL", 
    21 : "PP", 
    22 : "PQ", 
    23 : "QP", 
    24 : "QQ", 
    25 : "RCircular", 
    26 : "LCircular", 
    27 : "Linear", 
    28 : "Ptotal", 
    29 : "Plinear", 
    30 : "PFtotal", 
    31 : "PFlinear", 
    32 : "Pangle", 
}

def summary(msname, outfile=None, display=True):

    tab = pyrap.tables.table(msname)

    info = {
        "FIELD"     :   {},
        "SPW"       :   {},
        "ANT"       :   {},
        "MAXBL"     :   {},
        "SCAN"      :   {},
        "EXPOSURE"  :   {},
        "NROW"      :   tab.nrows(),
        "CORR"      :   {},
    }

    tabs = {
        'FIELD'     :   pyrap.tables.table(msname+'::FIELD'),
        'SPW'       :   pyrap.tables.table(msname+'::SPECTRAL_WINDOW'),
        'ANT'       :   pyrap.tables.table(msname+'::ANTENNA'),
    }

    state_tab = pyrap.tables.table(msname+'::STATE')
    info['FIELD']['INTENTS'] = state_tab.getcol('OBS_MODE')
    state_tab.close()

    fields = numpy.unique(tab.getcol("FIELD_ID"))
    info["FIELD"]["FIELD_ID"] = list(map(int, fields))
    nfields = len(fields)
    nant = tabs['ANT'].nrows()

    info['EXPOSURE'] = tab.getcell("EXPOSURE", 0)

    info['FIELD']['STATE_ID'] = [None]*nfields
    info['FIELD']['PERIOD'] = [None]*nfields
    for i, fid in enumerate(fields):
        ftab = tab.query('FIELD_ID=={0:d}'.format(fid))
        state_id = ftab.getcol('STATE_ID')[0]
        info['FIELD']['STATE_ID'][i] = int(state_id)
        scans = {}
        total_length = 0
        for scan in set(ftab.getcol('SCAN_NUMBER')):
            stab = ftab.query('SCAN_NUMBER=={0:d}'.format(scan))
            length = (stab.getcol('TIME').max() - stab.getcol('TIME').min())
            stab.close()
            scans[str(scan)] = length
            total_length += length

        info['SCAN'][str(fid)] = scans
        info['FIELD']['PERIOD'][i] = total_length
        ftab.close()
        
    for key, _tab in list(tabs.items()):
        if key == 'SPW':
            colnames = 'CHAN_FREQ MEAS_FREQ_REF REF_FREQUENCY TOTAL_BANDWIDTH NAME NUM_CHAN IF_CONV_CHAIN NET_SIDEBAND FREQ_GROUP_NAME'.split()
        else:
            colnames = _tab.colnames()
        for name in colnames:
            try:
                info[key][name] = _tab.getcol(name).tolist()
            except AttributeError:
                info[key][name] = _tab.getcol(name)
        _tab.close()

    # add correlation information
    tabcorr = pyrap.tables.table(msname+'::POLARIZATION')
    corr_type = tabcorr.getcol("CORR_TYPE")[0]
    info["CORR"]["NUM_CORR"] = int(tabcorr.getcol("NUM_CORR")[0])
    info["NCOR"] = int(info["CORR"]["NUM_CORR"])
    corrs = []
    for ct in corr_type:
        corrs.append( STOKES_TYPES[ct] )
    info["CORR"]["CORR_TYPE"] = corrs
    tabcorr.close()

    # get maximum baseline
    uv = tab.getcol("UVW")[:,:2]
    mb = numpy.sqrt((uv**2).sum(1)).max()
    info['MAXBL'] = mb
    tab.close()

    if display:
        print(info)
    
    if outfile:
        with codecs.open(outfile, 'w', 'utf8') as stdw:
            stdw.write(json.dumps(info, ensure_ascii=False))

    return info


def addcol(msname, colname=None, shape=None,
           data_desc_type='array', 
           valuetype=None, 
           init_with=None,
           coldesc=None,
           coldmi=None,
           clone='DATA',
           rowchunk=None,
           **kw):
    """ Add column to MS 
        msanme : MS to add colmn to
        colname : column name
        shape : shape
        valuetype : data type 
        data_desc_type : 'scalar' for scalar elements and array for 'array' elements
        init_with : value to initialise the column with 
    """
    tab = table(msname,readonly=False)

    if colname in tab.colnames():
        print('Column already exists')
        return 'exists'

    print(('Attempting to add %s column to %s'%(colname,msname)))

    valuetype = valuetype or 'complex'

    if coldesc:
        data_desc = coldesc
        shape = coldesc['shape']
    elif shape:
        data_desc = maketabdesc(makearrcoldesc(colname, 
                    init_with,
                    shape=shape,
                    valuetype=valuetype))
    elif valuetype == 'scalar':
        data_desc = maketabdesc(makearrcoldesc(colname, 
                    init_with,
                    valuetype=valuetype))
    elif clone:
        element = tab.getcell(clone, 0)
        try:
            shape = element.shape
            data_desc = maketabdesc(makearrcoldesc(colname, 
                        element.flatten()[0],
                        shape=shape,
                        valuetype=valuetype))
        except AttributeError:
            shape = []
            data_desc = maketabdesc(makearrcoldesc(colname, 
                        element,
                        valuetype=valuetype))
    
    colinfo = [data_desc, coldmi] if coldmi else [data_desc]
    tab.addcols(*colinfo)

    print('Column added successfuly.')

    if init_with is None:
        tab.close()
        return 'added'
    else:
        spwids = set(tab.getcol('DATA_DESC_ID'))
        for spw in spwids:
            print(('Initialising {0:s} column with {1}. DDID is {2:d}'.format(colname, init_with, spw)))
            tab_spw = tab.query('DATA_DESC_ID=={0:d}'.format(spw))
            nrows = tab_spw.nrows()

            rowchunk = rowchunk or nrows/10
            dshape = [0] + [a for a in shape]
            for row0 in range(0,nrows,rowchunk):
                nr = min(rowchunk,nrows-row0)
                dshape[0] = nr
                print(("Wrtiting to column  %s (rows %d to %d)"%(colname, row0, row0+nr-1)))
                tab_spw.putcol(colname,numpy.ones(dshape,dtype=type(init_with))*init_with,row0,nr)
            tab_spw.close()
    tab.close()

    return 'added'


def sumcols(msname, col1=None, col2=None, outcol=None, cols=None, subtract=False):
    """ Add col1 to col2, or sum columns in 'cols' list.
        If subtract, subtract col2 from col1
    """

    tab = table(msname, readonly=False)
    if outcol not in tab.colnames():
        print(('outcol {0:s} does not exist, will add it first.'.format(outcol)))
        addcol(msname, outcol, clone=col1 or cols[0])

    spws = set(tab.getcol('DATA_DESC_ID'))
    for spw in spws:
        tab_spw = tab.query('DATA_DESC_ID=={0:d}'.format(spw))
        nrows = tab_spw.nrows()
        rowchunk = nrows//10 if nrows > 10000 else nrows
        for row0 in range(0, nrows, rowchunk):
            nr = min(rowchunk, nrows-row0)
            print(("Wrtiting to column  %s (rows %d to %d)"%(outcol, row0, row0+nr-1)))
            if subtract:
                data = tab_spw.getcol(col1, row0, nr) - tab_spw.getcol(col2, row0, nr)
            else:
                cols = cols or [col1, col2]
                data = 0
                for col in cols:
                    data += tab.getcol(col, row0, nr)

            tab_spw.putcol(outcol, data, row0, nr)
        tab_spw.close()

    tab.close()


def copycol(msname, fromcol, tocol):
    """
        Copy data from one column to another
    """

    tab = table(msname, readonly=False)
    if tocol not in tab.colnames():
        addcol(msname, tocol, clone=fromcol)

    spws = set(tab.getcol('DATA_DESC_ID'))
    for spw in spws:
        tab_spw = tab.query('DATA_DESC_ID=={0:d}'.format(spw))
        nrows = tab_spw.nrows()
        rowchunk = nrows//10 if nrows > 5000 else nrows
        for row0 in range(0, nrows, rowchunk):
            nr = min(rowchunk, nrows-row0)
            data = tab_spw.getcol(fromcol, row0, nr)
            tab_spw.putcol(tocol, data, row0, nr)

        tab_spw.close()
    tab.close()


def compute_vis_noise(msname, sefd, spw_id=0):
    """Computes nominal per-visibility noise"""

    tab = table(msname)
    spwtab = table(msname + "::SPECTRAL_WINDOW")

    freq0 = spwtab.getcol("CHAN_FREQ")[spw_id, 0]
    wavelength = 300e+6/freq0
    bw = spwtab.getcol("CHAN_WIDTH")[spw_id, 0]
    dt = tab.getcol("EXPOSURE", 0, 1)[0]
    dtf = (tab.getcol("TIME", tab.nrows()-1, 1)-tab.getcol("TIME", 0, 1))[0]

    # close tables properly, else the calls below will hang waiting for a lock...
    tab.close()
    spwtab.close()

    print(("%s freq %.2f MHz (lambda=%.2fm), bandwidth %.2g kHz, %.2fs integrations, %.2fh synthesis"%(msname, 
        freq0*1e-6, wavelength, bw*1e-3, dt, dtf/3600)))
    noise = sefd/math.sqrt(abs(2*bw*dt))
    print(("SEFD of %.2f Jy gives per-visibility noise of %.2f mJy"%(sefd, noise*1000)))

    return noise


def verify_antpos (msname, fix=False, hemisphere=None):
    """Verifies antenna Y positions in MS. If Y coordinate convention is wrong, either fixes the positions (fix=True) or
    raises an error. hemisphere=-1 makes it assume that the observatory is in the Western hemisphere, hemisphere=1
    in the Eastern, or else tries to find observatory name using MS and pyrap.measure."""


    if not hemisphere:
        obs = table(msname+"::OBSERVATION").getcol("TELESCOPE_NAME")[0]
        print(("observatory is %s"%obs))
        try:
          hemisphere = 1 if dm.observatory(obs)['m0']['value'] > 0 else -1
        except:
          traceback.print_exc();
          print(("WARNING:: %s is unknown, or pyrap.measures is missing. Will not verify antenna positions."%obs))
          return 
    print(("antenna Y positions should be of sign %+d"%hemisphere))
    
    anttab = table(msname+"::ANTENNA", readonly=False)
    pos = anttab.getcol("POSITION")
    wrong = pos[:,1]<0 if hemisphere>0 else pos[:,1]>0
    nw = sum(wrong)

    if nw: 
        if not fix:
            os.abort("%s/ANTENNA has $nw incorrect Y antenna positions. Check your coordinate conversions (from UVFITS?), or run verify_antpos[fix=True]"%msname)
        pos[wrong,1] *= -1; 
        anttab.putcol("POSITION", pos)
        print(("WARNING:%s/ANTENNA: %s incorrect antenna positions were adjusted (Y sign flipped)"%(msname, nw)))
    else:
        print(("%s/ANTENNA: all antenna positions appear to have correct Y sign"%msname))


def prep(msname, verify=False):
    """Prepares MS for use with MeqTrees: adds imaging columns, adds BITFLAG columns, copies current flags
       to 'legacy' flagset
    """

    if verify:
        verify_antpos(msname, fix=verify);

    print("Adding imaging columns")
    pyrap.tables.addImagingColumns(msname)
    
    # check if addbitflagcol exists
    if spawn.find_executable("addbitflagcol"):
        print(("Adding bitflag column to %s"%msname))
        subprocess.check_call(['addbitflagcol', msname],
                         stderr=subprocess.PIPE if not isinstance(sys.stderr,file) else sys.stderr,  # noqa: F821 
                         stdout=subprocess.PIPE if not isinstance(sys.stdout,file) else sys.stdout)  # noqa: F821
    
    if spawn.find_executable("flag-ms.py"):
        print("Copying FLAG to bitflag 'legacy'")
        subprocess.check_call(['flag-ms.py', '-Y', '+L', '-f', 'legacy', '-c', msname],
                         stderr=subprocess.PIPE if not isinstance(sys.stderr,file) else sys.stderr,  # noqa: F821
                         stdout=subprocess.PIPE if not isinstance(sys.stdout,file) else sys.stdout)  # noqa: F821

        print("Flagging INFs/NaNs in data")
        subprocess.Popen(['flag-ms.py', '--nan', '-f', 'legacy', '--data-column', 'DATA', '-x', msname],
                         stderr=subprocess.PIPE if not isinstance(sys.stderr,file) else sys.stderr,  # noqa: F821
                         stdout=subprocess.PIPE if not isinstance(sys.stdout,file) else sys.stdout)  # noqa: F821


def addnoise(msname, column='MODEL_DATA',
             noise=0, sefd=551,
              rowchunk=None, 
              addToCol=None, 
              spw_id=None):
    """ Add Gaussian noise to MS, given a stdandard deviation (noise). 
        This noise can be also be calculated given SEFD value
    """

    tab = table(msname, readonly=False)

    multi_chan_noise = False
    if hasattr(noise, '__iter__'):
        multi_chan_noise = True
    elif hasattr(sefd, '__iter__'):
        multi_chan_noise = True
    else:
        noise = noise or compute_vis_noise(msname, sefd=sefd, spw_id=spw_id or 0)


    spws = set(tab.getcol('DATA_DESC_ID'))
    for spw in spws:
        tab_spw = tab.query('DATA_DESC_ID=={0:d}'.format(spw))
        nrows = tab_spw.nrows()
        nchan,ncor = tab_spw.getcell('DATA', 0).shape

        rowchunk = rowchunk or nrows/10
         
        for row0 in range(0, nrows, rowchunk):
            nr = min(rowchunk, nrows-row0)
            data = numpy.random.randn(nr, nchan, ncor) + 1j*numpy.random.randn(nr, nchan, ncor)
            if multi_chan_noise:
                noise = noise[numpy.newaxis,:,numpy.newaxis]
            data *= noise

            if addToCol: 
                data += tab_spw.getcol(addToCol, row0, nr)
                print(("%s + noise --> %s (rows %d to %d)"%(addToCol, column, row0, row0+nr-1)))
            else: 
                print(("Adding noise to column %s (rows %d to %d)"%(column, row0, row0+nr-1)))

            tab_spw.putcol(column, data, row0, nr)
        tab_spw.close()

    tab.close()

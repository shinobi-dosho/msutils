"""Core Measurement Set column operations (numpy + python-casacore only)."""
from __future__ import annotations

import codecs
import json
import math
import traceback
from contextlib import closing
from typing import Any, Optional, Sequence, Union

import numpy
import casacore.measures
from casacore.tables import table, maketabdesc, makearrcoldesc

from ._log import create_logger
from ._tables import open_table

__all__ = [
    "STOKES_TYPES",
    "summary",
    "addcol",
    "sumcols",
    "copycol",
    "compute_vis_noise",
    "verify_antpos",
    "addnoise",
]

LOGGER = create_logger(__name__)

dm = casacore.measures.measures()

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


def summary(msname: str, outfile: Optional[str] = None, display: bool = True) -> dict:
    """Collect a JSON-serializable summary of an MS (fields, SPWs, antennas, scans)."""

    with open_table(msname) as tab:
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

        with open_table(msname+'::STATE') as state_tab:
            info['FIELD']['INTENTS'] = state_tab.getcol('OBS_MODE')

        fields = numpy.unique(tab.getcol("FIELD_ID"))
        info["FIELD"]["FIELD_ID"] = list(map(int, fields))
        nfields = len(fields)

        info['EXPOSURE'] = tab.getcell("EXPOSURE", 0)

        info['FIELD']['STATE_ID'] = [None]*nfields
        info['FIELD']['PERIOD'] = [None]*nfields
        for i, fid in enumerate(fields):
            with closing(tab.query('FIELD_ID=={0:d}'.format(fid))) as ftab:
                state_id = ftab.getcol('STATE_ID')[0]
                info['FIELD']['STATE_ID'][i] = int(state_id)
                scans = {}
                total_length = 0
                for scan in set(ftab.getcol('SCAN_NUMBER')):
                    with closing(ftab.query('SCAN_NUMBER=={0:d}'.format(scan))) as stab:
                        length = (stab.getcol('TIME').max() - stab.getcol('TIME').min())
                    scans[str(scan)] = length
                    total_length += length

            info['SCAN'][str(fid)] = scans
            info['FIELD']['PERIOD'][i] = total_length

        with open_table(msname+'::FIELD') as field_tab, \
                open_table(msname+'::SPECTRAL_WINDOW') as spw_tab, \
                open_table(msname+'::ANTENNA') as ant_tab:
            tabs = {'FIELD': field_tab, 'SPW': spw_tab, 'ANT': ant_tab}
            for key, _tab in tabs.items():
                if key == 'SPW':
                    colnames = 'CHAN_FREQ MEAS_FREQ_REF REF_FREQUENCY TOTAL_BANDWIDTH NAME NUM_CHAN IF_CONV_CHAIN NET_SIDEBAND FREQ_GROUP_NAME'.split()
                else:
                    colnames = _tab.colnames()
                for name in colnames:
                    try:
                        info[key][name] = _tab.getcol(name).tolist()
                    except AttributeError:
                        info[key][name] = _tab.getcol(name)

        # add correlation information
        with open_table(msname+'::POLARIZATION') as tabcorr:
            corr_type = tabcorr.getcol("CORR_TYPE")[0]
            info["CORR"]["NUM_CORR"] = int(tabcorr.getcol("NUM_CORR")[0])
            info["NCOR"] = int(info["CORR"]["NUM_CORR"])
            info["CORR"]["CORR_TYPE"] = [STOKES_TYPES[ct] for ct in corr_type]

        # get maximum baseline
        uv = tab.getcol("UVW")[:,:2]
        info['MAXBL'] = numpy.sqrt((uv**2).sum(1)).max()

    if display:
        LOGGER.info(info)

    if outfile:
        with codecs.open(outfile, 'w', 'utf8') as stdw:
            stdw.write(json.dumps(info, ensure_ascii=False))

    return info


def addcol(msname: str, colname: Optional[str] = None, shape: Optional[Sequence[int]] = None,
           data_desc_type: str = 'array',
           valuetype: Optional[str] = None,
           init_with: Any = None,
           coldesc: Optional[dict] = None,
           coldmi: Optional[dict] = None,
           clone: Optional[str] = 'DATA',
           rowchunk: Optional[int] = None,
           **kw) -> str:
    """ Add column to MS
        msanme : MS to add colmn to
        colname : column name
        shape : shape
        valuetype : data type
        data_desc_type : 'scalar' for scalar elements and array for 'array' elements
        init_with : value to initialise the column with
    """
    with open_table(msname, readonly=False) as tab:
        if colname in tab.colnames():
            LOGGER.info('Column already exists')
            return 'exists'

        LOGGER.info('Attempting to add %s column to %s', colname, msname)

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

        LOGGER.info('Column added successfuly.')

        if init_with is None:
            return 'added'

        spwids = set(tab.getcol('DATA_DESC_ID'))
        for spw in spwids:
            LOGGER.info('Initialising {0:s} column with {1}. DDID is {2:d}'.format(colname, init_with, spw))
            with closing(tab.query('DATA_DESC_ID=={0:d}'.format(spw))) as tab_spw:
                nrows = tab_spw.nrows()

                rowchunk = rowchunk or max(nrows//10, 1)
                dshape = [0] + [a for a in shape]
                for row0 in range(0, nrows, rowchunk):
                    nr = min(rowchunk, nrows-row0)
                    dshape[0] = nr
                    LOGGER.info("Writing to column  %s (rows %d to %d)", colname, row0, row0+nr-1)
                    tab_spw.putcol(colname, numpy.ones(dshape, dtype=type(init_with))*init_with, row0, nr)

    return 'added'


def sumcols(msname: str, col1: Optional[str] = None, col2: Optional[str] = None,
            outcol: Optional[str] = None, cols: Optional[Sequence[str]] = None,
            subtract: bool = False) -> None:
    """ Add col1 to col2, or sum columns in 'cols' list.
        If subtract, subtract col2 from col1
    """

    with open_table(msname, readonly=False) as tab:
        if outcol not in tab.colnames():
            LOGGER.info('outcol {0:s} does not exist, will add it first.'.format(outcol))
            addcol(msname, outcol, clone=col1 or cols[0])

        spws = set(tab.getcol('DATA_DESC_ID'))
        for spw in spws:
            with closing(tab.query('DATA_DESC_ID=={0:d}'.format(spw))) as tab_spw:
                nrows = tab_spw.nrows()
                rowchunk = nrows//10 if nrows > 10000 else nrows
                for row0 in range(0, nrows, rowchunk):
                    nr = min(rowchunk, nrows-row0)
                    LOGGER.info("Writing to column  %s (rows %d to %d)", outcol, row0, row0+nr-1)
                    if subtract:
                        data = tab_spw.getcol(col1, row0, nr) - tab_spw.getcol(col2, row0, nr)
                    else:
                        cols = cols or [col1, col2]
                        data = 0
                        for col in cols:
                            data += tab_spw.getcol(col, row0, nr)

                    tab_spw.putcol(outcol, data, row0, nr)


def copycol(msname: str, fromcol: str, tocol: str) -> None:
    """
        Copy data from one column to another
    """

    with open_table(msname, readonly=False) as tab:
        if tocol not in tab.colnames():
            addcol(msname, tocol, clone=fromcol)

        spws = set(tab.getcol('DATA_DESC_ID'))
        for spw in spws:
            with closing(tab.query('DATA_DESC_ID=={0:d}'.format(spw))) as tab_spw:
                nrows = tab_spw.nrows()
                rowchunk = nrows//10 if nrows > 5000 else nrows
                for row0 in range(0, nrows, rowchunk):
                    nr = min(rowchunk, nrows-row0)
                    data = tab_spw.getcol(fromcol, row0, nr)
                    tab_spw.putcol(tocol, data, row0, nr)


def compute_vis_noise(msname: str, sefd: float, spw_id: int = 0) -> float:
    """Computes nominal per-visibility noise"""

    with open_table(msname) as tab, open_table(msname + "::SPECTRAL_WINDOW") as spwtab:
        freq0 = spwtab.getcol("CHAN_FREQ")[spw_id, 0]
        wavelength = 300e+6/freq0
        bw = spwtab.getcol("CHAN_WIDTH")[spw_id, 0]
        dt = tab.getcol("EXPOSURE", 0, 1)[0]
        dtf = (tab.getcol("TIME", tab.nrows()-1, 1)-tab.getcol("TIME", 0, 1))[0]

    LOGGER.info("%s freq %.2f MHz (lambda=%.2fm), bandwidth %.2g kHz, %.2fs integrations, %.2fh synthesis",
                msname, freq0*1e-6, wavelength, bw*1e-3, dt, dtf/3600)
    noise = sefd/math.sqrt(abs(2*bw*dt))
    LOGGER.info("SEFD of %.2f Jy gives per-visibility noise of %.2f mJy", sefd, noise*1000)

    return noise


def verify_antpos(msname: str, fix: bool = False, hemisphere: Optional[int] = None) -> None:
    """Verifies antenna Y positions in MS. If Y coordinate convention is wrong, either fixes the positions (fix=True) or
    raises an error. hemisphere=-1 makes it assume that the observatory is in the Western hemisphere, hemisphere=1
    in the Eastern, or else tries to find observatory name using MS and casacore.measures."""

    if not hemisphere:
        with open_table(msname+"::OBSERVATION") as obstab:
            obs = obstab.getcol("TELESCOPE_NAME")[0]
        LOGGER.info("observatory is %s", obs)
        try:
            hemisphere = 1 if dm.observatory(obs)['m0']['value'] > 0 else -1
        except Exception:
            traceback.print_exc()
            LOGGER.warning("%s is unknown, or casacore.measures is missing. Will not verify antenna positions.", obs)
            return
    LOGGER.info("antenna Y positions should be of sign %+d", hemisphere)

    with open_table(msname+"::ANTENNA", readonly=False) as anttab:
        pos = anttab.getcol("POSITION")
        wrong = pos[:,1]<0 if hemisphere>0 else pos[:,1]>0
        nw = sum(wrong)

        if nw:
            if not fix:
                raise RuntimeError(
                    "%s/ANTENNA has %d incorrect Y antenna positions. Check your coordinate conversions "
                    "(from UVFITS?), or run verify_antpos(fix=True)" % (msname, nw))
            pos[wrong,1] *= -1
            anttab.putcol("POSITION", pos)
            LOGGER.warning("%s/ANTENNA: %s incorrect antenna positions were adjusted (Y sign flipped)", msname, nw)
        else:
            LOGGER.info("%s/ANTENNA: all antenna positions appear to have correct Y sign", msname)


def addnoise(msname: str, column: str = 'MODEL_DATA',
             noise: Union[float, Sequence[float]] = 0,
             sefd: Union[float, Sequence[float]] = 551,
             rowchunk: Optional[int] = None,
             addToCol: Optional[str] = None,
             spw_id: Optional[int] = None) -> None:
    """ Add Gaussian noise to MS, given a stdandard deviation (noise).
        This noise can be also be calculated given SEFD value
    """

    with open_table(msname, readonly=False) as tab:
        multi_chan_noise = False
        if hasattr(noise, '__iter__'):
            multi_chan_noise = True
        elif hasattr(sefd, '__iter__'):
            multi_chan_noise = True
        else:
            noise = noise or compute_vis_noise(msname, sefd=sefd, spw_id=spw_id or 0)

        spws = set(tab.getcol('DATA_DESC_ID'))
        for spw in spws:
            with closing(tab.query('DATA_DESC_ID=={0:d}'.format(spw))) as tab_spw:
                nrows = tab_spw.nrows()
                nchan, ncor = tab_spw.getcell('DATA', 0).shape

                rowchunk = rowchunk or max(nrows//10, 1)

                for row0 in range(0, nrows, rowchunk):
                    nr = min(rowchunk, nrows-row0)
                    data = numpy.random.randn(nr, nchan, ncor) + 1j*numpy.random.randn(nr, nchan, ncor)
                    if multi_chan_noise:
                        noise = noise[numpy.newaxis,:,numpy.newaxis]
                    data *= noise

                    if addToCol:
                        data += tab_spw.getcol(addToCol, row0, nr)
                        LOGGER.info("%s + noise --> %s (rows %d to %d)", addToCol, column, row0, row0+nr-1)
                    else:
                        LOGGER.info("Adding noise to column %s (rows %d to %d)", column, row0, row0+nr-1)

                    tab_spw.putcol(column, data, row0, nr)

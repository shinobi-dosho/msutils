#from contextlib import ExitStack
import sys
import math
import json
import numpy
import logging

import casacore.measures

from MSUtils import msutils

import dask
import dask.array as da
from daskms import xds_from_ms, xds_to_table, xds_from_table

from bokeh.layouts import row, column
from bokeh.plotting import figure, output_file, save

def create_logger():
    """Create a console logger"""
    log = logging.getLogger(__name__)
    cfmt = logging.Formatter(('%(name)s - %(asctime)s %(levelname)s - %(message)s'))
    log.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(cfmt)
    log.addHandler(console)
    return log


LOGGER = create_logger()


def _get_ant_flags(names, antenna1, antenna2, flags):
    names = names[0]
    # chan and corr are assumed to have a single chunk
    # so we contract twice to access this single ndarray
    flags = flags[0][0]
    nant = len(names)
    fracs = numpy.zeros([nant,2], dtype=numpy.float64)
    for i in range(nant):
        flag_sum = flags[numpy.logical_or(antenna1==i, antenna2==i)]
        fracs[i,0] += flag_sum.sum()
        fracs[i,1] += numpy.product(flag_sum.shape)
    return fracs

def _get_flags(names, flags):
    names = names[0]
    # chan and corr are assumed to have a single chunk
    # so we contract twice to access this single ndarray
    flags = flags[0][0]
    num = len(names) # num can be scan, corr, field names
    fracs = numpy.zeros([num,2], dtype=numpy.float64)
    for i in range(num):
        fracs[i,0] = flags.sum()
        fracs[i,1] = numpy.product(flags.shape)
    return fracs

def _chunk(x, keepdims, axis):
    return x

def _combine(x, keepdims, axis):
    if isinstance(x, list):
        return sum(x)
    elif isinstance(x, numpy.ndarray):
        return x
    else:
        raise TypeError("Invalid type %s" % type(x))

def _aggregate(x, keepdims, axis):
    return _combine(x, keepdims, axis)

def _distance(xyz1, xyz2):
    """Distance between two points in a three dimension coordinate system"""
    x = xyz2[0] - xyz1[0]
    y = xyz2[1] - xyz1[1]
    z = xyz2[2] - xyz1[2]
    d2 = (x * x) + (y * y) + (z * z)
    d = numpy.sqrt(d2)
    return d

def wgs84_to_ecef(lon, lat, alt):
    """
    Convert wgs84(latitude (deg), longitude(deg), elevation(deg)) to
    Earth Centred Earth fixed coordinates (X(m), Y(m), Z(m)).
    Coordinates in the ITRF format should be converted to WGS84 coordinate system for consistency.
    This function is an implementation based on the following reference:
    https://docs.hisparc.nl/coordinates/HiSPARC_coordinates.pdf

    Parameters
    ----------:
    lat: :obj:`float`
        Latitude in degrees
    lon: :obj:`float`
        Longitude in degrees
    alt: :obj:`float`
        Altitude in metres

    Returns
    -------
    X, Y, Z:  ECEF coordinates in metres

    """

    # set up earth's shape ellipsoid approximation
    # semi major axis
    a = 6378137.0
    # flattening
    f = 1 / 298.257223563
    # semi-minor axis
    b = a - a * f
    # eccentricity
    e = numpy.sqrt((2 * f) - (f**2))
    # Normal: Distance between a location on the ellipsoid and the
    # intersection of its nromal and the ellipsoid's z-axis
    N = a / numpy.sqrt(1 - e**2 * numpy.sin(lat)**2)
    # altitude
    h = alt
    # transformation
    X = (N + h) * numpy.cos(lat) * numpy.cos(lon)
    Y = (N + h) * numpy.cos(lat) * numpy.sin(lon)
    Z = (((b**2 / a**2) * N) + h) * numpy.sin(lat)
    return X, Y, Z


def antenna_flags_field(msname, fields=None, antennas=None):
    ds_ant = xds_from_table(msname+"::ANTENNA")[0]
    ds_field = xds_from_table(msname+"::FIELD")[0]
    ds_obs = xds_from_table(msname+"::OBSERVATION")[0]

    ant_names = ds_ant.NAME.data.compute()
    field_names = ds_field.NAME.data.compute()
    ant_positions = ds_ant.POSITION.data.compute()
    LOGGER.info("Computing antenna flag stats data...")
    LOGGER.info(f"Antenna Names: {ant_names}")

    try:
        # Get observatory name and array centre
        obs_name = ds_obs.TELESCOPE_NAME.data.compute()[0]
        LOGGER.info(f"Observatory Name: {obs_name}")
        me = casacore.measures.measures()
        obs_cofa = me.observatory(obs_name)
        lon, lat, alt = (obs_cofa['m0']['value'],
                         obs_cofa['m1']['value'],
                         obs_cofa['m2']['value'])
        cofa = wgs84_to_ecef(lon, lat, alt)
    except:
        # Otherwise use the first id antenna as array centre
        LOGGER.warn("Using the first id antenna as array centre.")
        cofa = ant_positions[0]

    if fields:
        if isinstance(fields[0], str):
            field_ids = list(map(fields.index, fields))
        else:
            field_ids = fields
    else:
        field_ids = list(range(len(field_names)))

    if antennas:
        if isinstance(antennas[0], str):
            ant_ids = list(map(antennas.index, antennas))
        else:
            ant_ids = antennas
    else:
        ant_ids = list(range(len(ant_names)))

    nant = len(ant_ids)
    nfield = len(field_ids)
    fields_str = ", ".join(map(str, field_ids))
    missing_antennas = []

    flag_sum_computes = []
    for ant_id in ant_ids:
        ds = xds_from_ms(msname,
                chunks={'row': 100000},
                taql_where=f"ANTENNA1 IN [{ant_id}] or ANTENNA2 IN [{ant_id}]")
        if ds:
            ds = ds[0]
            flag_sums = da.blockwise(_get_ant_flags, ("row",),
                                     [ant_id], ("ant",),
                                     ds.ANTENNA1.data, ("row",),
                                     ds.ANTENNA2.data, ("row",),
                                     ds.FLAG.data, ("row","chan", "corr"),
                                     dtype=numpy.ndarray)

            flags_redux = da.reduction(flag_sums,
                                       chunk=_chunk,
                                       combine=_combine,
                                       aggregate=_aggregate,
                                       concatenate=False,
                                       dtype=numpy.float64)
            flag_sum_computes.append(flags_redux)
        else:
           missing_antennas.append(ant_id)
           LOGGER.warn(f"No data found for Antenna-Id: {ant_id}")


    stats = {}
    sum_per_antenna = dask.compute(flag_sum_computes)[0]
    ant_ids = [ant_id for ant_id in ant_ids if ant_id not in missing_antennas]
    for i,aid in enumerate(ant_ids):
        ant_stats = {}
        ant_pos = list(ant_positions[i])
        fraction = sum_per_antenna[i][0][0]/sum_per_antenna[i][0][1]
        ant_stats["name"] = ant_names[aid]
        ant_stats["position"] = ant_pos
        ant_stats["array_centre_dist"] = _distance(cofa, ant_pos)
        ant_stats["frac"] = fraction
        ant_stats["sum"] = sum_per_antenna[i][0][0]
        ant_stats["counts"] = sum_per_antenna[i][0][1]
        stats[aid] = ant_stats

    return stats

def scan_flags_field(msname, fields=None):
    ds_field = xds_from_table(msname+"::FIELD")[0]
    ds_obs = xds_from_table(msname+"::OBSERVATION")[0]
    field_names = ds_field.NAME.data.compute()
    LOGGER.info("Computing scan flag stats data...")

    if fields:
        if isinstance(fields[0], str):
            field_ids = list(map(fields.index, fields))
        else:
            field_ids = fields
    else:
        field_ids = list(range(len(field_names)))

    fields_str = ", ".join(map(str, field_ids))
    ds_mss = xds_from_ms(msname, group_cols=["SCAN_NUMBER"],
            chunks={'row': 100000}, taql_where="FIELD_ID IN [%s]" % fields_str)
    flag_sum_computes = []

    scan_ids = list(range(len(ds_mss)))
    scan_names = [str(ds.SCAN_NUMBER) for ds in ds_mss]
    nscans = len(scan_ids)
    LOGGER.info(f"Scan Names: {scan_names}")

    for ds in ds_mss:
        flag_sums = da.blockwise(_get_flags, ("row",),
                                    scan_ids, ("scan",),
                                    ds.FLAG.data, ("row","chan", "corr"),
                                    adjust_chunks={"row": nscans },
                                    dtype=numpy.ndarray)

        flags_redux = da.reduction(flag_sums,
                                 chunk=_chunk,
                                 combine=_combine,
                                 aggregate=_aggregate,
                                 concatenate=False,
                                 dtype=numpy.float64)
        flag_sum_computes.append(flags_redux)

    sum_per_scan_spw = dask.compute(flag_sum_computes)[0]
    stats = {}
    for i,sid in enumerate(scan_ids):
        scan_stats = {}
        sum_all = sum(sum_per_scan_spw[i])
        fraction = sum_all[0]/sum_all[1]
        scan_stats["name"] = scan_names[sid]
        scan_stats["frac"] = fraction
        scan_stats["sum"] = sum_all[0]
        scan_stats["counts"] = sum_all[1]
        stats[sid] = scan_stats

    return stats

def source_flags_field(msname, fields=None):
    ds_field = xds_from_table(msname+"::FIELD")[0]
    ds_obs = xds_from_table(msname+"::OBSERVATION")[0]
    field_names = ds_field.NAME.data.compute()
    LOGGER.info("Computing field flag stats data...")
    LOGGER.info(f"Field Names: {field_names}")

    if fields:
        if isinstance(fields[0], str):
            field_ids = list(map(list(field_names).index, fields))
        else:
            field_ids = fields
    else:
        field_ids = list(range(len(field_names)))

    fields_str = ", ".join(map(str, field_ids))
    flag_sum_computes = []
    nfields = len(field_ids)
    missing_fields = []

    for field_id in field_ids:
        ds = xds_from_ms(msname,
                chunks={'row': 100000}, taql_where="FIELD_ID IN [%s]" % str(field_id))
        if ds:
            ds = ds[0]
            flag_sums = da.blockwise(_get_flags, ("row",),
                                     [field_id], ("field",),
                                     ds.FLAG.data, ("row","chan", "corr"),
                                     dtype=numpy.ndarray)

            flags_redux = da.reduction(flag_sums,
                                       chunk=_chunk,
                                       combine=_combine,
                                       aggregate=_aggregate,
                                       concatenate=False,
                                       dtype=numpy.float64)
            flag_sum_computes.append(flags_redux)
        else:
           missing_fields.append(field_id)
           LOGGER.warn(f"No data for field-Id: {field_id}")

    stats = {}
    sum_per_field_spw = dask.compute(flag_sum_computes)[0]
    field_ids = [field_id for field_id in field_ids if field_id not in missing_fields]
    for i,fid in enumerate(field_ids):
        field_stats = {}
        sum_all = sum(sum_per_field_spw[i])
        fraction = sum_all[0]/sum_all[1]
        field_stats["name"] = field_names[fid]
        field_stats["frac"] = fraction
        field_stats["sum"] = sum_all[0]
        field_stats["counts"] = sum_all[1]
        stats[fid] = field_stats

    return stats


def correlation_flags_field(msname, fields=None):
    ds_field = xds_from_table(msname+"::FIELD")[0]
    ds_obs = xds_from_table(msname+"::OBSERVATION")[0]
    ds_pols = xds_from_table(msname+"::POLARIZATION")[0]
    corr_names = [msutils.STOKES_TYPES[corr] for corr in list(ds_pols.CORR_TYPE.data.compute()[0])]
    field_names = ds_field.NAME.data.compute()
    corr_ids = list(range(len(corr_names)))
    ncorrs = len(corr_ids)
    LOGGER.info("Computing correlation flag stats data...")
    LOGGER.info(f"Correlation Names: {corr_names}")

    if fields:
        if isinstance(fields[0], str):
            field_ids = list(map(list(field_names).index, fields))
        else:
            field_ids = fields
    else:
        field_ids = list(range(len(field_names)))

    fields_str = ", ".join(map(str, field_ids))
    ds_mss = xds_from_ms(msname, group_cols=["DATA_DESC_ID"],
                         chunks={'row': 100000},
                         taql_where="FIELD_ID IN [%s]" % fields_str)
    flag_sum_computes = []
    nfields = len(field_ids)

    for ds in ds_mss:
        flag_sums = da.blockwise(_get_flags, ("row",),
                                    corr_ids, ("corr",),
                                    ds.FLAG.data, ("row", "chan", "corr"),
                                    adjust_chunks={"corr": ncorrs},
                                    dtype=numpy.ndarray)

        flags_redux = da.reduction(flag_sums,
                                 chunk=_chunk,
                                 combine=_combine,
                                 aggregate=_aggregate,
                                 concatenate=False,
                                 dtype=numpy.float64)
        flag_sum_computes.append(flags_redux)

    sum_per_corr_spw = dask.compute(flag_sum_computes)[0]
    stats = {}
    for i,cid in enumerate(corr_ids):
        corr_stats = {}
        sum_all = [sum_per_corr_spw[0][i][0], sum_per_corr_spw[0][i][1]]
        fraction = sum_all[0]/sum_all[1]
        corr_stats["name"] = corr_names[cid]
        corr_stats["frac"] = fraction
        corr_stats["sum"] = sum_all[0]
        corr_stats["counts"] = sum_all[1]
        stats[cid] = corr_stats

    return stats

def _plot_flag_stats(antenna_stats, scan_stats, target_stats, corr_stats, outfile=None):
    """Plot antenna, corr, scan or target summary flag stats"""
    LOGGER.info("Plotting flag stats data.")
    plots={
           'fields': {'title':'Field RFI summary',
                      'x_label': 'Field',
                      'y_label': 'Flagged data (%)',
                      'rotate_xlabel':False},
           'antennas': {'title':'Antenna RFI summary',
                        'x_label': 'Antenna',
                        'y_label': 'Flagged data (%)',
                        'rotate_xlabel':True},
           'scans': {'title':'Scans RFI summary',
                     'x_label': 'Scans',
                     'y_label': 'Flagged data (%)',
                     'rotate_xlabel':True},
           'corrs': {'title':'Correlation RFI summary',
                            'x_label': 'Correlation',
                            'y_label': 'Flagged data (%)',
                            'rotate_xlabel':False}
          }

    plot_list = []
    if not outfile:
        outfile = 'default-flagging-summary-plots.html'
    for flag_stats in [antenna_stats, scan_stats, target_stats, corr_stats]:
        key = list(flag_stats.keys())[0]
        flag_data = list(flag_stats.values())[0]
        stats_keys = [fd['name'] for fd in flag_data.values()]
        flag_percentages = [fd['frac']*100 for fd in flag_data.values()]
        rotate_xlabel = plots[key]['rotate_xlabel']
        x_label=plots[key]['x_label']
        y_label=plots[key]['y_label']
        title=plots[key]['title']
        plotter = figure(x_range=stats_keys, x_axis_label=x_label, y_axis_label=y_label,
                         plot_width=600, plot_height=400, title=title, y_range=(0, 100),
                         tools="hover,box_zoom,wheel_zoom,pan,save,reset")
        plotter.vbar(x=stats_keys, top=flag_percentages, width=0.9)
        plotter.xgrid.grid_line_color = None
        plotter.y_range.start = 0
        plotter.title.align = 'center'
        plotter.hover.tooltips = [(key.title(), "@x"), ("%", "@top")]
        if rotate_xlabel:
            plotter.xaxis.major_label_orientation = math.pi/2
        plot_list.append(plotter)
    output_file(outfile)
    LOGGER.info(f"Output plots: {outfile}.")
    save(column(row(plot_list[2], plot_list[3]),
                row(plot_list[1], plot_list[0])))


def plot_statistics(msname, antennas=None, fields=None, htmlfile=None, outfile=None):
    """Plot stats data"""
    flag_data = save_statistics(msname, antennas=antennas, fields=fields, outfile=outfile)
    _plot_flag_stats(**flag_data, outfile=htmlfile)


def save_statistics(msname, antennas=None, fields=None, outfile=None):
    """Save flag statistics to a json file"""
    target_stats = {'fields': source_flags_field(msname, fields)}
    scan_stats = {'scans': scan_flags_field(msname, fields)}
    antenna_stats = {'antennas': antenna_flags_field(msname, fields, antennas)}
    corr_stats = {'corrs': correlation_flags_field(msname, fields)}
    flag_data = {'Flag stats': [scan_stats, antenna_stats, target_stats, corr_stats]}
    if not outfile:
        outfile = 'default-flag-statistics.json'
    LOGGER.info(f'Output json file: {outfile}.')
    with open(outfile, 'w') as f:
        json.dump(flag_data, f)
    flag_data = {'antenna_stats': antenna_stats, 'scan_stats': scan_stats,
                 'target_stats': target_stats, 'corr_stats': corr_stats}
    return flag_data

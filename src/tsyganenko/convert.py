"""convert: Wrappers for the conversion routines from Geopack."""
import geopack_tsyganenko as _geopack


def recalc(datetime, v_x, v_y, v_z):
    """
    Run RECALC_08 for the user so they don't have to import geopack in its entirety.

    Parameters
    ----------
    datetime : datetime.datetime
    v_x, v_y, v_z : float
        The X, Y and Z components of the solar wind velocity.
    """
    _geopack.recalc_08(datetime.year, datetime.timetuple().tm_yday,
                       datetime.hour, datetime.minute, datetime.second, v_x, v_y, v_z)


def car_to_sph(x, y, z):
    """Convert cartesian to spherical coordinates (in radians)"""
    r, theta, phi, _, _, _ = _geopack.sphcar_08(0., 0., 0., x, y, z, -1)
    return r, theta, phi


def sph_to_car(r, theta, phi):
    """Convert spherical (in radians) to cartesian coordinates"""
    _, _, _, x, y, z = _geopack.sphcar_08(r, theta, phi, 0., 0., 0., 1)
    return x, y, z


def coordinates(xin, yin, zin, coords_in, coords_out):
    """
    Convert cartesian coordinates from one coordinate system to another.
    
    Parameters
    ----------
    xin, yin, zin : float
        The x, y, and z locations in the system described by coords_in.
    coords_in, coords_out : string
        The strings describing the coordinate systems to convert from/to.
    """
    try:
        function = getattr(_geopack, "{}{}_08".format(
            coords_in.lower(), coords_out.lower()))
    except AttributeError:
        try:
            function = getattr(_geopack, "{}{}_08".format(
                coords_out.lower(), coords_in.lower()))
        except AttributeError:
            raise ValueError("Cannot convert {} to {}".format(
                coords_in.upper(), coords_out.upper()))
        else:
            xout, yout, zout, _, _, _ = function(0., 0., 0., xin, yin, zin, -1)
    else:
        _, _, _, xout, yout, zout = function(xin, yin, zin, 0., 0., 0., 1)

    if (xout, yout, zout) == (0., 0., 0.):
        print("\nHave you forgotten to call recalc_08?\n")

    return xout, yout, zout

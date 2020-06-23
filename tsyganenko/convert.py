"""convert: Wrappers for the conversion routines from Geopack."""
import tsyganenko as tsy


def car_to_sph(x, y, z):
    """Convert cartesian to spherical coordinates (in radians)"""
    r, theta, phi, _, _, _ = tsy.geopack.sphcar_08(0., 0., 0., x, y, z, -1)
    return r, theta, phi

def sph_to_car(r, theta, phi):
    """Convert spherical (in radians) to cartesian coordinates"""
    _, _, _, x, y, z = tsy.geopack.sphcar_08(r, theta, phi, 0., 0., 0., 1)
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
        function = getattr(tsy.geopack, "{}{}_08".format(
            coords_in.lower(), coords_out.lower()))
    except AttributeError:
        try:
            function = getattr(tsy.geopack, "{}{}_08".format(
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

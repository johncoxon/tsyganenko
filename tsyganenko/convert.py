"""convert: Wrappers for the conversion routines from Geopack."""
import tsyganenko as tsy


def sph_to_car(r, theta, phi):
    """Convert spherical (in radians) to cartesian coordinates"""
    _, _, _, x, y, z = tsy.geopack.sphcar_08(r, theta, phi, 0., 0., 0., 1)
    return x, y, z

def car_to_sph(x, y, z):
    """Convert cartesian to spherical coordinates (in radians)"""
    r, theta, phi, _, _, _ = tsy.geopack.sphcar_08(0., 0., 0., x, y, z, -1)
    return r, theta, phi

def gsw_to_gse(xgsw, ygsw, zgsw):
    """Convert GSW to GSE coordinates"""
    _, _, _, xgse, ygse, zgse = tsy.geopack.gswgse_08(xgsw, ygsw, zgsw, 0., 0., 0., 1)
    return xgse, ygse, zgse

def gse_to_gsw(xgse, ygse, zgse):
    """Convert GSE to GSW coordinates"""
    xgsw, ygsw, zgsw, _, _, _ = tsy.geopack.gswgse_08(0., 0., 0., xgsw, ygsw, zgsw, -1)
    return xgsw, ygsw, zgsw

def geo_to_mag(xgeo,ygeo,zgeo):
    """Convert GEO to MAG coordinates"""
    _, _, _, xmag, ymag, zmag = tsy.geopack.geomag_08(xgeo, ygeo, zgeo, 0., 0., 0., 1)
    return xmag, ymag, zmag

def mag_to_geo(xmag, ymag, zmag):
    """Convert MAG to GEO coordinates"""
    xgeo, ygeo, zgeo, _, _, _ = tsy.geopack.geomag_08(0., 0., 0., xmag, ymag, zmag, -1)
    return xgeo, ygeo, zgeo

def gei_to_geo(xgei, ygei, zgei):
    """Convert GEI to GEO coordinates"""
    _, _, _, xgeo, ygeo, zgeo = tsy.geopack.geigeo_08(xgei, ygei, zgei, 0., 0., 0., 1)
    return xgeo, ygeo, zgeo

def gei_to_geo(xgeo, ygeo, zgeo):
    """Convert GEI to GEO coordinates"""
    xgei, ygei, zgei, _, _, _ = tsy.geopack.geigeo_08(0., 0., 0., xgeo, ygeo, zgeo, -1)
    return xgei, ygei, zgei

def mag_to_sm(xmag, ymag, zmag):
    """Convert MAG to SM coordinates"""
    _, _, _, xsm, ysm, zsm = tsy.geopack.magsm_08(xmag, ymag, zmag, 0., 0., 0., 1)
    return xsm, ysm, zsm

def sm_to_mag(xsm, ysm, zsm):
    """Convert SM to MAG coordinates"""
    xmag, ymag, zmag, _, _, _ = tsy.geopack.magsm_08(0., 0., 0., xsm, ysm, zsm, -1)
    return xmag, ymag, zmag

def sm_to_gsw(xsm, ysm, zsm):
    """Convert SM to GSW coordinates"""
    _, _, _, xgsw, ygsw, zgsw = tsy.geopack.smgsw_08(xsm, ysm, zsm, 0., 0., 0., 1)
    return xgsw, ygsw, zgsw

def gsw_to_sm(xgsw, ygsw, zgsw):
    """Convert SM to GSW coordinates"""
    xsm, ysm, zsm, _, _, _ = tsy.geopack.smgsw_08(0., 0., 0., xgsw, ygsw, zgsw, -1)
    return xsm, ysm, zsm

def geo_to_gsw(xgeo, ygeo, zgeo):
    """Convert GEO to GSW coordinates"""
    _, _, _, xgsw, ygsw, zgsw = tsy.geopack.geogsw_08(xgeo, ygeo, zgeo, 0., 0., 0., 1)
    return xgsw, ygsw, zgsw

def gsw_to_geo(xgsw, ygsw, zgsw):
    """Convert GSW to GEO coordinates"""
    xgeo, ygeo, zgeo, _, _, _ = tsy.geopack.geogsw_08(0., 0., 0., xgsw, ygsw, zgsw, -1)
    return xgeo, ygeo, zgeo


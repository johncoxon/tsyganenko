"""
tsyganenko : a module to trace magnetic field lines using the Tsyganenko models

This package was initially written by Sebastien de Larquier (Virginia Tech).
In 2020, the package was updated by John Coxon (University of Southampton) to
add support for the latest release of Geopack-2008.for and Python 3 support.

Copyright (C) 2012 VT SuperDARN Lab

.. moduleauthor:: John Coxon

"""
import Geopack
import logging
import numpy as _np


# Declare the same Earth radius as used in Tsyganenko models (km)
RE = 6371.2


class Trace(object):
    """
    Trace magnetic field lines from a given start point.

    Parameters
    ----------
    lat : array_like
        Latitude of the start point (degrees).
    lon : array_like
        Longitude of the start point (degrees).
    rho : array_like
        Distance of the start point from the center of the Earth (km).
    coords : str, optional
        The coordinate system of the start point. Default is "geo".
    datetime : datetime, optional
        The date and time of the start point. If None, defaults to the
        current date and time.
    vswgse : list_like, optional
        Solar wind velocity in GSE coordinates (m/s, m/s, m/s).
    pdyn : float, optional
        Solar wind dynamic pressure (nPa). Default is 2.0.
    dst : int, optional
        Dst index (nT). Default is -5.
    byimf : float, optional
        IMF By (nT). Default is 0.0.
    bzimf : float, optional
        IMF Bz (nT). Default is -5.0.
    l_max : int, optional
        The maximum number of points to trace before stopping. Default is 5000.
    rmax : float, optional
        Upper trace boundary in Earth radii. Default is 60.0.
    rmin : float, optional
        Lower trace boundary in Earth radii. Default is 1.0.
    dsmax : float, optional
        Maximum tracing step size. Default is 0.01.
    err : float, optional
        Tracing step tolerance. Default is 0.000001.

    Attributes
    ----------
    lat[N/S]H : array_like
        Latitude (degrees) of the trace footpoint in the Northern/Southern
        Hemisphere.
    lon[N/S]H : array_like
        Longitude (degrees) of the trace footpoint in the Northern/Southern
        Hemisphere.
    rho[N/S]H : array_like
        Distance of the trace footpoint from the center of the Earth in
        Northern/Southern Hemisphere (km).

    Examples
    --------
        import numpy as np
        import tsyganenko
        # trace a series of points
        lats = np.arange(10, 90, 10)
        lons = np.zeros(len(lats))
        rhos = 6372.*np.ones(len(lats))
        trace = tsyganenko.Trace(lats, lons, rhos)
        # Print the results nicely
        print trace
        # Plot the traced field lines
        ax = trace.plot()
        # Or generate a 3d view of the traced field lines
        ax = trace.plot3d()
    """
    def __init__(self, lat, lon, rho, coords="geo", datetime=None,
                 vswgse=[-400., 0., 0.], pdyn=2., dst=-5., byimf=0., bzimf=-5.,
                 l_max=5000, rmax=60., rmin=1., dsmax=0.01, err=0.000001):
        from datetime import datetime as pydt

        self.lat = lat
        self.lon = lon
        self.rho = rho
        self.coords = coords
        self.vswgse = vswgse
        self.pdyn = pdyn
        self.dst = dst
        self.byimf = byimf
        self.bzimf = bzimf
        # If no datetime is provided, defaults to today
        if datetime is None:
            datetime = pydt.utcnow()
        self.datetime = datetime

        test_valid = self._test_valid()
        if not test_valid:
            self.__del__()

        self.trace()

    def _test_valid(self):
        """Test the validity of inputs to the Trace class and trace method"""
        if len(self.vswgse) != 3:
            raise ValueError("vswgse must have 3 elements")
        if self.coords.lower() != "geo":
            raise ValueError("{}: this coordinate system is not supported")\
                .format(self.coords.lower())
        if _np.isnan(self.pdyn) | _np.isnan(self.dst) | \
                _np.isnan(self.byimf) | _np.isnan(self.bzimf):
            raise ValueError("Input parameters are not numbers")

        # A provision for those who want to batch trace
        try:
            len_lat = len(self.lat)
        except TypeError:
            len_lat = 1
        try:
            len_lon = len(self.lon)
        except TypeError:
            len_lon = 1
        try:
            len_rho = len(self.rho)
        except TypeError:
            len_rho = 1
        try:
            len_dt = len(self.datetime)
        except TypeError:
            len_dt = 1

        # Make sure they're all the same length
        if not (len_lat == len_lon == len_rho == len_dt):
            raise ValueError(
                "lat, lon, rho and datetime must be the same length")

        return True

    def trace(self, lat=None, lon=None, rho=None, coords=None, datetime=None,
              vswgse=None, pdyn=None, dst=None, byimf=None, bzimf=None,
              l_max=5000, rmax=60., rmin=1., dsmax=0.01, err=0.000001):
        """Trace from the start point for both North/Southern Hemispheres"""

        # If new values are passed to this function, store existing values of
        # class attributes in case something is wrong and we need to revert
        # them, and then assign the attributes to the new values.
        if lat:
            _lat = self.lat
            self.lat = lat

        if lon:
            _lon = self.lon
            self.lon = lon

        if rho:
            _rho = self.rho
            self.rho = rho

        if coords:
            _coords = self.coords
            self.coords = coords

        if datetime is not None:
            _datetime = self.datetime
            self.datetime = datetime

        if vswgse:
            _vswgse = self.vswgse
            self.vswgse = vswgse

        if pdyn:
            _pdyn = self.pdyn
            self.pdyn = pdyn

        if dst:
            _dst = self.dst
            self.dst = dst

        if byimf:
            _byimf = self.byimf
            self.byimf = byimf

        if bzimf:
            _bzimf = self.bzimf
            self.bzimf = bzimf

        # Test that everything is in order, if not revert to existing values
        test_valid = self._test_valid()
        if not test_valid:
            if lat:
                self.lat = _lat
            if lon:
                self.lon = _lon
            if rho:
                self.rho = _rho
            if coords:
                self.coords = _coords
            if datetime is not None:
                self.datetime = _datetime
            if vswgse:
                self.vswgse = _vswgse
            if pdyn:
                self.pdyn = pdyn
            if dst:
                self.dst = dst
            if byimf:
                self.byimf = byimf
            if bzimf:
                self.bzimf = bzimf

        # Now that we have good attributes, assign the parameters to those.
        lat = self.lat
        lon = self.lon
        rho = self.rho
        coords = self.coords
        datetime = self.datetime
        vswgse = self.vswgse
        pdyn = self.pdyn
        dst = self.dst
        byimf = self.byimf
        bzimf = self.bzimf

        # Initialize trace arrays
        self.l_cnt = _np.zeros_like(lat)
        self.xTrace = _np.zeros((len(lat), 2*l_max))
        self.yTrace = self.xTrace.copy()
        self.zTrace = self.xTrace.copy()
        self.gsw = _np.zeros((len(lat), 3))
        self.latNH = self.l_cnt.copy()
        self.lonNH = self.l_cnt.copy()
        self.rhoNH = self.l_cnt.copy()
        self.latSH = self.l_cnt.copy()
        self.lonSH = self.l_cnt.copy()
        self.rhoSH = self.l_cnt.copy()

        # And now iterate through the desired points
        for ip in range(len(lat)):
            # This has to be called first
            Geopack.recalc_08(datetime[ip].year,
                              datetime[ip].timetuple().tm_yday,
                              datetime[ip].hour, datetime[ip].minute,
                              datetime[ip].second, *vswgse)

            # Convert spherical to cartesian
            r, theta, phi, xgeo, ygeo, zgeo = Geopack.sphcar_08(
                rho[ip]/RE, _np.radians(90.-lat[ip]), _np.radians(lon[ip]),
                0., 0., 0., 1)

            # Convert to GSW.
            if coords.lower() == "geo":
                _, _, _, xgsw, ygsw, zgsw = Geopack.geogsw_08(xgeo, ygeo, zgeo,
                                                              0., 0., 0., 1)

            self.gsw[ip, 0] = xgsw
            self.gsw[ip, 1] = ygsw
            self.gsw[ip, 2] = zgsw

            # Trace field line
            inmod = "IGRF_GSW_08"
            exmod = "T96_01"
            parmod = [pdyn, dst, byimf, bzimf, 0., 0., 0., 0., 0., 0.]

            # Towards SH and then towards NH
            for mapto in [-1, 1]:
                xfgsw, yfgsw, zfgsw, xarr, yarr, zarr, l_cnt \
                    = Geopack.trace_08(xgsw, ygsw, zgsw, mapto, dsmax, err,
                                       rmax, rmin, 0, parmod, exmod, inmod,
                                       l_max)

                # Convert back to spherical geographic coords
                xfgeo, yfgeo, zfgeo, _, _, _ = Geopack.geogsw_08(0., 0., 0.,
                                                                 xfgsw, yfgsw,
                                                                 zfgsw, -1)
                rhof, colatf, lonf, _, _, _ = Geopack.sphcar_08(0., 0., 0.,
                                                                xfgeo, yfgeo,
                                                                zfgeo, -1)

                # Get coordinates of traced point
                if mapto == 1:
                    self.latSH[ip] = 90. - _np.degrees(colatf)
                    self.lonSH[ip] = _np.degrees(lonf)
                    self.rhoSH[ip] = rhof*RE
                elif mapto == -1:
                    self.latNH[ip] = 90. - _np.degrees(colatf)
                    self.lonNH[ip] = _np.degrees(lonf)
                    self.rhoNH[ip] = rhof*RE

                # Store trace
                if mapto == -1:
                    self.xTrace[ip, 0:l_cnt] = xarr[l_cnt-1::-1]
                    self.yTrace[ip, 0:l_cnt] = yarr[l_cnt-1::-1]
                    self.zTrace[ip, 0:l_cnt] = zarr[l_cnt-1::-1]
                elif mapto == 1:
                    l_0 = int(_np.round(self.l_cnt[ip]))
                    self.xTrace[ip, l_0:l_0+l_cnt] = xarr[0:l_cnt]
                    self.yTrace[ip, l_0:l_0+l_cnt] = yarr[0:l_cnt]
                    self.zTrace[ip, l_0:l_0+l_cnt] = zarr[0:l_cnt]
                self.l_cnt[ip] += l_cnt

        # Resize trace output to more minimum possible length
        max_index = int(_np.round(self.l_cnt.max()))
        self.xTrace = self.xTrace[:, 0:max_index]
        self.yTrace = self.yTrace[:, 0:max_index]
        self.zTrace = self.zTrace[:, 0:max_index]

    def __str__(self):
        """Print object information in a nice way

        Written by Sebastien 2012-10
        """
        # Declare print format
        outstr = """
vswgse=[{:6.0f}, {:6.0f}, {:6.0f}]    [m/s]
pdyn={:3.0f}                        [nPa]
dst={:3.0f}                         [nT]
byimf={:3.0f}                       [nT]
bzimf={:3.0f}                       [nT]
                    """.format(self.vswgse[0],
                               self.vswgse[1],
                               self.vswgse[2],
                               self.pdyn,
                               self.dst,
                               self.byimf,
                               self.bzimf)
        outstr += "\nCoords: {}\n".format(self.coords)
        outstr += ("(latitude [degrees], longitude [degrees], "
                   "distance from center of the Earth [km])\n")

        # Print stuff
        for ip in range(len(self.lat)):
            outstr += """
({:6.3f}, {:6.3f}, {:6.3f}) @ {}
    --> NH({:6.3f}, {:6.3f}, {:6.3f})
    --> SH({:6.3f}, {:6.3f}, {:6.3f})""".format(
                self.lat[ip], self.lon[ip], self.rho[ip],
                self.datetime[ip].strftime("%H:%M UT (%d-%b-%y)"),
                self.latNH[ip], self.lonNH[ip], self.rhoNH[ip],
                self.latSH[ip], self.lonSH[ip], self.rhoSH[ip])

        return outstr

    def plot(self, ax=None, proj="xz", onlyPts=None, showPts=False,
             showEarth=True,  **kwargs):
        """Generate a 2D plot of the trace projected onto a given plane
        Graphic keywords apply to the plot method for the field lines

        Parameters
        ----------
        ax : matplotlib axes object, optional
            the object on which to plot
        proj : str, optional
            the projection plane in GSW coordinates
        onlyPts : list, optional
            if the trace countains multiple point,
            only show the specified indices (list)
        showEarth : bool, optional
            Toggle Earth disk visibility on/off
        showPts : bool, optional
            Toggle start points visibility on/off
        **kwargs :
            see matplotlib.axes.Axes.plot

        Returns
        -------
        ax : matplotlib axes object
        """
        from matplotlib import pyplot as plt
        from matplotlib.patches import Circle

        if (len(proj) != 2) or (proj[0] not in ["x", "y", "z"])\
                or (proj[1] not in ["x", "y", "z"]) or (proj[0] == proj[1]):
            raise ValueError("Invalid projection plane.")

        if ax is None:
            fig = plt.gcf()
            ax = fig.gca()
            ax.set_aspect("equal")

        # First plot a nice disk for the Earth
        if showEarth:
            circ = Circle(xy=(0, 0), radius=1, facecolor="0.8", edgecolor="k",
                          alpha=.5, zorder=0)
            ax.add_patch(circ)

        # Select indices to show
        if onlyPts is None:
            inds = _np.arange(len(self.lat))
        elif not isinstance(onlyPts, list):
            inds = [onlyPts]
        else:
            inds = onlyPts

        # Then plot the traced field line
        for ip, _ in enumerate(self.lat):
            # Select projection plane
            if proj[0] == "x":
                xx = self.xTrace[ip, :]
                xpt = self.gsw[ip, 0]
                ax.set_xlabel(r"$X_{GSW}$")
                xdir = [1, 0, 0]
            elif proj[0] == "y":
                xx = self.yTrace[ip, :]
                xpt = self.gsw[ip, 1]
                ax.set_xlabel(r"$Y_{GSW}$")
                xdir = [0, 1, 0]
            elif proj[0] == "z":
                xx = self.zTrace[ip, :]
                xpt = self.gsw[ip, 2]
                ax.set_xlabel(r"$Z_{GSW}$")
                xdir = [0, 0, 1]
            if proj[1] == "x":
                yy = self.xTrace[ip, :]
                ypt = self.gsw[ip, 0]
                ax.set_ylabel(r"$X_{GSW}$")
                ydir = [1, 0, 0]
            elif proj[1] == "y":
                yy = self.yTrace[ip, :]
                ypt = self.gsw[ip, 1]
                ax.set_ylabel(r"$Y_{GSW}$")
                ydir = [0, 1, 0]
            elif proj[1] == "z":
                yy = self.zTrace[ip, :]
                ypt = self.gsw[ip, 2]
                ax.set_ylabel(r"$Z_{GSW}$")
                ydir = [0, 0, 1]

            sign = 1 if -1 not in _np.cross(xdir, ydir) else -1
            if "x" not in proj:
                zz = sign*self.gsw[ip, 0]
                indMask = sign*self.xTrace[ip, :] < 0
            if "y" not in proj:
                zz = sign*self.gsw[ip, 1]
                indMask = sign*self.yTrace[ip, :] < 0
            if "z" not in proj:
                zz = sign*self.gsw[ip, 2]
                indMask = sign*self.zTrace[ip, :] < 0

            # Plot
            ax.plot(_np.ma.masked_array(xx, mask=~indMask),
                    _np.ma.masked_array(yy, mask=~indMask),
                    zorder=-1, **kwargs)
            ax.plot(_np.ma.masked_array(xx, mask=indMask),
                    _np.ma.masked_array(yy, mask=indMask),
                    zorder=1, **kwargs)
            if showPts:
                ax.scatter(xpt, ypt, c="k", s=40, zorder=zz)

    def plot3d(self, onlyPts=None, showEarth=True, showPts=False, disp=True,
               xyzlim=None, zorder=1, linewidth=2, color="b", **kwargs):
        """Generate a 3D plot of the trace
        Graphic keywords apply to the plot3d method for the field lines

        Parameters
        ----------
        onlyPts : Optional[ ]
            if the trace contains multiple points,
            only show the specified indices (list)
        showEarth : Optional[bool]
            Toggle Earth sphere visibility on/off
        showPts : Optional[bool]
            Toggle start points visibility on/off
        disp : Optional[bool]
            invoke plt.show()
        xyzlim : Optional[ ]
            3D axis limits
        zorder : Optional[int]
            3D layers ordering
        linewidth : Optional[int]
            field line width
        color : Optional[char]
            field line color
        **kwargs :
            see mpl_toolkits.mplot3d.axes3d.Axes3D.plot3D

        Returns
        -------
        ax :  matplotlib axes
            axes object

        Written by Sebastien 2012-10

        """
        from mpl_toolkits.mplot3d import proj3d
        from matplotlib import pyplot as plt

        fig = plt.gcf()
        ax = fig.gca(projection="3d")

        # First plot a nice sphere for the Earth
        if showEarth:
            u = _np.linspace(0, 2 * _np.pi, 179)
            v = _np.linspace(0, _np.pi, 179)
            tx = _np.outer(_np.cos(u), _np.sin(v))
            ty = _np.outer(_np.sin(u), _np.sin(v))
            tz = _np.outer(_np.ones(_np.size(u)), _np.cos(v))
            ax.plot_surface(tx, ty, tz, rstride=10, cstride=10, color="grey",
                            alpha=.5, zorder=0, linewidth=0.5)

        # Select indices to show
        if onlyPts is None:
            inds = _np.arange(len(self.lat))
        elif not isinstance(onlyPts, list):
            inds = [onlyPts]
        else:
            inds = onlyPts

        # Then plot the traced field line
        for ip in inds:
            plot_index = int(_np.round(self.l_cnt[ip]))
            ax.plot3D(self.xTrace[ip, 0:plot_index],
                      self.yTrace[ip, 0:plot_index],
                      self.zTrace[ip, 0:plot_index], zorder=zorder,
                      linewidth=linewidth, color=color, **kwargs)
            if showPts:
                ax.scatter3D(*self.gsw[ip, :], c="k")

        # Set plot limits
        if not xyzlim:
            xyzlim = _np.max([_np.max(ax.get_xlim3d()),
                             _np.max(ax.get_ylim3d()),
                             _np.max(ax.get_zlim3d())])
        ax.set_xlim3d([-xyzlim, xyzlim])
        ax.set_ylim3d([-xyzlim, xyzlim])
        ax.set_zlim3d([-xyzlim, xyzlim])

        if disp:
            plt.show()

        return ax

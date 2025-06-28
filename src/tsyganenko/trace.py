"""trace: Provides a class to easily trace field lines from start points"""
import datetime as _dt
import geopack_tsyganenko as _geopack
import numpy as _np
from matplotlib import pyplot as _plt
from matplotlib.patches import Circle as _Circle

# Declare the same Earth radius as used in Tsyganenko models (km)
earth_radius = 6371.2


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
    datetime : array_like
        The date and time of the start point. If None, defaults to the current date and time.
    vsw_gse : list_like, optional
        Solar wind velocity in GSE coordinates (m/s, m/s, m/s).
    pdyn : float, optional
        Solar wind dynamic pressure (nPa). Default is 2.0.
    dst : int, optional
        Dst index (nT). Default is -5.
    by_imf : float, optional
        IMF By (nT). Default is 0.0.
    bz_imf : float, optional
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
    lat, lon, rho, coords, time, vsw_gse, pdyn, dst, by_imf, bz_imf
    lat_[n/s] : array_like
        Latitude (degrees) of the trace footpoint in the Northern/Southern Hemisphere.
    lon_[n/s] : array_like
        Longitude (degrees) of the trace footpoint in the Northern/Southern Hemisphere.
    rho_[n/s] : array_like
        Distance of the trace footpoint from the center of the Earth in Northern/Southern Hemisphere (km).

    Example
    --------
        import numpy as np
        import tsyganenko as tsy
        # Trace a series of points
        lats = np.arange(10, 90, 10)
        lons = 0.
        rhos = tsy.earth_radius
        trace = tsy.Trace(lats, lons, rhos)
        # Print the results nicely
        print(trace)
        # Plot the traced field lines
        trace.plot()
        # Or generate a 3d view of the traced field lines
        trace.plot3d()
    """
    def __init__(self, lat, lon, rho, coords="geo", datetime=None,
                 vsw_gse=(-400., 0., 0.), pdyn=2., dst=-5., by_imf=0.,
                 bz_imf=-5., l_max=5000, rmax=60., rmin=1., dsmax=0.01,
                 err=0.000001):
        self.lat = lat
        self.lon = lon
        self.rho = rho
        self.coords = coords
        self.vsw_gse = vsw_gse
        self.pdyn = pdyn
        self.dst = dst
        self.by_imf = by_imf
        self.bz_imf = bz_imf

        # If no datetime is provided, defaults to today
        if datetime is None:
            self.time = _np.array([_dt.datetime.now()])
        elif isinstance(datetime, _dt.datetime):
            self.time = _np.array([datetime])
        else:
            self.time = datetime

        valid_inputs = self._test_valid()
        if not valid_inputs:
            raise ValueError("The inputs were not valid.")

        # Initialize trace arrays
        self.gsw = _np.zeros((len(self.lat), 3))
        self.lat_n = _np.zeros_like(self.lat)
        self.lon_n = _np.zeros_like(self.lat)
        self.rho_n = _np.zeros_like(self.lat)
        self.lat_s = _np.zeros_like(self.lat)
        self.lon_s = _np.zeros_like(self.lat)
        self.rho_s = _np.zeros_like(self.lat)
        self.trace_gsw = []

        self.trace(l_max=l_max, rmax=rmax, rmin=rmin, dsmax=dsmax, err=err)

    def __str__(self):
        """Print salient inputs alongside trace results for each trace"""
        outstr = """
vsw_gse=[{:6.0f},{:6.0f},{:6.0f}]   [m/s]
pdyn=    {:6.1f}                  [nPa]
dst=     {:6.1f}                  [nT]
by_imf=  {:6.1f}                  [nT]
bz_imf=  {:6.1f}                  [nT]

Coords: {}
(latitude [deg], longitude [deg], distance from center of the Earth [km])
""".format(self.vsw_gse[0], self.vsw_gse[1], self.vsw_gse[2], self.pdyn,
           self.dst, self.by_imf, self.bz_imf, self.coords)

        # Print the trace for each set of input coordinates.
        for ip in range(len(self.lat)):
            outstr += """
({:6.3f}, {:6.3f}, {:6.3f}) @ {}
    --> NH({:6.3f}, {:6.3f}, {:6.3f})
    --> SH({:6.3f}, {:6.3f}, {:6.3f})""".format(
                self.lat[ip], self.lon[ip], self.rho[ip],
                self.time[ip].strftime("%H:%M UT (%d-%b-%y)"),
                self.lat_n[ip], self.lon_n[ip], self.rho_n[ip],
                self.lat_s[ip], self.lon_s[ip], self.rho_s[ip])

        return outstr

    def trace(self, l_max=5000, rmax=60., rmin=1., dsmax=0.01, err=0.000001):
        """Trace from the start point for both North/Southern Hemispheres"""
        # Iterate through the desired points
        for ip in _np.arange(len(self.lat)):
            # This has to be called first
            _geopack.recalc_08(self.time[ip].year,
                               self.time[ip].timetuple().tm_yday,
                               self.time[ip].hour,
                               self.time[ip].minute,
                               self.time[ip].second, *self.vsw_gse)

            # Convert spherical to cartesian
            r, theta, phi, x, y, z = _geopack.sphcar_08(
                self.rho[ip]/earth_radius, _np.radians(90. - self.lat[ip]),
                _np.radians(self.lon[ip]), 0., 0., 0., 1)

            # Convert to GSW.
            if self.coords.lower() == "geo":
                _, _, _, xgsw, ygsw, zgsw = _geopack.geogsw_08(
                    x, y, z, 0., 0., 0., 1)

            self.gsw[ip, 0] = xgsw
            self.gsw[ip, 1] = ygsw
            self.gsw[ip, 2] = zgsw

            # Trace field line
            inmod = "IGRF_GSW_08"
            exmod = "T96_01"
            parmod = [self.pdyn, self.dst, self.by_imf, self.bz_imf, 0., 0., 0., 0., 0., 0.]

            # Towards NH and then towards SH
            x_trace_n, y_trace_n, z_trace_n = self._trace_to_hemisphere("north", ip, xgsw, ygsw, zgsw, dsmax, err, rmax,
                                                                        rmin, parmod, exmod, inmod, l_max)
            x_trace_s, y_trace_s, z_trace_s = self._trace_to_hemisphere("south", ip, xgsw, ygsw, zgsw, dsmax, err, rmax,
                                                                        rmin, parmod, exmod, inmod, l_max)

            # Combine the NH and SH traces into x/y/z arrays.
            x_trace = _np.concatenate((x_trace_n, x_trace_s))
            y_trace = _np.concatenate((y_trace_n, y_trace_s))
            z_trace = _np.concatenate((z_trace_n, z_trace_s))

            # Combine the combined arrays into an array of shape (:,3)
            # and add it to the list of traces.
            self.trace_gsw.append(_np.array((x_trace, y_trace, z_trace)).T)

    def _test_valid(self):
        """Test the validity of inputs to the Trace class and trace method"""
        if len(self.vsw_gse) != 3:
            raise ValueError("vsw_gse must have 3 elements")
        if self.coords.lower() != "geo":
            raise ValueError("{}: this coordinate system is not supported".format(self.coords.lower()))
        if _np.isnan(self.pdyn) | _np.isnan(self.dst) | _np.isnan(self.by_imf) | _np.isnan(self.bz_imf):
            raise ValueError("Input parameters are not numbers")

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
        len_dt = len(self.time)

        # Make the inputs into floating point arrays. Where an input is passed
        # once, make it into an array of that input (this allows passing e.g.
        # many latitudes for one longitude and rho).
        lens = _np.array((len_lat, len_lon, len_rho, len_dt))
        if len_lat == 1:
            self.lat = _np.ones(_np.max(lens), dtype=float) * self.lat
            len_lat = len(self.lat)
        else:
            self.lat = _np.array(self.lat, dtype=float)
        if len_lon == 1:
            self.lon = _np.ones(_np.max(lens), dtype=float) * self.lon
            len_lon = len(self.lon)
        else:
            self.lon = _np.array(self.lon, dtype=float)
        if len_rho == 1:
            self.rho = _np.ones(_np.max(lens), dtype=float) * self.rho
            len_rho = len(self.rho)
        else:
            self.rho = _np.array(self.rho, dtype=float)
        if len_dt == 1:
            self.time = _np.array([self.time[0] for _ in self.lat])
            len_dt = len(self.time)
        else:
            self.time = _np.array(self.time)

        # Make sure they're all the same length
        if not (len_lat == len_lon == len_rho == len_dt):
            raise ValueError(
                "lat, lon, rho and datetime must be the same length")

        return True

    def _trace_to_hemisphere(self, hemisphere, ip, xgsw, ygsw, zgsw, dsmax, err, rmax, rmin, parmod, exmod, inmod,
                             l_max):
        if hemisphere == "north":
            mapto = -1
        else:
            mapto = 1

        xfgsw, yfgsw, zfgsw, xarr, yarr, zarr, l_cnt = _geopack.trace_08(
            xgsw, ygsw, zgsw, mapto, dsmax, err, rmax, rmin, 0, parmod, exmod, inmod, l_max)

        # Convert back to spherical geographic coords
        xfgeo, yfgeo, zfgeo, _, _, _ = _geopack.geogsw_08(0., 0., 0., xfgsw, yfgsw, zfgsw, -1)
        rhof, colatf, lonf, _, _, _ = _geopack.sphcar_08(0., 0., 0., xfgeo, yfgeo, zfgeo, -1)

        if hemisphere == "north":
            self.lat_n[ip] = 90. - _np.degrees(colatf)
            self.lon_n[ip] = _np.degrees(lonf)
            self.rho_n[ip] = rhof * earth_radius

            x_trace = xarr[l_cnt - 1::-1]
            y_trace = yarr[l_cnt - 1::-1]
            z_trace = zarr[l_cnt - 1::-1]
        else:
            self.lat_s[ip] = 90. - _np.degrees(colatf)
            self.lon_s[ip] = _np.degrees(lonf)
            self.rho_s[ip] = rhof * earth_radius

            x_trace = xarr[0:l_cnt]
            y_trace = yarr[0:l_cnt]
            z_trace = zarr[0:l_cnt]

        return x_trace, y_trace, z_trace

    def plot(self, ax=None, proj="xz", only_pts=None, show_pts=False,
             show_earth=True,  **kwargs):
        """Generate a 2D plot of the trace projected onto a given plane
        Graphic keywords apply to the plot method for the field lines

        Parameters
        ----------
        ax : matplotlib axes object, optional
            The object on which to plot.
        proj : str, optional
            The GSW projection plane.
        only_pts : list_like, optional
            If the trace contains multiple points, only show those specified.
        show_earth : bool, optional
            Toggle Earth disk visibility.
        show_pts : bool, optional
            Toggle start points visibility.
        **kwargs :
            see matplotlib.axes.Axes.plot

        Returns
        -------
        ax : matplotlib axes object
        """
        if ((len(proj) != 2) or (proj[0] not in ["x", "y", "z"])
                or (proj[1] not in ["x", "y", "z"]) or (proj[0] == proj[1])):
            raise ValueError("Invalid projection plane.")

        if ax is None:
            fig = _plt.gcf()
            ax = fig.gca()
            ax.set_aspect("equal")

        # First plot a nice disk for the Earth
        if show_earth:
            earth = _Circle(xy=(0, 0), radius=1, facecolor="0.8", edgecolor="k",
                            alpha=.5, zorder=4)
            ax.add_patch(earth)

        # Select indices to show
        if only_pts is None:
            inds = _np.arange(len(self.lat))
        elif not isinstance(only_pts, list):
            inds = [only_pts]
        else:
            inds = only_pts

        # Then plot the traced field line
        for cnt, ip in enumerate(inds):
            # Select projection plane
            if proj[0] == "x":
                xx = self.trace_gsw[ip][:, 0]
                xpt = self.gsw[ip, 0]
                ax.set_xlabel(r"$X_{GSW}$")
            elif proj[0] == "y":
                xx = self.trace_gsw[ip][:, 1]
                xpt = self.gsw[ip, 1]
                ax.set_xlabel(r"$Y_{GSW}$")
            else:
                xx = self.trace_gsw[ip][:, 2]
                xpt = self.gsw[ip, 2]
                ax.set_xlabel(r"$Z_{GSW}$")

            if proj[1] == "x":
                yy = self.trace_gsw[ip][:, 0]
                ypt = self.gsw[ip, 0]
                ax.set_ylabel(r"$X_{GSW}$")
            elif proj[1] == "y":
                yy = self.trace_gsw[ip][:, 1]
                ypt = self.gsw[ip, 1]
                ax.set_ylabel(r"$Y_{GSW}$")
            else:
                yy = self.trace_gsw[ip][:, 2]
                ypt = self.gsw[ip, 2]
                ax.set_ylabel(r"$Z_{GSW}$")

            ax.plot(xx, yy, **kwargs)
            if show_pts:
                ax.scatter(xpt, ypt, c="k", zorder=4)

        # Set x limits to have the Sun to the left as per convention
        ax.set_xlim(ax.get_xlim()[::-1])

        return ax

    def plot3d(self, only_pts=None, show_earth=True, show_pts=False,
               xyzlim=None, **kwargs):
        """Generate a 3D plot of the trace
        Graphic keywords apply to the plot3d method for the field lines

        Parameters
        ----------
        only_pts : list_like, optional
            If the trace contains multiple points, only show those specified.
        show_earth : bool, optional
            Toggle Earth sphere visibility. Default is True.
        show_pts : bool, optional
            Toggle start points visibility. Default is False.
        xyzlim : tuple_like, optional
            3D axis limits.
        **kwargs :
            see mpl_toolkits.mplot3d.axes3d.Axes3D.plot3D

        Returns
        -------
        ax :  matplotlib axes
            axes object
        """
        fig = _plt.gcf()
        ax = fig.gca(projection="3d")

        # First plot a nice sphere for the Earth
        if show_earth:
            u = _np.linspace(0, 2 * _np.pi, 179)
            v = _np.linspace(0, _np.pi, 179)
            tx = _np.outer(_np.cos(u), _np.sin(v))
            ty = _np.outer(_np.sin(u), _np.sin(v))
            tz = _np.outer(_np.ones(_np.size(u)), _np.cos(v))
            ax.plot_surface(tx, ty, tz, rstride=10, cstride=10, color="grey",
                            alpha=.5, zorder=0, linewidth=0.5)

        # Select indices to show
        if only_pts is None:
            inds = _np.arange(len(self.lat))
        elif not isinstance(only_pts, list):
            inds = [only_pts]
        else:
            inds = only_pts

        # Then plot the traced field line
        for ip in inds:
            ax.plot3D(self.trace_gsw[ip][:, 0],
                      self.trace_gsw[ip][:, 1],
                      self.trace_gsw[ip][:, 2], **kwargs)
            if show_pts:
                ax.scatter3D(*self.gsw[ip, :], c="k")

        # Set plot limits if none are set. This is hugely convoluted
        # because you can't do ax.set_aspect("equal") for 3D plots
        if xyzlim is None:
            self._equal_aspect_3d(ax)
        else:
            ax.set(xlim3d=xyzlim, ylim3d=xyzlim, zlim3d=xyzlim)
        ax.set(xlabel=r"$X_{GSW}$", ylabel=r"$Y_{GSW}$", zlabel=r"$Z_{GSW}$")

        return ax

    @staticmethod
    def _equal_aspect_3d(ax):
        """Set limits on a 3D axis to get equal aspect ratio on each side"""
        lims = _np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
        diffs = _np.array([-1 * _np.subtract(*i) for i in lims])
        half_diff_max = _np.max(diffs) / 2.

        x_mid = lims[0][0] + (diffs[0] / 2.)
        y_mid = lims[1][0] + (diffs[1] / 2.)
        z_mid = lims[2][0] + (diffs[2] / 2.)

        if _np.argmax(diffs) == 0:
            ax.set_ylim3d(y_mid - half_diff_max, y_mid + half_diff_max)
            ax.set_zlim3d(z_mid - half_diff_max, z_mid + half_diff_max)
        elif _np.argmax(diffs) == 1:
            ax.set_xlim3d(x_mid - half_diff_max, x_mid + half_diff_max)
            ax.set_zlim3d(z_mid - half_diff_max, z_mid + half_diff_max)
        elif _np.argmax(diffs) == 2:
            ax.set_xlim3d(x_mid - half_diff_max, x_mid + half_diff_max)
            ax.set_ylim3d(y_mid - half_diff_max, y_mid + half_diff_max)

        return True

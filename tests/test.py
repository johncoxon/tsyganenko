import numpy as np
import pytest
from datetime import datetime
from pandas import read_csv
from pathlib import Path
from tsyganenko import Trace

year_ranges = ("1965-2015",)
lats = (-90, -60, -30, 0, 30, 60, 90)
lons = (0, 90, 180, 270)
rhos = (30000, 60000, 90000, 120000)
variables = ("lat_n", "lon_n", "rho_n", "lat_s", "lon_s", "rho_s")


@pytest.fixture
def benchmarks():
    benchmarks = {}

    for y in year_ranges:
        benchmarks[y] = {}

        for lat in lats:
            benchmarks[y][lat] = {}

            for lon in lons:
                benchmarks[y][lat][lon] = {}

                for rho in rhos:
                    csv_file = Path(__file__).parent / y / f"{lat}_{lon}_{rho}_{y}.csv"
                    benchmarks[y][lat][lon][rho] = read_csv(csv_file)

    return benchmarks


@pytest.fixture
def trace(year_range, lat, lon, rho):
    start, end = year_range.split("-")
    start = int(start)
    end = int(end)

    years = np.arange(start, end)
    dates = np.array([datetime(y, 1, 1) for y in years])
    ones_like_dates = np.ones_like(dates, dtype=float)

    return Trace(ones_like_dates * lat, ones_like_dates * lon, ones_like_dates * rho,
                 coords='geo', datetime=dates)


@pytest.mark.parametrize("variable", variables)
@pytest.mark.parametrize("rho", rhos)
@pytest.mark.parametrize("lon", lons)
@pytest.mark.parametrize("lat", lats)
@pytest.mark.parametrize("year_range", year_ranges)
def test(year_range, lat, lon, rho, variable, benchmarks, trace):
    """
    Run tests.

    Parameters
    ----------
    year_range : tuple
    lat : float
    lon : float
    rho : float
    variable : basestring
    benchmarks : dict
    trace : Trace
    """
    assert getattr(trace, variable) == pytest.approx(benchmarks[year_range][lat][lon][rho][variable],
                                                     rel=0.5, abs=10.0)

# Create a csv file which can be used for unit testing the BaseModel class.
import numpy as np
from datetime import datetime
from pandas import DataFrame
from pathlib import Path
from tsyganenko import Trace

lats = [-90, -60, -30, 0, 30, 60, 90]
lons = [0, 90, 180, 270]
rhos = [30000, 60000, 90000, 120000]

model_outputs = {}

start = 1965
end = 2015
year_range = np.arange(start, end)
dates = np.array([datetime(y, 1, 1) for y in year_range])
ones_like_dates = np.ones_like(dates, dtype=float)

indices = []

lat_list = []
lon_list = []
rho_list = []
dateses = []
lats_n = []
lons_n = []
rhos_n = []
lats_s = []
lons_s = []
rhos_s = []

for lat in lats:
    for lon in lons:
        for rho in rhos:
            trace = Trace(lat=ones_like_dates * lat, lon=ones_like_dates * lon, rho=ones_like_dates * rho,
                          coords='geo', datetime=dates)

            csv_dict = {"year": year_range,
                        "lat_n": trace.lat_n,
                        "lon_n": trace.lon_n,
                        "rho_n": trace.rho_n,
                        "lat_s": trace.lat_s,
                        "lon_s": trace.lon_s,
                        "rho_s": trace.rho_s
                        }

            dataframe = DataFrame(csv_dict)
            dataframe.to_csv(Path(__file__).parent / f"{start}-{end}" / f"{lat}_{lon}_{rho}_{start}-{end}.csv")

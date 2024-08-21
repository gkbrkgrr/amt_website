import os
from netCDF4 import Dataset
import pandas as pd
import xarray as xr
from ituamt import AMTPlotter, colormaps
from siphon.catalog import TDSCatalog
from datetime import datetime, timedelta
from xarray.backends import NetCDF4DataStore
import numpy as np
from cartopy import crs 
from matplotlib.colors import ListedColormap
from metpy.units import units
from metpy import calc
from wrf import smooth2d

def delete_png_files(directory):
    if os.path.exists(directory):
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".png"):
                    file_path = os.path.join(root, file)
                    os.remove(file_path)

def create_folders():
    folder_names = ["tempadv850", "temphgt850", "rh700", "rvo500", 
                    "avo_adv500", "vertical_v500", "rvo300", "jet300", 
                    "combined300", "pwat", "t2m", "wind10", 
                    "t2_depression", "thickness"]

    cwd = os.getcwd()
    if not os.path.exists("gfs_output_maps"): os.mkdir("gfs_output_maps")
    figures_path = os.path.join(cwd, "gfs_output_maps")

    for folder_name in folder_names:
        folder_path = os.path.join(figures_path, folder_name)
        if not os.path.exists(folder_path): os.mkdir(folder_path)

def download_gfs_plevs(levels = None, time_delta = 1,
                catalog: TDSCatalog = TDSCatalog('http://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_0p25deg/catalog.xml?dataset=grib/NCEP/GFS/Global_0p25deg/Best'),
                start_date = datetime.utcnow()) -> xr.DataArray:
    ds = list(catalog.datasets.values())[0]
    ncss = ds.subset()

    data_combined1 = []
    data_combined2 = []

    for level in levels:
        query1 = ncss.query()
        query1.lonlat_box(north=75, south=25, east=360, west=330).time_range(start_date, start_date + timedelta(days=time_delta))
        query1.accept('netcdf4')
        query1.variables("Geopotential_height_isobaric", "Temperature_isobaric", "Pressure_reduced_to_MSL_msl", 
                         "u-component_of_wind_isobaric", "v-component_of_wind_isobaric", "Relative_humidity_isobaric", "Absolute_vorticity_isobaric",
                         "Vertical_velocity_pressure_isobaric", "Precipitable_water_entire_atmosphere_single_layer")
        query1.vertical_level(level)

        data1 = ncss.get_data(query1)
        data1 = xr.open_dataset(NetCDF4DataStore(data1))
        data_combined1.append(data1)

        query2 = ncss.query()
        query2.lonlat_box(north=75, south=25, east=80, west=0).time_range(start_date, start_date + timedelta(days=time_delta))
        query2.accept('netcdf4')
        query2.variables("Geopotential_height_isobaric", "Temperature_isobaric", "Pressure_reduced_to_MSL_msl", 
                         "u-component_of_wind_isobaric", "v-component_of_wind_isobaric", "Relative_humidity_isobaric", "Absolute_vorticity_isobaric",
                         "Vertical_velocity_pressure_isobaric", "Precipitable_water_entire_atmosphere_single_layer")
        query2.vertical_level(level)

        data2 = ncss.get_data(query2)
        data2 = xr.open_dataset(NetCDF4DataStore(data2))
        data_combined2.append(data2)
    
    combined_data1 = xr.concat(data_combined1, dim='isobaric')
    combined_data2 = xr.concat(data_combined2, dim='isobaric')
    
    merged = xr.merge([combined_data1, combined_data2])

    return merged

def download_gfs_sfc(catalog: TDSCatalog = TDSCatalog('http://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_0p25deg/catalog.xml?dataset=grib/NCEP/GFS/Global_0p25deg/Best'),
                     start_date = datetime.utcnow(), time_delta = 1) -> xr.DataArray:
    
    variables = {"names": ["Temperature_height_above_ground", "Dewpoint_temperature_height_above_ground", "u-component_of_wind_height_above_ground", "v-component_of_wind_height_above_ground"],
                 "levels": [2., 2., 10., 10.]}
    
    ds = list(catalog.datasets.values())[0]
    ncss = ds.subset()

    data_combined = []

    for i in range(len(variables["names"])):
        query1 = ncss.query()
        query1.lonlat_box(north=75, south=25, east=360, west=330).time_range(start_date, start_date + timedelta(days=time_delta))
        query1.accept('netcdf4')
        query1.variables(variables["names"][i])
        query1.vertical_level(variables["levels"][i])

        data1 = ncss.get_data(query1)
        data1 = xr.open_dataset(NetCDF4DataStore(data1))
        data_combined.append(data1)

        query2 = ncss.query()
        query2.lonlat_box(north=75, south=25, east=80, west=0).time_range(start_date, start_date + timedelta(days=time_delta))
        query2.accept('netcdf4')
        query2.variables(variables["names"][i])
        query2.vertical_level(variables["levels"][i])

        data2 = ncss.get_data(query2)
        data2 = xr.open_dataset(NetCDF4DataStore(data2))
        data_combined.append(data2)

    merged = xr.merge(data_combined)

    return merged

class GFSParser():
    def __init__(self, ds_plevs: Dataset, ds_sfc: Dataset) -> None:
        try: 
            self.times = pd.to_datetime(ds_plevs["time"])
        except:
            self.times = pd.to_datetime(ds_plevs["time1"])
        self.init_date = self.times[0].strftime("%d-%m-%Y %H")
        self.lats = ds_plevs["latitude"]
        self.lons = ds_plevs["longitude"]
       
        self.pressure = ds_plevs["isobaric"]
        self.height = smooth2d(ds_plevs["Geopotential_height_isobaric"].metpy.convert_units("dam"), 355)
        self.temp = ds_plevs["Temperature_isobaric"]
        self.tempc = ds_plevs["Temperature_isobaric"].metpy.convert_units("degC")
        self.rh = ds_plevs["Relative_humidity_isobaric"]
        self.rh.attrs["units"] = "percent"
        
        self.rh_smoothed = calc.smooth_gaussian(self.rh, 30)
        self.pw = ds_plevs["Precipitable_water_entire_atmosphere_single_layer"].isel(isobaric=0)
        self.thickness = self.height[:, 1] - self.height[:, 4]

        self.u_ms = ds_plevs["u-component_of_wind_isobaric"]
        self.u_kt = ds_plevs["u-component_of_wind_isobaric"].metpy.convert_units("kt")
        self.v_ms = ds_plevs["v-component_of_wind_isobaric"]
        self.v_kt = ds_plevs["v-component_of_wind_isobaric"].metpy.convert_units("kt")
        self.ws_ms = calc.wind_speed(self.u_ms, self.v_ms)

        self.avo = (ds_plevs["Absolute_vorticity_isobaric"].values * 10**-5) * units["1/second"]
        self.rvo = calc.vorticity(self.u_ms, self.v_ms) * 10**5
        self.omega = ds_plevs["Vertical_velocity_pressure_isobaric"].metpy.convert_units("hPa/hour")

        self.t2 = calc.smooth_gaussian(ds_sfc["Temperature_height_above_ground"].isel(height_above_ground3=0).metpy.convert_units("degC"), 20)
        self.td2 = calc.smooth_gaussian(ds_sfc["Dewpoint_temperature_height_above_ground"].isel(height_above_ground4=0).metpy.convert_units("degC"), 20)
        self.t2depression = self.t2 - self.td2
        self.u10_ms = ds_sfc["u-component_of_wind_height_above_ground"].isel(height_above_ground2=0)
        self.u10_kt = ds_sfc["u-component_of_wind_height_above_ground"].isel(height_above_ground2=0).metpy.convert_units("kt")
        self.v10_ms = ds_sfc["v-component_of_wind_height_above_ground"].isel(height_above_ground2=0)
        self.v10_kt = ds_sfc["v-component_of_wind_height_above_ground"].isel(height_above_ground2=0).metpy.convert_units("kt")
        self.ws10_ms = calc.wind_speed(self.u10_ms, self.v10_ms)
        self.slp = calc.smooth_gaussian(ds_plevs["Pressure_reduced_to_MSL_msl"].metpy.convert_units("hPa").isel(isobaric=0), 50)

        self.tempc_adv = calc.smooth_gaussian(calc.advection(ds_plevs["Temperature_isobaric"], ds_plevs["u-component_of_wind_isobaric"], ds_plevs["v-component_of_wind_isobaric"]).metpy.convert_units("K/h"), 75)
        self.avo_adv = calc.smooth_gaussian(calc.advection(ds_plevs["Absolute_vorticity_isobaric"], ds_plevs["u-component_of_wind_isobaric"], ds_plevs["v-component_of_wind_isobaric"]).metpy.convert_units("1/hour**2"), 75)
        self.u_qvect, self.v_qvect = calc.q_vector(u=self.u_ms, v=self.v_ms, temperature=self.temp, pressure=self.pressure)
     
def make_plots(ds: xr.DataArray, figures_path: str, logo_path: str):
    for i in range(len(ds.times)):
        coords = (0.94, 0.1)
        temphgt850 = AMTPlotter(lons=ds.lons, lats=ds.lats, contourf_data=ds.tempc[i, 3], cproj = crs.PlateCarree(), contour_data=ds.height[i, 3], title=titles["temphgt850"], time=ds.times[i], contour_linewidths=0.5,
                                        cmap=colormaps["t2m"], cbar_label=f"({unit_list['tempc']})", contourf_levels=np.arange(-50, 40, 3), cbar_ticks=np.arange(-50, 40, 3), contour_levels=np.arange(100, 200, 3), map_extent=[-30, 80, 25, 75], data_source="GFS 0.25 Degree Forecast")
        temphgt850.add_logo(logo_path=logo_path, coords=coords)
        temphgt850.save_plot(path=os.path.join(figures_path, "temphgt850", f"temphgt850_{i}"))           
        temphgt850.close_plot()

        tempadv850 = AMTPlotter(lons=ds.lons, lats=ds.lats, contourf_data=ds.tempc_adv[i, 3], cproj=crs.PlateCarree(), contour_data=ds.slp[i], title=titles["tempadv850"], time=ds.times[i],
                                        cmap=colormaps["tempadv850"], cbar_label=f"({unit_list['tempc_adv']})", contourf_levels=np.arange(-0.2, 0.2, 0.05), cbar_ticks=np.arange(-0.2, 0.2, 0.05), 
                                        contour_linewidths=0.5, map_extent=[-30, 80, 25, 75], data_source="GFS 0.25 Degree Forecast")
        tempadv850.add_logo(logo_path=logo_path, coords=coords)
        tempadv850.save_plot(path=os.path.join(figures_path, "tempadv850", f"tempadv850_{i}"))           
        tempadv850.close_plot()

        rh700 = AMTPlotter(lons=ds.lons, lats=ds.lats, contourf_data=ds.rh[i, 2], cproj=crs.PlateCarree(), title=titles["rh700"], time=ds.times[i], contourf_levels=np.arange(60, 105, 5), cmap=colormaps["rh700"], 
                                cbar_extend="max", cbar_label=f"({unit_list['rh']})", contour_data=ds.rh_smoothed[i, 2], contour_levels=np.arange(60,95,15), contour_colors="blue", contour_linewidths=0.5, map_extent=[-30, 80, 25, 75], data_source="GFS 0.25 Degree Forecast")
        rh700.add_contour(data=ds.rh_smoothed[i, 2], levels=np.arange(15,60,15), colors="blue", linestyles="dashed", linewidths=0.5)
        rh700.add_logo(logo_path=logo_path, coords=coords)
        rh700.save_plot(path=os.path.join(figures_path, "rh700", f"rh700_{i}"))
        rh700.close_plot()

        rvo500 = AMTPlotter(lons=ds.lons, lats=ds.lats, contourf_data=ds.rvo[i, 1], cproj=crs.PlateCarree(), title=titles["rvo500"], time=ds.times[i], u=ds.u_kt[i, 1], v=ds.v_kt[i, 1], barb_gap=15,
                                cbar_label=f"({unit_list['rvo']})", cmap=ListedColormap(colormaps["holton"].colors[45:]), contourf_levels=np.concatenate((np.arange(-34, 32, 2), np.arange(35, 55, 5))), map_extent=[-30, 80, 25, 75], data_source="GFS 0.25 Degree Forecast")
                
        rvo500.add_logo(logo_path=logo_path, coords=coords)
        rvo500.save_plot(path=os.path.join(figures_path, "rvo500", f"rvo500_{i}"))
        rvo500.close_plot()

        avo_adv500 = AMTPlotter(lons=ds.lons, lats=ds.lats, contourf_data=ds.avo_adv[i, 1], contour_data=ds.height[i, 1], cproj=crs.PlateCarree(), title=titles["avo_adv500"], time=ds.times[i],
                                        cbar_label=f"({unit_list['avo']})", cmap=colormaps["avo_adv500"], contourf_levels=np.arange(-0.2, 0.2, 0.05), contour_levels=np.arange(500, 606, 3), cbar_ticks=np.arange(-0.2, 0.2, 0.05),
                                        contour_linewidths=0.5, map_extent=[-30, 80, 25, 75], data_source="GFS 0.25 Degree Forecast")
        avo_adv500.add_logo(logo_path=logo_path, coords=coords)
        avo_adv500.save_plot(path=os.path.join(figures_path, "avo_adv500", f"avo_adv500_{i}"))
        avo_adv500.close_plot()

        vertical_v500 = AMTPlotter(lons=ds.lons, lats=ds.lats, contourf_data=ds.omega[i, 1], contour_data=ds.height[i, 1], cproj=crs.PlateCarree(), title=titles["vertical_v500"], time=ds.times[i], 
                                        cbar_label=f"({unit_list['omega']})", cmap=colormaps["holton"], contourf_levels=np.arange(-46, 48, 2), cbar_ticks=np.arange(-46, 50, 4), contour_levels=np.arange(500, 606, 3), contour_linewidths=0.5, map_extent=[-30, 80, 25, 75], data_source="GFS 0.25 Degree Forecast")
        vertical_v500.add_logo(logo_path=logo_path, coords=coords)
        vertical_v500.save_plot(path=os.path.join(figures_path, "vertical_v500", f"vertical_v500_{i}"))
        vertical_v500.close_plot()

        rvo300 = AMTPlotter(lons=ds.lons, lats=ds.lats, contourf_data=ds.rvo[i, 0], cproj=crs.PlateCarree(), title=titles["rvo300"], time=ds.times[i], u=ds.u_kt[i, 0], v=ds.v_kt[i, 0], barb_gap=15,
                                    cbar_label=f"({unit_list['rvo']})", cmap=ListedColormap(colormaps["holton"].colors[45:]), contourf_levels=np.concatenate((np.arange(-34, 32, 2), np.arange(35, 55, 5))), map_extent=[-30, 80, 25, 75], data_source="GFS 0.25 Degree Forecast")
        rvo300.add_logo(logo_path=logo_path, coords=coords)
        rvo300.save_plot(path=os.path.join(figures_path, "rvo300", f"rvo300_{i}"))
        rvo300.close_plot()

        combined300 = AMTPlotter(lons=ds.lons, lats=ds.lats, contourf_data=ds.ws_ms[i, 0], contour_data=ds.height[i, 1], cproj=crs.PlateCarree(), title=titles["combined300"], time=ds.times[i],
                                    cbar_label=f"({unit_list['wind_ms']})", cmap=ListedColormap(colormaps["holton"].colors[130:]), contourf_levels=np.arange(20, 62, 2), cbar_ticks=np.arange(20, 64, 4), 
                                    contour_levels=np.arange(800, 1203, 3), contour_linewidths=1, map_extent=[-30, 80, 25, 75], data_source="GFS 0.25 Degree Forecast")
        combined300.add_contour(data=ds.slp[i], colors="blue", linewidths=1)
        combined300.add_logo(logo_path=logo_path, coords=coords)
        combined300.save_plot(path=os.path.join(figures_path, "combined300", f"combined300_{i}"))
        combined300.close_plot()

        jet300 = AMTPlotter(lons=ds.lons, lats=ds.lats, contourf_data=ds.ws_ms[i, 0], contour_data=ds.height[i, 0], cproj=crs.PlateCarree(), title=titles["jet300"], time=ds.times[i],
                                    cbar_label=f"({unit_list['wind_ms']})", cmap=ListedColormap(colormaps["holton"].colors[130:]), contourf_levels=np.arange(20, 62, 2), cbar_ticks=np.arange(20, 64, 4), 
                                    contour_levels=np.arange(800, 1203, 3), u=ds.u_kt[i, 0], v=ds.v_kt[i, 0], barb_gap=20, contour_linewidths=0.5, map_extent=[-30, 80, 25, 75], data_source="GFS 0.25 Degree Forecast")
        jet300.add_logo(logo_path=logo_path, coords=coords)
        jet300.save_plot(path=os.path.join(figures_path, "jet300", f"jet300_{i}"))
        jet300.close_plot()

        pwat = AMTPlotter(lons=ds.lons, lats=ds.lats, contourf_data=ds.pw[i], cproj=crs.PlateCarree(), title=titles["pwat"], time=ds.times[i],
                                cbar_label=f"({unit_list['precip']})", cmap=colormaps["pwat"], contourf_levels=np.arange(0, 51, 5), cbar_ticks=np.arange(0, 51, 5), cbar_extend="max", map_extent=[-30, 80, 25, 75], data_source="GFS 0.25 Degree Forecast")
        pwat.add_logo(logo_path=logo_path, coords=coords)
        pwat.save_plot(path=os.path.join(figures_path, "pwat", f"pwat_{i}"))
        pwat.close_plot()

        t2m = AMTPlotter(lons=ds.lons, lats=ds.lats, contourf_data=ds.t2[i], contour_data=ds.td2[i], cproj=crs.PlateCarree(), title=titles["t2m"], time=ds.times[i], contour_levels=np.arange(-40, 50, 5),
                            cbar_label=f"({unit_list['tempc']})", cmap=colormaps["t2m"], contourf_levels=np.arange(-40, 48, 3), cbar_ticks=np.arange(-40, 48, 3), contour_colors="white", contour_linewidths=0.5, map_extent=[-30, 80, 25, 75], data_source="GFS 0.25 Degree Forecast")
        t2m.add_logo(logo_path=logo_path, coords=coords)
        t2m.save_plot(path=os.path.join(figures_path, "t2m", f"t2m_{i}"))
        t2m.close_plot()

        wind10 = AMTPlotter(lons=ds.lons, lats=ds.lats, contourf_data=ds.ws10_ms[i], cproj=crs.PlateCarree(), title=titles["wind10"], time=ds.times[i], 
                        u=ds.u10_kt[i], v=ds.v10_kt[i], barb_gap=6, barb_length=5, contourf_levels=np.arange(0, 32, 2), cmap=colormaps["wind10"], cbar_extend="max", cbar_label=f"({unit_list['wind_ms']})", cbar_ticks=np.arange(0, 32, 2),
                        map_extent=[-30, 80, 25, 75], data_source="GFS 0.25 Degree Forecast")
        wind10.add_logo(logo_path=logo_path, coords=coords)
        wind10.save_plot(path=os.path.join(figures_path, "wind10", f"wind10_{i}"))
        wind10.close_plot()

        t2depression = AMTPlotter(lons=ds.lons, lats=ds.lats, contourf_data=ds.t2depression[i], contour_data=ds.slp[i], cproj=crs.PlateCarree(), title=titles["t2_depression"], time=ds.times[i], contour_levels=np.arange(800, 1200, 5),
                            cbar_label=f"({unit_list['tempc']})", cmap=colormaps["t2depression"], contourf_levels=np.arange(-3, 30, 3), cbar_ticks=np.arange(-3, 30, 3), contour_colors="white", 
                            u=ds.u10_kt[i], v=ds.v10_kt[i], barb_gap=6, barb_length=5, contour_linewidths=1, map_extent=[-30, 80, 25, 75], data_source="GFS 0.25 Degree Forecast")
        t2depression.save_plot(path=os.path.join(figures_path, "t2_depression", f"t2_depression_{i}"))
        t2depression.close_plot()

        thickness = AMTPlotter(lons=ds.lons, lats=ds.lats, contourf_data=ds.thickness[i], cproj=crs.PlateCarree(), title=titles["thickness"], time=ds.times[i], contourf_levels=np.arange(488, 648, 8), cmap=colormaps["holton"], 
                                cbar_extend="both", cbar_label=f"({unit_list['thickness']})", contour_data=ds.slp[i], contour_levels=np.arange(980, 1030, 4), contour_colors="white", contour_linewidths=0.5, map_extent=[-30, 80, 25, 75], data_source="GFS 0.25 Degree Forecast")
        thickness.add_contour(data=ds.height[i, 1], levels=np.arange(480., 600., 8.), colors="black", linewidths=1)
        thickness.add_logo(logo_path=logo_path, coords=coords)
        thickness.save_plot(path=os.path.join(figures_path, "thickness", f"thickness_{i}"))
        thickness.close_plot()

titles = {
        "eth850": "850 hPa Equivalent Potential Temperature",
        "tempadv850": "850 hPa Temperature Advection",
        "temphgt850": "850 hPa Temperature",
        "rh700": "700 hPa Relative Humidty",
        "temphgt700": "700 hPa Temperature",
        "rvo500": "500 hPa Rel. Vorticity",
        "avo_adv500": "500 hPa Abs. Vorticity Adv.",
        "temphgt500": "500 hPa Temperature & Height, MSLP",
        "vertical_v500": "500 hPa Vertical Velocity",
        "rvo300": "300 hPa Rel. Vorticity",
        "temphgt300": "300 hPa Temperature",
        "jet300": "300 hPa Jet Streams",
        "pwat": "Precipitable Water",
        "total_precip": "Acc. Total Precipitation",
        "t2m": "2m Temperature",
        "wind10": "10m Wind",
        "t2_depression": "2m Temperature Depression",
        "hourly_precip": "Hourly Precipitation",
        "sst": "Sea Surface Temperature",
        "tsk": "Surface Skin Temperature",
        "dbz": "Maximum Radar Reflectivity",
        "combined300": "300 hPa Combined Map",
        "isent320": "320K Isentropic Surface\nPotential Vorticity, Potential Temperature Heights",
        "thickness": "500-1000 hPa Topography, 500 hPa Heights & MSLP",
        "qvect850": "850 hPa Q-Vectors & Temperature Advection",
        }

unit_list = {
             "temp": "K", "tempc": "°C", "height": "dam", "td": "°C", "rh": "%", "pressure": "hPa", "wind_kt": "knot", "wind_ms": "m/s", "eth": "°C", "omega": "hPa/h", "avo": "1/h$^2$", 
             "rvo": "10$^{-5}$/s", "slp": "hPa", "precip": "mm", "tempc_adv": "K/h", "avo_adv": "1/h**2", "dbz": "dBz", "pvo": "PVU", "thickness": "meters"
             }

if __name__ == "__main__":
    start = datetime.now() 
    create_folders()
    delete_png_files(os.path.join(os.getcwd(), "gfs_output_maps"))
    ds = GFSParser(ds_plevs=download_gfs_plevs(levels=[30000, 50000, 70000, 85000, 100000], time_delta=10), ds_sfc=download_gfs_sfc(time_delta=10))
    make_plots(ds=ds, figures_path=os.path.join(os.getcwd(), "gfs_output_maps"), logo_path="C://Users//gkbrk//Desktop//son//amtlogo.png")
    stop = datetime.now()
    print(f"Duration: {stop - start}")
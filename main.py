from netCDF4 import Dataset
from ituamt import WRFParser, AMTPlotter, colormaps
from matplotlib.colors import ListedColormap
import os
import numpy as np
import glob
from datetime import datetime
import cartopy.crs as crs

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
        "isent290": "290K Isentropic Surface\nPotential Vorticity, Potential Temperature Heights",
        "thickness": "500-1000 hPa Topography, 500 hPa Heights & MSLP",
        }

unit_list = {
             "temp": "K", "tempc": "°C", "height": "dam", "td": "°C", "rh": "%", "pressure": "hPa", "wind_kt": "knot", "wind_ms": "m/s", "eth": "°C", "omega": "hPa/h", "avo": "1/h$^2$", 
             "rvo": "10$^{-5}$/s", "slp": "hPa", "precip": "mm", "tempc_adv": "K/h", "avo_adv": "1/h**2", "dbz": "dBz", "pvo": "PVU", "thickness": "meters"
             }

def create_folders():
    folder_names = {"d01": ["tempadv850", "temphgt850", "rh700", "rvo500", "avo_adv500", "temphgt500", "vertical_v500", "rvo300", "jet300", "combined300", "pwat", "total_precip", "t2m", "wind10", "t2_depression", "hourly_precip", "sst", "tsk", "dbz", "isent320", "thickness", "isent290"],
                    "d02": ["tempadv850", "temphgt850", "rh700", "rvo500", "vertical_v500", "rvo300", "pwat", "total_precip", "t2m", "wind10", "t2_depression", "hourly_precip", "tsk", "dbz"]}

    cwd = os.getcwd()
    if not os.path.exists("wrf_output_maps"): os.mkdir("wrf_output_maps")
    figures_path = os.path.join(cwd, "wrf_output_maps")

    for domain in list(folder_names.keys()):
        if not os.path.exists(os.path.join(figures_path, domain)): os.mkdir(os.path.join(figures_path, domain))
        for folder_name in folder_names[domain]:
            folder_path = os.path.join(figures_path, domain, folder_name)
            if not os.path.exists(folder_path): os.mkdir(folder_path)

def delete_png_files(directory):
    if os.path.exists(directory):
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".png"):
                    file_path = os.path.join(root, file)
                    os.remove(file_path)

def make_plots(domain: str, wrfout: WRFParser, figures_path: str, logo_path: str):
    cproj = crs.PlateCarree()
    if domain == "d01":
        coords = (0.94, 0.1)
        for i in range(len(wrfout.times)):
            tempadv850 = AMTPlotter(lons=wrfout.lons, lats=wrfout.lats, contourf_data=wrfout.tempc_adv[i, 0], cproj=cproj, contour_data=wrfout.slp[i], title=f"GFS Init {wrfout.init_date} UTC\n"+titles["tempadv850"], time=wrfout.times[i],
                                        cmap=colormaps["tempadv850"], cbar_label=f"({unit_list['tempc_adv']})", contourf_levels=np.arange(-0.2, 0.2, 0.05), cbar_ticks=np.arange(-0.2, 0.2, 0.05), 
                                        contour_linewidths=2, data_source="ITU AMT Model")
            tempadv850.add_logo(logo_path=logo_path, coords=coords)
            tempadv850.save_plot(path=os.path.join(figures_path, domain, "tempadv850", f"tempadv850_{i}"))           
            tempadv850.close_plot()

            temphgt850 = AMTPlotter(lons=wrfout.lons, lats=wrfout.lats, contourf_data=wrfout.temp850[i, 0], cproj=cproj, contour_data=wrfout.height[i, 0], title=f"GFS Init {wrfout.init_date} UTC\n"+titles["temphgt850"], time=wrfout.times[i],
                                        cmap=colormaps["t2m"], cbar_label=f"({unit_list['tempc']})", contourf_levels=np.arange(-50, 40, 3), cbar_ticks=np.arange(-50, 40, 3), contour_levels=np.arange(100, 200, 3), data_source="ITU AMT Model")
            temphgt850.add_logo(logo_path=logo_path, coords=coords)
            temphgt850.save_plot(path=os.path.join(figures_path, domain, "temphgt850", f"temphgt850_{i}"))           
            temphgt850.close_plot()

            rh700 = AMTPlotter(lons=wrfout.lons, lats=wrfout.lats, contourf_data=wrfout.rh[i, 1], cproj=cproj, title=f"GFS Init {wrfout.init_date} UTC\n"+titles["rh700"], time=wrfout.times[i], contourf_levels=np.arange(60, 105, 5), cmap=colormaps["rh700"], 
                                cbar_extend="max", cbar_label=f"({unit_list['rh']})", contour_data=wrfout.rh_smoothed[i, 1], contour_levels=np.arange(60,95,15), contour_colors="blue", contour_linewidths=1, data_source="ITU AMT Model")
            rh700.add_contour(data=wrfout.rh_smoothed[i, 1], levels=np.arange(15,60,15), colors="blue", linestyles="dashed", linewidths=1)
            rh700.add_logo(logo_path=logo_path, coords=coords)
            rh700.save_plot(path=os.path.join(figures_path, domain, "rh700", f"rh700_{i}"))
            rh700.close_plot()

            rvo500 = AMTPlotter(lons=wrfout.lons, lats=wrfout.lats, contourf_data=wrfout.rvo[i, 2], cproj=cproj, title=f"GFS Init {wrfout.init_date} UTC\n"+titles["rvo500"], time=wrfout.times[i], u=wrfout.u_kt[i, 2], v=wrfout.v_kt[i, 2], barb_gap=10,
                                    cbar_label=f"({unit_list['rvo']})", cmap=ListedColormap(colormaps["holton"].colors[45:]), contourf_levels=np.concatenate((np.arange(-34, 32, 2), np.arange(35, 55, 5))), data_source="ITU AMT Model")
                
            rvo500.add_logo(logo_path=logo_path, coords=coords)
            rvo500.save_plot(path=os.path.join(figures_path, domain, "rvo500", f"rvo500_{i}"))
            rvo500.close_plot()

            avo_adv500 = AMTPlotter(lons=wrfout.lons, lats=wrfout.lats, contourf_data=wrfout.avo_adv[i, 2], contour_data=wrfout.height[i, 2], cproj=cproj, title=f"GFS Init {wrfout.init_date} UTC\n"+titles["avo_adv500"], time=wrfout.times[i],
                                        cbar_label=f"({unit_list['avo']})", cmap=colormaps["avo_adv500"], contourf_levels=np.arange(-0.2, 0.2, 0.05), contour_levels=np.arange(500, 606, 3), cbar_ticks=np.arange(-0.2, 0.2, 0.05), data_source="ITU AMT Model")
            avo_adv500.add_logo(logo_path=logo_path, coords=coords)
            avo_adv500.save_plot(path=os.path.join(figures_path, domain, "avo_adv500", f"avo_adv500_{i}"))
            avo_adv500.close_plot()

            temphgt500 = AMTPlotter(lons=wrfout.lons, lats=wrfout.lats, contourf_data=wrfout.tempc[i, 2], contour_data=wrfout.height[i, 2], cproj=cproj, title=f"GFS Init {wrfout.init_date} UTC\n"+titles["temphgt500"], time=wrfout.times[i], u=wrfout.u_kt[i, 2], v=wrfout.v_kt[i, 2], barb_gap=10,
                                        cbar_label=f"({unit_list['tempc']})", cmap=colormaps["t2m"], contourf_levels=np.arange(-54, 6, 2), cbar_ticks=np.arange(-54, 6, 4), contour_levels=np.arange(500, 606, 3), data_source="ITU AMT Model")
            temphgt500.add_contour(data=wrfout.slp[i], colors="white")
            temphgt500.add_logo(logo_path=logo_path, coords=coords)
            temphgt500.save_plot(path=os.path.join(figures_path, domain, "temphgt500", f"temphgt500_{i}"))
            temphgt500.close_plot()

            vertical_v500 = AMTPlotter(lons=wrfout.lons, lats=wrfout.lats, contourf_data=wrfout.omega[i, 2], contour_data=wrfout.height[i, 2], cproj=cproj, title=f"GFS Init {wrfout.init_date} UTC\n"+titles["vertical_v500"], time=wrfout.times[i], 
                                        cbar_label=f"({unit_list['omega']})", cmap=colormaps["holton"], contourf_levels=np.arange(-46, 48, 2), cbar_ticks=np.arange(-46, 50, 4), contour_levels=np.arange(500, 606, 3), data_source="ITU AMT Model")
            vertical_v500.add_logo(logo_path=logo_path, coords=coords)
            vertical_v500.save_plot(path=os.path.join(figures_path, domain, "vertical_v500", f"vertical_v500_{i}"))
            vertical_v500.close_plot()

            rvo300 = AMTPlotter(lons=wrfout.lons, lats=wrfout.lats, contourf_data=wrfout.rvo[i, 3], cproj=cproj, title=f"GFS Init {wrfout.init_date} UTC\n"+titles["rvo300"], time=wrfout.times[i], u=wrfout.u_kt[i, 3], v=wrfout.v_kt[i, 3], barb_gap=10,
                                    cbar_label=f"({unit_list['rvo']})", cmap=ListedColormap(colormaps["holton"].colors[45:]), contourf_levels=np.concatenate((np.arange(-34, 32, 2), np.arange(35, 55, 5))), data_source="ITU AMT Model")
            rvo300.add_logo(logo_path=logo_path, coords=coords)
            rvo300.save_plot(path=os.path.join(figures_path, domain, "rvo300", f"rvo300_{i}"))
            rvo300.close_plot()

            combined300 = AMTPlotter(lons=wrfout.lons, lats=wrfout.lats, contourf_data=wrfout.ws_ms[i, 3], contour_data=wrfout.height[i, 3], cproj=cproj, title=f"GFS Init {wrfout.init_date} UTC\n"+titles["combined300"], time=wrfout.times[i],
                                    cbar_label=f"({unit_list['wind_ms']})", cmap=ListedColormap(colormaps["holton"].colors[130:]), contourf_levels=np.arange(20, 62, 2), cbar_ticks=np.arange(20, 64, 4), 
                                    contour_levels=np.arange(800, 1203, 3), data_source="ITU AMT Model")
            combined300.add_contour(data=wrfout.slp[i], colors="blue")
            combined300.add_logo(logo_path=logo_path, coords=coords)
            combined300.save_plot(path=os.path.join(figures_path, domain, "combined300", f"combined300_{i}"))
            combined300.close_plot()

            jet300 = AMTPlotter(lons=wrfout.lons, lats=wrfout.lats, contourf_data=wrfout.ws_ms[i, 3], contour_data=wrfout.height[i, 3], cproj=cproj, title=f"GFS Init {wrfout.init_date} UTC\n"+titles["jet300"], time=wrfout.times[i],
                                    cbar_label=f"({unit_list['wind_ms']})", cmap=ListedColormap(colormaps["holton"].colors[130:]), contourf_levels=np.arange(20, 62, 2), cbar_ticks=np.arange(20, 64, 4), 
                                    contour_levels=np.arange(800, 1203, 3), u=wrfout.u_kt[i, 3], v=wrfout.v_kt[i, 3], barb_gap=20, data_source="ITU AMT Model")
            jet300.add_logo(logo_path=logo_path, coords=coords)
            jet300.save_plot(path=os.path.join(figures_path, domain, "jet300", f"jet300_{i}"))
            jet300.close_plot()

            pwat = AMTPlotter(lons=wrfout.lons, lats=wrfout.lats, contourf_data=wrfout.pw[i], cproj=cproj, title=f"GFS Init {wrfout.init_date} UTC\n"+titles["pwat"], time=wrfout.times[i],
                                cbar_label=f"({unit_list['precip']})", cmap=colormaps["pwat"], contourf_levels=np.arange(0, 51, 5), cbar_ticks=np.arange(0, 51, 5), cbar_extend="max", data_source="ITU AMT Model")
            pwat.add_logo(logo_path=logo_path, coords=coords)
            pwat.save_plot(path=os.path.join(figures_path, domain, "pwat", f"pwat_{i}"))
            pwat.close_plot()

            total_precip = AMTPlotter(lons=wrfout.lons, lats=wrfout.lats, contourf_data=wrfout.total_precip[i], cproj=cproj, title=f"GFS Init {wrfout.init_date} UTC\n"+titles["total_precip"], time=wrfout.times[i], contourf_levels=np.arange(0.1, 105.1, 5),
                                        cbar_label=f"({unit_list['precip']})", cmap=ListedColormap(colormaps["holton_r"].colors[130:]), cbar_extend="max", cbar_ticks=np.arange(0.5, 110.5, 10), data_source="ITU AMT Model")
            total_precip.add_logo(logo_path=logo_path, coords=coords)
            total_precip.save_plot(path=os.path.join(figures_path, domain, "total_precip", f"total_precip_{i}"))
            total_precip.close_plot()

            if i <= len(wrfout.times) - 2:
                hourly_precip = AMTPlotter(lons=wrfout.lons, lats=wrfout.lats, contourf_data=wrfout.hourly_precip[i], cproj=cproj, title=f"GFS Init {wrfout.init_date} UTC\n"+titles["hourly_precip"], time=wrfout.times[i], 
                                            cmap=ListedColormap(colormaps["holton_r"].colors[130:]), contourf_levels=np.arange(0.5, 50.6, 5), cbar_extend="max", cbar_label=f"{unit_list['precip']}", data_source="ITU AMT Model")
                hourly_precip.add_logo(logo_path=logo_path, coords=coords)
                hourly_precip.save_plot(path=os.path.join(figures_path, domain, "hourly_precip", f"hourly_precip_{i}"))
                hourly_precip.close_plot()

            t2m = AMTPlotter(lons=wrfout.lons, lats=wrfout.lats, contourf_data=wrfout.t2[i], contour_data=wrfout.td2[i], cproj=cproj, title=f"GFS Init {wrfout.init_date} UTC\n"+titles["t2m"], time=wrfout.times[i], contour_levels=np.arange(-40, 50, 5),
                            cbar_label=f"({unit_list['tempc']})", cmap=colormaps["t2m"], contourf_levels=np.arange(-40, 48, 3), cbar_ticks=np.arange(-40, 48, 3), contour_colors="white", data_source="ITU AMT Model")
            t2m.add_logo(logo_path=logo_path, coords=coords)
            t2m.save_plot(path=os.path.join(figures_path, domain, "t2m", f"t2m_{i}"))
            t2m.close_plot()

            wind10 = AMTPlotter(lons=wrfout.lons, lats=wrfout.lats, contourf_data=wrfout.ws10_ms[i], cproj=cproj, title=f"GFS Init {wrfout.init_date} UTC\n"+titles["wind10"], time=wrfout.times[i], 
                        u=wrfout.u10[i], v=wrfout.v10[i], barb_gap=6, barb_length=5, contourf_levels=np.arange(0, 32, 2), cmap=colormaps["wind10"], cbar_extend="max", cbar_label=f"({unit_list['wind_ms']})", cbar_ticks=np.arange(0, 32, 2), data_source="ITU AMT Model")
            wind10.add_logo(logo_path=logo_path, coords=coords)
            wind10.save_plot(path=os.path.join(figures_path, domain, "wind10", f"wind10_{i}"))
            wind10.close_plot()

            t2depression = AMTPlotter(lons=wrfout.lons, lats=wrfout.lats, contourf_data=wrfout.t2_depression[i], contour_data=wrfout.slp[i], cproj=cproj, title=f"GFS Init {wrfout.init_date} UTC\n"+titles["t2_depression"], time=wrfout.times[i], contour_levels=np.arange(800, 1200, 5),
                            cbar_label=f"({unit_list['tempc']})", cmap=colormaps["t2depression"], contourf_levels=np.arange(-3, 30, 3), cbar_ticks=np.arange(-3, 30, 3), contour_colors="white", 
                            u=wrfout.u10[i], v=wrfout.v10[i], barb_gap=6, barb_length=5, data_source="ITU AMT Model")
            t2depression.add_logo(logo_path=logo_path, coords=coords)
            t2depression.save_plot(path=os.path.join(figures_path, domain, "t2_depression", f"t2_depression_{i}"))
            t2depression.close_plot()

            sst = AMTPlotter(lons=wrfout.lons, lats=wrfout.lats, contourf_data=wrfout.sst[i], contour_data=wrfout.slp[i], cproj=cproj, title=f"GFS Init {wrfout.init_date} UTC\n"+titles["sst"], time=wrfout.times[i], contour_levels=np.arange(800, 1200, 5),
                            cbar_label=f"({unit_list['tempc']})", cmap=colormaps["t2depression"], cbar_ticks=np.linspace(5, 35, 11), contourf_levels=np.linspace(5, 35, 11), data_source="ITU AMT Model")
            sst.add_logo(logo_path=logo_path, coords=coords)
            sst.save_plot(path=os.path.join(figures_path, domain, "sst", f"sst_{i}"))
            sst.close_plot()

            tsk = AMTPlotter(lons=wrfout.lons, lats=wrfout.lats, contourf_data=wrfout.tsk[i], contour_data=wrfout.slp[i], cproj=cproj, title=f"GFS Init {wrfout.init_date} UTC\n"+titles["tsk"], time=wrfout.times[i], contour_levels=np.arange(800, 1200, 5),
                            cbar_label=f"({unit_list['tempc']})", cmap=colormaps["t2m"], cbar_ticks=np.arange(-20, 70, 3), contourf_levels=np.arange(-20, 70, 3), data_source="ITU AMT Model")
            tsk.add_logo(logo_path=logo_path, coords=coords)
            tsk.save_plot(path=os.path.join(figures_path, domain, "tsk", f"tsk_{i}")) 
            tsk.close_plot()   

            dbz = AMTPlotter(lons=wrfout.lons, lats=wrfout.lats, contourf_data=wrfout.dbz[i], cproj=cproj, title=f"GFS Init {wrfout.init_date} UTC\n"+titles["dbz"], time=wrfout.times[i], 
                            cbar_label=f"({unit_list['dbz']})", cmap=colormaps["tigris"], cbar_ticks=np.arange(0, 80, 5),  contourf_levels=np.arange(0, 80, 5), cbar_extend="max", data_source="ITU AMT Model")
            dbz.add_logo(logo_path=logo_path, coords=coords)
            dbz.save_plot(path=os.path.join(figures_path, domain, "dbz", f"dbz_{i}"))
            dbz.close_plot()

            isent320 = AMTPlotter(lons=wrfout.lons, lats=wrfout.lats, contourf_data=wrfout.pvo[i, 1]/100, cproj=cproj, contour_data=wrfout.isent_pressures[i, 1], title=f"GFS Init {wrfout.init_date} UTC\n"+titles["isent320"], time=wrfout.times[i],
                                        cmap=colormaps["pvo"], cbar_label=f"({unit_list['pvo']})", u=wrfout.isent_u[i, 1], v=wrfout.isent_v[i, 1], barb_gap=10, barb_length=6, contourf_levels=np.concatenate((np.arange(0.1, 0.31, 0.1), np.arange(0.6, 6.61, 0.3), np.arange(7, 9.5, 0.5), [10])),
                                        cbar_ticks=np.concatenate((np.arange(0.1, 0.31, 0.1), np.arange(0.6, 6.61, 0.3), np.arange(7, 9.5, 0.5), [10])), barb_color="white", data_source="ITU AMT Model")
            isent320.add_logo(logo_path=logo_path, coords=coords)
            isent320.save_plot(path=os.path.join(figures_path, domain, "isent320", f"isent320_{i}"))
            isent320.close_plot()

            isent290 = AMTPlotter(lons=wrfout.lons, lats=wrfout.lats, contourf_data=wrfout.pvo[i, 0]/100, cproj=cproj, contour_data=wrfout.isent_pressures[i, 0], title=f"GFS Init {wrfout.init_date} UTC\n"+titles["isent290"], time=wrfout.times[i],
                                        cmap=colormaps["pvo"], cbar_label=f"({unit_list['pvo']})", u=wrfout.isent_u[i, 1], v=wrfout.isent_v[i, 1], barb_gap=10, barb_length=6, contourf_levels=np.concatenate((np.arange(0.1, 0.31, 0.1), np.arange(0.6, 6.61, 0.3), np.arange(7, 9.5, 0.5), [10])),
                                        cbar_ticks=np.concatenate((np.arange(0.1, 0.31, 0.1), np.arange(0.6, 6.61, 0.3), np.arange(7, 9.5, 0.5), [10])), barb_color="white", data_source="ITU AMT Model")
            isent290.add_logo(logo_path=logo_path, coords=coords)
            isent290.save_plot(path=os.path.join(figures_path, domain, "isent290", f"isent290_{i}"))
            isent290.close_plot()

            thickness = AMTPlotter(lons=wrfout.lons, lats=wrfout.lats, contourf_data=wrfout.thickness[i], cproj=cproj, title=f"GFS Init {wrfout.init_date} UTC\n"+titles["thickness"], time=wrfout.times[i], contourf_levels=np.arange(4880, 6048, 8), cmap=colormaps["holton"], 
                                    cbar_extend="both", cbar_label=f"({unit_list['thickness']})", contour_data=wrfout.slp[i], contour_levels=np.arange(980, 1030, 4), contour_colors="white", data_source="ITU AMT Model")
            thickness.add_contour(data=wrfout.height[i, 2], levels=np.arange(480., 600., 8.), colors="black")
            thickness.add_logo(logo_path=logo_path, coords=coords)
            thickness.save_plot(path=os.path.join(figures_path, domain, "thickness", f"thickness_{i}"))
            thickness.close_plot()
    elif domain == "d02":
        coords = (0.91, 0.1)
        for i in range(len(wrfout.times)):
            tempadv850 = AMTPlotter(lons=wrfout.lons, lats=wrfout.lats, contourf_data=wrfout.tempc_adv[i, 0], cproj=cproj, contour_data=wrfout.slp[i], title=f"GFS Init {wrfout.init_date} UTC\n"+titles["tempadv850"], time=wrfout.times[i],
                                        cmap=colormaps["tempadv850"], cbar_label=f"({unit_list['tempc_adv']})", contourf_levels=np.arange(-0.2, 0.2, 0.05), cbar_ticks=np.arange(-0.2, 0.2, 0.05), 
                                        contour_linewidths=2, data_source="ITU AMT Model")
            tempadv850.add_logo(logo_path=logo_path, coords=coords)
            tempadv850.save_plot(path=os.path.join(figures_path, domain, "tempadv850", f"tempadv850_{i}"))
            tempadv850.close_plot()

            temphgt850 = AMTPlotter(lons=wrfout.lons, lats=wrfout.lats, contourf_data=wrfout.temp850[i, 0], cproj=cproj, contour_data=wrfout.height[i, 0], title=f"GFS Init {wrfout.init_date} UTC\n"+titles["temphgt850"], time=wrfout.times[i],
                                        cmap=colormaps["t2m"], cbar_label=f"({unit_list['tempc']})", contourf_levels=np.arange(-50, 40, 3), cbar_ticks=np.arange(-50, 40, 3), contour_levels=np.arange(100, 200, 3), data_source="ITU AMT Model")
            temphgt850.add_logo(logo_path=logo_path, coords=coords)
            temphgt850.save_plot(path=os.path.join(figures_path, domain, "temphgt850", f"temphgt850_{i}"))
            temphgt850.close_plot()

            rh700 = AMTPlotter(lons=wrfout.lons, lats=wrfout.lats, contourf_data=wrfout.rh[i, 1], cproj=cproj, title=f"GFS Init {wrfout.init_date} UTC\n"+titles["rh700"], time=wrfout.times[i], contourf_levels=np.arange(60, 105, 5), cmap=colormaps["rh700"], 
                                cbar_extend="max", cbar_label=f"({unit_list['rh']})", contour_data=wrfout.rh_smoothed[i, 1], contour_levels=np.arange(60,95,15), contour_colors="blue", contour_linewidths=1, data_source="ITU AMT Model")
            rh700.add_contour(data=wrfout.rh_smoothed[i, 1], levels=np.arange(15,60,15), colors="blue", linestyles="dashed", linewidths=1)
            rh700.add_logo(logo_path=logo_path, coords=coords)
            rh700.save_plot(path=os.path.join(figures_path, domain, "rh700", f"rh700_{i}"))
            rh700.close_plot()

            rvo500 = AMTPlotter(lons=wrfout.lons, lats=wrfout.lats, contourf_data=wrfout.rvo[i, 2], cproj=cproj, title=f"GFS Init {wrfout.init_date} UTC\n"+titles["rvo500"], time=wrfout.times[i], u=wrfout.u_kt[i, 2], v=wrfout.v_kt[i, 2], barb_gap=10,
                                    cbar_label=f"({unit_list['rvo']})", cmap=ListedColormap(colormaps["holton"].colors[45:]), contourf_levels=np.concatenate((np.arange(-34, 32, 2), np.arange(35, 55, 5))), data_source="ITU AMT Model")
            rvo500.add_logo(logo_path=logo_path, coords=coords)    
            rvo500.save_plot(path=os.path.join(figures_path, domain, "rvo500", f"rvo500_{i}"))
            rvo500.close_plot()

            vertical_v500 = AMTPlotter(lons=wrfout.lons, lats=wrfout.lats, contourf_data=wrfout.omega[i, 2], contour_data=wrfout.height[i, 2], cproj=cproj, title=f"GFS Init {wrfout.init_date} UTC\n"+titles["vertical_v500"], time=wrfout.times[i], 
                                        cbar_label=f"({unit_list['omega']})", cmap=colormaps["holton"], contourf_levels=np.arange(-46, 48, 2), cbar_ticks=np.arange(-46, 50, 4), contour_levels=np.arange(500, 606, 3), data_source="ITU AMT Model")
            vertical_v500.add_logo(logo_path=logo_path, coords=coords)
            vertical_v500.save_plot(path=os.path.join(figures_path, domain, "vertical_v500", f"vertical_v500_{i}"))
            vertical_v500.close_plot()

            rvo300 = AMTPlotter(lons=wrfout.lons, lats=wrfout.lats, contourf_data=wrfout.rvo[i, 3], cproj=cproj, title=f"GFS Init {wrfout.init_date} UTC\n"+titles["rvo300"], time=wrfout.times[i], u=wrfout.u_kt[i, 3], v=wrfout.v_kt[i, 3], barb_gap=10,
                                    cbar_label=f"({unit_list['rvo']})", cmap=ListedColormap(colormaps["holton"].colors[45:]), contourf_levels=np.concatenate((np.arange(-34, 32, 2), np.arange(35, 55, 5))), data_source="ITU AMT Model")
            rvo300.add_logo(logo_path=logo_path, coords=coords)
            rvo300.save_plot(path=os.path.join(figures_path, domain, "rvo300", f"rvo300_{i}"))
            rvo300.close_plot()

            pwat = AMTPlotter(lons=wrfout.lons, lats=wrfout.lats, contourf_data=wrfout.pw[i], cproj=cproj, title=f"GFS Init {wrfout.init_date} UTC\n"+titles["pwat"], time=wrfout.times[i],
                                cbar_label=f"({unit_list['precip']})", cmap=colormaps["pwat"], contourf_levels=np.arange(0, 51, 5), cbar_ticks=np.arange(0, 51, 5), cbar_extend="max", data_source="ITU AMT Model")
            pwat.add_logo(logo_path=logo_path, coords=coords)
            pwat.save_plot(path=os.path.join(figures_path, domain, "pwat", f"pwat_{i}"))
            pwat.close_plot()

            total_precip = AMTPlotter(lons=wrfout.lons, lats=wrfout.lats, contourf_data=wrfout.total_precip[i], cproj=cproj, title=f"GFS Init {wrfout.init_date} UTC\n"+titles["total_precip"], time=wrfout.times[i], contourf_levels=np.arange(0.1, 105.1, 5),
                                        cbar_label=f"({unit_list['precip']})", cmap=ListedColormap(colormaps["holton_r"].colors[130:]), cbar_extend="max", cbar_ticks=np.arange(0.1, 110.1, 10), data_source="ITU AMT Model")
            total_precip.add_logo(logo_path=logo_path, coords=coords)
            total_precip.save_plot(path=os.path.join(figures_path, domain, "total_precip", f"total_precip_{i}"))
            total_precip.close_plot()

            if i <= len(wrfout.times) - 2:
                hourly_precip = AMTPlotter(lons=wrfout.lons, lats=wrfout.lats, contourf_data=wrfout.hourly_precip[i], cproj=cproj, title=f"GFS Init {wrfout.init_date} UTC\n"+titles["hourly_precip"], time=wrfout.times[i], 
                                            cmap=ListedColormap(colormaps["holton_r"].colors[130:]), contourf_levels=np.arange(0.5, 50.6, 5), cbar_extend="max", cbar_label=f"{unit_list['precip']}", data_source="ITU AMT Model")
                hourly_precip.add_logo(logo_path=logo_path, coords=coords)
                hourly_precip.save_plot(path=os.path.join(figures_path, domain, "hourly_precip", f"hourly_precip_{i}"))
                hourly_precip.close_plot()

            t2m = AMTPlotter(lons=wrfout.lons, lats=wrfout.lats, contourf_data=wrfout.t2[i], contour_data=wrfout.td2[i], cproj=cproj, title=f"GFS Init {wrfout.init_date} UTC\n"+titles["t2m"], time=wrfout.times[i], contour_levels=np.arange(-40, 50, 5),
                            cbar_label=f"({unit_list['tempc']})", cmap=colormaps["t2m"], contourf_levels=np.arange(-40, 48, 3), cbar_ticks=np.arange(-40, 48, 3), contour_colors="white", data_source="ITU AMT Model")
            t2m.add_logo(logo_path=logo_path, coords=coords)
            t2m.save_plot(path=os.path.join(figures_path, domain, "t2m", f"t2m_{i}"))
            t2m.close_plot()

            wind10 = AMTPlotter(lons=wrfout.lons, lats=wrfout.lats, contourf_data=wrfout.ws10_ms[i], cproj=cproj, title=f"GFS Init {wrfout.init_date} UTC\n"+titles["wind10"], time=wrfout.times[i], 
                        u=wrfout.u10[i], v=wrfout.v10[i], barb_gap=6, barb_length=5, contourf_levels=np.arange(0, 32, 2), cmap=colormaps["wind10"], cbar_extend="max", cbar_label=f"({unit_list['wind_ms']})", cbar_ticks=np.arange(0, 32, 2), data_source="ITU AMT Model")
            wind10.add_logo(logo_path=logo_path, coords=coords)
            wind10.save_plot(path=os.path.join(figures_path, domain, "wind10", f"wind10_{i}"))
            wind10.close_plot()

            t2depression = AMTPlotter(lons=wrfout.lons, lats=wrfout.lats, contourf_data=wrfout.t2_depression[i], contour_data=wrfout.slp[i], cproj=cproj, title=f"GFS Init {wrfout.init_date} UTC\n"+titles["t2_depression"], time=wrfout.times[i], contour_levels=np.arange(800, 1200, 5),
                            cbar_label=f"({unit_list['tempc']})", cmap=colormaps["t2depression"], contourf_levels=np.arange(-3, 30, 3), cbar_ticks=np.arange(-3, 30, 3), contour_colors="white", cbar_extend="max",
                            u=wrfout.u10[i], v=wrfout.v10[i], barb_gap=6, barb_length=5, data_source="ITU AMT Model")
            t2depression.add_logo(logo_path=logo_path, coords=coords)
            t2depression.save_plot(path=os.path.join(figures_path, domain, "t2_depression", f"t2_depression_{i}"))
            t2depression.close_plot()

            tsk = AMTPlotter(lons=wrfout.lons, lats=wrfout.lats, contourf_data=wrfout.tsk[i], contour_data=wrfout.slp[i], cproj=cproj, title=f"GFS Init {wrfout.init_date} UTC\n"+titles["tsk"], time=wrfout.times[i], contour_levels=np.arange(800, 1200, 5),
                            cbar_label=f"({unit_list['tempc']})", cmap=colormaps["t2m"], cbar_ticks=np.arange(-20, 70, 3), contourf_levels=np.arange(-20, 70, 3), data_source="ITU AMT Model")
            tsk.add_logo(logo_path=logo_path, coords=coords)
            tsk.save_plot(path=os.path.join(figures_path, domain, "tsk", f"tsk_{i}")) 
            tsk.close_plot() 

            dbz = AMTPlotter(lons=wrfout.lons, lats=wrfout.lats, contourf_data=wrfout.dbz[i], cproj=cproj, title=f"GFS Init {wrfout.init_date} UTC\n"+titles["dbz"], time=wrfout.times[i], 
                            cbar_label=f"({unit_list['dbz']})", cmap=colormaps["tigris"], cbar_ticks=np.arange(0, 80, 5),  contourf_levels=np.arange(0, 80, 5), cbar_extend="max", data_source="ITU AMT Model")
            dbz.add_logo(logo_path=logo_path, coords=coords)
            dbz.save_plot(path=os.path.join(figures_path, domain, "dbz", f"dbz_{i}"))
            dbz.close_plot()

def prepare_wrfout(wrfout_name: str):
    wrfout =  WRFParser(ds=Dataset(os.path.join(os.getcwd(), wrfout_name)))
    domain = wrfout_name[7:10]

    return wrfout, domain

def main():
    pattern = "wrfout_d??_"
    wrf_files = glob.glob(pattern + "*")
    logo_path = "C://Users//gkbrk//Desktop//son//amtlogo.png"

    delete_png_files(os.path.join(os.getcwd(), "wrf_output_maps"))
    create_folders() 

    for i in range(len(wrf_files)):
        wrfout, domain = prepare_wrfout(wrf_files[i])
        make_plots(domain=domain, wrfout=wrfout, figures_path=os.path.join(os.getcwd(), "wrf_output_maps"), logo_path=logo_path)

if __name__ == "__main__":
    start = datetime.now()
    main()
    stop = datetime.now()
    print(f"Duration: {stop - start}")

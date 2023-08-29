# #!/usr/bin/env python3

# import logging
# import os
# import pickle
from pathlib import Path
# from typing import Optional, Union

# import numpy as np
# import pandas as pd
# import toml
# from oxfel.longlist import make_default_longlist

# # from esme.analysis import (
# #     # BetaScan,
# #     # DispersionScan,
# #     # OpticalConfig,
# #     # SliceEnergySpreadMeasurement,
# #     # TDSScan,
# # )

# # from esme.calibration import TDSCalibrator, TrivialTDSCalibrator
# # from esme.channels import I1D_SCREEN_ADDRESS
# # from esme.dispersion import (
# #     DispersionMeasurer,
# #     DispersionScanConfiguration,
# #     QuadrupoleSetting,
# # )
# # from esme.measurement import (  # I1EnergySpreadMeasuringMachine,; I1EnergySpreadMeasuringMachineReplayer
# #     DataTaker,
# #     MeasurementRunner,
# #     ScanType,
# #     SetpointMachineSnapshots,
# # )
# # from esme.tds import TDSScanConfiguration

# # from esme.lattice import make_dummy_lookup_sequence


# LOG = logging.getLogger(__name__)


# class MissingMetadataInFileNameError(RuntimeError):
#     pass


# class MalformedESMEConfigFile(RuntimeError):
#     pass


# def _optics_config_from_dict(config: dict) -> OpticalConfig:
#     tds = config["optics"]["tds"]
#     screen_betax = config["optics"]["screen"]["betx"]

#     return OpticalConfig(
#         tds_bety=tds["bety"],
#         tds_alfy=tds["alfy"],
#         ocr_betx=screen_betax,
#     )


# def optics_config_from_toml(tds_name, config: dict) -> OpticalConfig:
#     tds = config[tds_name]["tds"]["optics"]
#     screen_betax = config["optics"]["screen"]["betx"]

#     oconf = OpticalConfig(
#         tds_bety=tds["bety"],
#         tds_alfy=tds["alfy"],
#         ocr_betx=screen_betax,
#     )

#     return oconf


# # def _files_from_config(config, scan_name) -> list[Path]:
# #     try:
# #         basepath = Path(config["data"]["basepath"])
# #     except KeyError:
# #         basepath = Path(".")

# #     fnames = config["data"][scan_name]["fnames"]
# #     paths = [basepath / f for f in fnames]
# #     from IPython import embed; embed()
# #     return paths


# def scan_files_from_toml(tom: Union[os.PathLike, dict]) -> tuple:
#     try:
#         tom = toml.load(tom)
#     except TypeError:
#         pass

#     dscan_paths = _files_from_config(tom, "dscan")
#     tscan_paths = _files_from_config(tom, "tscan")

#     try:
#         bscan_paths = _files_from_config(tom, "bscan")
#     except KeyError:
#         bscan_paths = None

#     return dscan_paths, tscan_paths, bscan_paths


# def title_from_toml(tom: Union[os.PathLike, dict]) -> tuple:
#     try:
#         tom = toml.load(tom)
#     except TypeError:
#         pass

#     try:
#         return tom["title"]
#     except KeyError:
#         return ""


# # def load_config(fname: os.PathLike) -> SliceEnergySpreadMeasurement:
# #     config = toml.load(fname)

# #     oconfig = _optics_config_from_dict(config)

# #     try:
# #         calib = config["optics"]["tds"]["calibration"]
# #     except KeyError:
# #         raise MalformedESMEConfigFile("Missing calibration information")

# #     try:
# #         percentages = calib["percentages"]
# #     except KeyError:
# #         raise MalformedESMEConfigFile("TDS % info is missing from esme file")

# #     if voltages := calib.get("voltages"):
# #         calibrator = TrivialTDSCalibrator(percentages, voltages)
# #     else:
# #         # This is because I do not get snapshots with the calibration
# #         # data, so I have to get a similar one from the data taken.
# #         # In the future this won't really be necessary as I can take
# #         # snapshots in the calibration stage.

# #         try:
# #             dispersion_setpoint = calib["dispersion_setpoint"]
# #         except KeyError:
# #             raise MalformedESMEConfigFile(
# #                 "Dispersion at which TDS was calibrated is"
# #                 " missing from esme run file"
# #             )

# #         tds_slopes = calib["tds_slopes"]
# #         tds_slopes_units = calib["tds_slope_units"]
# #         calibrator = TDSCalibrator(
# #             percentages,
# #             tds_slopes,
# #             dispersion_setpoint,
# #             tds_slope_units=tds_slopes_units,
# #         )

# #     dsnapshots, tsnapshots, bsnapshots = load_pickled_snapshots(config["data"])

# #     dscan = DispersionScan(
# #         dsnapshots,
# #         calibrator=calibrator,
# #     )

# #     tscan = TDSScan(
# #         tsnapshots,
# #         calibrator=calibrator,
# #     )

# #     bscan = None
# #     if bsnapshots is not None:
# #         bscan = BetaScan(
# #             bsnapshots,
# #             calibrator=calibrator,
# #         )

# #     return SliceEnergySpreadMeasurement(dscan, tscan, oconfig, bscan=bscan)


# def setpoint_snapshots_from_pcls(pcl_files):
#     result = []
#     for path in pcl_files:
#         with path.open("rb") as f:
#             result.append(pickle.load(f))
#     return result


# def make_measurement_runner(
#     fconfig, machine_area, outdir="./", measure_dispersion=False, replay_file=None
# ):
#     LOG.debug(f"Making MeasurementRunner instance from config file: {fconfig}")

#     # if measure_dispersion is not None:
#     #     measure_dispersion = make_dispersion_measurer(fconfig)

#     measure_dispersion = None

#     outdir = Path(outdir)
#     outdir.mkdir(exist_ok=True, parents=True)

#     if machine_area == "i1":
#         tscan_config = i1_tscan_config_from_scan_config_file(fconfig)
#         dscan_config = i1_dscan_config_from_scan_config_file(fconfig)

#         if replay_file:
#             machine = I1EnergySpreadMeasuringMachineReplayer(outdir, replay_file)
#         else:
#             # tds = I1TDS()
#             machine = I1EnergySpreadMeasuringMachine(outdir)
#     elif machine_area == "b2":
#         # tds = B2TDS()
#         machine = B2DEnergySpreadMeasuringMachine(outdir)
#         tscan_config = b2_tscan_config_from_scan_config_file(fconfig)
#         dscan_config = b2_dscan_config_from_scan_config_file(fconfig)
#     else:
#         raise ValueError("Unknown machine_area string:", machine_area)

#     return MeasurementRunner(
#         dscan_config=dscan_config,
#         tds_config=tscan_config,
#         outdir=outdir,
#         dispersion_measurer=measure_dispersion,
#         machine=machine,
#     )


# def make_data_taker(
#     fconfig, machine_area, outdir="./", measure_dispersion=False, replay_file=None
# ):
#     LOG.debug(f"Making MeasurementRunner instance from config file: {fconfig}")

#     # if measure_is not None:
#     #     measure_dispersion = make_dispersion_measurer(fconfig)

#     measure_dispersion = None

#     if machine_area == "i1":
#         tscan_config = i1_tscan_config_from_scan_config_file(fconfig)
#         dscan_config = i1_dscan_config_from_scan_config_file(fconfig)

#         if replay_file:
#             machine = I1EnergySpreadMeasuringMachineReplayer(outdir, replay_file)
#         else:
#             # tds = I1TDS()
#             machine = I1EnergySpreadMeasuringMachine(outdir)
#     elif machine_area == "b2":
#         # tds = B2TDS()
#         machine = B2DEnergySpreadMeasuringMachine(outdir)
#         tscan_config = b2_tscan_config_from_scan_config_file(fconfig)
#         dscan_config = b2_dscan_config_from_scan_config_file(fconfig)
#     else:
#         raise ValueError("Unknown machine_area string:", machine_area)

#     return DataTaker(
#         dscan_config=dscan_config, tds_config=tscan_config, machine=machine
#     )


# def get_config_sample_sizes(fconfig):
#     config = toml.load(fconfig)
#     bg_shots = config["measurement"]["nbackground"]
#     beam_shots = config["measurement"]["nbeam"]
#     return bg_shots, beam_shots


# def make_dispersion_measurer(fconfig):
#     config = toml.load(fconfig)
#     confd = config["dispersion"]
#     a1_voltages = np.linspace(
#         confd["a1_voltage_min"], confd["a1_voltage_max"], num=confd["a1_npoints"]
#     )
#     return DispersionMeasurer(a1_voltages, I1EnergySpreadMeasuringMachine("./"))


def find_scan_config(fconfig: Path, default_name):
    if fconfig.exists():
        return fconfig
    else:
        return Path(default_name)


# def rm_pcl(fpcl: os.PathLike, dry_run=True):
#     """Delete a .pcl snapshots file and all images it refers to."""
#     fpcl = Path(fpcl)
#     pdir = fpcl.resolve().parent

#     df = pd.read_pickle(fpcl)

#     image_file_names = df[I1_DUMP_SCREEN_ADDRESS]

#     for image_name in image_file_names:
#         image_path = pdir / image_name
#         if dry_run:
#             print(f"would rm '{image_path}'")
#         else:
#             try:
#                 image_path.unlink()
#             except FileNotFoundError as exc:
#                 LOG.warning(f"{exc.strerror}: {image_path}")

#     if dry_run:
#         print(f"would rm {fpcl}")
#     else:
#         fpcl.unlink()


# def rm_ims_from_pcl(fpcl: os.PathLike, im_names, dry_run=True):
#     fpcl = Path(fpcl)
#     pdir = fpcl.resolve().parent

#     with fpcl.open("rb") as f:
#         setpoint = pickle.load(f)

#     paths = setpoint.snapshots[I1D_SCREEN_ADDRESS]
#     mask = []
#     to_delete = []
#     for path in paths:
#         png_fname = Path(path).with_suffix(".png")
#         pcl_fname = Path(path).with_suffix(".pcl")
#         png_name = png_fname.name
#         pcl_name = pcl_fname.name
#         mask.append(png_name in im_names or pcl_name in im_names)
#         if mask[-1]:
#             to_delete.extend([pdir / png_fname, pdir / pcl_fname])

#     new_df = setpoint.snapshots[np.logical_not(mask)]
#     setpoint.snapshots = new_df

#     for fname in to_delete:
#         if dry_run:
#             print(f"Would delete: {fname}")
#         else:
#             fname.unlink()

#     if dry_run:
#         print(f"Would overrite {fpcl}")
#     else:
#         with fpcl.open("wb") as f:
#             pickle.dump(setpoint, f)


# def _loop_pcl_df_files(paths, scan_type):
#     for fdf in paths:
#         try:
#             snapshot = raw_df_pcl_to_setpoint_snapshots(fdf, scan_type)
#         except TypeError:
#             continue
#         else:
#             with fdf.open("wb") as f:
#                 pickle.dump(snapshot, f)


# def raw_df_pcl_to_setpoint_snapshots(fpcl, scan_type):
#     # try:
#     df = pd.read_pickle(fpcl)
#     # except EOFError:

#     if isinstance(df, SetpointMachineSnapshots):
#         LOG.info(f"{fpcl} is already a pickled SetpointMachineSnapshots instance")
#         raise TypeError("already a SetpointSnapshot")

#     # Measured dispersion
#     dx = df["MY_SCREEN_DX"].iloc[0]

#     try:
#         beta = df["MY_SCREEN_BETA"]
#     except KeyError:
#         beta = None
#     else:
#         beta = beta.iloc[0]
#         df = df.drop(columns="MY_SCREEN_BETA")

#     df = df.drop(columns="MY_SCREEN_DX")

#     dispersion_setpoints = np.array([0.6, 0.8, 1.0, 1.2])
#     isetpoint = np.abs(dispersion_setpoints - dx).argmin()

#     dispersion_setpoint = dispersion_setpoints[isetpoint]

#     return SetpointMachineSnapshots(
#         df,
#         scan_type,
#         dispersion_setpoint=dispersion_setpoint,
#         measured_dispersion=(dx, 0.0),
#         beta=beta,
#     )


# def toml_dfs_to_setpoint_snapshots(ftoml):
#     """This is for porting old pure dataframes (referred to in the
#     toml files) to the newer SetpointMachineSnapshots instances.

#     """
#     dscan_paths, tscan_paths, bscan_paths = scan_files_from_toml(ftoml)
#     if dscan_paths:
#         _loop_pcl_df_files(dscan_paths, ScanType.DISPERSION)
#     if tscan_paths:
#         _loop_pcl_df_files(tscan_paths, ScanType.TDS)
#     if bscan_paths:
#         _loop_pcl_df_files(bscan_paths, ScanType.BETA)


# def load_pickled_snapshots(
#     data_dict: dict,
# ) -> tuple[
#     list[SetpointMachineSnapshots],
#     list[SetpointMachineSnapshots],
#     Optional[list[SetpointMachineSnapshots]],
# ]:
#     """data_dict is of form coming
#     from toml file..."""
#     tscan = load_data_config_section(data_dict, "tscan")
#     dscan = load_data_config_section(data_dict, "dscan")
#     try:
#         bscan = load_data_config_section(data_dict, "bscan")
#     except KeyError:
#         bscan = None

#     return dscan, tscan, bscan


# def load_data_config_section(
#     data_dict: dict, scan_key: str
# ) -> list[SetpointMachineSnapshots]:
#     section = data_dict[scan_key]
#     if section.get("old_sergey_format", False):
#         return load_old_raw_df_snapshot_format(data_dict, scan_key)
#     return load_new_object_format(data_dict, scan_key)


# def load_new_object_format(data_dict, scan_key):
#     bad_images = data_dict.get("bad_images", [])
#     result = []
#     # This should be able to handle loading the old style format (metadata in the filename)
#     # And new style pickled format
#     basepath = Path(data_dict["basepath"])
#     paths = data_dict[scan_key]["fnames"]

#     for path in paths:
#         full_path = basepath / path
#         with full_path.open("rb") as f:
#             snapshots = pickle.load(f)

#         if not hasattr(snapshots, "screen_channel"):
#             snapshots.screen_channel = data_dict["screen_channel"]

#         snapshots.drop_bad_snapshots()
#         snapshots.drop_images(bad_images)
#         snapshots.resolve_image_path(full_path.parent)
#         result.append(snapshots)

#     return result


# def load_old_raw_df_snapshot_format(
#     data_dict: dict, scan_key: str
# ) -> list[SetpointMachineSnapshots]:
#     bad_images = data_dict.get("bad_images", [])

#     basepath = Path(data_dict["basepath"])
#     paths = data_dict[scan_key]["fnames"]
#     sps = data_dict[scan_key]["setpoint_dispersions"]

#     # Maybe we measured the dispersion at each point, but maybe not also.
#     try:
#         dispersions = data_dict[scan_key]["measured_dispersions"]
#     except:
#         dispersions = len(paths) * [None]
#     else:
#         dispersions = [tuple(d) for d in dispersions]

#     betas = data_dict[scan_key]["betas"]

#     result = []
#     for path, dispersion_sp, dispersion, beta in zip(paths, sps, dispersions, betas):
#         full_path = basepath / path
#         df = pd.read_pickle(full_path)
#         sn = SetpointMachineSnapshots(
#             snapshots=df,
#             scan_type=ScanType.from_name(scan_key),
#             dispersion_setpoint=dispersion_sp,
#             measured_dispersion=dispersion,
#             beta=beta,
#             screen_channel=data_dict["screen_channel"],
#         )
#         sn.resolve_image_path(full_path.parent)
#         result.append(sn)

#     return result


# def i1_dscan_config_from_scan_config_file(config_path: os.PathLike):
#     conf = toml.load(config_path)
#     return _dscan_config_from_scan_config(conf, "i1")


# def i1_tds_voltages_from_scan_config_file(config_path: os.PathLike):
#     conf = toml.load(config_path)
#     return conf["i1"]["tds"]["scan_voltages"]


# def i1_tds_amplitudes_from_scan_config_file(config_path: os.PathLike):
#     conf = toml.load(config_path)
#     return conf["i1"]["tds"]["scan_amplitudes"]


# def b2_tds_voltages_from_scan_config_file(config_path: os.PathLike):
#     conf = toml.load(config_path)
#     return conf["b2"]["tds"]["scan_voltages"]


# def b2_dscan_config_from_scan_config_file(config_path: os.PathLike):
#     conf = toml.load(config_path)
#     return _dscan_config_from_scan_config(conf, "b2")


# def _dscan_config_from_scan_config(
#     scan_config: dict, section
# ) -> DispersionScanConfiguration:
#     dscan = scan_config[section]["dscan"]

#     ref = dscan["reference_optics"]

#     try:
#         ref_k1s = ref["k1s"]
#     except KeyError:
#         ref_k1ls = ref["k1ls"]
#     else:
#         ref_k1ls = named_k1s_to_k1ls(ref["names"], ref_k1s)

#     reference_setting = QuadrupoleSetting(ref["names"], ref_k1ls, ref["dispersion"])

#     quads = dscan["quads"]
#     dscan_quad_names = quads["names"]

#     try:
#         quad_k1s = quads["k1s"]
#     except KeyError:
#         quad_k1ls = quads["k1ls"]
#     else:
#         quad_k1ls = named_k1s_to_k1ls(dscan_quad_names, quad_k1s)

#     scan_settings = []
#     for dispersion, scan_k1ls in zip(quads["dispersions"], quad_k1ls):
#         scan_settings.append(QuadrupoleSetting(dscan_quad_names, scan_k1ls, dispersion))

#     # from IPython import embed; embed()

#     tds_voltage = dscan["tds_voltage"]
#     return DispersionScanConfiguration(reference_setting, scan_settings, tds_voltage)


# def _tscan_config_from_scan_config_file(
#     key, config_path: os.PathLike
# ) -> TDSScanConfiguration:
#     conf = toml.load(config_path)
#     tscan = conf[key]["tscan"]
#     voltages = tscan["voltages"]
#     dispersion = tscan["dispersion_setpoint"]

#     dconf = _dscan_config_from_scan_config(conf, key)

#     assert tscan["dispersion_setpoint"] == dconf.reference_setting.dispersion

#     return TDSScanConfiguration(
#         voltages=tscan["voltages"], quad_setting=dconf.reference_setting
#     )

#     # return TDSScanConfiguration(
#     #     reference_amplitude=tds["reference_amplitude"],
#     #     scan_amplitudes=tds["scan_amplitudes"],
#     #     scan_dispersion=tds["scan_dispersion"])


# def i1_tscan_config_from_scan_config_file(config_path: os.PathLike):
#     return _tscan_config_from_scan_config_file("i1", config_path)


# def b2_tscan_config_from_scan_config_file(config_path: os.PathLike):
#     conf = toml.load(config_path)
#     return _tscan_config_from_scan_config_file("b2", config_path)


# def named_k1s_to_k1ls(names, k1s):
#     ll = make_default_longlist()
#     lengths = np.array(ll[names].LENGTH)
#     return lengths * k1s

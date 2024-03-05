"""Used for calibrating the measurement and also deriving measurement."""

import logging

# from oxfel.fel_track import Linac

# from esme.dispersion import QuadrupoleSetting

LOG = logging.getLogger(__name__)




# def _append_element_positions_and_names(df: pd.DataFrame, names: Iterable[str]):
#     cell = cell_to_injector_dump()
#     lengths = []
#     positions = []
#     for name in names:
#         try:
#             length = cell.element_attributes(name, "l")
#         except KeyError:
#             length = np.nan
#         try:
#             s = cell.element_s(name)
#         except KeyError:
#             s = np.nan

#         lengths.append(length)
#         positions.append(s)
#     return df.assign(length=lengths, s=positions)


# def injector_cell_from_snapshot(
#     snapshot: pd.Series, check=True, change_correctors=False
# ):
#     quad_mask = snapshot.index.str.contains(r"^Q(?:I?|L[NS])\.")
#     solenoid_mask = snapshot.index.str.contains(r"^SOL[AB]\.")
#     corrector_mask = snapshot.index.str.contains(r"^C[IK]?[XY]\.")
#     bends_mask = snapshot.index.str.contains(r"^B[BL]\.")
#     used_mask = quad_mask | solenoid_mask | corrector_mask | corrector_mask | bends_mask

#     if check:
#         xfel_mask = snapshot.index.str.startswith("XFEL.")
#         bpm_mask = snapshot.index.str.contains(r"^BPM[GAFRSCD]\.")
#         timestamp_mask = snapshot.index == "timestamp"
#         sextupole_mask = snapshot.index.str.contains(r"^SC\.")
#         other_kickers_mask = snapshot.index.str.contains(r"CB[LB]\.")
#         unused_mask = (
#             xfel_mask | bpm_mask | timestamp_mask | sextupole_mask | other_kickers_mask
#         )
#         my_added_metadata = snapshot.index.str.startswith("MY_")
#         the_rest_mask = ~(used_mask | unused_mask | my_added_metadata)
#         snapshot[the_rest_mask]

#         the_rest = snapshot[the_rest_mask]

#         expected = {"RF.23.I1", "BK.24.I1"}
#         if set(the_rest.index) != expected:
#             LOG.warning(
#                 f"Unexpected item in DF.  Expected: {expected}, got: {the_rest}"
#             )

#     from IPython import embed; embed()
#     cell 
#     cell = cell_to_injector_dump()

#     for element in cell:
#         if isinstance(element, Quadrupole):
#             # from IPython import embed; embed()
#             k1l = snapshot[element.id]
#             k1 = k1l * 1e-3 / element.l  # Convert to rad/m from mrad/m then to k1.
#             LOG.debug(
#                 f"setting quad strength from snapshot: {element.id}, {element.k1=} to {k1=}"
#             )
#             element.k1 = k1

#         if isinstance(element, (SBend, RBend)):
#             angle_mrad = snapshot[element.id]
#             angle = -angle_mrad * 1e-3
#             LOG.debug(
#                 f"setting dipole angle from snapshot: {element.id}, {element.angle=} to {angle=}"
#             )
#             element.angle = angle

#     # for quad_name, int_strength in snapshot[quad_mask].items():
#     #     try:
#     #         if quad_name.startswith("QLN."):
#     #             continue
#     #         if quad_name.startswith("QLS."):
#     #             continue
#     #         quad = cell[quad_name]
#     #     except KeyError:
#     #         LOG.debug("NOT setting quad strength from snapshot: %s", quad_name)
#     #     else:
#     #         k1 = int_strength * 1e-3 / quad.l  # Convert to rad/m from mrad/m then to k1.
#     #         LOG.debug(f"setting quad strength from snapshot: {quad_name=}, {quad.k1=} to {k1=}")
#     #         quad.k1 = k1

#     # dipole_mask = bends_mask
#     # if change_correctors:
#     #     dipole_mask |= corrector_mask
#     # for dipole_name, angle_mrad in snapshot[dipole_mask].items():
#     #     try:
#     #         dipole = cell[dipole_name]
#     #     except KeyError:
#     #         LOG.debug("NOT setting dipole angle from snapshot: %s", dipole_name)
#     #     else:
#     #         # I insert a negative sign here.  I guess convention is opposite?
#     #         angle = -angle_mrad * 1e-3
#     #         LOG.debug(f"setting dipole angle from snapshot: {dipole_name=}, {dipole.angle=} to {angle=}")
#     #         dipole.angle = angle

#     return cell


# def apply_quad_setting_to_lattice(lattice# : Linac
#                                   , qset# : QuadrupoleSetting
#                                   ):
#     LOG.debug(
#         "Applying quadrupole strengths from QuadrupoleSetting instance to OCELOT SectionLattice."
#     )
#     for section in lattice.sections:
#         sequence = section.lattice.sequence
#         for element in sequence:
#             if not isinstance(element, Quadrupole):
#                 continue
#             try:
#                 k1l = qset.k1l_from_name(element.id)
#             except ValueError:
#                 continue

#             LOG.debug(f"{element.id} k1 before: {element.k1}")
#             element.k1 = k1l / element.l
#             LOG.debug(f"{element.id} k1 after: {element.k1}")


# def make_to_i1d_lattice(twiss0, outdir="./"):
#     all_sections = [sections.G1, sections.A1, sections.AH1, sections.LH, sections.I1D]
#     return Linac(all_sections, twiss0, outdir=outdir)



"""L2 lattice section definition"""
from copy import deepcopy
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from ocelot.cpbd.beam import Twiss
    from ocelot.cpbd.elements import Cavity, Drift, Hcor, Marker, Monitor, Quadrupole, SBend, TDCavity, Vcor
    from ocelot.utils.section_track import SectionCell

tws = Twiss()
# tws.beta_x  = 6.669645808992628
# tws.beta_y  = 8.65178360432342
# tws.alpha_x = -0.7191959870340436
# tws.alpha_y = -1.3700291435722858
tws.beta_x = 7.622631562086844
tws.beta_y = 9.26531262686501
tws.alpha_x = -0.883149418027471
tws.alpha_y = -1.4181947820106364
tws.E = 0.7

tws = Twiss()
tws.beta_x = 7.48319123649
tws.beta_y = 8.79758475319
tws.alpha_x = -0.721071482665
tws.alpha_y = -1.31392272539
tws.E = 0.7
tws.s = 229.3007540000002

# Drifts
d_1 = Drift(l=0.75275, eid='D_1')
d_2 = Drift(l=1.981646, eid='D_2')
d_3 = Drift(l=0.11165, eid='D_3')
d_4 = Drift(l=0.15, eid='D_4')
d_5 = Drift(l=0.15165, eid='D_5')
d_6 = Drift(l=0.12165, eid='D_6')
d_8 = Drift(l=0.14165, eid='D_8')
d_9 = Drift(l=5.11825, eid='D_9')
d_10 = Drift(l=0.3459, eid='D_10')
d_17 = Drift(l=0.2475, eid='D_17')
d_18 = Drift(l=0.0432, eid='D_18')
d_19 = Drift(l=0.085, eid='D_19')
d_20 = Drift(l=0.6795, eid='D_20')
d_141 = Drift(l=5.25378, eid='D_141')
d_142 = Drift(l=0.248, eid='D_142')
d_143 = Drift(l=0.15277, eid='D_143')
d_144 = Drift(l=0.13165, eid='D_144')
d_145 = Drift(l=0.62763, eid='D_145')
d_146 = Drift(l=0.13567, eid='D_146')
d_147 = Drift(l=1.77888, eid='D_147')
d_149 = Drift(l=0.10065, eid='D_149')
d_150 = Drift(l=0.56265, eid='D_150')
d_151 = Drift(l=0.08163, eid='D_151')
d_152 = Drift(l=0.3439, eid='D_152')
d_153 = Drift(l=0.3505, eid='D_153')
d_154 = Drift(l=0.655693, eid='D_154')
d_155 = Drift(l=8.507518, eid='D_155')
d_156 = Drift(l=7.2e-05, eid='D_156')
d_157 = Drift(l=0.865, eid='D_157')
d_158 = Drift(l=0.31, eid='D_158')
d_159 = Drift(l=0.325073, eid='D_159')
d_161 = Drift(l=8.507446, eid='D_161')
d_163 = Drift(l=0.1, eid='D_163')
d_164 = Drift(l=0.47848, eid='D_164')
d_165 = Drift(l=0.4028, eid='D_165')
d_166 = Drift(l=0.15735, eid='D_166')
d_167 = Drift(l=0.14465, eid='D_167')
d_168 = Drift(l=0.15002, eid='D_168')
d_169 = Drift(l=0.93165, eid='D_169')
d_170 = Drift(l=0.78165, eid='D_170')
d_172 = Drift(l=0.17888, eid='D_172')
d_175 = Drift(l=6.39998, eid='D_175')
d_176 = Drift(l=0.17165, eid='D_176')
d_177 = Drift(l=0.19165, eid='D_177')
d_178 = Drift(l=0.3899, eid='D_178')
d_179 = Drift(l=0.247, eid='D_179')
d_180 = Drift(l=0.43477, eid='D_180')
d_181 = Drift(l=0.38593, eid='D_181')
d_182 = Drift(l=0.4, eid='D_182')
d_183 = Drift(l=0.17835, eid='D_183')
d_184 = Drift(l=0.09065, eid='D_184')
d_185 = Drift(l=0.3223, eid='D_185')
d_186 = Drift(l=1.90937, eid='D_186')
d_188 = Drift(l=2.23165, eid='D_188')
d_189 = Drift(l=1.78153, eid='D_189')
d_190 = Drift(l=1.629, eid='D_190')
d_193 = Drift(l=3.17888, eid='D_193')
d_196 = Drift(l=0.90012, eid='D_196')
d_197 = Drift(l=0.62888, eid='D_197')
d_200 = Drift(l=1.54988, eid='D_200')
d_201 = Drift(l=1.655, eid='D_201')
d_202 = Drift(l=0.15285, eid='D_202')
d_203 = Drift(l=1.75545, eid='D_203')
d_204 = Drift(l=0.3501, eid='D_204')
d_205 = Drift(l=1.1789, eid='D_205')
d_213 = Drift(l=1.079, eid='D_213')
d_214 = Drift(l=0.15275, eid='D_214')
d_215 = Drift(l=0.83167, eid='D_215')
d_216 = Drift(l=0.33165, eid='D_216')
d_218 = Drift(l=0.35, eid='D_218')
d_221 = Drift(l=0.53165, eid='D_221')

# Quadrupoles
qd_231_b1 = Quadrupole(l=0.2367, k1=1.655730773130545, eid='QD.231.B1')
qd_232_b1 = Quadrupole(l=0.2367, k1=-1.2501718111533586, eid='QD.232.B1')
qd_233_b1 = Quadrupole(l=0.2367, k1=-0.4397248284748627, eid='QD.233.B1')
q_249_l2 = Quadrupole(l=0.2136, k1=0.42491290683520594, eid='Q.249.L2')
q_261_l2 = Quadrupole(l=0.2136, k1=-0.39384852715355806, eid='Q.261.L2')
q_273_l2 = Quadrupole(l=0.2136, k1=0.387775731741573, eid='Q.273.L2')
q_285_l2 = Quadrupole(l=0.2136, k1=-0.42475373689138574, eid='Q.285.L2')
q_297_l2 = Quadrupole(l=0.2136, k1=0.488189811329588, eid='Q.297.L2')
q_309_l2 = Quadrupole(l=0.2136, k1=-0.6058097570224719, eid='Q.309.L2')
q_321_l2 = Quadrupole(l=0.2136, k1=0.488189811329588, eid='Q.321.L2')
q_333_l2 = Quadrupole(l=0.2136, k1=-0.6058097570224719, eid='Q.333.L2')
q_345_l2 = Quadrupole(l=0.2136, k1=0.6525829808052434, eid='Q.345.L2')
q_357_l2 = Quadrupole(l=0.2136, k1=-0.38908541666666663, eid='Q.357.L2')
q_369_l2 = Quadrupole(l=0.2136, k1=0.43128398220973785, eid='Q.369.L2')
q_381_l2 = Quadrupole(l=0.2136, k1=-0.38391488764044945, eid='Q.381.L2')
qd_387_b2 = Quadrupole(l=0.2367, k1=0.3351733848753697, eid='QD.387.B2')
qd_388_b2 = Quadrupole(l=0.2367, k1=0.3559964321926489, eid='QD.388.B2')
qd_391_b2 = Quadrupole(l=0.2367, k1=-0.7255245525982257, eid='QD.391.B2')
qd_392_b2 = Quadrupole(l=0.2367, k1=0.19699609040980143, eid='QD.392.B2')
qd_415_b2 = Quadrupole(l=0.2367, k1=0.19466236079425434, eid='QD.415.B2')
qd_417_b2 = Quadrupole(l=0.2367, k1=-0.7490104947190536, eid='QD.417.B2')
qd_418_b2 = Quadrupole(l=0.2367, k1=0.6364796899028307, eid='QD.418.B2')
qd_425_b2 = Quadrupole(l=0.2367, k1=-1.2989425141529363, eid='QD.425.B2')
qd_427_b2 = Quadrupole(l=0.2367, k1=0.943028920574567, eid='QD.427.B2')
qd_431_b2 = Quadrupole(l=0.2367, k1=0.435183027460921, eid='QD.431.B2')
qd_434_b2 = Quadrupole(l=0.2367, k1=-0.5278581909590199, eid='QD.434.B2')
qd_437_b2 = Quadrupole(l=0.2367, k1=0.4055492834811999, eid='QD.437.B2')
qd_440_b2 = Quadrupole(l=0.2367, k1=-0.6685246721588509, eid='QD.440.B2')
qd_444_b2 = Quadrupole(l=0.2367, k1=-0.4582186615969582, eid='QD.444.B2')
qd_448_b2 = Quadrupole(l=0.2367, k1=0.8960955487959443, eid='QD.448.B2')
qd_452_b2 = Quadrupole(l=0.2367, k1=-1.2632843840304184, eid='QD.452.B2')
qd_456_b2 = Quadrupole(l=0.2367, k1=0.8960955487959443, eid='QD.456.B2')
qd_459_b2 = Quadrupole(l=0.2367, k1=-1.2632843840304184, eid='QD.459.B2')
qd_463_b2 = Quadrupole(l=0.2367, k1=-0.5696070975918884, eid='QD.463.B2')
qd_464_b2 = Quadrupole(l=0.2367, k1=1.298267850021124, eid='QD.464.B2')
qd_465_b2 = Quadrupole(l=0.2367, k1=-0.2468610054921842, eid='QD.465.B2')

# SBends
bb_393_b2 = SBend(l=0.5, angle=0.0411897704, e2=0.0411897704, tilt=1.570796327, eid='BB.393.B2')
bb_402_b2 = SBend(l=0.5, angle=-0.0411897704, e1=-0.0411897704, tilt=1.570796327, eid='BB.402.B2')
bb_404_b2 = SBend(l=0.5, angle=-0.0411897704, e2=-0.0411897704, tilt=1.570796327, eid='BB.404.B2')
bb_413_b2 = SBend(l=0.5, angle=0.0411897704, e1=0.0411897704, tilt=1.570796327, eid='BB.413.B2')

# Hcors
ccx_232_b1 = Hcor(l=0.1, eid='CCX.232.B1')
cx_249_l2 = Hcor(eid='CX.249.L2')
cx_273_l2 = Hcor(eid='CX.273.L2')
cx_297_l2 = Hcor(eid='CX.297.L2')
cx_321_l2 = Hcor(eid='CX.321.L2')
cx_345_l2 = Hcor(eid='CX.345.L2')
cx_369_l2 = Hcor(eid='CX.369.L2')
ccx_388_b2 = Hcor(l=0.1, eid='CCX.388.B2')
ccx_392_b2 = Hcor(l=0.1, eid='CCX.392.B2')
ccx_415_b2 = Hcor(l=0.1, eid='CCX.415.B2')
ccx_418_b2 = Hcor(l=0.1, eid='CCX.418.B2')
ccx_425_b2 = Hcor(l=0.1, eid='CCX.425.B2')
ccx_431_b2 = Hcor(l=0.1, eid='CCX.431.B2')
ccx_441_b2 = Hcor(l=0.1, eid='CCX.441.B2')
ccx_447_b2 = Hcor(l=0.1, eid='CCX.447.B2')
ccx_454_b2 = Hcor(l=0.1, eid='CCX.454.B2')
ccx_456_b2 = Hcor(l=0.1, eid='CCX.456.B2')
ccx_464_b2 = Hcor(l=0.1, eid='CCX.464.B2')
ccx_465_b2 = Hcor(l=0.1, eid='CCX.465.B2')

# Vcors
cbb_229_b1d = Vcor(eid='CBB.229.B1D')
ccy_232_b1 = Vcor(l=0.1, eid='CCY.232.B1')
cy_261_l2 = Vcor(eid='CY.261.L2')
cy_285_l2 = Vcor(eid='CY.285.L2')
cy_309_l2 = Vcor(eid='CY.309.L2')
cy_333_l2 = Vcor(eid='CY.333.L2')
cy_357_l2 = Vcor(eid='CY.357.L2')
cy_381_l2 = Vcor(eid='CY.381.L2')
ccy_387_b2 = Vcor(l=0.1, eid='CCY.387.B2')
ccy_391_b2 = Vcor(l=0.1, eid='CCY.391.B2')
cbb_403_b2 = Vcor(eid='CBB.403.B2')
cbb_405_b2 = Vcor(eid='CBB.405.B2')
cbb_414_b2 = Vcor(eid='CBB.414.B2')
ccy_416_b2 = Vcor(l=0.1, eid='CCY.416.B2')
ccy_418_b2 = Vcor(l=0.1, eid='CCY.418.B2')
ccy_426_b2 = Vcor(l=0.1, eid='CCY.426.B2')
ccy_434_b2 = Vcor(l=0.1, eid='CCY.434.B2')
ccy_448_b2 = Vcor(l=0.1, eid='CCY.448.B2')
ccy_460_b2 = Vcor(l=0.1, eid='CCY.460.B2')
ccy_464_b2 = Vcor(l=0.1, eid='CCY.464.B2')

# Cavitys
DIP_KICK = 1
c_a3_1_1_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A3.1.1.L2',
)
c_a3_1_2_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A3.1.2.L2',
)
c_a3_1_3_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A3.1.3.L2',
)
c_a3_1_4_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A3.1.4.L2',
)
c_a3_1_5_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A3.1.5.L2',
)
c_a3_1_6_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A3.1.6.L2',
)
c_a3_1_7_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A3.1.7.L2',
)
c_a3_1_8_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A3.1.8.L2',
)
c_a3_2_1_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A3.2.1.L2',
)
c_a3_2_2_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A3.2.2.L2',
)
c_a3_2_3_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A3.2.3.L2',
)
c_a3_2_4_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A3.2.4.L2',
)
c_a3_2_5_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A3.2.5.L2',
)
c_a3_2_6_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A3.2.6.L2',
)
c_a3_2_7_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A3.2.7.L2',
)
c_a3_2_8_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A3.2.8.L2',
)
c_a3_3_1_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A3.3.1.L2',
)
c_a3_3_2_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A3.3.2.L2',
)
c_a3_3_3_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A3.3.3.L2',
)
c_a3_3_4_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A3.3.4.L2',
)
c_a3_3_5_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A3.3.5.L2',
)
c_a3_3_6_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A3.3.6.L2',
)
c_a3_3_7_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A3.3.7.L2',
)
c_a3_3_8_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A3.3.8.L2',
)
c_a3_4_1_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A3.4.1.L2',
)
c_a3_4_2_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A3.4.2.L2',
)
c_a3_4_3_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A3.4.3.L2',
)
c_a3_4_4_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A3.4.4.L2',
)
c_a3_4_5_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A3.4.5.L2',
)
c_a3_4_6_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A3.4.6.L2',
)
c_a3_4_7_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A3.4.7.L2',
)
c_a3_4_8_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A3.4.8.L2',
)
c_a4_1_1_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A4.1.1.L2',
)
c_a4_1_2_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A4.1.2.L2',
)
c_a4_1_3_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A4.1.3.L2',
)
c_a4_1_4_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A4.1.4.L2',
)
c_a4_1_5_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A4.1.5.L2',
)
c_a4_1_6_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A4.1.6.L2',
)
c_a4_1_7_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A4.1.7.L2',
)
c_a4_1_8_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A4.1.8.L2',
)
c_a4_2_1_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A4.2.1.L2',
)
c_a4_2_2_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A4.2.2.L2',
)
c_a4_2_3_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A4.2.3.L2',
)
c_a4_2_4_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A4.2.4.L2',
)
c_a4_2_5_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A4.2.5.L2',
)
c_a4_2_6_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A4.2.6.L2',
)
c_a4_2_7_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A4.2.7.L2',
)
c_a4_2_8_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A4.2.8.L2',
)
c_a4_3_1_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A4.3.1.L2',
)
c_a4_3_2_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A4.3.2.L2',
)
c_a4_3_3_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A4.3.3.L2',
)
c_a4_3_4_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A4.3.4.L2',
)
c_a4_3_5_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A4.3.5.L2',
)
c_a4_3_6_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A4.3.6.L2',
)
c_a4_3_7_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A4.3.7.L2',
)
c_a4_3_8_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A4.3.8.L2',
)
c_a4_4_1_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A4.4.1.L2',
)
c_a4_4_2_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A4.4.2.L2',
)
c_a4_4_3_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A4.4.3.L2',
)
c_a4_4_4_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A4.4.4.L2',
)
c_a4_4_5_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A4.4.5.L2',
)
c_a4_4_6_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A4.4.6.L2',
)
c_a4_4_7_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A4.4.7.L2',
)
c_a4_4_8_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A4.4.8.L2',
)
c_a5_1_1_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A5.1.1.L2',
)
c_a5_1_2_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A5.1.2.L2',
)
c_a5_1_3_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A5.1.3.L2',
)
c_a5_1_4_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A5.1.4.L2',
)
c_a5_1_5_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A5.1.5.L2',
)
c_a5_1_6_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A5.1.6.L2',
)
c_a5_1_7_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A5.1.7.L2',
)
c_a5_1_8_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A5.1.8.L2',
)
c_a5_2_1_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A5.2.1.L2',
)
c_a5_2_2_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A5.2.2.L2',
)
c_a5_2_3_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A5.2.3.L2',
)
c_a5_2_4_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A5.2.4.L2',
)
c_a5_2_5_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A5.2.5.L2',
)
c_a5_2_6_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A5.2.6.L2',
)
c_a5_2_7_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A5.2.7.L2',
)
c_a5_2_8_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A5.2.8.L2',
)
c_a5_3_1_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A5.3.1.L2',
)
c_a5_3_2_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A5.3.2.L2',
)
c_a5_3_3_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A5.3.3.L2',
)
c_a5_3_4_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A5.3.4.L2',
)
c_a5_3_5_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A5.3.5.L2',
)
c_a5_3_6_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A5.3.6.L2',
)
c_a5_3_7_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A5.3.7.L2',
)
c_a5_3_8_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A5.3.8.L2',
)
c_a5_4_1_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A5.4.1.L2',
)
c_a5_4_2_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A5.4.2.L2',
)
c_a5_4_3_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A5.4.3.L2',
)
c_a5_4_4_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A5.4.4.L2',
)
c_a5_4_5_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A5.4.5.L2',
)
c_a5_4_6_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A5.4.6.L2',
)
c_a5_4_7_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A5.4.7.L2',
)
c_a5_4_8_l2 = Cavity(
    l=1.0377,
    v=0.01770833333,
    freq=1300000000.0,
    vx_up=(-5.6813e-05 + 1.0751e-05j) * DIP_KICK,
    vy_up=(-4.1091e-05 + 5.739e-07j) * DIP_KICK,
    vxx_up=(0.00099943 - 0.00081401j),
    vxy_up=(0.0034065 - 0.0004146j),
    vx_down=(-2.4014e-05 + 1.2492e-05j) * DIP_KICK,
    vy_down=(3.6481e-05 + 7.9888e-06j) * DIP_KICK,
    vxx_down=(-0.004057 - 0.0001369j),
    vxy_down=(0.0029243 - 1.2891e-05j),
    eid='C.A5.4.8.L2',
)
tdsb_428_b2 = TDCavity(l=1.5, freq=2800000000.0, eid='TDSB.428.B2')
tdsb_430_b2 = TDCavity(l=1.5, freq=2800000000.0, eid='TDSB.430.B2')

# Monitors
bpma_233_b1 = Monitor(eid='BPMA.233.B1')
bpmc_249_l2 = Monitor(eid='BPMC.249.L2')
bpmc_261_l2 = Monitor(eid='BPMC.261.L2')
bpmr_273_l2 = Monitor(eid='BPMR.273.L2')
bpmr_285_l2 = Monitor(eid='BPMR.285.L2')
bpmc_297_l2 = Monitor(eid='BPMC.297.L2')
bpmr_309_l2 = Monitor(eid='BPMR.309.L2')
bpmc_321_l2 = Monitor(eid='BPMC.321.L2')
bpmr_333_l2 = Monitor(eid='BPMR.333.L2')
bpmc_345_l2 = Monitor(eid='BPMC.345.L2')
bpmc_357_l2 = Monitor(eid='BPMC.357.L2')
bpmr_369_l2 = Monitor(eid='BPMR.369.L2')
bpmr_381_l2 = Monitor(eid='BPMR.381.L2')
bpma_387_b2 = Monitor(eid='BPMA.387.B2')
bpma_390_b2 = Monitor(eid='BPMA.390.B2')
bpmf_393_b2 = Monitor(eid='BPMF.393.B2')
bpms_404_b2 = Monitor(eid='BPMS.404.B2')
bpmf_414_b2 = Monitor(eid='BPMF.414.B2')
bpma_418_b2 = Monitor(eid='BPMA.418.B2')
bpma_426_b2 = Monitor(eid='BPMA.426.B2')
bpma_432_b2 = Monitor(eid='BPMA.432.B2')
bpma_440_b2 = Monitor(eid='BPMA.440.B2')
bpma_444_b2 = Monitor(eid='BPMA.444.B2')
bpma_448_b2 = Monitor(eid='BPMA.448.B2')
bpma_452_b2 = Monitor(eid='BPMA.452.B2')
bpma_455_b2 = Monitor(eid='BPMA.455.B2')
bpma_459_b2 = Monitor(eid='BPMA.459.B2')
bpma_462_b2 = Monitor(eid='BPMA.462.B2')
bpma_465_b2 = Monitor(eid='BPMA.465.B2')

# Markers
tora_232_b1 = Marker(eid='TORA.232.B1')
tora_387_b2 = Marker(eid='TORA.387.B2')
otra_392_b2 = Marker(eid='OTRA.392.B2')
otrs_404_b2 = Marker(eid='OTRS.404.B2')
match_414_b2 = Marker(eid='MATCH.414.B2')
tora_415_b2 = Marker(eid='TORA.415.B2')
engrd_419_b2 = Marker(eid='ENGRD.419.B2')
otra_426_b2 = Marker(eid='OTRA.426.B2')
match_428_b2 = Marker(eid='MATCH.428.B2')
otra_438_b2 = Marker(eid='OTRA.438.B2')
match_446_b2 = Marker(eid='MATCH.446.B2')
otra_446_b2 = Marker(eid='OTRA.446.B2')
otrb_450_b2 = Marker(eid='OTRB.450.B2')
otrb_454_b2 = Marker(eid='OTRB.454.B2')
otrb_457_b2 = Marker(eid='OTRB.457.B2')
otrb_461_b2 = Marker(eid='OTRB.461.B2')
ensub_466_b2 = Marker(eid='ENSUB.466.B2')


d_154_2 = Drift(l=0.100073, eid='D_180')
d_154_1 = Drift(d_154.l - d_154_2.l)
stlat_393_b2 = Marker()
d_154_n = (d_154_1, stlat_393_b2, d_154_2)

# power supplies

#
qd_231_b1.ps_id = 'QD.20.B1'
qd_232_b1.ps_id = 'QD.21.B1'
qd_233_b1.ps_id = 'QD.22.B1'
q_249_l2.ps_id = 'Q.A3.1.L2'
q_261_l2.ps_id = 'Q.A3.2.L2'
q_273_l2.ps_id = 'Q.A3.3.L2'
q_285_l2.ps_id = 'Q.A3.4.L2'
q_297_l2.ps_id = 'Q.A4.1.L2'
q_309_l2.ps_id = 'Q.A4.2.L2'
q_321_l2.ps_id = 'Q.A4.3.L2'
q_333_l2.ps_id = 'Q.A4.4.L2'
q_345_l2.ps_id = 'Q.A5.1.L2'
q_357_l2.ps_id = 'Q.A5.2.L2'
q_369_l2.ps_id = 'Q.A5.3.L2'
q_381_l2.ps_id = 'Q.A5.4.L2'
qd_387_b2.ps_id = 'QD.1.B2'
qd_388_b2.ps_id = 'QD.2.B2'
qd_391_b2.ps_id = 'QD.3.B2'
qd_392_b2.ps_id = 'QD.4.B2'
qd_415_b2.ps_id = 'QD.6.B2'
qd_417_b2.ps_id = 'QD.7.B2'
qd_418_b2.ps_id = 'QD.8.B2'
qd_425_b2.ps_id = 'QD.9.B2'
qd_427_b2.ps_id = 'QD.10.B2'
qd_431_b2.ps_id = 'QD.11.B2'
qd_434_b2.ps_id = 'QD.12.B2'
qd_437_b2.ps_id = 'QD.13.B2'
qd_440_b2.ps_id = 'QD.14.B2'
qd_444_b2.ps_id = 'QD.15.B2'
qd_448_b2.ps_id = 'QD.16.B2'
qd_452_b2.ps_id = 'QD.17.B2'
qd_456_b2.ps_id = 'QD.18.B2'
qd_459_b2.ps_id = 'QD.19.B2'
qd_463_b2.ps_id = 'QD.21.B2'
qd_464_b2.ps_id = 'QD.22.B2'
qd_465_b2.ps_id = 'QD.23.B2'

#

#

#
c_a3_1_1_l2.ps_id = 'C.A3.L2'
c_a3_1_2_l2.ps_id = 'C.A3.L2'
c_a3_1_3_l2.ps_id = 'C.A3.L2'
c_a3_1_4_l2.ps_id = 'C.A3.L2'
c_a3_1_5_l2.ps_id = 'C.A3.L2'
c_a3_1_6_l2.ps_id = 'C.A3.L2'
c_a3_1_7_l2.ps_id = 'C.A3.L2'
c_a3_1_8_l2.ps_id = 'C.A3.L2'
c_a3_2_1_l2.ps_id = 'C.A3.L2'
c_a3_2_2_l2.ps_id = 'C.A3.L2'
c_a3_2_3_l2.ps_id = 'C.A3.L2'
c_a3_2_4_l2.ps_id = 'C.A3.L2'
c_a3_2_5_l2.ps_id = 'C.A3.L2'
c_a3_2_6_l2.ps_id = 'C.A3.L2'
c_a3_2_7_l2.ps_id = 'C.A3.L2'
c_a3_2_8_l2.ps_id = 'C.A3.L2'
c_a3_3_1_l2.ps_id = 'C.A3.L2'
c_a3_3_2_l2.ps_id = 'C.A3.L2'
c_a3_3_3_l2.ps_id = 'C.A3.L2'
c_a3_3_4_l2.ps_id = 'C.A3.L2'
c_a3_3_5_l2.ps_id = 'C.A3.L2'
c_a3_3_6_l2.ps_id = 'C.A3.L2'
c_a3_3_7_l2.ps_id = 'C.A3.L2'
c_a3_3_8_l2.ps_id = 'C.A3.L2'
c_a3_4_1_l2.ps_id = 'C.A3.L2'
c_a3_4_2_l2.ps_id = 'C.A3.L2'
c_a3_4_3_l2.ps_id = 'C.A3.L2'
c_a3_4_4_l2.ps_id = 'C.A3.L2'
c_a3_4_5_l2.ps_id = 'C.A3.L2'
c_a3_4_6_l2.ps_id = 'C.A3.L2'
c_a3_4_7_l2.ps_id = 'C.A3.L2'
c_a3_4_8_l2.ps_id = 'C.A3.L2'
c_a4_1_1_l2.ps_id = 'C.A4.L2'
c_a4_1_2_l2.ps_id = 'C.A4.L2'
c_a4_1_3_l2.ps_id = 'C.A4.L2'
c_a4_1_4_l2.ps_id = 'C.A4.L2'
c_a4_1_5_l2.ps_id = 'C.A4.L2'
c_a4_1_6_l2.ps_id = 'C.A4.L2'
c_a4_1_7_l2.ps_id = 'C.A4.L2'
c_a4_1_8_l2.ps_id = 'C.A4.L2'
c_a4_2_1_l2.ps_id = 'C.A4.L2'
c_a4_2_2_l2.ps_id = 'C.A4.L2'
c_a4_2_3_l2.ps_id = 'C.A4.L2'
c_a4_2_4_l2.ps_id = 'C.A4.L2'
c_a4_2_5_l2.ps_id = 'C.A4.L2'
c_a4_2_6_l2.ps_id = 'C.A4.L2'
c_a4_2_7_l2.ps_id = 'C.A4.L2'
c_a4_2_8_l2.ps_id = 'C.A4.L2'
c_a4_3_1_l2.ps_id = 'C.A4.L2'
c_a4_3_2_l2.ps_id = 'C.A4.L2'
c_a4_3_3_l2.ps_id = 'C.A4.L2'
c_a4_3_4_l2.ps_id = 'C.A4.L2'
c_a4_3_5_l2.ps_id = 'C.A4.L2'
c_a4_3_6_l2.ps_id = 'C.A4.L2'
c_a4_3_7_l2.ps_id = 'C.A4.L2'
c_a4_3_8_l2.ps_id = 'C.A4.L2'
c_a4_4_1_l2.ps_id = 'C.A4.L2'
c_a4_4_2_l2.ps_id = 'C.A4.L2'
c_a4_4_3_l2.ps_id = 'C.A4.L2'
c_a4_4_4_l2.ps_id = 'C.A4.L2'
c_a4_4_5_l2.ps_id = 'C.A4.L2'
c_a4_4_6_l2.ps_id = 'C.A4.L2'
c_a4_4_7_l2.ps_id = 'C.A4.L2'
c_a4_4_8_l2.ps_id = 'C.A4.L2'
c_a5_1_1_l2.ps_id = 'C.A5.L2'
c_a5_1_2_l2.ps_id = 'C.A5.L2'
c_a5_1_3_l2.ps_id = 'C.A5.L2'
c_a5_1_4_l2.ps_id = 'C.A5.L2'
c_a5_1_5_l2.ps_id = 'C.A5.L2'
c_a5_1_6_l2.ps_id = 'C.A5.L2'
c_a5_1_7_l2.ps_id = 'C.A5.L2'
c_a5_1_8_l2.ps_id = 'C.A5.L2'
c_a5_2_1_l2.ps_id = 'C.A5.L2'
c_a5_2_2_l2.ps_id = 'C.A5.L2'
c_a5_2_3_l2.ps_id = 'C.A5.L2'
c_a5_2_4_l2.ps_id = 'C.A5.L2'
c_a5_2_5_l2.ps_id = 'C.A5.L2'
c_a5_2_6_l2.ps_id = 'C.A5.L2'
c_a5_2_7_l2.ps_id = 'C.A5.L2'
c_a5_2_8_l2.ps_id = 'C.A5.L2'
c_a5_3_1_l2.ps_id = 'C.A5.L2'
c_a5_3_2_l2.ps_id = 'C.A5.L2'
c_a5_3_3_l2.ps_id = 'C.A5.L2'
c_a5_3_4_l2.ps_id = 'C.A5.L2'
c_a5_3_5_l2.ps_id = 'C.A5.L2'
c_a5_3_6_l2.ps_id = 'C.A5.L2'
c_a5_3_7_l2.ps_id = 'C.A5.L2'
c_a5_3_8_l2.ps_id = 'C.A5.L2'
c_a5_4_1_l2.ps_id = 'C.A5.L2'
c_a5_4_2_l2.ps_id = 'C.A5.L2'
c_a5_4_3_l2.ps_id = 'C.A5.L2'
c_a5_4_4_l2.ps_id = 'C.A5.L2'
c_a5_4_5_l2.ps_id = 'C.A5.L2'
c_a5_4_6_l2.ps_id = 'C.A5.L2'
c_a5_4_7_l2.ps_id = 'C.A5.L2'
c_a5_4_8_l2.ps_id = 'C.A5.L2'
tdsb_428_b2.ps_id = 'TDSB.B2'
tdsb_430_b2.ps_id = 'TDSB.B2'

#
bb_393_b2.ps_id = 'BB.1.B2'
bb_402_b2.ps_id = 'BB.1.B2'
bb_404_b2.ps_id = 'BB.1.B2'
bb_413_b2.ps_id = 'BB.1.B2'


def make_cell():

    return deepcopy(SectionCell(
        # Lattice
        cell=[
            d_1,
            cbb_229_b1d,
            d_2,
            qd_231_b1,
            d_3,
            ccx_232_b1,
            d_4,
            tora_232_b1,
            d_5,
            qd_232_b1,
            d_6,
            ccy_232_b1,
            d_4,
            bpma_233_b1,
            d_8,
            qd_233_b1,
            d_9,
            c_a3_1_1_l2,
            d_10,
            c_a3_1_2_l2,
            d_10,
            c_a3_1_3_l2,
            d_10,
            c_a3_1_4_l2,
            d_10,
            c_a3_1_5_l2,
            d_10,
            c_a3_1_6_l2,
            d_10,
            c_a3_1_7_l2,
            d_10,
            c_a3_1_8_l2,
            d_17,
            q_249_l2,
            d_18,
            cx_249_l2,
            d_19,
            bpmc_249_l2,
            d_20,
            c_a3_2_1_l2,
            d_10,
            c_a3_2_2_l2,
            d_10,
            c_a3_2_3_l2,
            d_10,
            c_a3_2_4_l2,
            d_10,
            c_a3_2_5_l2,
            d_10,
            c_a3_2_6_l2,
            d_10,
            c_a3_2_7_l2,
            d_10,
            c_a3_2_8_l2,
            d_17,
            q_261_l2,
            d_18,
            cy_261_l2,
            d_19,
            bpmc_261_l2,
            d_20,
            c_a3_3_1_l2,
            d_10,
            c_a3_3_2_l2,
            d_10,
            c_a3_3_3_l2,
            d_10,
            c_a3_3_4_l2,
            d_10,
            c_a3_3_5_l2,
            d_10,
            c_a3_3_6_l2,
            d_10,
            c_a3_3_7_l2,
            d_10,
            c_a3_3_8_l2,
            d_17,
            q_273_l2,
            d_18,
            cx_273_l2,
            d_19,
            bpmr_273_l2,
            d_20,
            c_a3_4_1_l2,
            d_10,
            c_a3_4_2_l2,
            d_10,
            c_a3_4_3_l2,
            d_10,
            c_a3_4_4_l2,
            d_10,
            c_a3_4_5_l2,
            d_10,
            c_a3_4_6_l2,
            d_10,
            c_a3_4_7_l2,
            d_10,
            c_a3_4_8_l2,
            d_17,
            q_285_l2,
            d_18,
            cy_285_l2,
            d_19,
            bpmr_285_l2,
            d_20,
            c_a4_1_1_l2,
            d_10,
            c_a4_1_2_l2,
            d_10,
            c_a4_1_3_l2,
            d_10,
            c_a4_1_4_l2,
            d_10,
            c_a4_1_5_l2,
            d_10,
            c_a4_1_6_l2,
            d_10,
            c_a4_1_7_l2,
            d_10,
            c_a4_1_8_l2,
            d_17,
            q_297_l2,
            d_18,
            cx_297_l2,
            d_19,
            bpmc_297_l2,
            d_20,
            c_a4_2_1_l2,
            d_10,
            c_a4_2_2_l2,
            d_10,
            c_a4_2_3_l2,
            d_10,
            c_a4_2_4_l2,
            d_10,
            c_a4_2_5_l2,
            d_10,
            c_a4_2_6_l2,
            d_10,
            c_a4_2_7_l2,
            d_10,
            c_a4_2_8_l2,
            d_17,
            q_309_l2,
            d_18,
            cy_309_l2,
            d_19,
            bpmr_309_l2,
            d_20,
            c_a4_3_1_l2,
            d_10,
            c_a4_3_2_l2,
            d_10,
            c_a4_3_3_l2,
            d_10,
            c_a4_3_4_l2,
            d_10,
            c_a4_3_5_l2,
            d_10,
            c_a4_3_6_l2,
            d_10,
            c_a4_3_7_l2,
            d_10,
            c_a4_3_8_l2,
            d_17,
            q_321_l2,
            d_18,
            cx_321_l2,
            d_19,
            bpmc_321_l2,
            d_20,
            c_a4_4_1_l2,
            d_10,
            c_a4_4_2_l2,
            d_10,
            c_a4_4_3_l2,
            d_10,
            c_a4_4_4_l2,
            d_10,
            c_a4_4_5_l2,
            d_10,
            c_a4_4_6_l2,
            d_10,
            c_a4_4_7_l2,
            d_10,
            c_a4_4_8_l2,
            d_17,
            q_333_l2,
            d_18,
            cy_333_l2,
            d_19,
            bpmr_333_l2,
            d_20,
            c_a5_1_1_l2,
            d_10,
            c_a5_1_2_l2,
            d_10,
            c_a5_1_3_l2,
            d_10,
            c_a5_1_4_l2,
            d_10,
            c_a5_1_5_l2,
            d_10,
            c_a5_1_6_l2,
            d_10,
            c_a5_1_7_l2,
            d_10,
            c_a5_1_8_l2,
            d_17,
            q_345_l2,
            d_18,
            cx_345_l2,
            d_19,
            bpmc_345_l2,
            d_20,
            c_a5_2_1_l2,
            d_10,
            c_a5_2_2_l2,
            d_10,
            c_a5_2_3_l2,
            d_10,
            c_a5_2_4_l2,
            d_10,
            c_a5_2_5_l2,
            d_10,
            c_a5_2_6_l2,
            d_10,
            c_a5_2_7_l2,
            d_10,
            c_a5_2_8_l2,
            d_17,
            q_357_l2,
            d_18,
            cy_357_l2,
            d_19,
            bpmc_357_l2,
            d_20,
            c_a5_3_1_l2,
            d_10,
            c_a5_3_2_l2,
            d_10,
            c_a5_3_3_l2,
            d_10,
            c_a5_3_4_l2,
            d_10,
            c_a5_3_5_l2,
            d_10,
            c_a5_3_6_l2,
            d_10,
            c_a5_3_7_l2,
            d_10,
            c_a5_3_8_l2,
            d_17,
            q_369_l2,
            d_18,
            cx_369_l2,
            d_19,
            bpmr_369_l2,
            d_20,
            c_a5_4_1_l2,
            d_10,
            c_a5_4_2_l2,
            d_10,
            c_a5_4_3_l2,
            d_10,
            c_a5_4_4_l2,
            d_10,
            c_a5_4_5_l2,
            d_10,
            c_a5_4_6_l2,
            d_10,
            c_a5_4_7_l2,
            d_10,
            c_a5_4_8_l2,
            d_17,
            q_381_l2,
            d_18,
            cy_381_l2,
            d_19,
            bpmr_381_l2,
            d_141,
            tora_387_b2,
            d_142,
            bpma_387_b2,
            d_143,
            qd_387_b2,
            d_144,
            ccy_387_b2,
            d_145,
            qd_388_b2,
            d_146,
            ccx_388_b2,
            d_147,
            bpma_390_b2,
            d_143,
            qd_391_b2,
            d_149,
            ccy_391_b2,
            d_150,
            qd_392_b2,
            d_151,
            ccx_392_b2,
            d_152,
            otra_392_b2,
            d_153,
            bpmf_393_b2,
            d_154_n,
            bb_393_b2,
            d_155,
            bb_402_b2,
            d_156,
            cbb_403_b2,
            d_157,
            bpms_404_b2,
            d_158,
            otrs_404_b2,
            d_159,
            bb_404_b2,
            d_156,
            cbb_405_b2,
            d_161,
            bb_413_b2,
            d_156,
            cbb_414_b2,
            d_163,
            match_414_b2,
            d_164,
            bpmf_414_b2,
            d_165,
            tora_415_b2,
            d_166,
            qd_415_b2,
            d_167,
            ccx_415_b2,
            d_168,
            ccy_416_b2,
            d_169,
            qd_417_b2,
            d_170,
            ccx_418_b2,
            d_4,
            ccy_418_b2,
            d_172,
            bpma_418_b2,
            d_143,
            qd_418_b2,
            d_144,
            engrd_419_b2,
            d_175,
            ccx_425_b2,
            d_176,
            qd_425_b2,
            d_177,
            ccy_426_b2,
            d_178,
            otra_426_b2,
            d_179,
            bpma_426_b2,
            d_180,
            qd_427_b2,
            d_181,
            match_428_b2,
            tdsb_428_b2,
            d_182,
            tdsb_430_b2,
            d_183,
            qd_431_b2,
            d_184,
            ccx_431_b2,
            d_185,
            bpma_432_b2,
            d_186,
            qd_434_b2,
            d_144,
            ccy_434_b2,
            d_188,
            qd_437_b2,
            d_189,
            otra_438_b2,
            d_190,
            bpma_440_b2,
            d_143,
            qd_440_b2,
            d_144,
            ccx_441_b2,
            d_193,
            bpma_444_b2,
            d_143,
            qd_444_b2,
            d_189,
            match_446_b2,
            otra_446_b2,
            d_196,
            ccx_447_b2,
            d_197,
            bpma_448_b2,
            d_143,
            qd_448_b2,
            d_144,
            ccy_448_b2,
            d_200,
            otrb_450_b2,
            d_201,
            bpma_452_b2,
            d_202,
            qd_452_b2,
            d_203,
            otrb_454_b2,
            d_204,
            ccx_454_b2,
            d_205,
            bpma_455_b2,
            d_143,
            qd_456_b2,
            d_144,
            ccx_456_b2,
            d_200,
            otrb_457_b2,
            d_190,
            bpma_459_b2,
            d_143,
            qd_459_b2,
            d_144,
            ccy_460_b2,
            d_200,
            otrb_461_b2,
            d_213,
            bpma_462_b2,
            d_214,
            qd_463_b2,
            d_215,
            ccx_464_b2,
            d_216,
            qd_464_b2,
            d_144,
            ccy_464_b2,
            d_218,
            ccx_465_b2,
            d_172,
            bpma_465_b2,
            d_143,
            qd_465_b2,
            d_221,
            ensub_466_b2,
        ]
    )
)

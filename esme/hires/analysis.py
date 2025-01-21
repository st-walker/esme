def transform_pixel_widths(
    pixel_widths: npt.NDArray,
    pixel_widths_errors: npt.NDArray,
    *,
    pixel_units: str = "px",
    to_variances: bool = True,
    dimension: str = "x",
) -> tuple[npt.NDArray, npt.NDArray]:
    """The fits used in the paper are linear relationships between the variances
    (i.e. pixel_std^2) and the square of the independent variable (either
    voltage squared V^2 or dipsersion D^2). This function takes D or V and sigma
    and transforms these variables to D^2 or V^2 and pixel width *variances* in
    units of um, so that the resulting fit for the standard deviations will be
    linear in the independent variable.

    """
    # Convert to ufloats momentarily for the error propagation.
    widths = np.array(
        [ufloat(v, e) for (v, e) in zip(pixel_widths, pixel_widths_errors)]
    )

    if pixel_units == "px":
        scale = 1
    elif pixel_units == "um" and dimension == "x":
        scale = PIXEL_SCALE_X_UM
    elif pixel_units == "um" and dimension == "y":
        scale = PIXEL_SCALE_Y_UM
    elif pixel_units == "m" and dimension == "x":
        scale = PIXEL_SCALE_X_M
    elif pixel_units == "m" and dimension == "y":
        scale = PIXEL_SCALE_Y_M
    else:
        raise ValueError(f"unknown unit or dimension: {pixel_units=}, {dimension=}")

    widths *= scale
    if to_variances:
        widths = widths**2

    # Extract errors form ufloats to go back to a 2-tuple of arrays.
    widths, errors = zip(*[(w.nominal_value, w.std_dev) for w in widths])
    return np.array(widths), np.array(errors)


class SliceWidthsFitter:
    def __init__(self, dscan_widths, tscan_widths, bscan_widths=None, *, avmapping):
        self.dscan_widths = dscan_widths
        self.tscan_widths = tscan_widths
        self.bscan_widths = bscan_widths
        self.avmapping = avmapping

    def dispersion_scan_fit(self, sigma_r=None, emittance=None) -> ValueWithErrorT:
        widths_with_errors = list(self.dscan_widths.values())
        widths = [x.n for x in widths_with_errors]
        errors = [x.s for x in widths_with_errors]
        dx = np.array(list(self.dscan_widths.keys()))
        dx2 = dx**2

        # widths, errors = transform_units_for_pixel_widths(widths, errors)
        widths2_m2, errors2_m2 = transform_pixel_widths(
            widths, errors, pixel_units="m", to_variances=True
        )
        a_v, b_v = linear_fit(dx2, widths2_m2, errors2_m2)
        return a_v, b_v

    def tds_scan_fit(self) -> tuple[ValueWithErrorT, ValueWithErrorT]:
        widths_with_errors = list(self.tscan_widths.values())
        widths = [x.n for x in widths_with_errors]
        errors = [x.s for x in widths_with_errors]
        amplitudes = np.array(list(self.tscan_widths.keys()))
        try:
            voltages = self.avmapping(amplitudes)
        except TypeError:
            raise TypeError()

        voltages2 = voltages**2
        widths2_m2, errors2_m2 = transform_pixel_widths(
            widths, errors, pixel_units="m", to_variances=True
        )
        a_v, b_v = linear_fit(voltages2, widths2_m2, errors2_m2)
        return a_v, b_v

    def beta_scan_fit(self) -> tuple[ValueWithErrorT, ValueWithErrorT]:
        widths_with_errors = list(self.bscan_widths.values())
        widths = [x.n for x in widths_with_errors]
        errors = [x.s for x in widths_with_errors]
        beta = np.array(list(self.bscan_widths.keys()))
        widths2_m2, errors2_m2 = transform_pixel_widths(
            widths, errors, pixel_units="m", to_variances=True
        )
        a_beta, b_beta = linear_fit(beta, widths2_m2, errors2_m2)
        return a_beta, b_beta

    def all_fit_parameters(
        self,
        beam_energy,
        tscan_dispersion,
        dscan_voltage,
        optics_fixed_points,
        sigma_z=None,
        sigma_z_rms=None,
    ) -> DerivedBeamParameters:
        a_v, b_v = self.tds_scan_fit()
        a_d, b_d = self.dispersion_scan_fit()

        # Values and errors, here we just say there is 0 error in the
        # dispersion, voltage and energy, not strictly true of course.
        energy = beam_energy, 0.0  # in eV
        dispersion = tscan_dispersion, 0.0
        voltage = dscan_voltage, 0.0

        try:
            a_beta, b_beta = self.beta_scan_fit()
        except (TypeError, AttributeError):
            a_beta = b_beta = None

        return DerivedBeamParameters(
            a_v=a_v,
            b_v=b_v,
            a_d=a_d,
            b_d=b_d,
            reference_energy=energy,
            reference_dispersion=dispersion,
            reference_voltage=voltage,
            oconfig=optics_fixed_points,
            a_beta=a_beta,
            b_beta=b_beta,
            sigma_z=sigma_z,
            sigma_z_rms=sigma_z_rms,
        )


@dataclass
class DerivedBeamParameters:
    """Table 2 from the paper"""

    # Stored as tuples, nominal value with error.
    a_v: ValueWithErrorT
    b_v: ValueWithErrorT
    a_d: ValueWithErrorT
    b_d: ValueWithErrorT
    reference_energy: ValueWithErrorT
    reference_dispersion: ValueWithErrorT
    reference_voltage: ValueWithErrorT
    oconfig: OpticalConfig
    a_beta: Optional[ValueWithErrorT] = None
    b_beta: Optiona[ValueWithErrorT] = None
    sigma_z: Optional[ValueWithErrorT] = None
    sigma_z_rms: Optional[ValueWithErrorT] = None

    def a_d_derived(self, sigma_r, emittance: Optional[float] = None):
        if emittance is None:
            emittance = self.emitx
        return np.sqrt(sigma_r**2 + (emittance * self.oconfig.beta_screen) ** 2)

    def set_a_d_to_known_value(self, sigma_r, emittance: Optional[float] = None):
        new_ad = self.a_d_derived(sigma_r, emittance=emittance)
        self.a_d = new_ad

    def set_a_v_to_known_value(self, sigma_r, emittance: Optional[float] = None):
        new_ad = self.a_d_derived(sigma_r, emittance=emittance)
        self.a_d = new_ad

    @property
    def sigma_e(self) -> ValueWithErrorT:
        energy0 = ufloat(*self.reference_energy)
        dx0 = ufloat(*self.reference_dispersion)
        # Convert to ufloat for correct error propagation before converting back
        # to tuples at the end.
        av = ufloat(*self.a_v)
        ad = ufloat(*self.a_d)
        result = (energy0 / dx0) * usqrt(av - ad)
        return result.n, result.s

    @property
    def sigma_e_alt(self) -> ValueWithErrorT:
        energy0 = ufloat(*self.reference_energy)
        dx0 = ufloat(*self.reference_dispersion)
        v0 = ufloat(*self.reference_voltage)
        # Convert to ufloat for correct error propagation before converting back
        # to tuples at the end.
        bd = ufloat(*self.b_d)
        bv = ufloat(*self.b_v)
        result = (energy0 / dx0) * usqrt(bd * dx0**2 - bv * v0**2)
        return result.n, result.s

    @property
    def sigma_i(self) -> ValueWithErrorT:
        """This is the average beamsize in the TDS, returned in metres"""
        k = TDS_WAVENUMBER
        dx0 = ufloat(*self.reference_dispersion)
        energy0 = ufloat(*self.reference_energy)
        e0_joules = energy0 * e
        bv = ufloat(*self.b_v)
        try:
            result = (e0_joules / (dx0 * e * k)) * usqrt(bv)
        except ValueError:
            return np.nan, np.nan
        return result.n, result.s

    @property
    def sigma_i_alt(self) -> ValueWithErrorT:
        """This is the average beamsize in the TDS, returned in metres"""
        av = ufloat(*self.a_v)
        ad = ufloat(*self.a_d)
        bd = ufloat(*self.b_d)

        dx0 = ufloat(*self.reference_dispersion)
        v0 = abs(ufloat(*self.reference_voltage))
        e0j = ufloat(*self.reference_energy) * e  # Convert to joules
        k = TDS_WAVENUMBER
        try:
            result = (e0j / (dx0 * e * k * v0)) * usqrt(ad - av + dx0**2 * bd)
        except ValueError:
            return np.nan, np.nan
        return result.n, result.s

    @property
    def sigma_b(self) -> ValueWithErrorT:
        bety = self.oconfig.beta_tds
        alfy = self.oconfig.alpha_tds
        gamy = self.oconfig.gamma_tds
        length = TDS_LENGTH
        sigma_i = ufloat(*self.sigma_i)
        b_beta = sigma_i**2 / (bety + 0.25 * length**2 * gamy - length * alfy)
        result = usqrt(b_beta * self.oconfig.beta_screen)
        return result.n, result.s

    @property
    def sigma_r(self) -> ValueWithErrorT:
        # if self.known_sigma_r:
        #     return self.known_sigma_r, 0
        ad = ufloat(*self.a_d)
        sigma_b = ufloat(*self.sigma_b)
        result = usqrt(ad - sigma_b**2)
        return result.n, result.s

    @property
    def emitx(self) -> ValueWithErrorT:
        gam0 = ufloat(*self.reference_energy) / ELECTRON_MASS_EV
        sigma_b = ufloat(*self.sigma_b)
        result = sigma_b**2 * gam0 / self.oconfig.beta_screen
        return result.n, result.s

    @property
    def sigma_b_alt(self) -> ValueWithErrorT:
        ab = ufloat(*self.a_beta)
        ad = ufloat(*self.a_d)
        bd = ufloat(*self.b_d)
        d0 = ufloat(*self.reference_dispersion)
        result = usqrt(ad + bd * d0**2 - ab)
        return result.n, result.s

    @property
    def emitx_alt(self) -> ValueWithErrorT:
        bb = ufloat(*self.b_beta)
        gamma0 = ufloat(*self.reference_energy) / ELECTRON_MASS_EV
        result = bb * gamma0
        return result.n, result.s

    @property
    def sigma_r_alt(self) -> ValueWithErrorT:
        # if self.known_sigma_r:
        #     return self.known_sigma_r, 0
        ab = ufloat(*self.a_beta)
        bd = ufloat(*self.b_d)
        d0 = ufloat(*self.reference_dispersion)
        result = usqrt(ab - bd * d0**2)
        return result.n, result.s

    @property
    def sigma_e_from_tds(self) -> ValueWithErrorT:
        sigma_i = ufloat(*self.sigma_i)
        in_ev = abs(ufloat(*self.reference_voltage)) * TDS_WAVENUMBER * sigma_i
        return in_ev.n, in_ev.s

    @property
    def sigma_e_from_tds_alt(self) -> ValueWithErrorT:
        sigma_i = ufloat(*self.sigma_i_alt)
        in_ev = abs(ufloat(*self.reference_voltage)) * TDS_WAVENUMBER * sigma_i
        return in_ev.n, in_ev.s

    def fit_parameters_to_df(self) -> pd.DataFrame:
        dx0 = self.reference_dispersion
        v0 = self.reference_voltage
        e0 = self.reference_energy

        av, bv = self.a_v, self.b_v
        ad, bd = self.a_d, self.b_d

        pdict = {
            "V_0": v0,
            "D_0": dx0,
            "E_0": e0,
            "A_V": av,
            "B_V": bv,
            "A_D": ad,
            "B_D": bd,
        }

        if self.a_beta and self.b_beta:
            pdict |= {"A_beta": self.a_beta, "B_beta": self.b_beta}

        values = []
        errors = []
        for key, pair in pdict.items():
            values.append(pair[0])
            errors.append(pair[1])

        return pd.DataFrame({"values": values, "errors": errors}, index=pdict.keys())

    def _beam_parameters_to_df(self) -> pd.DataFrame:
        pdict = {
            "sigma_e": self.sigma_e,
            "sigma_i": self.sigma_i,
            "sigma_e_from_tds": self.sigma_e_from_tds,
            "sigma_b": self.sigma_b,
            "sigma_r": self.sigma_r,
            "emitx": self.emitx,
        }

        values = []
        errors = []
        for key, pair in pdict.items():
            values.append(pair[0])
            errors.append(pair[1])

        return pd.DataFrame({"values": values, "errors": errors}, index=pdict.keys())

    def _alt_beam_parameters_to_df(self) -> pd.DataFrame:
        pdict = {
            "sigma_e": self.sigma_e_alt,
            "sigma_i": self.sigma_i_alt,
            "sigma_e_from_tds": self.sigma_e_from_tds_alt,
        }
        if self.a_beta and self.b_beta:
            pdict |= {
                "sigma_b": self.sigma_b_alt,
                "sigma_r": self.sigma_r_alt,
                "emitx": self.emitx_alt,
            }
        values = []
        errors = []
        for key, pair in pdict.items():
            values.append(pair[0])
            errors.append(pair[1])

        return pd.DataFrame(
            {"alt_values": values, "alt_errors": errors}, index=pdict.keys()
        )

    def beam_parameters_to_df(self) -> pd.DataFrame:
        # sigma_t in picosecondsm
        params = self._beam_parameters_to_df()
        alt_params = self._alt_beam_parameters_to_df()
        if self.sigma_z is not None:
            self.sigma_t
            params.loc["sigma_z"] = {
                "values": self.sigma_z[0],
                "errors": self.sigma_z[1],
            }
            params.loc["sigma_t"] = {
                "values": self.sigma_t[0],
                "errors": self.sigma_t[1],
            }

        return pd.concat([params, alt_params], axis=1)

    @property
    def sigma_t(self):
        stn = self.sigma_z[0] / c
        ste = self.sigma_z[1] / c
        return stn, ste

    @property
    def sigma_t_rms(self):
        stn = self.sigma_z_rms[0] / c
        ste = self.sigma_z_rms[1] / c
        return stn, ste

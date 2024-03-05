"""Module relating to the measurement of dispersion"""







class BaseDispersionMeasurer:
    pass


class DispersionMeasurer(BaseDispersionMeasurer):
    def __init__(self, a1_voltages: list[float], machine):
        self.a1_voltages = a1_voltages
        self.machine = machine
        # if machine is None:
        # self.machine = EnergySpreadMeasuringMachine(SNAPSHOT_TEMPL)

    def measure(self):
        raise NotImplementedError()


class BasicDispersionMeasurer(BaseDispersionMeasurer):
    def measure(self) -> tuple[float, float]:
        dispersion = _repeat_float_input_until_valid("Enter dispersion in m: ")
        dispersion_unc = _repeat_float_input_until_valid("Enter dispersion unc in m: ")
        return dispersion, dispersion_unc


def _repeat_float_input_until_valid(prompt):
    while True:
        given = input(prompt)
        try:
            dispersion = float(given)
        except ValueError:
            print(f"Invalid dispersion: {given=}, go again")
            continue
        else:
            return dispersion

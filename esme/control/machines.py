from .kickers import KickerController




class SimpleLPSDevice:
    def __init__(self, kickerop: KickerController) -> None:
        self.kickerop = kickerop

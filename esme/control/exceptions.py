"""A set of exceptions to be used when interacting with the machine directly."""


class DOOCSError(RuntimeError):
    def __init__(self, address):
        self.address = address


class DOOCSReadError(DOOCSError):
    def __str__(self):
        return f"Read Error with channel: {self.address}"

class DOOCSWriteError(DOOCSError):
    def __init__(self, address: str, value: str):
        super().__init__(address)
        self.value = value

    def __str__(self) -> str:
        return f"Write error with channel: {self.address=}, {value=}"

class DOOCSUnexpectedReadValueError(DOOCSError):
    def __init__(self, address, value):
        super().__init__(address)
        self.value = value

    def __str__(self) -> str:
        return f"Unexpected read value: {self.address=}, {self.value=}"

class EuXFELUserError(RuntimeError):
    """Base class for the user trying to do something wrong"""



class EuXFELMachineError(RuntimeError):
    """Base class for the machine being in some bad state not
    conducive to successul GUI operation

    """

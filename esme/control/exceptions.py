

class EuXFELUserError(RuntimeError):
    """Base class for the user trying to do something wrong"""



class EuXFELMachineError(RuntimeError):
    """Base class for the machine being in some bad state not
    conducive to successul GUI operation

    """


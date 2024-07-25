import esme.gui.widgets.common as co
from esme.core import DiagnosticRegion


def main():
    f = co.get_machine_manager_factory()
    reader = f.make_machine_reader_manager(DiagnosticRegion.I1)
    print(reader.full_read())


if __name__ == "__main__":
    main()

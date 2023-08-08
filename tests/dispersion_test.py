from esme.dispersion import DispersionScanConfiguration

def test_ScanOpticsConfiguration_init(quad_setting, quad_setting2, quad_setting3):
    inst = DispersionScanConfiguration(quad_setting, [quad_setting2, quad_setting3])

    assert inst.reference_setting == quad_setting
    assert inst.scan_settings == [quad_setting2, quad_setting3]

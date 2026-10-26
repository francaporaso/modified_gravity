"""Config: void threshold, void types, cosmology and the keys of features to come.

Run with: python -m pytest lensing/tests/test_config.py
"""

import pytest
import toml

from lensing import cosmology
from lensing.settings import Config

TEMPLATE = "lensing/config.tmplt.toml"


def load(tmp_path, edit=lambda c: None, gravity="gr", threshold=None):
    c = toml.load(TEMPLATE)
    edit(c)
    path = tmp_path / "cfg.toml"
    path.write_text(toml.dumps(c))
    return Config(str(path), gravity, threshold)


def test_threshold_of_the_config_is_used_unless_one_is_given(tmp_path):
    cfg = load(tmp_path)
    assert cfg.lensname == "~/cats/L768/voids_LCDM_09.dat" and cfg.voidthr == "09"
    cfg = load(tmp_path, threshold="08", gravity="mg")
    assert cfg.lensname == "~/cats/L768/voids_fR_08.dat" and cfg.voidthr == "08"

    # the config says 08: not overridden by a default of the command line
    def thr08(c):
        c["lenses"]["gr"]["name"] = "~/cats/L768/voids_LCDM_08.dat"

    cfg = load(tmp_path, thr08)
    assert cfg.lensname.endswith("_08.dat") and cfg.voidthr == "08"


def test_threshold_is_only_replaced_in_the_name_of_the_file(tmp_path):
    def edit(c):
        c["lenses"]["gr"]["name"] = "~/cats/run_01/L768_v2/voids_LCDM_09.dat"

    cfg = load(tmp_path, edit, threshold="08")
    assert cfg.lensname == "~/cats/run_01/L768_v2/voids_LCDM_08.dat"
    cfg = load(tmp_path, edit)
    assert cfg.lensname == "~/cats/run_01/L768_v2/voids_LCDM_09.dat" and cfg.voidthr == "09"


def test_file_without_threshold(tmp_path):
    def edit(c):
        c["lenses"]["gr"]["name"] = "~/cats/voids_LCDM.dat"

    assert load(tmp_path, edit).voidthr == "NA"
    with pytest.raises(ValueError, match="no '_dd' threshold"):
        load(tmp_path, edit, threshold="08")


def test_void_types_are_checked(tmp_path):
    def edit(c):
        c["lenses"]["voidtype"] = ["R", "mixd"]

    with pytest.raises(ValueError, match="voidtype"):
        load(tmp_path, edit)


def test_jackknife_seed(tmp_path):
    assert load(tmp_path).jk_seed == 0

    def edit(seed):
        return lambda c: c["profile"].update(jk_seed=seed)

    assert load(tmp_path, edit(5)).jk_seed == 5
    for bad in (-1, 1.5):
        with pytest.raises(ValueError, match="jk_seed"):
            load(tmp_path, edit(bad))


def test_keys_of_features_to_come_are_read(tmp_path):
    cfg = load(tmp_path)
    assert cfg.plot is False and cfg.nback is None and cfg.Ode0 == 0.6911

    def edit(c):
        c["run"]["plot"] = True
        c["profile"]["nback"] = 10.0

    cfg = load(tmp_path, edit)
    assert cfg.plot is True and cfg.nback == 10.0


def test_old_keys_are_ignored(tmp_path):
    # configs from before: fullshape, randname
    def edit(c):
        c["lenses"]["fullshape"] = True
        c["lenses"]["gr"]["randname"] = "~/cats/L768/voids_rand_LCDM_09.dat"

    load(tmp_path, edit)


def test_cosmology_must_be_flat():
    cosmo = cosmology.make_cosmo(1.0, 0.3089, 0.0486, 0.6911)
    assert cosmo.Om0 == 0.3089 and cosmo.h == 1.0
    cosmology.make_cosmo(0.7, 0.3, 0.05)  # Ode0 not given: nothing to check
    with pytest.raises(ValueError, match="flat"):
        cosmology.make_cosmo(1.0, 0.3089, 0.0486, 0.5)

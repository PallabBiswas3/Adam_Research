import numpy as np
from robust_learning.synthetic import make_long_tail,retention_by_stratum

def test_determinism_and_label_flags():
    a=make_long_tail(17);b=make_long_tail(17)
    np.testing.assert_array_equal(a["x"],b["x"])
    np.testing.assert_array_equal(a["observed_y"],b["observed_y"])
    assert np.array_equal(a["observed_y"]!=a["oracle"]["true_y"],a["oracle"]["flipped"])
    assert len(a["x"])==820

def test_zero_corruption():
    d=make_long_tail(3,noise_rate=0)
    assert not d["oracle"]["flipped"].any()
    m=retention_by_stratum(np.ones(len(d["x"]),dtype=bool),d["oracle"])
    assert m["clean_tail"]["n"]==60
    assert m["noisy"]["retained"] is None

from strategy import G0Random, G1BeamSearch, G2Adaptive


def test_g0_name():
    assert G0Random().name == 'G0'


def test_g0_generates_nodes():
    nodes = G0Random().generate_nodes(5)
    assert len(nodes) == 5
    assert all(n > 0 for n in nodes)


def test_g1_name():
    assert G1BeamSearch().name == 'G1'


def test_g1_generates_nodes():
    nodes = G1BeamSearch().generate_nodes(10)
    assert len(nodes) == 10


def test_g2_name():
    assert G2Adaptive().name == 'G2'


def test_g2_generates_nodes():
    nodes = G2Adaptive().generate_nodes(8)
    assert len(nodes) == 8

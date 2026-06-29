import json
from pathlib import Path

from eval.node_cache import (
    build_cache,
    build_node_info,
    get_global_cache,
    get_node_info,
    get_nodes_by_complexity,
    get_stable_nodes,
    load_cache,
    save_cache,
)


def test_build_cache():
    cache = build_cache()
    assert len(cache) == 256


def test_get_node_info():
    info = get_node_info(1)
    assert info.n_eml == 1
    assert info.x_slot == 0


def test_get_nodes_by_complexity():
    nodes = get_nodes_by_complexity(1)
    assert all(n.n_eml == 1 for n in nodes)
    assert len(nodes) > 0


def test_get_stable_nodes():
    stable = get_stable_nodes()
    assert all(n.stable for n in stable)


def test_global_cache():
    c1 = get_global_cache()
    c2 = get_global_cache()
    assert c1 is c2


def test_structure_description():
    info = build_node_info(0b1001)
    assert 'n_eml=1' in info.structure_description
    assert 'x_slot=R' in info.structure_description


def test_save_and_load_cache(tmp_path: Path):
    path = tmp_path / 'cache.json'
    save_cache(path)
    loaded = load_cache(path)
    assert len(loaded) == 256
    assert loaded[1].byte == 1

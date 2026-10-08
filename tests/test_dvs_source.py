"""Compare the DVS adapter with its exact, licensed upstream reference."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
from typing import Dict
import warnings

import numpy as np
import pytest

from reproduce import paths
from reproduce.vision_preparation import (
    _seg_index, _seg_to_frame, integrate_to_frames, load_aedat_v3,
)


@pytest.fixture(scope="module")
def provenance():
    return json.loads((paths.ROOT / "data/manifests/dvs_preprocessing_source.json").read_text())


def test_distributed_dvs_source_and_license_identities(provenance):
    for relative, expected in provenance["upstream"]["license_files"].items():
        assert hashlib.sha256((paths.ROOT / relative).read_bytes()).hexdigest() == expected
    source = (paths.ROOT / provenance["local_file"]).read_text()
    functions = {node.name: node for node in ast.parse(source).body
                 if isinstance(node, ast.FunctionDef)}
    for entry in provenance["function_mapping"]:
        segment = ast.get_source_segment(source, functions[entry["local"]])
        assert hashlib.sha256(segment.encode()).hexdigest() == entry["local_sha256"]


@pytest.fixture(scope="module")
def upstream(provenance):
    package = importlib.util.find_spec("spikingjelly")
    if package is None:
        pytest.skip("Install pinned state-space extra for SpikingJelly source comparison")
    filename = Path(package.origin).parent / "datasets/__init__.py"
    raw = filename.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == provenance["upstream"]["source_sha256"]
    source = raw.decode().replace("\r\n", "\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        parsed = ast.parse(source)
    names = {entry["upstream"] for entry in provenance["function_mapping"]}
    selected = [node for node in parsed.body
                if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(selected) == len(names)
    for node in selected:
        entry = next(item for item in provenance["function_mapping"]
                     if item["upstream"] == node.name)
        assert hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest() == entry["upstream_sha256"]
    namespace = {"np": np, "struct": struct, "Dict": Dict}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(filename), "exec"), namespace)
    return namespace


def test_aedat_polarity_packets_and_timestamp_overflow_match_upstream(tmp_path, upstream):
    path = tmp_path / "events.aedat"
    packets = []
    for overflow, events in ((0, [(1, 2, 0, 7), (127, 126, 1, 31)]),
                             (2, [(4, 5, 1, 11), (63, 64, 0, 19)])):
        header = struct.pack("<HHIIIIII", 1, 0, 8, 4, overflow, len(events), len(events), len(events))
        payload = b"".join(struct.pack("<II", (x << 17) | (y << 2) | (p << 1) | 1, t)
                           for x, y, p, t in events)
        packets.append(header + payload)
    # A non-polarity packet is consumed without changing decoded events.
    packets.insert(1, struct.pack("<HHIIIIII", 0, 0, 8, 4, 0, 1, 1, 1) + bytes(8))
    path.write_bytes(b"#!AER-DAT3.1\r\n#!END-HEADER\r\n" + b"".join(packets))
    actual = load_aedat_v3(path)
    expected = upstream["load_aedat_v3"](path)
    for name in ("t", "x", "y", "p"):
        np.testing.assert_array_equal(actual[name], expected[name])
    assert actual["t"].tolist() == [7, 31, (2 << 31) | 11, (2 << 31) | 19]


@pytest.mark.parametrize("split_by", ["time", "number"])
def test_regular_boundaries_and_frames_match_upstream(split_by, upstream):
    rng = np.random.RandomState(19)
    events = {"t": np.arange(513, dtype=np.int64),
              "x": rng.randint(0, 8, 513), "y": rng.randint(0, 8, 513),
              "p": rng.randint(0, 2, 513)}
    expected_bounds = upstream["cal_fixed_frames_number_segment_index"](events["t"], split_by, 8)
    actual_bounds = _seg_index(events["t"], split_by, 8)
    for actual, expected in zip(actual_bounds, expected_bounds):
        np.testing.assert_array_equal(actual, expected)
    expected = upstream["integrate_events_by_fixed_frames_number"](events, split_by, 8, 8, 8)
    actual = integrate_to_frames(events, split_by, 8, 8, 8)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32


def test_repeated_event_accumulation_matches_upstream(upstream):
    events = {"t": np.arange(4096), "x": np.zeros(4096, dtype=int),
              "y": np.ones(4096, dtype=int), "p": np.arange(4096) % 2}
    actual = _seg_to_frame(events, 2, 2, 0, 4096)
    expected = upstream["integrate_events_segment_to_frame"](
        events["x"], events["y"], events["p"], 2, 2, 0, 4096)
    np.testing.assert_array_equal(actual, expected)
    assert actual[:, 1, 0].tolist() == [2048, 2048]

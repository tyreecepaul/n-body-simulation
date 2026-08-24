"""
Loader for the N-body simulation's binary output.

File layout (all little-endian, native float32/uint32). Format "NBS2":

    Header (28 bytes):
        char[4]  magic = b"NBS2"
        uint32   n_bodies
        uint32   total_steps
        uint32   output_every
        uint32   num_frames        -- expected frame count
        uint32   completed_frames  -- frames actually written (crash-safe)
        float32  dt

    Then, written once (mass/radius are invariant over the run):
        float32[n_bodies]  mass
        float32[n_bodies]  radius

    Then num_frames frames, each frame = n_bodies records of 4 float32:
        [x, y, vx, vy]

    Frame 0 is the initial state (before any steps). Frame k (k >= 1)
    is the state after step (k * output_every).

If the writer was killed mid-run, completed_frames reflects how many
frames actually made it to disk, so this loader returns a (safely)
truncated result instead of raising.
"""

import numpy as np

_MAGIC = b"NBS2"

_HEADER_DTYPE = np.dtype([
    ("magic", "S4"),
    ("n_bodies", "u4"),
    ("total_steps", "u4"),
    ("output_every", "u4"),
    ("num_frames", "u4"),
    ("completed_frames", "u4"),
    ("dt", "f4"),
])

_FLOATS_PER_BODY_PER_FRAME = 4  # x, y, vx, vy


def load_sim(path):
    """
    Load a simulation output file.

    Returns a dict with:
        header: structured numpy record (n_bodies/total_steps/output_every/
                num_frames/completed_frames/dt)
        pos:    (frames_loaded, n_bodies, 2) float32 -- x, y
        vel:    (frames_loaded, n_bodies, 2) float32 -- vx, vy
        mass:   (n_bodies,) float32 -- constant across the run
        radius: (n_bodies,) float32 -- constant across the run

    `frames_loaded` is `header["completed_frames"]` when that's usable,
    otherwise it's derived from how much frame data is actually present
    in the file (for files written before completed_frames was kept live,
    or in the rare case a crash happened before the very first update).
    """
    with open(path, "rb") as f:
        raw_header = f.read(_HEADER_DTYPE.itemsize)
        if len(raw_header) < _HEADER_DTYPE.itemsize:
            raise ValueError(
                f"File too short to contain a header: {path} "
                f"({len(raw_header)} bytes)"
            )

        header = np.frombuffer(raw_header, dtype=_HEADER_DTYPE)[0]
        if bytes(header["magic"]) != _MAGIC:
            raise ValueError(
                f"Unrecognized file format (magic={header['magic']!r}, "
                f"expected {_MAGIC!r}). This loader only reads the NBS2 "
                f"format; the file may be from an older writer version "
                f"or may not be a simulation output file at all."
            )

        n = int(header["n_bodies"])
        frames = int(header["num_frames"])

        mass = np.frombuffer(f.read(n * 4), dtype=np.float32).copy()
        radius = np.frombuffer(f.read(n * 4), dtype=np.float32).copy()
        if mass.size != n or radius.size != n:
            raise ValueError(
                f"File truncated before mass/radius block finished: "
                f"got mass={mass.size}, radius={radius.size}, expected {n}. "
                f"File may have been cut off very early in writing."
            )

        frame_data = np.frombuffer(f.read(), dtype=np.float32)

    floats_per_frame = n * _FLOATS_PER_BODY_PER_FRAME
    expected = frames * floats_per_frame

    completed = int(header["completed_frames"])
    if completed > 0:
        # Trust the live-updated counter; it reflects exactly how many
        # frames were fully written, regardless of what's physically
        # present past that point.
        actual_frames = min(completed, frame_data.size // floats_per_frame)
    else:
        # Older file (no live counter) or a crash before the very first
        # per-frame update landed: fall back to inferring from size.
        actual_frames = frame_data.size // floats_per_frame

    if actual_frames < frames:
        print(
            f"Warning: truncated file. Using {actual_frames}/{frames} "
            f"frames."
        )

    if actual_frames == 0:
        raise ValueError(
            f"No complete frames found in {path} "
            f"(frame data size={frame_data.size} float32s, "
            f"needs at least {floats_per_frame} per frame)."
        )

    frame_data = frame_data[: actual_frames * floats_per_frame]
    data = frame_data.reshape(actual_frames, n, _FLOATS_PER_BODY_PER_FRAME)

    return {
        "header": header,
        "pos": data[:, :, 0:2],
        "vel": data[:, :, 2:4],
        "mass": mass,
        "radius": radius,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <sim_output.bin>")
        sys.exit(1)

    sim = load_sim(sys.argv[1])
    h = sim["header"]
    print(f"n_bodies={h['n_bodies']} total_steps={h['total_steps']} "
          f"output_every={h['output_every']} num_frames={h['num_frames']} "
          f"completed_frames={h['completed_frames']} dt={h['dt']}")
    print(f"pos shape: {sim['pos'].shape}")
    print(f"mass shape: {sim['mass'].shape}, radius shape: {sim['radius'].shape}")
    print(f"any NaN in pos: {np.isnan(sim['pos']).any()}")
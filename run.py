from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

from load_sim import load_sim


def _resolve_input_path(input_path: str | None) -> Path:
	if input_path is not None:
		path = Path(input_path)
		if not path.is_file():
			raise FileNotFoundError(f"Input file not found: {path}")
		return path

	sim_output_dir = Path("sim_output")
	candidates = sorted(sim_output_dir.glob("*.bin"), key=lambda p: p.stat().st_mtime)
	if not candidates:
		raise FileNotFoundError(
			"No .bin simulation files found in sim_output. "
			"Pass an explicit input file path."
		)
	return candidates[-1]


def _writer_available(name: str) -> bool:
	from matplotlib import animation

	try:
		return animation.writers.is_available(name)
	except Exception:
		return False


def _choose_format(requested: str) -> str:
	if requested == "mp4":
		if not _writer_available("ffmpeg"):
			raise RuntimeError("MP4 requested but ffmpeg writer is unavailable.")
		return "mp4"

	if requested == "gif":
		if not _writer_available("pillow"):
			raise RuntimeError("GIF requested but pillow writer is unavailable.")
		return "gif"

	if _writer_available("ffmpeg"):
		return "mp4"
	if _writer_available("pillow"):
		return "gif"

	raise RuntimeError("No supported animation writer found (ffmpeg or pillow).")


def _resolve_output_path(output: str | None, input_path: Path, fmt: str) -> Path:
	if output is None:
		return input_path.with_suffix(f".{fmt}")

	out = Path(output)
	if out.suffix.lower() not in {".mp4", ".gif"}:
		out = out.with_suffix(f".{fmt}")
	return out


def _compute_plot_bounds(pos: np.ndarray) -> tuple[float, float, float, float]:
	min_xy = np.min(pos.reshape(-1, 2), axis=0)
	max_xy = np.max(pos.reshape(-1, 2), axis=0)
	center = 0.5 * (min_xy + max_xy)
	span = max(max_xy[0] - min_xy[0], max_xy[1] - min_xy[1])
	span = max(span, 1e-6)
	half = 0.55 * span
	return center[0] - half, center[0] + half, center[1] - half, center[1] + half


def _marker_sizes(radius: np.ndarray, max_marker: float, min_marker: float = 5.0) -> np.ndarray:
	r = np.maximum(radius, 0.0)
	r_scale = np.percentile(r, 95)
	if r_scale <= 0:
		r_scale = 1.0
	sizes = min_marker + (r / r_scale) * max_marker
	return np.clip(sizes, 5.0, max_marker + min_marker)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Export N-body simulation visualization.")
	parser.add_argument("input", nargs="?", help="Path to .bin simulation output file.")
	parser.add_argument(
		"-o",
		"--output",
		help="Output animation file path. Default: input filename with chosen extension.",
	)
	parser.add_argument(
		"--format",
		choices=["auto", "mp4", "gif"],
		default="auto",
		help="Export format. Default: auto (prefer mp4, fallback gif).",
	)
	parser.add_argument("--fps", type=int, default=30, help="Output frames per second.")
	parser.add_argument("--dpi", type=int, default=120, help="Output DPI.")
	parser.add_argument(
		"--stride",
		type=int,
		default=1,
		help="Use every Nth simulation frame to speed up rendering.",
	)
	parser.add_argument(
		"--max-marker",
		type=float,
		default=140.0,
		help="Added marker size scale for large bodies (max size = min-marker + max-marker).",
	)
	parser.add_argument(
		"--min-marker",
		type=float,
		default=5.0,
		help="Floor marker size for the smallest bodies.",
	)
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	if args.fps <= 0:
		raise ValueError("--fps must be positive.")
	if args.dpi <= 0:
		raise ValueError("--dpi must be positive.")
	if args.stride <= 0:
		raise ValueError("--stride must be positive.")
	if args.max_marker <= 0:
		raise ValueError("--max-marker must be positive.")
	if args.min_marker < 0:
		raise ValueError("--min-marker must be non-negative.")

	input_path = _resolve_input_path(args.input)

	try:
		import matplotlib.pyplot as plt
		from matplotlib import animation
		from matplotlib.colors import LinearSegmentedColormap, Normalize
	except ImportError as exc:
		raise RuntimeError(
			"matplotlib is required for visualization export. "
			"Install with: pip install matplotlib pillow"
		) from exc

	chosen_format = _choose_format(args.format)
	output_path = _resolve_output_path(args.output, input_path, chosen_format)

	sim = load_sim(str(input_path))
	h = sim["header"]

	pos = sim["pos"][:: args.stride]
	vel = sim["vel"][:: args.stride]
	# mass/radius are constant across the run (written once in the file),
	# so they're 1D (n_bodies,) arrays now -- no per-frame stride/indexing.
	radius = sim["radius"]

	speeds = np.linalg.norm(vel, axis=2)
	speed_max = float(np.nanmax(speeds))
	if not np.isfinite(speed_max) or speed_max <= 0:
		speed_max = 1.0
	norm = Normalize(vmin=0.0, vmax=speed_max)
	magenta_white = LinearSegmentedColormap.from_list("magenta_white", ["#ff00e5", "#ffffff"])

	sizes = _marker_sizes(radius, args.max_marker, args.min_marker)
	x_min, x_max, y_min, y_max = _compute_plot_bounds(pos)

	fig, ax = plt.subplots(figsize=(16, 16))
	fig.patch.set_facecolor("black")
	ax.set_facecolor("black")
	fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
	ax.set_aspect("equal", adjustable="box")
	ax.set_xlim(x_min, x_max)
	ax.set_ylim(y_min, y_max)
	ax.set_axis_off()

	scatter = ax.scatter(
		pos[0, :, 0],
		pos[0, :, 1],
		c=speeds[0],
		s=sizes,
		cmap=magenta_white,
		norm=norm,
		alpha=1.0,
		edgecolors="none",
	)

	total_steps = int(h["total_steps"])
	output_every = int(h["output_every"])
	dt = float(h["dt"])
	stride_step = args.stride * output_every

	info_text = ax.text(
		0.98,
		0.98,
		"",
		transform=ax.transAxes,
		va="top",
		ha="right",
		fontsize=11,
		color="#d4fbff",
	)

	def _update(i: int):
		scatter.set_offsets(pos[i])
		scatter.set_array(speeds[i])
		# sizes are constant across frames now, so no set_sizes() call needed.
		sim_step = i * stride_step
		sim_time = sim_step * dt
		info_text.set_text(
			f"frame {i + 1}/{len(pos)}\\n"
			f"step {sim_step}/{total_steps}\\n"
			f"time {sim_time:.3f}"
		)
		return scatter, info_text

	anim = animation.FuncAnimation(
		fig,
		_update,
		frames=len(pos),
		interval=1000 / args.fps,
		blit=False,
	)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	if output_path.suffix.lower() == ".mp4":
		writer = animation.FFMpegWriter(fps=args.fps, bitrate=2400)
	elif output_path.suffix.lower() == ".gif":
		writer = animation.PillowWriter(fps=args.fps)
	else:
		raise RuntimeError(f"Unsupported output extension: {output_path.suffix}")

	anim.save(str(output_path), writer=writer, dpi=args.dpi)
	plt.close(fig)

	print(f"Input:  {input_path}")
	print(f"Output: {output_path}")
	print(f"Frames: {len(pos)}")
	print(f"FPS:    {args.fps}")
	print(f"Format: {output_path.suffix[1:]}")
	return 0


if __name__ == "__main__":
	try:
		raise SystemExit(main())
	except Exception as exc:
		print(f"Error: {exc}", file=sys.stderr)
		raise SystemExit(1)
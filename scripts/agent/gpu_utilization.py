# Copyright 2026 the Awesome-Perception-Aware-RLVR authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""GPU utilization of a training run's rollouts (generation phase), from nvidia-smi samples.

Run ``record`` next to the training on the same machine: it samples the GPUs once a second and notes when each step
ends (a new line in the run's ``experiment_log.jsonl``, written by ``trainer.logger`` with ``file``). ``report`` then
places each step's generation window, ``timing_s.gen`` long from the step's start (``timing_s.step`` before its end;
generation comes first in a step), and reports per step:

- utilization: the mean ``utilization.gpu`` over the window and the GPUs;
- active: the share of GPU-seconds with utilization of at least ``--active`` percent (10), and the mean utilization
  of those (their product is the utilization);
- busiest / idlest: the active share of the busiest and of the idlest GPU;
- gen_s, step_s: the step's generation and total time.

    python scripts/agent/gpu_utilization.py record --log <ckpt dir>/experiment_log.jsonl --out <dir> [--gpus 0,1,2,3]
    python scripts/agent/gpu_utilization.py report --dir <dir> [--skip 1]
"""

import argparse
import csv
import json
import os
import subprocess
import time


def record(args: argparse.Namespace) -> None:
    os.makedirs(args.out, exist_ok=True)
    query = ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits"]
    if args.gpus:
        query.append(f"--id={args.gpus}")
    lines_seen, training_steps = 0, 0
    with (
        open(os.path.join(args.out, "samples.csv"), "a", newline="") as samples,
        open(os.path.join(args.out, "steps.csv"), "a", newline="") as steps,
        open(os.path.join(args.out, "timing.jsonl"), "a") as timing,
    ):
        sample_writer, step_writer = csv.writer(samples), csv.writer(steps)
        while args.steps is None or training_steps < args.steps:
            now = time.time()
            for line in subprocess.run(query, capture_output=True, text=True, check=True).stdout.splitlines():
                index, utilization, memory = (field.strip() for field in line.split(","))
                sample_writer.writerow([f"{now:.3f}", index, utilization, memory])
            if os.path.exists(args.log):
                with open(args.log) as log:
                    lines = [line for line in log if line.strip()]
                for line in lines[lines_seen:]:
                    row = json.loads(line)
                    if "timing_s" in row:  # a training step (validation rows have no timings)
                        step_writer.writerow([f"{now:.3f}", row["step"]])
                        timing.write(json.dumps({"step": row["step"], "timing_s": row["timing_s"]}) + "\n")
                        training_steps += 1
                lines_seen = len(lines)
            for handle in (samples, steps, timing):
                handle.flush()
            time.sleep(max(0.0, args.interval - (time.time() - now)))


def report(args: argparse.Namespace) -> None:
    samples: dict[float, dict[str, float]] = {}
    with open(os.path.join(args.dir, "samples.csv")) as handle:
        for when, index, utilization, _ in csv.reader(handle):
            samples.setdefault(float(when), {})[index] = float(utilization)
    with open(os.path.join(args.dir, "steps.csv")) as handle:
        ends = {int(step): float(when) for when, step in csv.reader(handle)}
    with open(os.path.join(args.dir, "timing.jsonl")) as handle:
        timings = {}
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if "timing_s" in row:  # validation rows have none
                    timings[row["step"]] = row["timing_s"]

    print("step  gen_s  step_s  utilization  active  util_when_active  busiest  idlest")
    totals = []
    for step in sorted(ends):
        if step <= args.skip or step not in timings:
            continue
        start = ends[step] - timings[step]["step"]
        window = [values for when, values in samples.items() if start <= when <= start + timings[step]["gen"]]
        if not window:
            continue
        gpus = sorted(window[0])
        cells = [values[gpu] for values in window for gpu in gpus if gpu in values]
        active = [value for value in cells if value >= args.active]
        per_gpu = [sum(values.get(gpu, 0) >= args.active for values in window) / len(window) for gpu in gpus]
        row = (
            sum(cells) / len(cells),
            len(active) / len(cells),
            sum(active) / len(active) if active else 0.0,
            max(per_gpu),
            min(per_gpu),
        )
        totals.append((timings[step]["gen"], timings[step]["step"], *row))
        print(
            f"{step:4d} {timings[step]['gen']:6.0f} {timings[step]['step']:7.0f} {row[0]:11.1f}% {row[1]:6.0%} "
            f"{row[2]:16.1f}% {row[3]:8.0%} {row[4]:7.0%}"
        )
    if totals:
        means = [sum(column) / len(totals) for column in zip(*totals)]
        print(
            f"mean {means[0]:6.0f} {means[1]:7.0f} {means[2]:11.1f}% {means[3]:6.0%} {means[4]:16.1f}% "
            f"{means[5]:8.0%} {means[6]:7.0%}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    record_parser = commands.add_parser("record", help="sample the GPUs while a run trains")
    record_parser.add_argument("--log", required=True, help="the run's experiment_log.jsonl")
    record_parser.add_argument("--out", required=True, help="directory for the samples")
    record_parser.add_argument("--gpus", default=None, help="nvidia-smi ids of the run's GPUs (default: all)")
    record_parser.add_argument("--interval", type=float, default=1.0, help="seconds between samples")
    record_parser.add_argument("--steps", type=int, default=None, help="stop after this many steps")
    report_parser = commands.add_parser("report", help="utilization of each step's generation")
    report_parser.add_argument("--dir", required=True, help="the directory of record")
    report_parser.add_argument("--skip", type=int, default=1, help="leave out the first steps (warm-up)")
    report_parser.add_argument("--active", type=float, default=10.0, help="utilization (percent) counted as active")
    args = parser.parse_args()
    record(args) if args.command == "record" else report(args)


if __name__ == "__main__":
    main()

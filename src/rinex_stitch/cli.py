from __future__ import annotations

import argparse
import logging
import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple


@dataclass
class EpochBlock:
    time: datetime
    lines: List[str]


@dataclass
class RinexFile:
    path: Path
    header_lines: List[str]
    version: str
    epoch_blocks: List[EpochBlock]


RINEX_VERSION_RE = re.compile(r"^(?P<ver>\d\.\d{2})\s+RINEX VERSION / TYPE")
END_OF_HEADER = "END OF HEADER"

# RINEX v3 epoch header line, starts with '>'
R3_EPOCH_RE = re.compile(
    r"^>\s*(?P<year>\d{4})\s+(?P<mon>\d{1,2})\s+(?P<day>\d{1,2})\s+"
    r"(?P<hour>\d{1,2})\s+(?P<min>\d{1,2})\s+(?P<sec>\d{1,2})(?:\.(?P<frac>\d+))?\s+"
)

# RINEX v2 epoch header line (no '>'), two-digit year
R2_EPOCH_RE = re.compile(
    r"^\s*(?P<year>\d{2})\s+(?P<mon>\d{1,2})\s+(?P<day>\d{1,2})\s+"
    r"(?P<hour>\d{1,2})\s+(?P<min>\d{1,2})\s+(?P<sec>\d{1,2})(?:\.(?P<frac>\d+))?\s+"
)


def _parse_header(lines: Iterator[str]) -> Tuple[List[str], str]:
    header: List[str] = []
    version = ""
    for line in lines:
        header.append(line)
        if not version:
            m = RINEX_VERSION_RE.search(line)
            if m:
                version = m.group("ver")
        if END_OF_HEADER in line:
            break
    if not header or END_OF_HEADER not in header[-1]:
        raise ValueError("Invalid RINEX: missing END OF HEADER")
    if not version:
        # Default: try v3 if '>' epoch lines appear, else v2
        version = "3.00"
    return header, version


def _to_datetime_v2(y2: int, mon: int, day: int, hour: int, minute: int, sec: int, frac: Optional[str]) -> datetime:
    # RINEX v2 uses two-digit years. Map 80-99 -> 1900s, 00-79 -> 2000s
    year = 2000 + y2 if y2 < 80 else 1900 + y2
    micro = int(float("0." + frac) * 1_000_000) if frac else 0  # type: ignore
    return datetime(year, mon, day, hour, minute, sec, micro, tzinfo=timezone.utc)


def _to_datetime_v3(year: int, mon: int, day: int, hour: int, minute: int, sec: int, frac: Optional[str]) -> datetime:
    micro = int(float("0." + frac) * 1_000_000) if frac else 0  # type: ignore
    return datetime(year, mon, day, hour, minute, sec, micro, tzinfo=timezone.utc)


def _is_epoch_line_v2(line: str) -> Optional[datetime]:
    m = R2_EPOCH_RE.match(line)
    if not m:
        return None
    try:
        y2 = int(m.group("year"))
        mon = int(m.group("mon"))
        day = int(m.group("day"))
        hour = int(m.group("hour"))
        minute = int(m.group("min"))
        sec = int(m.group("sec"))
        frac = m.group("frac")
        return _to_datetime_v2(y2, mon, day, hour, minute, sec, frac)
    except Exception:
        return None


def _is_epoch_line_v3(line: str) -> Optional[datetime]:
    if not line.startswith(">"):
        return None
    m = R3_EPOCH_RE.match(line)
    if not m:
        return None
    try:
        year = int(m.group("year"))
        mon = int(m.group("mon"))
        day = int(m.group("day"))
        hour = int(m.group("hour"))
        minute = int(m.group("min"))
        sec = int(m.group("sec"))
        frac = m.group("frac")
        return _to_datetime_v3(year, mon, day, hour, minute, sec, frac)
    except Exception:
        return None


def parse_rinex_file(path: Path) -> RinexFile:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        lines_iter = iter(f)
        header, ver = _parse_header(lines_iter)
        is_v3 = ver.startswith("3")

        epoch_blocks: List[EpochBlock] = []
        current_lines: List[str] = []
        current_time: Optional[datetime] = None

        def flush_current():
            nonlocal current_lines, current_time
            if current_time is not None and current_lines:
                epoch_blocks.append(EpochBlock(current_time, current_lines))
            current_lines = []
            current_time = None

        for line in lines_iter:
            ts = _is_epoch_line_v3(line) if is_v3 else _is_epoch_line_v2(line)
            if ts is not None:
                # New epoch starts: flush previous block
                flush_current()
                current_time = ts
                current_lines = [line]
            else:
                if current_lines is not None:
                    current_lines.append(line)

        # Flush tail
        flush_current()

    if not epoch_blocks:
        raise ValueError(f"No epochs parsed in {path}")

    return RinexFile(path=path, header_lines=header, version=ver, epoch_blocks=epoch_blocks)


# ----- Header utilities -----

def _label_index(header: List[str], label: str) -> Optional[int]:
    for i, ln in enumerate(header):
        if label in ln:
            return i
    return None


def _content_for_label(line: str, label: str) -> str:
    idx = line.find(label)
    if idx >= 0:
        return line[:idx].rstrip()
    return line[:60].rstrip()


def _make_header_line(content: str, label: str) -> str:
    content = content[:60]
    return f"{content:<60}{label}\n"


@dataclass
class SessionHeader:
    observer: Optional[str] = None
    agency: Optional[str] = None
    ant_type: Optional[str] = None
    ant_serial: Optional[str] = None
    dH: Optional[float] = None
    dE: Optional[float] = None
    dN: Optional[float] = None


def _extract_session_defaults(header: List[str]) -> SessionHeader:
    sh = SessionHeader()
    i = _label_index(header, "OBSERVER / AGENCY")
    if i is not None:
        content = _content_for_label(header[i], "OBSERVER / AGENCY")
        sh.observer = content[:20].strip() or None
        sh.agency = content[20:].strip() or None
    i_type = _label_index(header, "ANTENNA: TYPE")
    if i_type is not None:
        content = _content_for_label(header[i_type], "ANTENNA: TYPE")
        sh.ant_type = content.strip() or None
    else:
        i_old = _label_index(header, "ANT # / TYPE")
        if i_old is not None:
            content = _content_for_label(header[i_old], "ANT # / TYPE")
            parts = content.split()
            if parts:
                sh.ant_serial = parts[0]
                if len(parts) > 1:
                    sh.ant_type = " ".join(parts[1:])
    i_serial = _label_index(header, "ANTENNA: SERIAL NO")
    if i_serial is not None:
        content = _content_for_label(header[i_serial], "ANTENNA: SERIAL NO")
        sh.ant_serial = content.strip() or sh.ant_serial
    i_dhen = _label_index(header, "ANTENNA: DELTA H/E/N")
    if i_dhen is not None:
        content = _content_for_label(header[i_dhen], "ANTENNA: DELTA H/E/N")
        parts = content.split()
        try:
            if len(parts) >= 1:
                sh.dH = float(parts[0])
            if len(parts) >= 2:
                sh.dE = float(parts[1])
            if len(parts) >= 3:
                sh.dN = float(parts[2])
        except ValueError:
            pass
    return sh


def _prompt(default: Optional[str], prompt_text: str) -> Optional[str]:
    try:
        val = input(f"{prompt_text} [{default or ''}]: ").strip()
        return default if val == "" else val
    except (EOFError, KeyboardInterrupt):
        return default


def _prompt_float(default: Optional[float], prompt_text: str) -> Optional[float]:
    while True:
        sdef = "" if default is None else str(default)
        try:
            val = input(f"{prompt_text} [{sdef}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            return default
        if val == "":
            return default
        try:
            return float(val)
        except ValueError:
            print("Please enter a number.")


def interactive_update_header(orig: List[str]) -> List[str]:
    header = orig[:]
    defaults = _extract_session_defaults(header)
    print("-- Verify session info (blank to keep) --")
    observer = _prompt(defaults.observer, "Observer")
    agency = _prompt(defaults.agency, "Agency/Company")
    ant_type = _prompt(defaults.ant_type, "Antenna Type")
    ant_serial = _prompt(defaults.ant_serial, "Antenna Serial")
    dH = _prompt_float(defaults.dH, "ARP Height (dH, m)")
    dE = _prompt_float(defaults.dE, "ARP East (dE, m)")
    dN = _prompt_float(defaults.dN, "ARP North (dN, m)")

    idx = _label_index(header, "OBSERVER / AGENCY")
    if idx is not None and (observer is not None or agency is not None):
        obs = observer or ""
        ag = agency or ""
        content = f"{obs:<20}{ag:<40}"[:60]
        header[idx] = _make_header_line(content, "OBSERVER / AGENCY")

    idx = _label_index(header, "ANTENNA: TYPE")
    if idx is not None and (ant_type is not None):
        header[idx] = _make_header_line(ant_type or "", "ANTENNA: TYPE")
    else:
        idx_old = _label_index(header, "ANT # / TYPE")
        if idx_old is not None and (ant_type is not None or ant_serial is not None):
            serial = ant_serial or ""
            atype = ant_type or ""
            content = f"{serial} {atype}".strip()
            header[idx_old] = _make_header_line(content, "ANT # / TYPE")

    idx = _label_index(header, "ANTENNA: SERIAL NO")
    if idx is not None and ant_serial is not None:
        header[idx] = _make_header_line(ant_serial or "", "ANTENNA: SERIAL NO")

    idx = _label_index(header, "ANTENNA: DELTA H/E/N")
    if idx is not None and (dH is not None or dE is not None or dN is not None):
        h = defaults.dH if dH is None else dH
        e = defaults.dE if dE is None else dE
        n = defaults.dN if dN is None else dN
        parts: List[str] = []
        if h is not None:
            parts.append(f"{h:14.4f}".strip())
        if e is not None:
            parts.append(f"{e:14.4f}".strip())
        if n is not None:
            parts.append(f"{n:14.4f}".strip())
        content = " ".join(parts)
        header[idx] = _make_header_line(content, "ANTENNA: DELTA H/E/N")

    return header


def detect_interval_seconds(blocks: List[EpochBlock]) -> float:
    dts = []
    for a, b in zip(blocks, blocks[1:]):
        dt = (b.time - a.time).total_seconds()
        if dt > 0:
            dts.append(round(dt, 6))
    if not dts:
        return 0.0
    # Use the mode (most common delta)
    c = Counter(dts)
    interval, _ = max(c.items(), key=lambda kv: kv[1])
    return float(interval)


def split_blocks_by_gaps(blocks: List[EpochBlock], interval: float, tolerance: float) -> List[List[EpochBlock]]:
    if not blocks:
        return []
    if interval <= 0:
        # No interval info; no splitting
        return [blocks]
    segments: List[List[EpochBlock]] = []
    current: List[EpochBlock] = [blocks[0]]
    for prev, nxt in zip(blocks, blocks[1:]):
        dt = (nxt.time - prev.time).total_seconds()
        if dt > interval + tolerance:
            segments.append(current)
            current = [nxt]
        else:
            current.append(nxt)
    segments.append(current)
    return segments


def _header_label(line: str) -> str:
    # Label occupies columns 61-80 in RINEX headers
    return line[60:].strip()


IGNORE_HEADER_LABELS = {
    "PGM / RUN BY / DATE",
    "TIME OF FIRST OBS",
    "TIME OF LAST OBS",
}


def _normalized_header(header: List[str]) -> List[str]:
    norm: List[str] = []
    for ln in header:
        label = _header_label(ln)
        if label in IGNORE_HEADER_LABELS:
            continue
        # Normalize trailing whitespace in content area
        content = ln[:60].rstrip()
        norm.append(f"{content}|{label}")
    return norm


def headers_compatible(h1: List[str], h2: List[str]) -> bool:
    return _normalized_header(h1) == _normalized_header(h2)


@dataclass
class Segment:
    header: List[str]
    source_files: List[Path]
    blocks: List[EpochBlock]

    @property
    def start(self) -> datetime:
        return self.blocks[0].time

    @property
    def end(self) -> datetime:
        return self.blocks[-1].time


def merge_adjacent_segments(segments: List[Segment], interval: float, tolerance: float) -> List[Segment]:
    if not segments:
        return []
    segments = sorted(segments, key=lambda s: s.start)
    merged: List[Segment] = []
    cur = segments[0]
    for nxt in segments[1:]:
        # Merge if headers match and adjacency within tolerance
        if headers_compatible(cur.header, nxt.header):
            gap = (nxt.start - cur.end).total_seconds()
            if 0 <= gap <= interval + tolerance:
                # Also avoid duplicate epoch if exactly zero gap and same time
                blocks = cur.blocks.copy()
                if nxt.start == cur.end:
                    # Drop first block of next if duplicate timestamp
                    nxt_blocks = nxt.blocks[1:]
                else:
                    nxt_blocks = nxt.blocks

                cur = Segment(
                    header=cur.header,
                    source_files=cur.source_files + nxt.source_files,
                    blocks=blocks + nxt_blocks,
                )
                continue
        merged.append(cur)
        cur = nxt
    merged.append(cur)
    return merged


def _common_stem(paths: List[Path]) -> str:
    if not paths:
        return "segment"
    stems = [p.stem for p in paths]
    # Use the longest common prefix of stems
    prefix = os.path.commonprefix(stems).rstrip("-_ .")
    return prefix or stems[0]


def _fmt_ts(dt: datetime) -> str:
    return dt.strftime("%Y%m%dT%H%M%S")


NAME_TEMPLATE_DEFAULT = "{stem}_{start:%Y%m%dT%H%M%S}_{end:%Y%m%dT%H%M%S}.rnx"


def _render_name_template(template: str, stem: str, start: datetime, end: datetime) -> str:
    # Support placeholders: {stem}, {start}, {end} with optional strftime, e.g. {start:%Y%m%d}
    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        fmt = match.group(2)
        if key == "stem":
            return stem
        if key == "start":
            return start.strftime(fmt or "%Y%m%dT%H%M%S")
        if key == "end":
            return end.strftime(fmt or "%Y%m%dT%H%M%S")
        return match.group(0)

    pattern = re.compile(r"\{(stem|start|end)(?::([^}]+))?\}")
    name = pattern.sub(repl, template)
    # Basic sanitization
    bad = [os.sep]
    if os.altsep:
        bad.append(os.altsep)
    for ch in bad:
        name = name.replace(ch, "_")
    name = name.strip().replace(" ", "_")
    return name


def write_segment(
    seg: Segment,
    out_dir: Path,
    name_hint: Optional[str] = None,
    dry_run: bool = False,
    name_template: str = NAME_TEMPLATE_DEFAULT,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    hint = name_hint or _common_stem(seg.source_files)
    fname = _render_name_template(name_template, hint, seg.start, seg.end)
    out_path = out_dir / fname
    if dry_run:
        return out_path
    with out_path.open("w", encoding="utf-8") as w:
        for ln in seg.header:
            w.write(ln)
        # Ensure header ends with newline if not already
        if not seg.header[-1].endswith("\n"):
            w.write("\n")
        for blk in seg.blocks:
            for ln in blk.lines:
                w.write(ln)
    return out_path


def build_cli(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split and merge RINEX observation files at epoch gaps and boundaries.")
    p.add_argument("inputs", nargs="+", help="RINEX observation files (supports shell globs expanded by shell)")
    p.add_argument("--output-dir", default="out", help="Directory to write output files (default: out)")
    p.add_argument("--interval", type=float, default=None, help="Sampling interval in seconds (auto-detect by default)")
    p.add_argument("--tolerance", type=float, default=0.1, help="Tolerance in seconds for gap/adjacency detection")
    p.add_argument("--no-merge", action="store_true", help="Disable cross-file merging")
    p.add_argument(
        "--name-template",
        default=NAME_TEMPLATE_DEFAULT,
        help=(
            "Output filename template with placeholders {stem}, {start}, {end}; "
            "use strftime like {start:%Y%m%dT%H%M%S}"
        ),
    )
    p.add_argument("--dry-run", action="store_true", help="Plan only; do not write files")
    p.add_argument("--verify-header", action="store_true", help="Interactively verify and update session header fields (Observer, Agency, Antenna, ARP)")
    p.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    return p.parse_args(argv)


def setup_logging(level: str) -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=lvl, format="%(levelname)s: %(message)s")


def main(argv: Optional[List[str]] = None) -> int:
    args = build_cli(argv)
    setup_logging(args.log_level)

    input_paths: List[Path] = []
    for arg in args.inputs:
        p = Path(arg)
        if p.exists():
            input_paths.append(p)
        else:
            logging.warning("Path does not exist: %s", arg)
    if not input_paths:
        logging.error("No valid input files provided")
        return 2

    parsed: List[RinexFile] = []
    for pth in sorted(input_paths):
        try:
            rf = parse_rinex_file(pth)
            parsed.append(rf)
            logging.info("Parsed %s: version=%s epochs=%d", pth.name, rf.version, len(rf.epoch_blocks))
        except Exception as e:
            logging.error("Failed to parse %s: %s", pth, e)
            return 2

    # Determine interval if not provided
    interval: float
    if args.interval is None:
        all_blocks = [blk for rf in parsed for blk in rf.epoch_blocks]
        interval = detect_interval_seconds(all_blocks)
        if interval <= 0:
            logging.warning("Could not auto-detect interval; splitting disabled without --interval")
            interval = 0.0
        else:
            logging.info("Auto-detected interval: %.3f s", interval)
    else:
        interval = float(args.interval)
        logging.info("Using provided interval: %.3f s", interval)

    # Optionally verify/update session header fields once (applied to all outputs)
    updated_header_template: Optional[List[str]] = None
    if args.verify_header:
        base_header = parsed[0].header_lines
        updated_header_template = interactive_update_header(base_header)

    # Split within files
    per_file_segments: List[Segment] = []
    for rf in parsed:
        splits = split_blocks_by_gaps(rf.epoch_blocks, interval, args.tolerance)
        if len(splits) > 1:
            logging.info("Split %s into %d segments due to gaps", rf.path.name, len(splits))
        for i, seg_blocks in enumerate(splits, start=1):
            hdr = updated_header_template if updated_header_template is not None else rf.header_lines
            per_file_segments.append(Segment(header=hdr, source_files=[rf.path], blocks=seg_blocks))
            if len(splits) > 1:
                logging.debug("  Segment %d: %s -> %s (%d epochs)", i, _fmt_ts(seg_blocks[0].time), _fmt_ts(seg_blocks[-1].time), len(seg_blocks))

    # Merge across files
    final_segments: List[Segment]
    if args.no_merge or interval <= 0:
        final_segments = sorted(per_file_segments, key=lambda s: s.start)
        if args.no_merge:
            logging.info("Cross-file merging disabled by flag")
    else:
        before = len(per_file_segments)
        final_segments = merge_adjacent_segments(per_file_segments, interval, args.tolerance)
        after = len(final_segments)
        if after < before:
            logging.info("Merged segments across files: %d -> %d", before, after)
        else:
            logging.info("No cross-file merges performed")

    out_dir = Path(args.output_dir)
    written: List[Path] = []
    for seg in final_segments:
        hint = _common_stem(seg.source_files)
        out_path = write_segment(
            seg,
            out_dir,
            hint,
            dry_run=args.dry_run,
            name_template=args.name_template,
        )
        if args.dry_run:
            logging.info("Would write: %s (%s -> %s, %d epochs)", out_path.name, _fmt_ts(seg.start), _fmt_ts(seg.end), len(seg.blocks))
        else:
            logging.info("Wrote: %s (%s -> %s, %d epochs)", out_path.name, _fmt_ts(seg.start), _fmt_ts(seg.end), len(seg.blocks))
            written.append(out_path)

    if args.dry_run:
        logging.info("Dry run complete; no files written")
    else:
        logging.info("Done: %d file(s) written to %s", len(written), out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

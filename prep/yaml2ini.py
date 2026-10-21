#!/usr/bin/env python3
"""Flatten a YAML mapping (and one level of nesting) to key=value ini."""
import argparse
import sys

try:
    import yaml
except ImportError:
    yaml = None


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "." if prefix else k + "."))
        elif isinstance(v, list):
            out[key] = ",".join(str(x) for x in v)
        else:
            out[k if not prefix else key] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("yaml_path")
    ap.add_argument("-o", "--output", default="-")
    args = ap.parse_args()
    if yaml is None:
        sys.stderr.write("pyyaml is required\n")
        return 1
    with open(args.yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    flat = flatten(data)
    lines = [f"{k}={v}" for k, v in flat.items()]
    text = "\n".join(lines) + "\n"
    if args.output == "-":
        sys.stdout.write(text)
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Reconciles config/atmos-gl.json's SHAPE against config/atmos-gl.json.tmpl:
adds any section/option the template defines that the live file is missing,
and drops any live section/option the template no longer defines -- without
touching the VALUE of anything already present in both (a schema sync, not a
values reset).

Recurses into nested dicts (data_collector.datasources,
data_collector.channel_enabled) so additions/removals inside those are
caught too. Lists and scalars are left as opaque leaf values and never
overwritten -- e.g. satellites.sat_names is genuine live data, not schema,
and survives untouched even though its key exists in both files.

Run automatically by `make bootstrap-config` (via `make up`/`make prod`)
whenever the live file already exists. A missing live file is still just a
straight copy of the template -- handled by the Makefile before this script
would even have anything to reconcile against.
"""
import argparse
import copy
import json
import sys


def sync_shape(live: dict, template: dict) -> tuple[dict, list[str], list[str]]:
    """Returns (synced, added, removed); added/removed are dotted-path labels."""
    added: list[str] = []
    removed: list[str] = []

    def _sync(live_node: dict, template_node: dict, prefix: str) -> dict:
        result = {}
        for key, tmpl_val in template_node.items():
            path = f"{prefix}.{key}" if prefix else key
            if key in live_node:
                live_val = live_node[key]
                if isinstance(tmpl_val, dict) and isinstance(live_val, dict):
                    result[key] = _sync(live_val, tmpl_val, path)
                else:
                    result[key] = live_val
            else:
                result[key] = copy.deepcopy(tmpl_val)
                added.append(path)
        for key in live_node:
            if key not in template_node:
                removed.append(f"{prefix}.{key}" if prefix else key)
        return result

    synced = _sync(live, template, "")
    return synced, added, removed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", default="config/atmos-gl.json")
    parser.add_argument("--template", default="config/atmos-gl.json.tmpl")
    args = parser.parse_args()

    with open(args.template) as f:
        template = json.load(f)
    with open(args.live) as f:
        live = json.load(f)

    synced, added, removed = sync_shape(live, template)

    if not added and not removed:
        print(f"{args.live}: already in sync with {args.template}")
        return 0

    with open(args.live, "w") as f:
        json.dump(synced, f, indent=2)

    if added:
        print(f"{args.live}: added {len(added)} key(s) from template: {', '.join(added)}")
    if removed:
        print(f"{args.live}: removed {len(removed)} key(s) no longer in template: {', '.join(removed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

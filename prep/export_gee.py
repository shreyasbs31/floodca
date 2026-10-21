#!/usr/bin/env python3
"""Earth Engine exports for FloodCA (SPEC Sec 4.4, 4.5, 4.6).

Run on a Mac after `earthengine authenticate`. Do not run GEE auth on the cluster.

Setup (once):
    python3 -m venv ~/.venvs/floodca
    source ~/.venvs/floodca/bin/activate
    pip install earthengine-api
    earthengine authenticate
    export GEE_PROJECT=your-ee-project-id

JRC permanent-water layer only (after the max_extent band fix):
    python3 scripts/export_gee.py --project "$GEE_PROJECT" --drive-folder FloodCA --jrc-only

Refresh event metadata only (dates, severity, AOI flood fraction, clear_perc).
Does not start Drive raster exports:
    python3 scripts/export_gee.py --project "$GEE_PROJECT" --list-only \\
        --out-json gee_event_list.json

Full original run (WorldCover + JRC + GFD rasters + listing):
    python3 prep/export_gee.py --project "$GEE_PROJECT" --drive-folder FloodCA

One extra GFD event, flooded and jrc_perm_water only. Does not list events, and does not export WorldCover or the top 8:
    python3 prep/export_gee.py --project "$GEE_PROJECT" --drive-folder FloodCA --only-event 4198

After Drive tasks finish, download the GeoTIFFs into data/raw/gee/ and copy them:
    rsync -avP data/raw/gee/ \\
        "$FCA_LOGIN:$FCA_DATA/raw/gee/"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone


def _init(project: str | None):
    import ee

    kwargs = {}
    if project:
        kwargs["project"] = project
    try:
        ee.Initialize(**kwargs)
    except Exception as exc:
        sys.stderr.write(
            f"ee.Initialize failed ({exc}). Run `earthengine authenticate` "
            "and pass --project or GEE_PROJECT.\n"
        )
        raise
    return ee


def _export_image(ee, image, region, desc, folder, scale, crs):
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=desc[:90],
        folder=folder,
        fileNamePrefix=desc[:90],
        region=region,
        scale=scale,
        crs=crs,
        maxPixels=1_000_000_000,
        fileFormat="GeoTIFF",
    )
    task.start()
    return {"description": desc, "id": task.id, "state": task.status().get("state")}


def _fmt_date(ee, img, prop):
    """Format an EE date property, or None if missing."""
    return ee.Algorithms.If(
        img.propertyNames().contains(prop),
        ee.Date(img.get(prop)).format("YYYY-MM-dd"),
        None,
    )


def _event_feature(ee, img, aoi):
    """One Feature with GFD properties plus AOI reducers (SPEC 4.4 listing)."""
    flood = img.select("flooded")
    perm = img.select("jrc_perm_water")
    land = perm.eq(0)
    land_flood = flood.eq(1).And(land)
    stacked = ee.Image.cat(
        [
            flood.rename("flooded"),
            perm.rename("perm"),
            land.rename("land"),
            land_flood.rename("land_flood"),
            img.select("clear_perc"),
            img.select("clear_views"),
        ]
    )
    stats = stacked.reduceRegion(
        reducer=ee.Reducer.mean().combine(ee.Reducer.sum(), sharedInputs=True),
        geometry=aoi,
        scale=250,
        maxPixels=1_000_000_000,
        bestEffort=True,
        tileScale=4,
    )
    land_sum = ee.Number(stats.get("land_sum"))
    land_flood_sum = ee.Number(stats.get("land_flood_sum"))
    land_flood_frac = ee.Algorithms.If(
        land_sum.gt(0), land_flood_sum.divide(land_sum), None
    )
    clear_land = (
        img.select("clear_perc").updateMask(land).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=250,
            maxPixels=1_000_000_000,
            bestEffort=True,
            tileScale=4,
        )
    )
    return ee.Feature(
        None,
        {
            "id": img.get("id"),
            "system_index": img.get("system:index"),
            "began": _fmt_date(ee, img, "system:time_start"),
            "ended": _fmt_date(ee, img, "system:time_end"),
            "began_property": "system:time_start",
            "ended_property": "system:time_end",
            "dfo_main_cause": img.get("dfo_main_cause"),
            "dfo_severity": img.get("dfo_severity"),
            "dfo_country": img.get("dfo_country"),
            "dfo_dead": img.get("dfo_dead"),
            "dfo_displaced": img.get("dfo_displaced"),
            "glide_index": img.get("glide_index"),
            "aoi_flooded_frac": stats.get("flooded_mean"),
            "aoi_perm_frac": stats.get("perm_mean"),
            "aoi_land_flood_frac": land_flood_frac,
            "aoi_clear_perc_mean": stats.get("clear_perc_mean"),
            "aoi_clear_perc_land_mean": clear_land.get("clear_perc"),
            "aoi_clear_views_mean": stats.get("clear_views_mean"),
        },
    )


def _collect_events(ee, collection, aoi, label):
    n = int(collection.size().getInfo())
    print(f"{label}: {n}")
    if n == 0:
        return [], None
    first = ee.Image(collection.first())
    meta = {
        "property_names": first.propertyNames().getInfo(),
        "band_names": first.bandNames().getInfo(),
    }
    fc = ee.FeatureCollection(collection.map(lambda img: _event_feature(ee, img, aoi)))
    feats = fc.getInfo().get("features", [])
    rows = []
    for f in feats:
        props = dict(f.get("properties") or {})
        rows.append(props)
    rows.sort(key=lambda r: (r.get("id") is None, r.get("id") if r.get("id") is not None else 0))
    return rows, meta


def _print_shortlist(rows, n=8):
    """Rank by AOI land-flood fraction, then severity, then land clear_perc."""

    def key(r):
        frac = r.get("aoi_land_flood_frac")
        sev = r.get("dfo_severity")
        clr = r.get("aoi_clear_perc_land_mean")
        return (
            -(frac if isinstance(frac, (int, float)) else -1.0),
            -(sev if isinstance(sev, (int, float)) else 0.0),
            -(clr if isinstance(clr, (int, float)) else 0.0),
        )

    ranked = sorted(rows, key=key)
    print("ranked udupi_coast events (land_flood_frac, severity, clear_perc_land):")
    for i, r in enumerate(ranked[:n], 1):
        print(
            f"  {i:2d}. id={r.get('id')} {r.get('began')} to {r.get('ended')} "
            f"sev={r.get('dfo_severity')} land_flood={r.get('aoi_land_flood_frac')} "
            f"clear_perc_land={r.get('aoi_clear_perc_land_mean')} "
            f"cause={r.get('dfo_main_cause')}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", default=os.environ.get("GEE_PROJECT"))
    ap.add_argument("--drive-folder", default="FloodCA")
    ap.add_argument("--lon-min", type=float, default=74.5)
    ap.add_argument("--lat-min", type=float, default=13.0)
    ap.add_argument("--lon-max", type=float, default=75.5)
    ap.add_argument("--lat-max", type=float, default=14.0)
    ap.add_argument("--buffer-deg", type=float, default=0.02)
    ap.add_argument("--out-json", default="gee_event_list.json")
    ap.add_argument(
        "--max-events",
        type=int,
        default=8,
        help="max GFD Drive exports for the primary AOI (ignored with --list-only)",
    )
    ap.add_argument(
        "--list-only",
        action="store_true",
        help="write event metadata JSON only; do not start Drive exports",
    )
    ap.add_argument(
        "--jrc-only",
        action="store_true",
        help="start only the JRC GSW max_extent Drive export",
    )
    ap.add_argument(
        "--only-event",
        type=int,
        default=0,
        help="export only this GFD event id; skips the event listing, WorldCover, and the top-8 list",
    )
    args = ap.parse_args()
    if not args.project:
        sys.stderr.write("Pass --project or set GEE_PROJECT.\n")
        return 1
    if args.list_only and args.jrc_only:
        sys.stderr.write("Use only one of --list-only or --jrc-only.\n")
        return 1

    ee = _init(args.project)
    west = args.lon_min - args.buffer_deg
    south = args.lat_min - args.buffer_deg
    east = args.lon_max + args.buffer_deg
    north = args.lat_max + args.buffer_deg
    aoi = ee.Geometry.Rectangle([args.lon_min, args.lat_min, args.lon_max, args.lat_max])
    buffered = ee.Geometry.Rectangle([west, south, east, north])

    summary = {
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "project": args.project,
        "aoi": [args.lon_min, args.lat_min, args.lon_max, args.lat_max],
        "mode": "jrc-only" if args.jrc_only else ("list-only" if args.list_only else "all"),
        "tasks": [],
        "gfd_udupi": [],
        "gfd_kerala2018": [],
        "gfd_udupi_image_meta": None,
        "band_names_gfd_sample": None,
        "notes": (
            "Tier A/B/C is a human checkpoint (SPEC 4.4 / P-8). "
            "GFD has no began/ended properties; dates are system:time_start "
            "and system:time_end. JRC GSW band is max_extent (not extent)."
        ),
    }

    if args.jrc_only:
        # JRC GSW v1.4 bands: occurrence, change_abs, change_norm, seasonality,
        # recurrence, transition, max_extent. There is no band named extent.
        jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("max_extent").clip(buffered)
        summary["tasks"].append(
            _export_image(
                ee,
                jrc,
                buffered,
                "floodca_jrc_gsw_extent_udupi",
                args.drive_folder,
                30,
                "EPSG:4326",
            )
        )
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")
        print(json.dumps(summary, indent=2))
        print(f"wrote {args.out_json}")
        print("JRC-only task started. Download floodca_jrc_gsw_extent_udupi.tif when READY.")
        return 0

    gfd = ee.ImageCollection("GLOBAL_FLOOD_DB/MODIS_EVENTS/V1")
    events = gfd.filterBounds(aoi)
    if args.only_event:
        summary["mode"] = "only-event"
        img = ee.Image(events.filter(ee.Filter.eq("id", args.only_event)).first())
        stacked = img.select(["flooded", "jrc_perm_water"]).clip(aoi)
        summary["tasks"].append(
            _export_image(
                ee, stacked, aoi, f"floodca_gfd_udupi_{args.only_event}",
                args.drive_folder, 250, "EPSG:4326",
            )
        )
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")
        print(json.dumps(summary, indent=2))
        print(f"wrote {args.out_json}")
        print("only-event: no event listing. Watch the one Drive export, then rsync the GeoTIFF.")
        return 0

    udupi_rows, udupi_meta = _collect_events(ee, events, aoi, "GFD events intersecting udupi_coast")
    summary["gfd_udupi"] = udupi_rows
    if udupi_meta:
        summary["gfd_udupi_image_meta"] = udupi_meta
        summary["band_names_gfd_sample"] = udupi_meta.get("band_names")
    _print_shortlist(udupi_rows)

    kerala_aoi = ee.Geometry.Rectangle([76.0, 9.0, 77.0, 11.0])
    kerala = gfd.filterBounds(kerala_aoi).filterDate("2018-08-01", "2018-09-01")
    kerala_rows, _kmeta = _collect_events(
        ee, kerala, kerala_aoi, "GFD events Kerala box Aug 2018"
    )
    summary["gfd_kerala2018"] = kerala_rows

    if not args.list_only:
        wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").clip(buffered)
        summary["tasks"].append(
            _export_image(
                ee, wc, buffered, "floodca_worldcover_v200_udupi", args.drive_folder, 10, "EPSG:4326"
            )
        )
        jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("max_extent").clip(buffered)
        summary["tasks"].append(
            _export_image(
                ee,
                jrc,
                buffered,
                "floodca_jrc_gsw_extent_udupi",
                args.drive_folder,
                30,
                "EPSG:4326",
            )
        )
        exported = 0
        for info in udupi_rows:
            if exported >= args.max_events:
                break
            eid = info.get("id")
            img = ee.Image(events.filter(ee.Filter.eq("id", eid)).first())
            stacked = img.select(["flooded", "jrc_perm_water"]).clip(aoi)
            tag = str(eid if eid is not None else exported).replace("/", "_")
            summary["tasks"].append(
                _export_image(
                    ee,
                    stacked,
                    aoi,
                    f"floodca_gfd_udupi_{tag}"[:90],
                    args.drive_folder,
                    250,
                    "EPSG:4326",
                )
            )
            exported += 1
        if kerala_rows:
            img = ee.Image(kerala.first())
            stacked = img.select(["flooded", "jrc_perm_water"]).clip(kerala_aoi)
            summary["tasks"].append(
                _export_image(
                    ee, stacked, kerala_aoi, "floodca_gfd_kerala2018", args.drive_folder, 250, "EPSG:4326"
                )
            )

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out_json}")
    if args.list_only:
        print("list-only: no Drive exports started. rsync gee_event_list.json to scratch.")
    else:
        print("Watch Drive exports, then rsync GeoTIFFs to ${FCA_SCRATCH}/data/raw/gee/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

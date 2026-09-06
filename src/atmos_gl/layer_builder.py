#!/usr/bin/env python3
import argparse
import json
import logging
import sys
import os
import signal
import asyncio
import multiprocessing
import threading
import time
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Library imports
from atmos_gl.lib.config import AtmosGLConfig
from atmos_gl.lib.logging import setup_logging, set_loglevel
from atmos_gl.db.process_status_adapter import ProcessStatusAdapter
from atmos_gl.collectors import FIELD_COLLECTOR_CLASSES, CACHE_COLLECTORS
from atmos_gl.round_robin_order import RoundRobinOrder


# Task imports
from atmos_gl.tasks.common import MapData, LAYER_CYCLE_SECONDS, MultiHourRenderMixin
from atmos_gl.tasks.clouds import CloudUpdater
from atmos_gl.tasks.isobars import IsobarUpdater
from atmos_gl.tasks.wind import WindUpdater
from atmos_gl.tasks.precipitation import PrecipitationUpdater
from atmos_gl.tasks.sst import SSTUpdater
from atmos_gl.tasks.greenhouse_gases import GhgUpdater
from atmos_gl.tasks.air_quality import AirQualityUpdater
from atmos_gl.tasks.currents import CurrentsUpdater
from atmos_gl.tasks.jetstream import JetStreamUpdater
from atmos_gl.tasks.waves import WavesUpdater
from atmos_gl.tasks.scalar_field import ScalarFieldUpdater, SPECS
from atmos_gl.tasks.markers import MarkerUpdater
from atmos_gl.tasks.fire_weather import FireWeatherUpdater
from atmos_gl.tasks.flood_risk import FloodRiskUpdater

logger = logging.getLogger("atmos_gl.layer_builder")

# Seconds between fan-out cycles. Every cycle dispatches all updaters; per-hour freshness
# checks make a steady-state (nothing-changed) cycle cheap, so this is just the
# responsiveness window for picking up new data or deleted output. Canonical definition
# is tasks.common.LAYER_CYCLE_SECONDS (Updater.layer_status() needs it too, and
# tasks/common.py can't import this module without a cycle).
CYCLE_SECONDS = LAYER_CYCLE_SECONDS

# Watchdog threshold (see LayerBuilder._check_watchdog for the incident this guards
# against). Hardcoded, not config-driven -- this is an internal safety net, not
# something an admin should routinely retune (matches CYCLE_SECONDS's own
# precedent). 20 minutes comfortably clears worst-case sequential dispatch on
# performance_tier="low" (one worker, up to 11 multi-hour sections) plus a real
# backlog.
WATCHDOG_STALE_SECONDS = 20 * 60

# section -> updater class. The parent dispatches one task per entry; each worker process
# looks up the class it must build by section name. Order is informational only now —
# updaters render in parallel, not in sequence.
TASK_CLASSES = {
    "isobars": IsobarUpdater,
    "precipitation": PrecipitationUpdater,
    "clouds": CloudUpdater,
    "wind": WindUpdater,
    "sst": SSTUpdater,
    "greenhouse_gases": GhgUpdater,
    "air_quality": AirQualityUpdater,
    "currents": CurrentsUpdater,
    "jetstream": JetStreamUpdater,
    "waves": WavesUpdater,
    "temperature": partial(ScalarFieldUpdater, spec=SPECS["temperature"]),
    "ozone": partial(ScalarFieldUpdater, spec=SPECS["ozone"]),
    "stormwatch": partial(ScalarFieldUpdater, spec=SPECS["stormwatch"]),
    "pwat": partial(ScalarFieldUpdater, spec=SPECS["pwat"]),
    "markers": MarkerUpdater,
    "fires": FireWeatherUpdater,
    "flood_risk": FloodRiskUpdater,
}


# common.performance_tier's fallback when unset/unknown -- the one default value
# every reader of it (LayerBuilder.__init__, refresh_settings(), and
# workers_for_tier()'s own fallback branch below) must agree on.
_DEFAULT_TIER = "medium"


def workers_for_tier(tier: str, cpu_count: int | None) -> int:
    """Effective ProcessPoolExecutor worker count for common.performance_tier.

    "low" -> 1: fully sequential, the one value that actually guarantees no
    concurrent-process pile-up regardless of core count.
    "medium" (also the fallback for an unset/unknown value) -> half the cores,
    floored at 2 so it's never accidentally equal to "low".
    "high" -> today's pre-existing hardcoded formula, unchanged -- a complete
    no-op relative to current behavior."""
    n = cpu_count or 4
    if tier == "low":
        return 1
    if tier == "high":
        return min(len(TASK_CLASSES), n)
    return max(2, n // 2)


def _updater_class(entry):
    """The plain class behind a TASK_CLASSES entry, unwrapping the partial() binding
    the four ScalarFieldUpdater-based sections use."""
    return entry.func if isinstance(entry, partial) else entry


def build_layer_channel_keys(field_collector_classes, cache_collector_classes) -> dict:
    """Maps a layer's TASK_CLASSES section name (e.g. "isobars") to the channel_key
    that feeds it (e.g. "gfs_atmos"). Derived from the collector classes' own
    `products`/`channel_key` rather than hand-duplicated, so the two can't drift apart.

    Two consumers: the Data Status UI (routes/status.py, imports this from here) uses
    it to gray out every layer a disabled channel backs; LayerBuilder's own dispatch
    (see dispatchable_sections()) uses the SAME mapping so a channel-disabled layer is
    both grayed out AND never rendered -- one source of truth for both."""
    mapping = {}
    for CollectorCls in field_collector_classes:
        if getattr(CollectorCls, "channel_key", None):
            for product_name in CollectorCls.products:
                mapping[product_name] = getattr(CollectorCls, "channel_key", None)
    for CollectorCls in cache_collector_classes:
        if getattr(CollectorCls, "channel_key", None):
            # settings_section (CollectorBase) is the real TASK_CLASSES/layer key when
            # a collector's own `section` differs from it (e.g. the greenhouse_gases
            # layer's two collectors, each independently scheduled/reported but
            # sharing one settings section) -- falls back to `section` for every
            # other collector, which doesn't set settings_section at all.
            layer_key = getattr(CollectorCls, "settings_section", None) or CollectorCls.section
            # setdefault, not assignment: every existing layer has exactly one
            # collector per layer_key, so this is a no-op behaviour change for them.
            # Where two collectors share one layer_key (greenhouse_gases' GEOS-CF +
            # EGG4 baseline pair), the FIRST one registered in cache_collector_classes
            # wins -- deliberately CamsGhgForecastCollector, since it's the layer's true
            # hard dependency (both modes need it; EGG4 only gates Anomaly, which
            # already self-gates in GhgUpdater.run() independent of channel_enabled).
            mapping.setdefault(layer_key, getattr(CollectorCls, "channel_key", None))
    return mapping


def dispatchable_sections(channel_enabled: dict, layer_channel_keys: dict, all_sections) -> list:
    """`all_sections` filtered down to the ones LayerBuilder should actually spawn a
    worker for this cycle: a section whose mapped channel_key is explicitly disabled in
    data_collector.channel_enabled is excluded -- no worker spawned at all, not even
    once to discover there's nothing to render. A section with no channel_key mapping
    (not every layer maps to exactly one toggleable channel -- e.g. markers) is always
    dispatchable. A channel_key present in layer_channel_keys but not yet written to
    channel_enabled defaults to enabled, matching _serialize()'s existing convention."""
    return [
        s for s in all_sections
        if channel_enabled.get(layer_channel_keys.get(s), True)
    ]


# Sections rendered per-forecast-hour (mix in MultiHourRenderMixin) vs. once per cycle
# (sst/clouds/markers -- no per-hour concept). Only multi-hour sections participate in
# the round-robin dispatch below: each round renders at most ONE hour per section, so a
# section with a large backlog can't monopolise the render pool's workers for its whole
# catch-up -- every section advances roughly evenly instead of depth-first through
# whichever ones got dispatched first (architecture review candidate "interleave
# per-hour rendering across layers").
MULTI_HOUR_SECTIONS = [
    name for name, entry in TASK_CLASSES.items()
    if issubclass(_updater_class(entry), MultiHourRenderMixin)
]
SINGLE_SHOT_SECTIONS = [name for name in TASK_CLASSES if name not in MULTI_HOUR_SECTIONS]

# Internal-only listener for reprioritising the round-robin order (architecture review
# candidate "test changed processes much more quickly") -- never called directly.
# routes/layer_builder.py on map_api (the app's one public API surface, port 9000)
# proxies POST/GET /api/layer_builder/priority to this over the agl docker network;
# nothing outside this container talks to this port. Not published to the host.
ORDER_SERVER_PORT = 9100


class _OrderRequestHandler(BaseHTTPRequestHandler):
    """`order` is bound per-subclass by _start_order_server() below (BaseHTTPRequestHandler
    instantiates a fresh handler object per request, so the shared RoundRobinOrder has to
    live on the class, not the instance)."""

    order: RoundRobinOrder

    def _reply(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/priority":
            self._reply(200, {"order": self.order.current()})
        else:
            self._reply(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._reply(400, {"error": "invalid JSON"})
            return

        if self.path == "/priority":
            sections = payload.get("sections")
            if not isinstance(sections, list) or not sections:
                self._reply(400, {"error": "sections must be a non-empty list"})
                return
            try:
                self.order.reorder(sections)
            except ValueError as e:
                self._reply(400, {"error": str(e)})
                return
            self._reply(200, {"order": self.order.current()})
        elif self.path == "/priority/reset":
            self.order.reset()
            self._reply(200, {"order": self.order.current()})
        else:
            self._reply(404, {"error": "not found"})

    def log_message(self, fmt, *args):
        logger.debug("order-server: " + fmt % args)


def _start_order_server(order: RoundRobinOrder, port: int = ORDER_SERVER_PORT) -> ThreadingHTTPServer:
    """Starts the priority-reorder listener in a daemon thread. `port=0` (tests only)
    binds an OS-assigned free port -- read it back via server.server_address[1]."""
    handler_cls = type("_BoundOrderRequestHandler", (_OrderRequestHandler,), {"order": order})
    server = ThreadingHTTPServer(("0.0.0.0", port), handler_cls)
    thread = threading.Thread(target=server.serve_forever, name="order-server", daemon=True)
    thread.start()
    logger.info(f"Round-robin priority endpoint listening on :{server.server_address[1]}")
    return server


def _worker_init(config_path):
    """Runs once per worker PROCESS at spawn. The child never calls main(), so it must
    configure its own logging — at the configured level so worker render logs match the
    parent's verbosity."""
    setup_logging()
    try:
        level = AtmosGLConfig(config_path).get_setting("common", "log_level")
        if level:
            set_loglevel(level)
    except Exception:
        pass


def _render_worker(config_path, section, baseline, max_hours=None):
    """Runs in a SEPARATE PROCESS.

    Rebuilds config + map_data from the config path (no live objects cross the process
    boundary, and config edits are picked up automatically), injects the pre-resolved
    GFS/RTOFS baseline so the worker never re-probes NOMADS, then builds the one updater
    for `section` and renders it.

    Each process owns its own cartopy / matplotlib / GEOS state, so renders run truly in
    parallel — what the thread model could not do safely (those C libraries are not
    thread-safe and segfaulted under concurrency).

    max_hours is forwarded to run() unconditionally -- every TASK_CLASSES updater
    accepts it now (single-shot layers ignore it; multi-hour ones cap the backlog they
    drain this call to that many hours). Returns (section, error, plotted): error is
    None on success (repr(exception) on failure, and one failing layer can't poison the
    gather since it's caught here rather than raised); plotted is however many hours
    run() actually rendered (0 for single-shot layers, or an exception).
    """
    try:
        cfg = AtmosGLConfig(config_path)
        md = MapData(cfg)
        md.shared_state = {}
        if baseline.get("gfs"):
            md.shared_state["gfs_baseline"] = baseline["gfs"]
        if baseline.get("rtofs"):
            md.shared_state["rtofs_baseline"] = baseline["rtofs"]
        plotted = TASK_CLASSES[section](cfg, md).run(max_hours=max_hours)
        return (section, None, plotted or 0)
    except Exception as e:
        return (section, repr(e), 0)


class LayerBuilder:
    enabled = False

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = AtmosGLConfig(config_path)
        self.map_data = MapData(self.config)
        # Own ProcessStatusAdapter, used ONLY to record process_status for the Data Status UI
        # after each cycle (see _handle_results). Rendering itself happens in worker
        # processes with their own fieldstore/db connections; this one never touches
        # render data.
        self.process_status_adapter = ProcessStatusAdapter()

        # Ensure this folder exists
        data_dir = os.path.join(
            self.config.get_setting("common", "workdir", "."), "data"
        )
        os.makedirs(data_dir, exist_ok=True)

        # Shared state holds the GFS/RTOFS baseline the primer resolves each cycle.
        self.map_data.shared_state = {}

        signal.signal(signal.SIGUSR1, self.handle_force_refresh)

        # One in-process updater, used ONLY to resolve the baseline once per cycle (a
        # lightweight NOMADS probe, no rendering). All rendering happens in worker
        # processes. Built in start_scheduler once the region/config are current.
        self._primer = None

        # Render is CPU-bound, so cap workers at core count (never more than the number
        # of layers), scaled down further by common.performance_tier. See
        # workers_for_tier()'s docstring for the tier -> count mapping; "high" matches
        # this line's own pre-existing formula exactly.
        self._tier = self.config.get_setting("common", "performance_tier", _DEFAULT_TIER)
        self._max_workers = workers_for_tier(self._tier, os.cpu_count())
        self._pool = None

        # In-memory priority order for the multi-hour round-robin dispatch below --
        # defaults to MULTI_HOUR_SECTIONS' declared order, reorderable live via
        # /api/layer_builder/priority (see _start_order_server()). Resets on restart,
        # deliberately -- this is a dev-speed tool, not persisted config.
        self.order = RoundRobinOrder(MULTI_HOUR_SECTIONS)
        self._order_server = None

        # Watchdog state: section -> time.time() of its last dispatch RESULT (success,
        # failure, or broken pool -- "returned a result at all" is the liveness signal,
        # not "rendered successfully"). Only multi-hour sections are tracked; see
        # _handle_results/_check_watchdog. Starts empty, so a freshly-started process
        # gives every section a free grace period until it's actually been dispatched
        # once.
        self._last_dispatch_ts: dict[str, float] = {}

        # Static (derived from the collector class registries, not config), so computed
        # once here rather than every dispatch cycle -- see dispatchable_sections().
        self._layer_channel_keys = build_layer_channel_keys(
            FIELD_COLLECTOR_CLASSES, CACHE_COLLECTORS
        )

    def refresh_settings(self):
        self.config.load()
        self.enabled = self.config.get_setting("layer_builder", "enabled")
        # Adjust log level if changed
        log_level = self.config.get_setting("common", "log_level")
        if log_level:
            set_loglevel(log_level)

        # A tier change recomputes _max_workers and, if a pool already exists (this is
        # not the very first refresh_settings() call, which runs before start_scheduler()
        # creates the initial pool), recreates it live -- reusing the exact recovery
        # mechanism _dispatch_round() already uses for a crashed pool, just triggered
        # deliberately instead of only reactively.
        tier = self.config.get_setting("common", "performance_tier", _DEFAULT_TIER)
        if tier != self._tier:
            self._tier = tier
            self._max_workers = workers_for_tier(tier, os.cpu_count())
            if self._pool is not None:
                self._recycle_pool()

    def handle_force_refresh(self, signum, frame):
        """SIGUSR1: drop the cached GFS/RTOFS datum so the next cycle re-resolves it."""
        logger.debug("External trigger (SIGUSR1): clearing cached baselines")
        ss = getattr(self.map_data, "shared_state", None)
        if isinstance(ss, dict):
            ss.pop("gfs_baseline", None)
            ss.pop("rtofs_baseline", None)

    def _new_pool(self):
        """Create a fresh spawn-based process pool. 'spawn' (not fork) gives each worker a
        clean interpreter: fork would inherit the parent's GEOS/PROJ/matplotlib state and
        re-introduce the very C-library hazards the process model exists to escape."""
        logger.info(
            f"Starting render process pool (max_workers={self._max_workers}, spawn)"
        )
        return ProcessPoolExecutor(
            max_workers=self._max_workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_worker_init,
            initargs=(self.config_path,),
        )

    def _recycle_pool(self):
        """Discard the current pool and start a fresh one at the current
        _max_workers -- shared by both the crash-recovery path (_dispatch_round,
        after a BrokenProcessPool) and refresh_settings()'s live tier-change path."""
        try:
            self._pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        self._pool = self._new_pool()

    def _resolve_baselines(self):
        """Resolve the GFS/RTOFS datums ONCE, up front (cleared first so a long-lived
        process can't pin to an ever-older run), and return them as a plain dict to hand to
        every worker — so all workers inherit one datum instead of each re-probing NOMADS."""
        ss = self.map_data.shared_state
        ss.pop("gfs_baseline", None)
        ss.pop("rtofs_baseline", None)
        for label, resolve in (
            ("GFS", self._primer.get_gfs_state),
            ("RTOFS", self._primer.get_rtofs_state),
        ):
            try:
                resolve()
            except Exception as e:
                logger.warning(f"{label} baseline pre-resolve failed: {e}")
        return {"gfs": ss.get("gfs_baseline"), "rtofs": ss.get("rtofs_baseline")}

    def _handle_results(self, sections, results):
        """Log per-task errors and record process_status for the Data Status UI (one row
        per dispatched section, success or failure). `sections` is the same ordered list
        futures were built from, so zip(sections, results) reliably pairs each result with
        its task even in the edge case where a result is a bare Exception (e.g. the
        executor itself died) rather than _render_worker's own (section, error, plotted)
        tuple.

        Returns (broken, plotted_by_section). broken is True if the pool broke (a worker
        died) and must be recreated. plotted_by_section maps each section to how many
        hours it actually rendered this dispatch -- the round-robin loop in
        start_scheduler() uses it to drop a multi-hour section once it stops reporting
        progress, rather than looping it forever.
        """
        broken = False
        plotted_by_section = {}
        for section, r in zip(sections, results):
            if section in MULTI_HOUR_SECTIONS:
                # Watchdog liveness signal (_check_watchdog): a result coming back at
                # all -- success, failure, or a broken pool -- proves the round-robin
                # loop is still actively dispatching this section. This is exactly the
                # signal that silently stopped for `isobars` for ~27h in the incident
                # this guards against: nothing crashed, no error was ever logged, it
                # just quietly stopped being included in dispatch.
                self._last_dispatch_ts[section] = time.time()
            if isinstance(r, BrokenProcessPool):
                broken = True
                self.process_status_adapter.record_process_run(
                    section, "layer", success=False, error="render pool broke"
                )
            elif isinstance(r, Exception):
                logger.error(f"Render dispatch error: {r!r}")
                self.process_status_adapter.record_process_run(
                    section, "layer", success=False, error=repr(r)
                )
            elif r and r[1]:
                logger.error(f"Task '{r[0]}' failed in worker: {r[1]}")
                self.process_status_adapter.record_process_run(
                    section, "layer", success=False, error=r[1]
                )
            else:
                self.process_status_adapter.record_process_run(section, "layer", success=True)
                if r and len(r) > 2:
                    plotted_by_section[section] = r[2] or 0
        if broken:
            logger.error("Render worker died (BrokenProcessPool); recreating pool")
        return broken, plotted_by_section

    def _check_watchdog(self):
        """Force a restart if a multi-hour section has silently stopped being
        dispatched -- the failure mode found live 2026-08-10/11, where isobars
        dropped out of round-robin rotation for ~27h with no error, no crash, and no
        effect on any other section, while every other multi-hour section kept
        rendering normally. Comparing rendered forecast RUN across layers (e.g.
        isobars vs wind) was considered and rejected: GfsAtmosCollector ingests every
        atmos product from one shared grib2 download per hour, so their CATALOG data
        is never out of sync with each other -- only isobars' RENDER of that data
        fell behind, which per-section dispatch-liveness (this check) catches
        directly and a cross-layer run comparison would not have.

        _last_dispatch_ts (updated in _handle_results the instant a section's future
        resolves, success or failure) only gains an entry once a section has actually
        been dispatched -- so a section absent from it (freshly restarted process, or
        one still waiting its turn behind a large backlog) is never flagged, only one
        that WAS seen and then stopped updating.

        Deliberately no backoff/loop-guard: always self-heals via os._exit(1) --
        Docker's `restart: unless-stopped` on this service brings it back up, no
        Docker-socket access or cross-container call needed. If this recurs rapidly,
        the restart loop itself is loud and visible enough (docker compose ps, these
        very log lines) to investigate as its own problem, rather than something
        worth pre-emptively engineering backoff for before it's ever been observed.
        """
        now = time.time()
        stale = {
            s: now - ts for s, ts in self._last_dispatch_ts.items()
            if now - ts > WATCHDOG_STALE_SECONDS
        }
        if not stale:
            return
        ages = ", ".join(
            f"{s}={(now - ts) / 60:.1f}m"
            for s, ts in sorted(self._last_dispatch_ts.items())
        )
        logger.error(
            f"Watchdog: {sorted(stale)} not dispatched in over "
            f"{WATCHDOG_STALE_SECONDS / 60:.0f} minutes -- forcing a restart to "
            f"recover. All tracked section ages: {ages}"
        )
        os._exit(1)

    async def _dispatch_round(self, loop, sections, baseline, max_hours_by_section):
        """Dispatch one future per section in `sections` (each capped to
        max_hours_by_section[section] hours -- None for single-shot layers, 1 for a
        multi-hour layer's round-robin turn), gather, record process_status, and
        recreate the pool if a worker died. Returns plotted_by_section."""
        futures = [
            loop.run_in_executor(
                self._pool, _render_worker, self.config_path, section, baseline,
                max_hours_by_section[section],
            )
            for section in sections
        ]
        results = await asyncio.gather(*futures, return_exceptions=True)
        broken, plotted_by_section = self._handle_results(sections, results)
        # Liveness heartbeat for the System Status section (Global tab) -- written
        # once per ROUND (not just once per outer cycle in start_scheduler()), since a
        # single outer cycle can take many minutes to drain a large multi-hour
        # backlog (see _run_dispatch_cycle's docstring); a heartbeat that only ticked
        # once per outer cycle would read as "stale" during a long but healthy
        # backlog-catchup, indistinguishable from an actually-dead process.
        self.process_status_adapter.record_process_run(
            "layer_builder", "service", success=True
        )
        if broken:
            self._recycle_pool()
        return plotted_by_section

    async def _run_dispatch_cycle(self, loop, baseline):
        """One cycle's worth of rendering, given an already-resolved baseline.

        Single-shot layers (sst/clouds/markers/greenhouse_gases/air_quality) ride
        along on EVERY round, not just the first. Each one's own freshness check
        (Updater._is_render_fresh) short-circuits to a cheap no-op when nothing's
        actually stale, so this costs little -- but it matters because a large
        multi-hour backlog can otherwise take many rounds (tens of minutes) to fully
        drain, and this whole method isn't called again (config re-read, single-shot
        layers re-dispatched) until it does. Previously single-shot layers dropped
        out after round 1, so a config/data change to one of them sat unpicked-up for
        however long the multi-hour backlog behind it happened to take to clear (see
        issue #240 -- found live: an air_quality settings change went unreflected for
        30+ minutes behind an in-progress isobars/waves/etc. backfill).

        Multi-hour layers dispatch in ROUNDS -- one hour per section per round -- so a
        section with a large backlog can't monopolise the render pool's workers for its
        whole catch-up; every section advances roughly evenly instead of depth-first
        through whichever ones happened to dispatch first (architecture review
        candidate "interleave per-hour rendering across layers"). A round drops a
        multi-hour section once it stops reporting progress; the cycle itself ends
        once every multi-hour section has stopped -- single-shot layers place no
        bound on this, they simply ride along for free on however many rounds the
        multi-hour sections need.
        """
        # A section whose backing channel is manually disabled never gets its worker
        # process spawned at all -- not even once to discover there's nothing to
        # render. channel_enabled is re-read fresh every cycle (it's live-toggleable
        # via the Data Status page's own instant-save endpoint); _layer_channel_keys
        # is static (derived from the collector class registries) and computed once.
        channel_enabled = self.config.get_setting("data_collector", "channel_enabled", {}) or {}
        all_sections = dispatchable_sections(
            channel_enabled, self._layer_channel_keys,
            SINGLE_SHOT_SECTIONS + MULTI_HOUR_SECTIONS,
        )
        single_shot = [s for s in SINGLE_SHOT_SECTIONS if s in all_sections]
        multi_hour_pending = {s: 1 for s in MULTI_HOUR_SECTIONS if s in all_sections}
        if not single_shot and not multi_hour_pending:
            return

        while True:
            # Priority order is read fresh each round, so a reorder() made mid-cycle
            # (via POST /api/layer_builder/priority) takes effect on the very next
            # round without touching one already dispatched to the process pool --
            # see round_robin_order.py's module docstring.
            ordered_multi_hour = self.order.ordered(multi_hour_pending)
            sections = single_shot + ordered_multi_hour
            max_hours_by_section = {s: None for s in single_shot}
            max_hours_by_section.update({s: 1 for s in ordered_multi_hour})
            plotted_by_section = await self._dispatch_round(
                loop, sections, baseline, max_hours_by_section
            )
            multi_hour_pending = {
                s: 1 for s in multi_hour_pending
                if plotted_by_section.get(s, 0) > 0
            }
            if not multi_hour_pending:
                break

    async def start_scheduler(self):
        # Initial refresh so the region/config are current before the primer is built.
        self.refresh_settings()
        self.map_data.refresh()
        self._primer = TASK_CLASSES[next(iter(TASK_CLASSES))](self.config, self.map_data)
        self._pool = self._new_pool()
        self._order_server = _start_order_server(self.order)
        loop = asyncio.get_running_loop()

        try:
            while True:
                self.refresh_settings()
                # Coarse liveness heartbeat for the System Status section (Global tab) --
                # independent of `enabled`, so it keeps ticking even while rendering
                # itself is disabled (the only case that reaches here without also
                # ticking _dispatch_round's own per-round heartbeat below). A dead
                # container simply stops advancing this row.
                self.process_status_adapter.record_process_run(
                    "layer_builder", "service", success=True
                )

                if self.enabled:
                    self.map_data.refresh()

                    # Resolve the datum once, then dispatch every updater to its own
                    # process. Workers rebuild config per task, so config edits are picked
                    # up automatically — no rebuild bookkeeping here. No should_run gating;
                    # each updater's per-hour freshness check skips already-current work, so
                    # a steady-state cycle is cheap and a changed/deleted layer re-renders
                    # promptly, now-hour first — and now genuinely in parallel.
                    # refresh_settings/baseline-resolve still only happen once per OUTER
                    # cycle -- a very large backlog still delays picking up config/baseline
                    # changes until every section's rounds finish, unchanged from before
                    # this file's per-hour round-robin dispatch existed.
                    baseline = self._resolve_baselines()
                    await self._run_dispatch_cycle(loop, baseline)
                    self._check_watchdog()
                else:
                    logger.info("Layer-builder scheduler disabled: skipping")

                await asyncio.sleep(CYCLE_SECONDS)
        finally:
            if self._pool is not None:
                self._pool.shutdown(wait=False, cancel_futures=True)
            if self._order_server is not None:
                self._order_server.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Atmos GL Layer Builder Scheduler")
    parser.add_argument("--config", required=True, help="Path to atmos-gl.json")
    args = parser.parse_args()

    setup_logging()
    layer_builder = LayerBuilder(args.config)

    try:
        asyncio.run(layer_builder.start_scheduler())
    except KeyboardInterrupt:
        logger.info("Scheduler gracefully stopped.")
        sys.exit(130)


if __name__ == "__main__":
    main()
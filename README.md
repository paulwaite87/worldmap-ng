# Atmos GL

## What is this?

A Docker Compose–based system that gives you a live, interactive 3D globe in your browser —
built on MapLibre GL JS — showing real-time and forecast weather and world events: clouds,
isobars, wind, precipitation, precipitable water (moisture), sea surface temperature, ocean
currents, wave height, air temperature, ozone, storm watch (CAPE/CIN), earthquakes, volcanoes,
tropical storms, lightning strikes, shipping traffic, satellites and more.

A Python/FastAPI backend continuously pulls forecast data from NOAA/NCEP (GFS atmospheric and
wave models, RTOFS ocean currents) and live event feeds (USGS earthquakes, NOAA/NHC/JTWC storm
tracks, Smithsonian/USGS volcanic activity data, AIS shipping, lightning strikes, satellite
orbits), storing everything in PostGIS and rendering it onto the globe as you watch.

### Our Blue Marble
![Atmos GL Example](docs/atmos-gl-blue-marble.png)

## Quick Start

### Prerequisites: Docker Installation

Before running this project, you must have Docker and Docker Compose installed on your system.
For Ubuntu users, it is highly recommended to install Docker via the official Docker repository
rather than the default apt archives to ensure you have the latest version compatible with
modern systemd and container features. You can verify your installation by running
docker --version in your terminal.

If you need some guidance on this a good place to look is
here https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-20-04

Despite the '20-04' at the end of the link, this tutorial is also fine for later versions of Ubuntu.

To avoid having to use sudo with every command, ensure your user is added to the docker group.
After installation, run `sudo usermod -aG docker $USER` and log out and back in for the changes
to take effect. This will allow you to manage containers and orchestration seamlessly while
working within the repository.

Everything here is Docker-based, so while these instructions lean Linux, it should run
wherever Docker Desktop does.

### Install and Run

Run this to download everything needed and set it up in `~/atmos-gl` (pass a path as
an argument if you'd rather install somewhere else):

    curl -fsSL https://raw.githubusercontent.com/paulwaite87/atmos-gl/master/install.sh | bash

This fetches `docker-compose.yml`, an `atmos-gl.sh` control script, and the reference
data the map needs, then creates a `.env` and `config/atmos-gl.json` for you (both
left alone on future re-runs, so it's always safe to run this again later to pick up
updates to everything else).

Edit `.env` and fill in your API keys — see [Map Tiles API Key](#map-tiles-api-key),
[Shipping Data API Key](#shipping-data-api-key),
[Lightning Strikes API Key](#lightning-strikes-api-key),
[NASA FIRMS API Key](#nasa-firms-api-key),
[Copernicus CDS/ADS API Key](#copernicus-cdsads-api-key),
[NASA Earthdata Login Token](#nasa-earthdata-login-token)

The map tiles key is MANDATORY for the globe's basemap to render at all. All others 
are optional (you can enable them later once you have keys).

Then start everything:

    cd ~/atmos-gl
    ./atmos-gl.sh start

By default most layers are disabled to begin with — see [Setting it Up](#setting-it-up) below.

To stop everything:

    ./atmos-gl.sh stop

`atmos-gl.sh` also has `restart`, `update` (pulls the latest images), `status` (ship/lightning
counts per region) and `logs` commands — run it with no arguments for the full list. If you
ever need to report a problem, `./atmos-gl.sh logs save` writes a timestamped log file with
any API keys automatically redacted, safe to attach to a GitHub issue.

(If you've cloned the repo instead of using the installer, the Makefile has equivalents for
all of this — see [Developer's Corner](#developers-corner) below.)

### Setting it Up
The configuration file is called `atmos-gl.json` and it lives in the `config` folder.
You can either edit this file directly, or browse to `http://localhost:9000/config` to use
the configuration webpage there. If you do use that page, and save some changes they
will overwrite your `atmos-gl.json` — which is fine, but if you want to preserve your own
hand-edits, make a backup copy of the file first.

The configurator and the Data Status page (below) both require signing in as an admin —
see [Account & Personal Settings](#account--personal-settings) for how an account
becomes an admin.

The live globe itself is at `http://localhost:8180`.

Here is a shot of the homepage for the configurator at `http://localhost:9000/config`

![Configuration Homepage](docs/atmos-gl-conf-home.png)

I would suggest first setting your starting Latitude and Longitude to a view you want to
see at startup and then in the `Show` tab just enabling some of the atmospheric layers
like `Isobars`, `Wind` and `Preciptation` as a starting point.

If you change something in `config/atmos-gl.json` by hand rather than through the web UI,
restart the backend to pick it up:

    ./atmos-gl.sh restart

### Account & Personal Settings
The live globe itself (`http://localhost:8180`) needs no account at all — it's open to
anyone who can reach it. Signing in (Google or GitHub, via the hamburger menu top-right)
unlocks two things beyond that:

* **Personal settings** — a signed-in user can save their own overrides for most sliders/
  toggles/colours (camera start position, palettes, opacity, thresholds and the like),
  layered on top of the shared `config/atmos-gl.json` rather than replacing it: your
  override wins for you, everyone else still gets the shared default. Settings that
  affect what the backend actually renders (not just how it's displayed) stay admin-only,
  shared config.
* **Admin access** — one or more accounts can be designated admin (via an email
  allowlist in `.env`). Only an admin account can reach the configurator
  (`http://localhost:9000/config`, which edits the shared `atmos-gl.json` default
  everyone else's overrides sit on top of) and the Data Status page.

The account menu also has a settings link, sign-out, account deletion, and shows the
running release version in its footer.

### API Keys
Some of the data resources we are lucky enough to have free access to are only available
if you have an API Key. This is perfectly reasonable because it allows the folks running
and maintaining the resource to rate-limit data being served. None of the API Keys used
in this project are hard to obtain, but even so it is entirely optional apart from the
first one below, which is needed for you to see the map.

#### Map Tiles API Key
This is MANDATORY.
The globe's basemap imagery (satellite/street tiles) is served by MapTiler, and needs its own
free API key. Sign up at https://www.maptiler.com/, grab a key from your account dashboard,
and put it in `.env` as `MAPTILER_API_KEY`. Without this the globe has nothing to render its
basemap with.

#### Shipping Data API Key
This is optional.
The `shipping_collector` needs an API Key to access the AIS stream carrying shipping messages.

To obtain one, head on over to https://aisstream.io/documentation on that page you will see
a link to `Sign In` (https://aisstream.io/authenticate) which will ask you to sign in to their
Github. Obviously if you don't have a Github account you will have to sign up for that first.

The process of obtaining the API Key is easy once you are signed in. There is a link `API Keys`
and you can create one there. Copy the key, and then back in the root directory edit the
file named `.env` and replace the `AIS_API_KEY` placeholder there with your newly minted
API Key. You will now be able to go into the Atmos GL Configurator and on the `Show` tab
in the `Background Processes` group enable either or both the Shipping and Lightning processes.

#### Lightning Strikes API Key
This is optional.
It is the API keyfor the `lightning_collector` and it's a similar deal, but also easy. 
You just need to create an account on https://openweathermap.org and the link to acquire 
an API Key is right there on the homepage. Just be aware it will take some hours before 
the key is made active.

In your `.env` file do as above and put the key in for the `OPENWEATHER_API_KEY` setting.
No quotes around the key are required.

Once the `lightning_collector` process is enabled and running, you will find that the table
in the database called `lightning_strikes` will acquire data, though it also gets culled
every few hours (`strike_expiry_hours` setting in that section) so won't get too populated.

#### NASA FIRMS API Key
This is optional.
It's for the `Wildfires` layer, sourced from NASA's FIRMS (Fire Information for Resource
Management System) — near-real-time active-fire detections from satellite, updated
roughly hourly.

Sign up for a free key at https://firms.modaps.eosdis.nasa.gov/api/map_key/ — just an
email address, no approval wait, the key is generated and emailed to you instantly.
Back in the root directory edit `.env` and put the key in for the `FIRMS_API_KEY`
setting, same as the others above.

With climate change seemingly setting various regions of the planet on fire, this one
is a really informative layer to have running. Once the key is in place, enable
`Wildfires` in the Atmos GL Configurator's `Show` tab, under the `Events` group.

#### Copernicus CDS/ADS API Key
This is optional.
It is for the `Greenhouse Gases` and `Air Quality` layers — without it both stay disabled.
Volcanoes' [Smoke Plume](#smoke-plume) overlay needs it too, since it's fetched
alongside Air Quality's own CAMS variables in the same request.

Register for a free account at https://ads.atmosphere.copernicus.eu/ (click
`Login/Register` — the same login also works for Copernicus's other data stores, so
if you've registered for CDS before, that account works here too). Your API key/token
is shown on your profile page once logged in. Put it in `.env` as `CDSAPI_KEY` — that
exact name, no `AIS_`/`OPENWEATHER_`-style prefix, since it's the literal name the
`cdsapi` Python library itself reads.

For viewing `Greenhouse Gases` one extra step is required which registration alone 
doesn't cover: **accepting the CC-BY licence**. This is a separate, per-dataset 
licence, not an account-level setting, so you won't find it under your profile. 

To accept it: open either dataset's page —
[cams-global-greenhouse-gas-forecasts](https://ads.atmosphere.copernicus.eu/datasets/cams-global-greenhouse-gas-forecasts)
or
[cams-global-ghg-reanalysis-egg4](https://ads.atmosphere.copernicus.eu/datasets/cams-global-ghg-reanalysis-egg4)

Click the `Download` tab, and scroll to the very bottom of the request-builder
form. There's a licence checkbox/accept button down there, separate from the account
terms you saw at sign-up. Both datasets require the *same* licence (`CC-BY-4.0`), so
accepting it once, on either page, covers both — no need to repeat it on the second.

Once the key is set (and, for Greenhouse Gases, the licence accepted), select
`Greenhouse Gases` in the Atmos GL Configurator's `Show` tab under the `Climate`
group, and/or `Air Quality` under the `Events` group.

#### NASA Earthdata Login Token
This is optional.
It's for [Flood Risk](#flood-risk)'s `Live` mode (observed current flooding) — `Historical`
mode needs no credential at all, so the layer still works without this if you'd rather
skip it.

Register a free account at https://urs.earthdata.nasa.gov, log in, open your profile,
and use `Generate Token` to create a Bearer token. Put it in `.env` as `EARTHDATA_TOKEN`.
Unlike the other keys above, this token expires after around 60 days — there's no
automatic renewal, so if `Live` mode suddenly stops updating, regenerate one and drop
the new value in.

### Forecasting
The map has a time scrubber built right into it — play, step forward/back, or drag through
the available forecast hours (configurable, default 24) for any layer that supports 
forecasting. The most useful elements which will show forecasts are of course Precipitation, 
Precipitable Water and Isobars, but others will do so as well such as Stormwatch, Temperature,
Waves and Wind. A notable exception is Clouds which are really only eye-candy as far as 
meteorology is concerned. They are built up over 24 hours as photo swathes by the polar 
orbiting NOAA satellites, so are not computed out into the future like the above datasets. 
The Global tab also has forecast-stepping controls that let you play forward through upcoming 
hours automatically.

For the elements which support forecasting, the number of hours into the future which you
can display is a configurable property of the `Data Collector`. In the configuration UI
screen go to the `Background` tab, and it is the `Cache hours` setting under
`Data Collector Properties`.

### Day and Night
There is a `Terminator` layer (in the `Show` tab's `Miscellaneous` group) which shades the
night side of the globe with a soft transition at the terminator line, for a realistic view
of what's happening on the planet day and night. It has its own opacity, colour and edge
softness settings if you'd like to tune the look, and can simply be switched off if you'd
rather have an unshaded view of whatever layers you have enabled.

### Place Markers
A static layer of city/town and marine feature labels (in the `Miscellaneous` group of
the `Show` tab), with priority-based label collision handling so busy areas don't turn
into an unreadable jumble of overlapping text as you zoom in and out. Turn on
`Weather popup` and hovering a marker shows a live-sampled temperature, humidity and
wind reading for that exact point, sourced from the same GFS atmos data every other
weather layer uses — a quick way to check conditions for a specific city without
needing any colourising layer switched on at all.

### Coastline Outline (Landmass)
A thin outline layer (`Miscellaneous` group) tracing coastlines and lake shores,
independent of any basemap style. Useful mainly alongside data layers or basemaps
where landmass boundaries otherwise get lost — it has its own line colour, a
contrasting halo colour (so it stays legible over any combination of basemap and data
layers underneath), line width and opacity.

### Watching It Work
To tail the logs of everything:

    ./atmos-gl.sh logs

Or just one service:

    ./atmos-gl.sh logs layer_builder

A healthy log might look something like this.
    
    data_collector-1       | 2026-07-11 08:57:09,922 [INFO] atmos_gl.collectors.gfs_atmos: Data Collector (gfs): 20260710 12Z, hours 009..056; stored 8 field(s).
    data_collector-1       | 2026-07-11 08:57:13,713 [INFO] atmos_gl.collectors.gfs_waves: Data Collector (waves): 20260710 12Z, hours 009..056; stored 1 field(s).
    layer_builder-1        | 2026-07-11 09:00:38,398 [INFO] atmos_gl.tasks.common: currents: rendered 1 hour(s) (51 available, stopped early after 7 examined).
    layer_builder-1        | 2026-07-11 09:00:39,710 [INFO] atmos_gl.tasks.waves: Waves: wrote swell velocity texture f018.
    layer_builder-1        | 2026-07-11 09:00:39,803 [INFO] atmos_gl.tasks.common: waves: rendered 1 hour(s) (49 available, stopped early after 11 examined).
    layer_builder-1        | 2026-07-11 09:00:41,787 [INFO] atmos_gl.tasks.isobars: Finished Isobars texture f019.
    layer_builder-1        | 2026-07-11 09:00:41,827 [INFO] atmos_gl.tasks.common: isobars: rendered 1 hour(s) (49 available, stopped early after 12 examined).
    layer_builder-1        | 2026-07-11 09:00:56,080 [INFO] atmos_gl.tasks.precipitation: Finished Precipitation texture f019 (low smoothing).
    layer_builder-1        | 2026-07-11 09:00:56,714 [INFO] atmos_gl.tasks.common: precipitation: rendered 1 hour(s) (49 available, stopped early after 12 examined).
    layer_builder-1        | 2026-07-11 09:03:38,143 [INFO] atmos_gl.tasks.scalar_field: Finished temperature texture f019.

The `data_collector` continuously fetches fresh data in the background regardless of
which layers you have switched on, so it's ready the moment you enable something. The
`layer_builder` is what turns collected data into the images/textures the globe actually
displays, cycling through every enabled layer, one forecast hour at a time, so all your
enabled layers make visible progress together rather than one finishing its whole backlog
before the next one starts.

For ship/lightning counts per region, run:

    ./atmos-gl.sh status

### Starting Fresh
If you really want a fresh start, stop the stack and remove the contents of the `data`
folder — you may need `sudo` depending on how your containers are set up.

## Map Layers
Apart from shipping there are, of course, other elements to the map display.
The full list is:

* Clouds
* Isobars
* Wind speed & direction
* Jet stream
* Precipitation
* Precipitable water (atmospheric moisture)
* Sea surface temperature
* Ocean currents
* Wave height & direction
* Air temperature
* Ozone layer density
* Storm watch
* Greenhouse gases (CO2/CH4)
* Lightning strikes
* Active storms
* Earthquakes
* Volcanoes (with a Smoke Plume/SO2 overlay)
* Wildfires
* World Events (conflict, explosions and high-level diplomacy)
* Troublespots (multi-source convergence zones)
* Air Quality (PM2.5/PM10/Smoke/SO2)
* Flood Risk (live observed inundation / historical hazard)
* Shipping
* Flight Radar
* Satellites
* Place markers
* Landmass outline (coastlines/lakes)

Each of these has its own configuration options.

Hopefully the settings in each section are fairly self-explanatory.

In the web configuration UI, the `Show` tab controls what gets shown on the map. If 
something is disabled, then the following tabs will show that section disabled, to avoid 
cluttering the interface. A checked layer whose underlying `Data Collector` is itself
switched off (see [Data Collector](#data-collector) below) is greyed out too, with an
explanation — a nudge that the checkbox alone won't get you data, since there's nothing
being collected for it to show.

The data for these elements is also updated according to a frequency determined by a 
`Runs per day `setting. This is to restrict load on the remote servers, which only update 
their data every few hours at most anyway. Data collection itself, though, always runs in the
background regardless of whether a layer is switched on — so the moment you enable
something it's ready to display rather than waiting for a fresh fetch.

### Weather
This group loosely comprises day-to-day atmospheric (and wind-driven ocean surface)
activity — the layers we have here are usually the most interesting from the point of
view of direct daily impact on our activities.

Of these clouds are the outlier in that it isn't a true data layer (see below). For
the rest the data is measured by satellites every 6 hours ('runs' termed 00Z, 06Z, 12Z, 18Z).
After each set of measurements has been downloaded, super-computers get to work
generating an hourly forecast using sophisticated models for each element measured
ending up with up to 384 hours prediction into the future for some of the data.

#### Clouds
Clouds are a global image built up over 24 hours by NOAA satellites photographing a 
'slice' of Earth from space. At any given time, therefore, we have a partial image 
of how the clouds look right now, depending on where the satellite is in its sweep. 
So to make sure we always have a full global image we grab imagery from 24h in the 
past guarnteeing a full sweep is available.

Clouds are therefore regarded as 'eye candy' and the images we have won't perfectly
reflect the more up to date real data (precipitation, isobars etc) that we have.
![Clouds](docs/atmos-gl-clouds.png)

#### Precipitation
Often mis-labelled as rainfall, even though that's what it mostly is, it does also
cover snow, sleet, hail etc. It's probably one of the most interesting layers from
the point of view of the amateur meteorologist given it often affects our daily plans
in life! Given it can be displayed forecasted, it's quite useful in that regard.
![Precipitation](docs/atmos-gl-precipitation.png)

#### Precipitable Water
This shows the total amount of water vapour sitting in a column of atmosphere — not rain
itself, but the fuel that heavy rain and atmospheric rivers need. Rather than colourising
the whole globe, it only highlights potential problem areas: anywhere below a configurable
`Critical Moisture Threshold` is left transparent, and only the genuinely moist regions
above it get coloured, brightest where moisture is most extreme. By default it uses the
same colour palette as Precipitation, so the two layers visually reinforce each other when
shown together, though a couple of other palettes are available if you'd prefer something
distinct.
![Precipitable Water](docs/atmos-gl-precipitable-water.png)

#### Isobars
The cornerstone of meteorology it shows what the pressure is doing in the atmosphere
and hence how the air masses are moving. Coupled with wind and precipitation
layers it really does show you how the weather is shaping up. Once again it can be
forecasted which makes it very useful. You can see examples of isobars in the Precipitation
image above.

#### Wind
Wind is depicted windy.com-style: animated flowing particle trails over a colourised
speed heatmap, so you can see both direction and intensity of the wind at a glance. This
layer is quite good paired with isobars where you can see the effect of differing air
pressure. If you want to see Precipitation at the same time as Wind then I recommend
you set `Heatmap opacity` of the underlying wind speed canvas to zero.
![Wind](docs/atmos-gl-wind.png)

#### Jet Stream
The high-altitude core of fastest wind (around the 250mb level), shown as animated
particle trails the same way Wind is, but with no underlying heatmap — the trails
themselves, colour-coded by speed, are the whole picture. It shares Wind/Currents'
particle engine, and reads the same GFS atmos data, just at a much higher, faster-moving
level. A `Palette` selector lets you pick between a few different colour treatments
(`stratosphere`, `aurora`, `inferno`) for however you like the jet core to look against
the basemap.

#### Waves
This one is a colourisation depicting wave height across the planet, GFS-sourced (same
forecast cycle as Wind) rather than the Ocean Currents layer's separate RTOFS model. It
gets quite interesting when you watch waves interacting with a storm, or a tsunami
eventuates from an earthquake. Unlike Wind's heatmap it doesn't cover landmasses, so it's
usually safe to leave showing alongside other layers rather than needing `Heatmap opacity`
zeroed out.
![Waves](docs/atmos-gl-waves.png)

#### Storm Tracks
A storm is depicted as a track history followed by a prediction cone showing where
the storm might go next according to the computational models. You can hover over
the storm track to see its name and various other details.

To get the data we scan two sources:
* NHC (National Hurricane Center)
    Responsible for tracking storms in the North Atlantic (AL) and Eastern North
    Pacific (EP). Their servers also typically host data for the Central North
    Pacific (CP), which is technically handled by the CPHC in Hawaii.

* JTWC (Joint Typhoon Warning Center): This is a joint U.S. Navy and Air Force
    command responsible for tracking tropical cyclones everywhere else on Earth
    including the Western North Pacific (typhoons), the Indian Ocean, and the
    Southern Hemisphere.

The major advantage of the ATCF (Automated Tropical Cyclone Forecast) system is
that it's a shared standard. Even though the NHC and JTWC are entirely different
organizations with different jurisdictions, they both output their data using
the same comma-delimited columns.

There are a lot of configurable items on the panel for storms, so you can get
these looking how you like to see them.
![Storms](docs/atmos-gl-storms.png)

#### Lightning
Lightning strikes are of course very brief, so we need a shorter expiry for them to
avoid them building up and obliterating parts of the map. You have an `Expiry hours`
slider in the configuration panel for these to let you tune that. Having them show
for a few hours is good to see how clusters of them are forming in stormy weather.
We also have a colour code to give an idea of timing:
* ![Bolt New](ui/images/bolt_white.png) Strikes within the last 15 minutes
* ![Bolt New](ui/images/bolt_yellow.png) Within the last hour
* ![Bolt New](ui/images/bolt_red.png) Older than an hour (but not yet expired)
![Lightning](docs/atmos-gl-lightning.png)

### Events
These are one-off events and cover Earthquakes and Volcanoes.

#### Earthquakes
These are one of the most interesting elements to put onto the map because it allows
you to visualise clusters of quakes appearing and providing a pattern of activity.
The symbol used comes in two colours, one for a very recent earthquake and one for
those older. You can set the `Recent activity hours` which determines this switch in
the configuration UI. The expiry hours can also be set there. Symbols:
* ![EQ recent](ui/images/earthquake_new.png) Recent earthquake activity
* ![EQ older](ui/images/earthquake_old.png) Older earthquakes
![Earthquakes](docs/atmos-gl-quakes.png)

#### Volcanoes
Rather than a static historical catalog, this is an event-driven feed of currently
active volcanoes, refreshed regularly like Storms and Earthquakes. Two sources are
combined:

* The Smithsonian/USGS Global Volcanism Program's **Weekly Volcanic Activity Report** —
  a global, curated feed of volcanoes with genuinely new or ongoing activity this week.
* **USGS HANS**, which adds richer real-time alert-level detail for the ~60 US volcanoes
  USGS actively monitors.

Markers are colour-coded straight from the report itself:
* ![Volcano new](ui/images/volcano_new.png) New activity this week
* ![Volcano continuing](ui/images/volcano_continuing.png) Continuing activity

Hover a marker for its country, activity type and report description, plus — for US
volcanoes HANS is watching — its USGS alert level and colour code. A volcano drops off
the map automatically once it's no longer reported as active in either source (after 14
days of inactivity), so there's nothing to filter or configure beyond `Icon zoom`.
![Volcanoes](docs/atmos-gl-volcanoes.png)

##### Smoke Plume
Also in Volcano Properties, `Show Smoke Plume` overlays a global, volcano-specific SO2
(sulphur dioxide) total-column heatmap, in Dobson Units, rendered in its own bright
cyan-to-magenta palette — deliberately not the [Air Quality](#air-quality) layer's
green-to-purple gradient, so the two never look ambiguous if you have both showing at
once. Copernicus CAMS models this as a genuinely separate variable from Air Quality's
own general `SO2` option (which includes industrial and other non-volcanic sources
too), each with its own opacity/threshold controls, so you don't need to visit Air
Quality at all to use Smoke Plume. It's normal for the overlay to show nothing at
all some of the time — CAMS only populates this variable when its model detects
volcanic SO2 somewhere on the planet, so an empty overlay is telling you exactly
that, not a stale or broken layer. Like Air Quality, it needs a
[Copernicus CDS/ADS API Key](#copernicus-cdsads-api-key).
![Volcanoes](docs/atmos-gl-volcanoes-smoke-plume.png)

#### Wildfires
If you have a [NASA FIRMS API Key](#nasa-firms-api-key) then you can enable the
`Wildfires` layer in the `Events` tab. The configuration settings for this allow you
to tune the underlying heatmap (showing fire risk) visibility, and also filter out
wildfire detections so the map isn't swamped with them. The satellite which is the
source of this data is simply measuring heat anomalies and the radiative power of
them, so it isn't a guarantee an actual fire is burning at that location. Position
is also fairly crude, being a 375 square meter area for each blob on the map.

That said it does provide a very interesting view of how dry the vegetation is across
the globe and hence how high the fire risk is, as well as highlighting areas where
fires are most probably burning. When you see a group of many together spread across
a line or in a cluster, there is a high probability is is a wildfire.
![Wildfires](docs/atmos-gl-wildfires.png)

#### World Events
Sourced from the [GDELT Project](https://www.gdeltproject.org)'s real-time Event
Database, refreshed roughly every 15 minutes. Rather than showing GDELT's entire
firehose of hundreds of thousands of daily events, this layer is filtered down to a
curated, high-signal set of four categories:

* **Explosion** — suicide, car and roadside bombings
* **Conflict** — armed conflict, occupation, ceasefire violations
* **Targeted / mass violence** — abductions, assassinations, mass killings, use of
  weapons of mass destruction
* **Diplomatic Meeting** — a genuinely high-level summit or negotiation (NATO, the UN,
  G7/G8/G20, the EU, and similar bodies), not just any two officials on a routine call

Each category has its own marker colour, and can be individually shown or hidden in
`World Events Properties`, alongside marker size, opacity, and how many days of history
to display. Hovering a marker shows who was involved, where, when, how many sources
reported it, and a link to read the original article. On first setup the layer
backfills a configurable window of recent history (three days by default) so it isn't
empty while waiting for new data to arrive, and self-heals if the collector is ever
offline for a while.

#### Troublespots
A derived layer, not its own data source: it flags areas where at least two of four
independently-collected feeds converge within the same geographic cell and time window —

* **Earthquakes**
* **Wildfires**
* **Volcanic Activity**
* **World Events** (any of the four categories above, counted once regardless of category)

Convergence is banded by how many of the four types overlap in a cell — **Elevated** (2),
**High** (3), **Severe** (4) — each rendered as a smooth, hatched boundary outlined in
that band's colour, so it reads as a heatmap-style zone rather than a hard grid. Hovering
a zone breaks down exactly which sources contributed and how many reports came from each.
Cell size and time window are both configurable.

Because it has no collector of its own, the `Show` checkbox is only available (not greyed
out) once at least two of its four source layers' Data Collectors are enabled — with only
zero or one contributing, a troublespot can never form.

#### Air Quality
Needs a [Copernicus CDS/ADS API Key](#copernicus-cdsads-api-key), the same one used by
Greenhouse Gases below. Shows near-real-time air quality sourced from Copernicus CAMS's
global atmospheric composition forecast. A `Variable` selector switches between:

* `PM2.5` — fine particulate matter, the pollutant most closely tied to respiratory and
  cardiovascular health effects.
* `PM10` — coarser particulate matter (dust, pollen, larger combustion particles).
* `Smoke (AOD)` — Aerosol Optical Depth, a column-wide measure most useful for tracking
  wildfire smoke plumes as they spread. Being dimensionless, it has no official
  health-based threshold the way surface PM2.5/PM10 do.
* `SO2` — Sulphur Dioxide (total column, in Dobson Units), from all sources (industrial,
  general atmospheric, etc.) — a genuinely different CAMS variable from the
  [Volcanoes](#volcanoes) layer's `Show Smoke Plume` overlay, which tracks a
  volcano-specific SO2 field instead. Rendered in the same green-to-purple gradient as
  `PM2.5`/`PM10`/`Smoke (AOD)` above, deliberately distinct from Smoke Plume's own
  cyan-to-magenta palette.

The `PM2.5`/`PM10` dropdown entries show the concentration widely considered "Unhealthy
for Sensitive Groups" by the US EPA's Air Quality Index, and that same value is the
default for the `Critical threshold` slider — readings below it fade out via a smooth
transparency ramp, so the map highlights genuinely elevated readings rather than
colourising the whole globe. All four variables render every cycle, so switching
between them in the config UI applies instantly, no render wait. Colours follow the
familiar green → yellow → orange → red → purple AQI convention used by most phone
weather apps' air-quality widgets.
![Air Quality](docs/atmos-gl-air-quality.png)

#### Flood Risk
A land-only layer with two independently-sourced modes sharing one `Mode` selector — they
show genuinely different things, not two views of the same data:

* `Live (Observed Inundation)` — actual flooding detected right now, from NASA's LANCE
  MODIS satellite flood product. Needs a [NASA Earthdata Login Token](#nasa-earthdata-login-token).
  This used to be a Copernicus GloFAS ensemble river-discharge *forecast* instead, but that
  was abandoned entirely after repeated out-of-memory crashes and unfixable network
  flakiness against ECMWF/Copernicus's shared backend — observed satellite detection turned
  out to be both more reliable to collect and arguably more useful than a forecast anyway.
  Optical satellite flood detection can occasionally mistake terrain shadow in mountainous
  regions (low sun angle across steep valleys) for flooding — a spatial filter requiring
  flagged pixels to sit near known water suppresses most of this, though not perfectly, so
  don't be surprised by the odd stray pixel in hill country.
* `Historical (JRC 100yr Hazard)` — a fixed, terrain-derived flood hazard classification
  from the Copernicus/JRC Global River Flood Hazard Maps (100-year return period), no
  credential needed. Four severity categories from a shade under 1m depth up to over 10m,
  mosaicked once from ~90m-resolution tiles and cached forever, since it doesn't change
  day to day the way `Live` does.

Since the two modes measure different things on different scales, switching `Mode` doesn't
just recolour the same data — it swaps which of the two independently-rendered layers you're
looking at, applying instantly with no render wait either way.

### Climate
This area is quite fascinating as it covers the entire planet. The data is sourced
from https://nomads.ncep.noaa.gov/ which contains a staggering amount of publicly
available data. Currently we are just dipping our toes in those waters and providing
Sea Surface Temperature, Air Temperature, Ocean Currents and
the Ozone Layer data resolved to a 0.25 degree grid (with interpolation/smoothing
as required).

Each of Sea Surface Temperature, Ocean Currents, Air Temperature, Ozone, Storm Watch
and Greenhouse Gases can be shown independently — the `Show` tab presents them as
regular checkboxes, so you're free to enable as many at once as you like (SST Anomaly
alongside Ocean Currents, for instance). Since most of these colourise the entire
planet, though, showing several at once usually turns into an overlapping mashup of
colour rather than anything readable — `Opacity` on each layer is there to help you
find a combination that works, and in general it's still worth keeping most other
colourising layers (Precipitation, Precipitable Water) switched off while you're
looking at Climate data, though marker elements like Earthquakes and Shipping are
no problem to leave on alongside any of these.

#### SST
Sea Surface Temperature, sourced straight from NOAA's data. A fascinating visualisation
of what's happening across our oceans. Here is an example showing `Absolute SST`.
![SST Absolute](docs/atmos-gl-sst-absolute.png)
But there is another mode available `Anomaly SST` which shows the variation from historical
`normal` for sea surface temperature. Here's an example of that view:
![SST Anomaly](docs/atmos-gl-sst-anomaly.png)

#### Air Temperature
Air/land temperatures are a measure of what's going on in our atmosphere, resolved
globally at the same grid resolution as everything else here.
![Air Temperature](docs/atmos-gl-air-temperature.png)

#### Ocean Currents
These are depicted as curves with arrows showing the flows going on in our oceans
on a real-time basis. This is one layer which could be shown together with others
such as Isobars, Clouds and Precipitation as it isn't a colouration layer.
![Ocean Curents](docs/atmos-gl-ocean-currents.png)

#### Ozone Layer
Another interesting climate layer to have a look at. There are a couple of palettes to
choose from, both built around a setting called `Critical Ozone Threshold`. The `du`
stands for "Dobson Units" which is what the ozone layer density is measured in. A value
of `220.0` is considered the threshold for a "hole" in the ozone layer. Any ozone reading
below that setting is coloured (brightest at the very worst readings), and anything above
it fades to a dim, near-transparent "safe" tone — so you only really see the problem areas.
![Ozone](docs/atmos-gl-ozone.png)

#### Storm Watch
This layer shows where there is a likelihood of a storm forming. It uses the CAPE
measurement (Convective Available Potential Energy) which, as the name suggests is
a value which expresses how much energy is available in the atmosphere, but it
also looks at the CIN (Convective Inhibition) measurement. CIN represents the
amount of energy required to overcome the negatively buoyant, stable "cap" or
"lid" in the lower atmosphere. It is essentially the barrier that a rising air
parcel must break through before it can tap into the CAPE (Convective Available
Potential Energy) above it. Combining both gives us a reasonable idea of the
actual potential for storm formation.
![Storm Watch](docs/atmos-gl-stormwatch.png)

#### Greenhouse Gases (CO2/CH4)
Needs a [Copernicus CDS/ADS API Key](#copernicus-cdsads-api-key). Shows global CO2 or
CH4 (methane) levels — a species toggle switches between the two — in one of two modes:

* `Absolute` shows today's levels, sourced from Copernicus CAMS's daily global
  greenhouse gas forecast.
* `Anomaly` shows how today's levels compare against a historical baseline year of
  your choosing (`Baseline year`, 2003–2020), computed by comparing the current
  reading against Copernicus's CAMS EGG4 reanalysis for that year — unlike Sea
  Surface Temperature above, no one publishes a ready-made CO2/CH4 anomaly, so this
  is worked out for you here.

Both modes and both species render every cycle, so switching any of the settings
applies instantly without waiting for a fresh calculation. `Absolute` mode's colour
scale (min/max and palette) is independently configurable per species, since CO2 and
CH4 sit on very different numeric ranges (parts-per-million vs. parts-per-billion);
`Anomaly` mode's colour scale is worked out automatically from the data each cycle.

### Shipping
Ships are shown as small icons pointing in their current heading, colour-coded
by vessel type: red for tankers, green for cargo, and violet/purple for passenger and
other vessel types.
* ![Tankers](ui/images/red_ship_base.png) Tankers
* ![Cargo](ui/images/green_ship_base.png) Cargo
* ![Passenger](ui/images/purple_ship_base.png) Passenger/other

Smaller vessels only appear once you're zoomed in reasonably close, with progressively
larger ones visible from further out — this keeps busy shipping lanes from turning into
an unreadable wall of icons at low zoom.
![Shipping](docs/atmos-gl-shipping.png)

#### Shipping Data Acquisition
Ships broadcast data in the form of messages continuously at regular intervals. The main
message they emit is a `PositionReport` which contains information as to latitude and longitude,
current heading and speed. This message is usually fairly frequent. The other message of
interest to us is the `ShipStaticData` which has details of the ship itself such as name,
size, draught, type and IMO number (International Maritime Organization number). This message
is broadcast much less frequently, but the data is extremely useful to identify the type of
vessel and its current loading state (draught).

The `shipping_collector` listens for both types of message and will gradually populate your
database `ships` table with them. It does this by slicing the globe up into segments by
longitude, and then listening in each slice defined as a bounding box. The listen duration
varies according to how busy each slice is expected to be, based on shipping lanes and the
area of ocean it's looking at.

At any given instant either a `ShipStaticData` or `PositionReport` message might come in. If it's
a `PositionReport` the message is fairly specific to position, heading, speed etc. and contains
no details about the ship itself. The `shipping_collector` will look for an existing `ships`
record in our database with the same `mmsi` identifier, and if found add the new position info.
It also logs the position in the tracking table `ship_position` so we can display vessel tracks.
If it doesn't find an existing `ships` record it creates a `shadow` record with scant data about
the ship, basically just the name and the `mmsi` identifier. At some point we would hope to
back-fill that data when a `ShipStaticData` is acquired for it.

### Flight Radar
Live aircraft positions overlaid on the globe, sourced from [adsb.lol](https://adsb.lol)'s
free community ADS-B receiver network. Each aircraft is drawn as a simplified plane
silhouette — a distinct straight-wing shape for gliders/balloons/drones — tinted by
broad size/class, from near-white for widebody jets down to light blue for helicopters,
military types and anything unrecognised, and rotated to match its heading. Positions
are dead-reckoned smoothly from each aircraft's last known speed and track between
updates, rather than snapping every time fresh data arrives.

Hover over an aircraft for its flight number, aircraft type/class, airline (inferred
from the ICAO callsign for major/regional carriers — not shown for private/GA traffic,
which usually broadcasts its own registration as the callsign instead), registration, a
climb/descend/level/landed status, current altitude, the autopilot's selected target
altitude (once one is set and reachable), ground speed, heading and ICAO hex code. Like
Shipping, low-altitude and on-the-ground traffic only reveals itself once you're zoomed
in reasonably close, so busy airspace around major airports doesn't turn into a wall of
icons from further out.
![Flight Radar](docs/atmos-gl-flightradar.png)

#### Flight Radar Data Acquisition
Like every other layer, Flight Radar is now backed by a `data_collector` process and a
database table — the browser never talks to adsb.lol directly. A dedicated collector
continuously sweeps the globe: it samples fastest wherever a viewer's browser is
actually looking right now (read from the same request your browser makes to fetch
aircraft), and falls back to a coarser, slower background sweep everywhere else, so
there's always some picture of global air traffic even where nobody's currently
watching. adsb.lol's request tolerance is limited, so this all happens within a fixed,
shared request budget — the collector, not your browser, is adsb.lol's only consumer.
Your browser just polls the database periodically, the same way every other data
layer on the map works.

### Data Collector
The data collector is a separate background process which collects data for:
* Quakes
* Storms
* Volcanoes
* Satellites
* SST
* Greenhouse Gases 
  * CO2 - Carbon Dioxide
  * CH4 - Methane
* Air Quality
  * PM2.5 - Fine Particulate Matter 2.5 microns
  * PM10 - Coarse Particulate Matter 10 microns
  * Smoke AOD - Aerosol Optical Depth of smoke from fire
  * SO2 - Sulphur Dioxide from all sources
* GFS Atmos:
  * Isobars
  * Precipitation
  * Temperature
  * Ozone
  * Wind
  * Stormwatch
* Clouds
* Waves
* Currents
* Shipping
* Lightning
* Flight Radar
* Markers (Cities, Towns etc)

There is a status page in the configuration UI which shows the amount of data currently
collected, and also how much of that data has been crunched into front-end content. You
can also opt out of collecting data for them, depending on what you are interested in
seeing. Be aware though, that data generally takes some hours to collect, so if you
enable something don't expect to be seeing anything on the globe for a while. If you have
everything enabled the system is configured to be a light footprint in terms of remote
download requests, so all you scarifice there is some disk space. That said, if you are
genuinely never going to look at some of the data, it makes no sense to collect it.
![Data Status](docs/atmos-gl-conf-status.png)

If you're running on a small VPS or a machine you don't want pegged, the `Global` tab's
`Performance tier` setting (Low/Medium/High) caps how many layers `layer_builder` renders
concurrently — lower it if rendering is starving other things on the host of CPU/RAM;
raise it for faster rendering on a beefier machine.

### Satellites
Plotting satellite paths is something that XPlanet did, and since that venerable project
inspired this one, we do it here too.

There are literally thousands of objects up there whizzing around the Planet, not to mention 
the mega-clusters like Starlink etc. so to keep things sane there's a multi-select with a 
handful of the most popular ones you can display.

There is also a text box for you to add a comma-separated list of satellite names to cater
for a few you want which aren't in the list. Just be aware any you add have to come from
the groups currently downloaded from Celestrak: `resource`, `science`, `stations`, and
`weather`. You can see what's in those by browsing to the following URL, replacing `{group}`
with one of those 4 group names:
    https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT=tle
![Satellites](docs/atmos-gl-satellites.png)

## Developer's Corner
If you want to tinker with the code (and maybe help improve it with some PRs) this
section aims to provide you with a few helpful instructions.

### Clone the Repository

    cd /your/preferred/workspace
    git clone -v https://github.com/paulwaite87/atmos-gl

After that, most things can be done via the Makefile. To see what is available:

    make help

Use `make up` to start up the stack with your local source bind-mounted in
(via `docker-compose.override.yml`) so code edits just need `make reload` to take effect —
no image rebuild required. `make build`/`make rebuild` are only needed for dependency or
Dockerfile changes.

### Control
Everything is managed through the Makefile.

    make up

Gets your stack running (an alias for `docker compose up -d`). And to stop it:

    make stop

Once running, browse to `http://localhost:8180` for the live globe.

If you change something in `config/atmos-gl.json` by hand rather than through the web UI,
or if you just want to restart the backend services:

    make reload

`make prod`/`make prod-down` do the same job as `atmos-gl.sh start`/`stop` — running exactly
as a package consumer would, against the published images rather than a local build.

### Logging
To tail the logs of everything:

    make logs

Or just one service:

    make logs service=layer_builder

See [Watching It Work](#watching-it-work) above for what a healthy log cycle looks like.

One useful command for a quick data snapshot is:

    make status

That prints a summary of everything currently stored in the database — ships and lightning per
region, plus earthquakes, volcanic activity, fires, world events, storms, satellites, weather
model cache coverage, place markers, and flight radar — one heading per data domain.

### Testing a Changed Layer Faster (Round-Robin Priority)
`layer_builder` renders the multi-hour layers (isobars, precipitation, wind, currents,
jetstream, waves, temperature, ozone, stormwatch, pwat, fires) in rounds — one forecast
hour per section per round — cycling through them in a fixed declared order. If you're
iterating on one layer's render code, waiting for it to come back around can be slow.

Bump a layer to the front of the next round without disturbing whatever's already in
flight:

    curl -X POST http://localhost:9000/api/layer_builder/priority \
        -H 'Content-Type: application/json' \
        -d '{"sections": ["precipitation"]}'

Check the current order:

    curl http://localhost:9000/api/layer_builder/priority

Reset back to the default order:

    curl -X POST http://localhost:9000/api/layer_builder/priority/reset

This is in-memory only — it resets to the default order whenever `layer_builder`
restarts (e.g. after `make reload`).

### Everything Else
`make test`, `make lint` and `make bash` are all there for you too — run `make help` for
the full list.

See `CLAUDE.md` at the repo root for the architectural conventions this project follows
(collector/render task layout, testing approach, git workflow) before diving in. Feel free
to clone the repo, update the code and give us a pull request!

### Regions
The database will be seeded with a few regions, which is a legacy thing from before this
project forked from `worldmap-desktop`. It serves no real purpose now except to provide
a breakdown of ships and lightning strikes per region when you do `./atmos-gl.sh status`.

However, if you want to add a region, we will need to get nerdy and insert data into your 
database. The format of an SQL statement which will do just that is:

    INSERT INTO map_region (label, boundary) VALUES ('My Region', ST_MakeEnvelope(-7.346384, 42.490591, 10.854976, 51.487329, 4326));

Copy this somewhere that you can change it in an editor.

For the coords, go to https://tools.mofei.life/bbox#1/0/0 and navigate to wherever is
centre of the region you want on the World map there. Zoom in and then pull a bounding-box
with SHIFT-drag. In the WGS84 box `Copy` the bounding box coords and paste those (minus 
the square brackets) into your INSERT.

The co-ordinate ordering is already correct. Give your INSERT a new appropriate label,
(replacing 'My Region', then copy that SQL statement onto your clipboard and execute
this command:

    docker compose exec atmos_gl_db psql -U agl atmos_gl

(If you've cloned the repo, `make psql` is a shortcut for this.)

That will get you into the Atmos GL database PSQL shell. Paste your INSERT into that and
hit enter. Bingo, a brand new region. The Atmos GL configurator should read your new region
and allow you to select it.

Just hit Ctrl-d to get out of the database.

"""
FastAPI web interface for the Maritime Intelligence Platform.

Run with:
    pip install fastapi uvicorn
    python -m uvicorn src.web.app:app --reload --port 8000
"""

from datetime import datetime

try:
    from fastapi import FastAPI, Query
    from fastapi.responses import HTMLResponse, JSONResponse
except ImportError:
    raise ImportError(
        "FastAPI is required for the web interface. "
        "Install with: pip install fastapi uvicorn"
    )

from src.config.constants import ROUTES, PORTS, VESSEL_TYPES, BEAUFORT_SCALE, VESSELS, knots_to_beaufort
from src.services.sailing_ban_checker import SailingBanChecker
from src.services.route_analysis import analyze_route_risk, get_route_sample_points
from src.data_collection.demo_data import generate_demo_route_conditions, DEMO_SCENARIOS

app = FastAPI(
    title="Greek Maritime Intelligence Platform",
    description="Ferry cancellation prediction from sea-state and wind conditions",
    version="0.1.0",
)

checker = SailingBanChecker()
ml_checker = None

def _get_ml_checker():
    """Lazily initialize ML checker."""
    global ml_checker
    if ml_checker is None:
        ml_checker = SailingBanChecker(use_ml=True)
    return ml_checker


@app.get("/", response_class=HTMLResponse)
def index():
    """Serve the main dashboard page with route map."""
    return MAP_HTML


@app.get("/api/routes")
def list_routes():
    """List all configured routes with port coordinates."""
    result = []
    for route_id, info in ROUTES.items():
        origin_code, dest_code = route_id.split("-")
        origin = PORTS.get(origin_code, {})
        dest = PORTS.get(dest_code, {})
        result.append({
            "id": route_id,
            "origin": {**info, "code": origin_code, "lat": origin.get("lat"), "lon": origin.get("lon")},
            "destination": {"code": dest_code, "name": info["destination"], "lat": dest.get("lat"), "lon": dest.get("lon")},
            "distance_nm": info["distance_nm"],
            "sea_area": info["sea_area"],
            "exposed": info.get("exposed", True),
        })
    return result


@app.get("/api/check")
def check_routes(
    vessel: str = Query("CONVENTIONAL", description="Vessel type"),
    scenario: str = Query("auto", description="Demo scenario: calm, storm, meltemi, auto"),
    days: int = Query(2, ge=1, le=7, description="Forecast days"),
):
    """Check all routes using demo data (no internet required)."""
    results = []
    for route_id, route_info in ROUTES.items():
        weather_data = generate_demo_route_conditions(
            forecast_days=days, scenario=scenario,
        )
        result = checker.check_route(route_id, weather_data, vessel)

        # Add summary stats
        hourly = result.get("hourly", [])
        result["max_wind"] = max((h["wind_speed_knots"] for h in hourly), default=0)
        result["max_wave"] = max((h["wave_height_m"] for h in hourly), default=0)
        result["max_beaufort"] = knots_to_beaufort(result["max_wind"])

        # Don't send full hourly in list view
        result["hourly_count"] = len(hourly)
        del result["hourly"]
        results.append(result)

    status_order = {"BAN_LIKELY": 0, "AT_RISK": 1, "CLEAR": 2}
    results.sort(key=lambda r: status_order.get(r["overall_status"], 3))

    return {
        "timestamp": datetime.now().isoformat(),
        "vessel_type": vessel,
        "scenario": scenario,
        "forecast_days": days,
        "routes": results,
        "summary": {
            "total": len(results),
            "ban_likely": sum(1 for r in results if r["overall_status"] == "BAN_LIKELY"),
            "at_risk": sum(1 for r in results if r["overall_status"] == "AT_RISK"),
            "clear": sum(1 for r in results if r["overall_status"] == "CLEAR"),
        },
    }


@app.get("/api/check/{route_id}")
def check_single_route(
    route_id: str,
    vessel: str = Query("CONVENTIONAL"),
    scenario: str = Query("auto"),
    days: int = Query(2, ge=1, le=7),
):
    """Check a single route with full hourly details."""
    if route_id not in ROUTES:
        return JSONResponse(
            status_code=404,
            content={"error": f"Unknown route: {route_id}", "available": list(ROUTES.keys())},
        )

    weather_data = generate_demo_route_conditions(forecast_days=days, scenario=scenario)
    return checker.check_route(route_id, weather_data, vessel)


@app.get("/api/scenarios")
def list_scenarios():
    """List available demo scenarios."""
    return DEMO_SCENARIOS


@app.get("/api/ports")
def list_ports():
    """List all ports with coordinates."""
    return PORTS


@app.get("/api/check-ml")
def check_routes_ml(
    vessel: str = Query("CONVENTIONAL"),
    scenario: str = Query("auto"),
    days: int = Query(2, ge=1, le=7),
):
    """Check all routes with ML predictions (requires trained models)."""
    c = _get_ml_checker()
    results = []
    for route_id in ROUTES:
        weather_data = generate_demo_route_conditions(forecast_days=days, scenario=scenario)
        result = c.check_route(route_id, weather_data, vessel)

        hourly = result.get("hourly", [])
        result["max_wind"] = max((h["wind_speed_knots"] for h in hourly), default=0)
        result["max_wave"] = max((h["wave_height_m"] for h in hourly), default=0)
        result["max_beaufort"] = knots_to_beaufort(result["max_wind"])
        result["hourly_count"] = len(hourly)
        del result["hourly"]
        results.append(result)

    status_order = {"BAN_LIKELY": 0, "AT_RISK": 1, "CLEAR": 2}
    results.sort(key=lambda r: status_order.get(r["overall_status"], 3))

    return {
        "timestamp": datetime.now().isoformat(),
        "vessel_type": vessel,
        "scenario": scenario,
        "ml_enabled": c.ml_predictor is not None,
        "routes": results,
    }


@app.get("/api/calibrate")
def calibrate_thresholds_endpoint():
    """Analyze ground truth data and suggest threshold calibrations."""
    try:
        from src.models.ml_predictor import calibrate_thresholds
        return calibrate_thresholds()
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/ground-truth/stats")
def ground_truth_stats():
    """Get ground truth data statistics."""
    try:
        from src.data_collection.ground_truth import GroundTruthCollector
        collector = GroundTruthCollector()
        return collector.get_stats()
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/health")
def health_check():
    """Health check endpoint — verifies all components."""
    status = {"status": "ok", "timestamp": datetime.now().isoformat()}

    # Check ML models
    try:
        c = _get_ml_checker()
        status["ml_available"] = c.ml_predictor is not None
    except Exception:
        status["ml_available"] = False

    # Check ground truth data
    try:
        from src.data_collection.ground_truth import GroundTruthCollector
        collector = GroundTruthCollector()
        stats = collector.get_stats()
        status["ground_truth_records"] = stats.get("total", 0)
    except Exception:
        status["ground_truth_records"] = 0

    # Check cache
    from src.utils.cache import api_cache
    status["cache_size"] = api_cache.size

    # Check rate limiter
    from src.utils.rate_limiter import api_rate_limiter
    status["rate_limit_remaining"] = api_rate_limiter.remaining

    status["routes_configured"] = len(ROUTES)
    status["ports_configured"] = len(PORTS)
    status["vessels_configured"] = len(VESSELS)

    return status


@app.get("/api/route-analysis/{route_id}")
def route_analysis(
    route_id: str,
    wind_speed: float = Query(20.0, description="Wind speed in knots"),
    wind_direction: float = Query(0.0, description="Wind direction (0=N, 90=E)"),
    wave_height: float = Query(2.0, description="Wave height in meters"),
    vessel: str = Query("CONVENTIONAL"),
):
    """Detailed route risk analysis with wind angle, shelter, and waypoints."""
    if route_id not in ROUTES:
        return JSONResponse(
            status_code=404,
            content={"error": f"Unknown route: {route_id}"},
        )
    return analyze_route_risk(route_id, wind_speed, wind_direction, wave_height, vessel)


@app.get("/api/vessels")
def list_vessels():
    """List all known vessels with their specific thresholds."""
    return VESSELS


@app.get("/api/route-points/{route_id}")
def route_points(
    route_id: str,
    interval_nm: float = Query(20.0, ge=5, le=100),
):
    """Get sample points along a route for multi-point weather checking."""
    if route_id not in ROUTES:
        return JSONResponse(status_code=404, content={"error": f"Unknown route: {route_id}"})
    return get_route_sample_points(route_id, interval_nm=interval_nm)


# ---------------------------------------------------------------------------
# Embedded HTML dashboard with Leaflet.js map
# ---------------------------------------------------------------------------
MAP_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Maritime Intelligence Platform</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0a1628; color: #e0e0e0; }
  #header { background: #1a2a4a; padding: 12px 24px; display: flex;
            justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; }
  #header h1 { font-size: 18px; color: #4fc3f7; }
  #controls { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  select, button { padding: 6px 12px; border-radius: 4px; border: 1px solid #345;
                   background: #1e3050; color: #e0e0e0; font-size: 13px; cursor: pointer; }
  button { background: #2196F3; border: none; font-weight: bold; }
  button:hover { background: #1976D2; }
  .btn-sm { padding: 4px 8px; font-size: 11px; background: #37474f; }
  .btn-sm.active { background: #2196F3; }
  #main { display: flex; flex-direction: column; height: calc(100vh - 52px); }
  #map { flex: 1; min-height: 300px; }
  #bottom { display: flex; background: #0d1f3c; max-height: 45vh; overflow: hidden; }
  #table-panel { flex: 1; overflow-y: auto; }
  #chart-panel { width: 400px; padding: 12px; display: none; }
  #chart-panel.visible { display: block; }
  #summary { background: #0d1f3c; padding: 8px 24px; display: flex;
             gap: 16px; align-items: center; font-size: 13px; flex-wrap: wrap; }
  .stat { display: flex; align-items: center; gap: 4px; }
  .dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
  .dot-red { background: #f44336; }
  .dot-yellow { background: #ff9800; }
  .dot-green { background: #4caf50; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th { background: #1a2a4a; position: sticky; top: 0; padding: 5px 10px;
       text-align: left; color: #90caf9; font-size: 11px; }
  td { padding: 4px 10px; border-bottom: 1px solid #1a2a4a; cursor: pointer; }
  tr:hover td { background: #1a2a4a; }
  .status-BAN_LIKELY { color: #f44336; font-weight: bold; }
  .status-AT_RISK { color: #ff9800; font-weight: bold; }
  .status-CLEAR { color: #4caf50; }
  #timer { font-size: 11px; color: #78909c; }
  @media (max-width: 768px) {
    #bottom { flex-direction: column; }
    #chart-panel { width: 100%; max-height: 200px; }
  }
</style>
</head>
<body>
<div id="header">
  <h1>Greek Maritime Intelligence</h1>
  <div id="controls">
    <select id="vessel">
      <option value="CONVENTIONAL">Conventional</option>
      <option value="HIGH_SPEED">High Speed</option>
      <option value="CATAMARAN">Catamaran</option>
      <option value="SMALL">Small Craft</option>
    </select>
    <select id="scenario">
      <option value="auto">Auto (seasonal)</option>
      <option value="calm_summer">Calm</option>
      <option value="storm">Storm</option>
      <option value="meltemi">Meltemi</option>
    </select>
    <button onclick="refresh()">Refresh</button>
    <button class="btn-sm" id="autoBtn" onclick="toggleAutoRefresh()">Auto: OFF</button>
    <span id="timer"></span>
  </div>
</div>
<div id="main">
  <div id="map"></div>
  <div id="summary"></div>
  <div id="bottom">
    <div id="table-panel"><table><thead><tr>
      <th>Route</th><th>From</th><th>To</th><th>Status</th>
      <th>Wind</th><th>Wave</th><th>Bf</th>
    </tr></thead><tbody id="tbody"></tbody></table></div>
    <div id="chart-panel"><canvas id="routeChart"></canvas></div>
  </div>
</div>

<script>
const map = L.map('map').setView([37.5, 25.0], 7);
L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
  attribution: 'CartoDB', maxZoom: 18
}).addTo(map);

let routeLines = [], portMarkers = [], wpMarkers = [];
let autoInterval = null, routeChart = null, lastData = null;
const colors = { BAN_LIKELY: '#f44336', AT_RISK: '#ff9800', CLEAR: '#4caf50' };

function toggleAutoRefresh() {
  const btn = document.getElementById('autoBtn');
  if (autoInterval) {
    clearInterval(autoInterval);
    autoInterval = null;
    btn.textContent = 'Auto: OFF';
    btn.classList.remove('active');
    document.getElementById('timer').textContent = '';
  } else {
    autoInterval = setInterval(refresh, 60000);
    btn.textContent = 'Auto: 1m';
    btn.classList.add('active');
  }
}

async function showRouteDetail(routeId) {
  const vessel = document.getElementById('vessel').value;
  const scenario = document.getElementById('scenario').value;
  const resp = await fetch('/api/check/' + routeId + '?vessel=' + vessel + '&scenario=' + scenario);
  const data = await resp.json();

  // Show waypoints on map
  wpMarkers.forEach(m => map.removeLayer(m));
  wpMarkers = [];
  try {
    const pts = await (await fetch('/api/route-points/' + routeId)).json();
    pts.forEach((p, i) => {
      if (i > 0 && i < pts.length - 1) {
        const m = L.circleMarker([p.lat, p.lon], {
          radius: 3, fillColor: '#fff', fillOpacity: 0.5, color: '#fff', weight: 1
        }).addTo(map).bindTooltip(p.name);
        wpMarkers.push(m);
      }
    });
  } catch(e) {}

  // Build chart
  const panel = document.getElementById('chart-panel');
  panel.classList.add('visible');

  const hourly = data.hourly || [];
  const labels = hourly.map(h => h.time ? h.time.substring(11, 16) : '');
  const winds = hourly.map(h => h.wind_speed_knots);
  const waves = hourly.map(h => h.wave_height_m);

  if (routeChart) routeChart.destroy();
  routeChart = new Chart(document.getElementById('routeChart'), {
    type: 'line',
    data: {
      labels,
      datasets: [
        { label: 'Wind (kn)', data: winds, borderColor: '#4fc3f7', backgroundColor: 'rgba(79,195,247,0.1)',
          fill: true, tension: 0.3, yAxisID: 'y' },
        { label: 'Wave (m)', data: waves, borderColor: '#ff9800', backgroundColor: 'rgba(255,152,0,0.1)',
          fill: true, tension: 0.3, yAxisID: 'y1' }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { title: { display: true, text: data.route_name || routeId, color: '#e0e0e0' },
                 legend: { labels: { color: '#aaa', font: { size: 10 } } } },
      scales: {
        x: { ticks: { color: '#888', maxTicksLimit: 12, font: { size: 9 } }, grid: { color: '#1a2a4a' } },
        y: { position: 'left', title: { display: true, text: 'Wind (kn)', color: '#4fc3f7' },
             ticks: { color: '#4fc3f7' }, grid: { color: '#1a2a4a' } },
        y1: { position: 'right', title: { display: true, text: 'Wave (m)', color: '#ff9800' },
              ticks: { color: '#ff9800' }, grid: { drawOnChartArea: false } }
      }
    }
  });
}

async function refresh() {
  const vessel = document.getElementById('vessel').value;
  const scenario = document.getElementById('scenario').value;
  document.getElementById('timer').textContent = 'loading...';

  const [resp, routeResp] = await Promise.all([
    fetch('/api/check?vessel=' + vessel + '&scenario=' + scenario),
    fetch('/api/routes')
  ]);
  const data = await resp.json();
  const routeDefs = await routeResp.json();
  lastData = data;

  routeLines.forEach(l => map.removeLayer(l));
  portMarkers.forEach(m => map.removeLayer(m));
  wpMarkers.forEach(m => map.removeLayer(m));
  routeLines = []; portMarkers = []; wpMarkers = [];

  const s = data.summary;
  document.getElementById('summary').innerHTML =
    '<span><b>' + data.routes.length + '</b> routes | ' + data.scenario + ' | ' + data.vessel_type + '</span>' +
    '<span class="stat"><span class="dot dot-red"></span>' + s.ban_likely + '</span>' +
    '<span class="stat"><span class="dot dot-yellow"></span>' + s.at_risk + '</span>' +
    '<span class="stat"><span class="dot dot-green"></span>' + s.clear + '</span>' +
    '<span style="margin-left:auto;font-size:11px;color:#607d8b">' + new Date().toLocaleTimeString() + '</span>';

  const tbody = document.getElementById('tbody');
  tbody.innerHTML = '';
  const drawnPorts = {};

  data.routes.forEach(r => {
    const rd = routeDefs.find(d => d.id === r.route_id);
    if (!rd || !rd.origin.lat) return;

    const color = colors[r.overall_status] || '#888';
    const line = L.polyline(
      [[rd.origin.lat, rd.origin.lon], [rd.destination.lat, rd.destination.lon]],
      { color, weight: 3, opacity: 0.8 }
    ).addTo(map);
    line.bindPopup(
      '<b>' + r.route_name + '</b><br>Status: <b style="color:' + color + '">' + r.overall_status + '</b>' +
      '<br>Wind: ' + r.max_wind.toFixed(0) + ' kn (Bf ' + r.max_beaufort + ')' +
      '<br>Wave: ' + r.max_wave.toFixed(1) + ' m' +
      '<br><a href="#" onclick="showRouteDetail(\\''+r.route_id+'\\');return false">View forecast chart</a>'
    );
    line.on('click', () => showRouteDetail(r.route_id));
    routeLines.push(line);

    [[rd.origin.code, rd.origin.lat, rd.origin.lon, rd.origin.origin],
     [rd.destination.code, rd.destination.lat, rd.destination.lon, rd.destination.name]
    ].forEach(([code, lat, lon, name]) => {
      if (!drawnPorts[code] && lat) {
        const m = L.circleMarker([lat, lon], {
          radius: 5, fillColor: '#4fc3f7', fillOpacity: 0.9, color: '#fff', weight: 1
        }).addTo(map).bindTooltip(name || code);
        portMarkers.push(m);
        drawnPorts[code] = true;
      }
    });

    const tr = document.createElement('tr');
    tr.onclick = () => showRouteDetail(r.route_id);
    tr.innerHTML =
      '<td>' + r.route_id + '</td>' +
      '<td>' + (r.route_name.split(' \\u2192 ')[0]||'') + '</td>' +
      '<td>' + (r.route_name.split(' \\u2192 ')[1]||'') + '</td>' +
      '<td class="status-' + r.overall_status + '">' + r.overall_status + '</td>' +
      '<td>' + r.max_wind.toFixed(0) + ' kn</td>' +
      '<td>' + r.max_wave.toFixed(1) + ' m</td>' +
      '<td>' + r.max_beaufort + '</td>';
    tbody.appendChild(tr);
  });

  document.getElementById('timer').textContent = 'updated ' + new Date().toLocaleTimeString();
}

refresh();
</script>
</body>
</html>"""

"""
DICOM viewer utilities: embedded canvas viewer (no external deps) +
lightweight HTTP server for raw DICOM serving.
"""
import http.server
import socketserver
import threading
import os
import json
import base64
import logging

logger = logging.getLogger(__name__)

_lock   = threading.Lock()
_server = None
_served_dir = None

VIEWER_PORT = 8502


class _CORSHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass  # silence request log noise


def start_server(directory: str, port: int = VIEWER_PORT) -> int:
    """
    Start (or restart) the DICOM HTTP server serving `directory`.
    Returns the port number.  Safe to call multiple times.
    """
    global _server, _served_dir

    with _lock:
        if _server is not None and _served_dir == directory:
            return port

        if _server is not None:
            try:
                _server.shutdown()
            except Exception:
                pass

        handler = lambda *a, **kw: _CORSHandler(*a, directory=directory, **kw)

        for attempt_port in [port, port + 1, port + 2]:
            try:
                srv = socketserver.TCPServer(("127.0.0.1", attempt_port), handler)
                srv.allow_reuse_address = True
                t = threading.Thread(target=srv.serve_forever, daemon=True)
                t.start()
                _server = srv
                _served_dir = directory
                logger.info(f"DICOM server on port {attempt_port} → {directory}")
                return attempt_port
            except OSError:
                continue

    raise RuntimeError("Could not bind DICOM server on any port")


def save_uploads_to_dir(files_data: list, directory: str) -> list:
    """
    Write (bytes, filename) tuples to `directory`.
    Returns list of filenames written.
    """
    os.makedirs(directory, exist_ok=True)
    written = []
    for b, name in files_data:
        safe = os.path.basename(name)
        path = os.path.join(directory, safe)
        with open(path, "wb") as fh:
            fh.write(b)
        written.append(safe)
    return written


# ── DWV viewer HTML template ───────────────────────────────────────────────
_DWV_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  *{{margin:0;padding:0;box-sizing:border-box}}
  html,body{{width:100%;height:100%;background:#0d0d0d;overflow:hidden;
    font-family:'Courier New',monospace;color:#ccc}}

  /* ── toolbar ── */
  #tb{{
    height:36px;background:#141414;border-bottom:1px solid #222;
    display:flex;align-items:center;gap:5px;padding:0 10px;
  }}
  #tb .logo{{color:#0EA5E9;font-size:11px;font-weight:bold;
    letter-spacing:2px;margin-right:6px}}
  .tbtn{{
    background:#1e1e1e;color:#999;border:1px solid #333;
    border-radius:3px;padding:2px 9px;cursor:pointer;font-size:11px;
    font-family:inherit;transition:all .15s;
  }}
  .tbtn:hover{{background:#282828;color:#fff}}
  .tbtn.on{{background:#0369a1;border-color:#0EA5E9;color:#fff}}
  #hint{{font-size:10px;color:#444;margin-left:auto}}
  #slice-pos{{font-size:11px;color:#0EA5E9;min-width:70px;text-align:right}}

  /* ── viewer area ── */
  #dv{{
    width:100%;
    height:calc(100vh - 36px);
    position:relative;
  }}

  /* loading overlay */
  #ov{{
    position:absolute;inset:0;background:#0d0d0d;
    display:flex;flex-direction:column;align-items:center;
    justify-content:center;gap:10px;z-index:9;
  }}
  .spin{{
    width:28px;height:28px;border:2px solid #222;
    border-top-color:#0EA5E9;border-radius:50%;
    animation:sp .7s linear infinite;
  }}
  @keyframes sp{{to{{transform:rotate(360deg)}}}}
  #ov-msg{{font-size:12px;color:#0EA5E9;letter-spacing:1px}}
</style>
</head>
<body>

<div id="tb">
  <span class="logo">SPINE AI</span>
  <button class="tbtn on" id="b0" onclick="setT('WindowLevel',0)">W/L</button>
  <button class="tbtn"    id="b1" onclick="setT('ZoomAndPan',1)">Zoom/Pan</button>
  <button class="tbtn"    id="b2" onclick="setT('Scroll',2)">Scroll</button>
  <button class="tbtn"           onclick="rst()">Reset</button>
  <span id="hint">wheel=slices &nbsp;|&nbsp; drag=W/L &nbsp;|&nbsp; shift+drag=zoom</span>
  <span id="slice-pos"></span>
</div>

<div id="dv"></div>

<div id="ov">
  <div class="spin"></div>
  <div id="ov-msg">Loading DICOM series...</div>
</div>

<script src="https://cdn.jsdelivr.net/npm/dwv@0.33.0/dist/dwv.min.js"></script>
<script>
var app = new dwv.App();
app.init({{
  dataViewConfigs: {{'*': [{{divId:'dv'}}]}},
  tools: {{Scroll:{{}}, ZoomAndPan:{{}}, WindowLevel:{{}}}}
}});

var BT = ['b0','b1','b2'];
function setT(name, idx) {{
  try {{ app.setTool(name); }} catch(e) {{}}
  BT.forEach(function(id,i){{
    document.getElementById(id).classList.toggle('on', i===idx);
  }});
}}

function rst() {{
  try {{ app.resetLayout(); }} catch(e) {{}}
}}

/* hide overlay on load */
app.addEventListener('load', function() {{
  document.getElementById('ov').style.display = 'none';
  setT('WindowLevel', 0);
}});

/* progress */
app.addEventListener('loadprogress', function(e) {{
  var p = e.loaded || 0;
  document.getElementById('ov-msg').textContent = 'Loading... ' + p + '%';
}});

/* error */
app.addEventListener('error', function(e) {{
  document.getElementById('ov-msg').textContent = 'Load error — check DICOM files';
  document.getElementById('spin') && (document.querySelector('.spin').style.display='none');
}});

/* slice counter */
app.addEventListener('positionchange', function(e) {{
  try {{
    var v = e.value;
    if (v && v.length > 2) {{
      document.getElementById('slice-pos').textContent = 'Slice ' + (Math.round(v[2])+1);
    }}
  }} catch(ex) {{}}
}});

/* receive jump command from Streamlit parent */
window.addEventListener('message', function(e) {{
  if (!e.data || e.data.cmd !== 'jumpSlice') return;
  try {{
    var idx = parseInt(e.data.index, 10);
    var lg = app.getActiveLayerGroup();
    if (!lg) return;
    var vc = lg.getActiveViewLayer().getViewController();
    if (vc.setCurrentScrollIndex) {{
      vc.setCurrentScrollIndex(idx);
    }} else {{
      var pos = vc.getCurrentPosition();
      vc.setCurrentPosition({{i: pos.i || 0, j: pos.j || 0, k: idx}});
    }}
  }} catch(ex) {{ console.warn('jump error', ex); }}
}});

var urls = {urls_json};
app.loadURLs(urls);
</script>
</body>
</html>"""


def build_viewer_html(filenames: list, port: int = VIEWER_PORT) -> str:
    """Return the DWV viewer HTML with DICOM URLs injected."""
    urls = [f"http://localhost:{port}/{name}" for name in sorted(filenames)]
    return _DWV_TEMPLATE.format(urls_json=json.dumps(urls))


def prepare_viewer(files_data: list,
                   viewer_dir: str = "data/viewer_temp",
                   port: int = VIEWER_PORT) -> tuple:
    """
    Full setup: save files, start server, generate HTML.
    Returns (actual_port, viewer_html).
    """
    filenames = save_uploads_to_dir(files_data, viewer_dir)
    actual_port = start_server(os.path.abspath(viewer_dir), port)
    html = build_viewer_html(filenames, actual_port)

    # Write HTML to disk so components.iframe can serve it from same origin
    html_path = os.path.join(viewer_dir, "viewer.html")
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(html)

    return actual_port, f"http://localhost:{actual_port}/viewer.html"


# ── Self-contained Canvas viewer (no CDN, no HTTP, embeds PNG frames as b64) ──

_CANVAS_VIEWER_HTML = """\
<!DOCTYPE html><html><head><meta charset="utf-8"><style>
*{margin:0;padding:0;box-sizing:border-box}
html,body{width:100%;height:100%;background:#0d0d0d;overflow:hidden;
  font-family:'Courier New',monospace;color:#ccc;user-select:none}
#tb{height:36px;background:#141414;border-bottom:1px solid #222;
  display:flex;align-items:center;gap:5px;padding:0 10px}
.logo{color:#0EA5E9;font-size:11px;font-weight:700;letter-spacing:2px;margin-right:6px}
.tbtn{background:#1e1e1e;color:#999;border:1px solid #333;border-radius:3px;
  padding:2px 9px;cursor:pointer;font-size:11px;font-family:inherit;transition:all .15s}
.tbtn:hover{background:#282828;color:#fff}
.tbtn.on{background:#0369a1;border-color:#0EA5E9;color:#fff}
#hint{font-size:10px;color:#444;margin-left:auto}
#sinfo{font-size:11px;color:#0EA5E9;min-width:90px;text-align:right}
#dv{width:100%;height:calc(100vh - 36px);display:flex;
  align-items:center;justify-content:center}
canvas{display:block;cursor:crosshair}
</style></head><body>
<div id="tb">
  <span class="logo">SPINE AI</span>
  <button class="tbtn on" id="bwl" onclick="setMode('wl','bwl')">W/L</button>
  <button class="tbtn"    id="bzm" onclick="setMode('zoom','bzm')">Zoom</button>
  <button class="tbtn"    id="bpn" onclick="setMode('pan','bpn')">Pan</button>
  <button class="tbtn"           onclick="reset()">Reset</button>
  <span id="hint">wheel=slices&nbsp;|&nbsp;drag=W/L&nbsp;|&nbsp;shift+drag=zoom</span>
  <span id="sinfo">Slice 1 / 1</span>
</div>
<div id="dv"><canvas id="cv"></canvas></div>
<script>
var frames = FRAMES_PLACEHOLDER;
var N = frames.length, idx = 0;
var zoom = 1.0, pan = {x:0,y:0}, wl = {wc:0.5,ww:1.0};
var mode = 'wl';
var drag = {on:false,btn:0,sx:0,sy:0};

var cv  = document.getElementById('cv');
var ctx = cv.getContext('2d');
var dv  = document.getElementById('dv');

/* Preload images */
var imgs = frames.map(function(src) {
  var img = new Image();
  img.onload = function() { render(); };
  img.src = src;
  return img;
});

function resize() {
  cv.width  = dv.clientWidth  || window.innerWidth;
  cv.height = dv.clientHeight || (window.innerHeight - 36);
  render();
}
window.addEventListener('resize', resize);
setTimeout(resize, 80);

function render() {
  var img = imgs[idx];
  if (!img || !img.complete || !img.naturalWidth) return;
  var W = cv.width, H = cv.height;
  var sc = Math.min(W / img.naturalWidth, H / img.naturalHeight) * zoom;
  var iw = img.naturalWidth * sc, ih = img.naturalHeight * sc;
  var ox = (W - iw) / 2 + pan.x, oy = (H - ih) / 2 + pan.y;

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#0d0d0d';
  ctx.fillRect(0, 0, W, H);

  var br = Math.max(0, (wl.wc * 2) * 100).toFixed(0) + '%';
  var ct = (1.0 / Math.max(0.05, wl.ww) * 100).toFixed(0) + '%';
  ctx.filter = 'brightness(' + br + ') contrast(' + ct + ')';
  ctx.drawImage(img, ox, oy, iw, ih);
  ctx.filter = 'none';

  document.getElementById('sinfo').textContent = 'Slice ' + (idx+1) + ' / ' + N;
}

/* Mouse wheel — scroll slices */
cv.addEventListener('wheel', function(e) {
  e.preventDefault();
  idx = Math.max(0, Math.min(N-1, idx + (e.deltaY > 0 ? 1 : -1)));
  render();
}, {passive:false});

/* Mouse drag */
cv.addEventListener('mousedown', function(e) {
  drag.on=true; drag.btn=e.button;
  drag.sx=e.clientX; drag.sy=e.clientY;
  if (e.button===1) e.preventDefault();
});
window.addEventListener('mousemove', function(e) {
  if (!drag.on) return;
  var dx=e.clientX-drag.sx, dy=e.clientY-drag.sy;
  drag.sx=e.clientX; drag.sy=e.clientY;
  var m = (e.shiftKey) ? 'zoom' : (e.altKey||drag.btn===1) ? 'pan' : mode;
  if (drag.btn===2) m='zoom';
  if (m==='wl' && drag.btn===0) {
    wl.ww = Math.max(0.02, wl.ww + dx*0.006);
    wl.wc = Math.max(0.01, Math.min(1.99, wl.wc - dy*0.006));
  } else if (m==='zoom') {
    zoom = Math.max(0.1, zoom - dy*0.012);
  } else if (m==='pan') {
    pan.x+=dx; pan.y+=dy;
  }
  render();
});
window.addEventListener('mouseup', function() { drag.on=false; });
cv.addEventListener('contextmenu', function(e){ e.preventDefault(); });

/* Keyboard */
window.addEventListener('keydown', function(e) {
  if (e.key==='ArrowDown'||e.key==='ArrowRight') { idx=Math.min(N-1,idx+1); render(); }
  else if (e.key==='ArrowUp'||e.key==='ArrowLeft') { idx=Math.max(0,idx-1); render(); }
});

/* postMessage from Streamlit (level button clicks) */
window.addEventListener('message', function(e) {
  if (e.data && e.data.cmd==='jumpSlice') {
    idx = Math.max(0, Math.min(N-1, parseInt(e.data.index,10)||0));
    render();
  }
});

function setMode(m, bid) {
  mode=m;
  ['bwl','bzm','bpn'].forEach(function(id){
    document.getElementById(id).classList.toggle('on', id===bid);
  });
}
function reset() { zoom=1; pan={x:0,y:0}; wl={wc:0.5,ww:1.0}; render(); }
</script></body></html>
"""


def build_embedded_viewer(png_frames: list) -> str:
    """
    Return a self-contained HTML DICOM viewer with PNG slices embedded as base64.
    png_frames: list of bytes objects (PNG-encoded axial slices).
    """
    encoded = []
    for b in png_frames:
        if isinstance(b, (bytes, bytearray)):
            b64 = base64.b64encode(b).decode('ascii')
            encoded.append(f"data:image/png;base64,{b64}")
        else:
            encoded.append("")  # skip invalid frames

    frames_json = json.dumps([f for f in encoded if f])
    return _CANVAS_VIEWER_HTML.replace("FRAMES_PLACEHOLDER", frames_json)

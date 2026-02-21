"""
Local Flask web server for FaceGallery.

Endpoints:
  GET  /                → person list (auth required)
  GET  /login           → PIN login page
  POST /login           → authenticate
  GET  /logout          → clear session
  GET  /persons         → browse by person (AJAX: JSON)
  GET  /photos          → photo browsing (optional ?person_id=)
  GET  /photo/<id>      → serves the actual image
  GET  /thumb/<photo_id>  → thumbnail
  GET  /face_thumb/<face_id> → face thumbnail
  POST /export/zip      → download ZIP of selected photos
  GET  /api/persons     → JSON person list
  GET  /api/photos      → JSON photo list (optional ?person_id=)
"""

import io
import logging
import os
import mimetypes
import json
import threading
from pathlib import Path
from functools import wraps
from typing import Optional, List

from ..core.scanner import PhotoScanner

logger = logging.getLogger(__name__)

_flask_app = None
_server_thread: Optional[threading.Thread] = None
_server_running = False

# Thread-safe scan progress store
import time as _time
_scan_progress = {
    "running": False,
    "current": 0,
    "total": 0,
    "message": "",
    "faces": 0,
    "started_at": None,
    "finished_at": None,
    "summary": None,
}
_scan_progress_lock = threading.Lock()


def create_flask_app(db, app_core):
    """Build and return the Flask application."""
    try:
        from flask import (Flask, request, session, redirect, url_for,
                           render_template_string, send_file, jsonify,
                           Response, abort)
    except ImportError:
        logger.error("Flask is not installed. Web server disabled.")
        return None

    from ..utils.helpers import verify_pin, hash_pin, bytes_to_embedding
    from ..face_engine.clusterer import cluster_embeddings
    import numpy as np

    # Project root is two levels up from this file's directory (src/web -> src -> root)
    project_root = Path(__file__).resolve().parent.parent.parent
    static_dir = project_root / "resources"
    
    UPLOADS_ROOT_DIR = project_root / "uploads"
    UPLOADS_ROOT_DIR.mkdir(parents=True, exist_ok=True) # Ensure it exists
    
    # Correctly configure the static folder
    flask_app = Flask(
        __name__,
        static_folder=str(static_dir),
        static_url_path="/static"
    )
    flask_app.secret_key = os.urandom(32)
    flask_app.config["SESSION_COOKIE_HTTPONLY"] = True

    # ── auth decorator ────────────────────────────────────────────────── #
    def login_required(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if not session.get("username"):
                return redirect(url_for("login"))
            return f(*args, **kwargs)
        return decorated

    def admin_required(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if not session.get("username"):
                return redirect(url_for("login"))
            if session.get("role") != "admin":
                abort(403)
            return f(*args, **kwargs)
        return decorated

    def uploader_allowed(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if not session.get("username"):
                return redirect(url_for("login"))
            if session.get("role") == "admin" or session.get("can_upload"):
                return f(*args, **kwargs)
            abort(403)
        return decorated

    def get_user_allowed_ids():
        username = session.get("username")
        if not username: return None
        user = db.get_user(username)
        if not user: return None
        if user["role"] == "admin": return None
        allowed = db.get_user_permissions(user["user_id"])
        return allowed if allowed else None

    def is_photo_allowed(photo_id, allowed_ids):
        if allowed_ids is None: return True
        placeholders = ','.join(['?'] * len(allowed_ids))
        row = db.fetchone(f"""
            SELECT 1 FROM person_face_mappings pfm
            JOIN faces f ON f.face_id = pfm.face_id
            WHERE f.photo_id = ? AND pfm.person_id IN ({placeholders})
        """, (photo_id, *allowed_ids))
        if row: return True
        
        # Allow uploader to see their own photos even if unnamed
        if session.get("can_upload"):
            photo = db.get_photo(photo_id)
            if photo:
                uid = session.get("user_id")
                user_folders = db.get_scan_folders(created_by=uid)
                if any(photo["path"].startswith(f["path"]) for f in user_folders):
                    return True
        return False

    def get_nav():
        username = session.get("username", "")
        role = session.get("role")
        can_upload = session.get("can_upload")
        
        links = ""
        if role == "admin":
            links = '<a href="/admin">⚙️ Admin</a>'
            links += '<a href="/admin/naming">👤 Name Unknown</a>'
        elif can_upload:
            links = '<a href="/admin/folders">📤 My Folders</a>'
            links += '<a href="/admin/naming">👤 Name Unknown</a>'
            
        return NAV_HTML.replace("{username}", username).replace("{admin_link}", links)

    def is_face_allowed(face_id, allowed_ids):
        if allowed_ids is None: return True
        placeholders = ','.join(['?'] * len(allowed_ids))
        row = db.fetchone(f"""
            SELECT 1 FROM person_face_mappings pfm
            WHERE pfm.face_id = ? AND pfm.person_id IN ({placeholders})
        """, (face_id, *allowed_ids))
        if row: return True
        
        if session.get("can_upload"):
            face = db.get_face(face_id)
            if face:
                return is_photo_allowed(face["photo_id"], allowed_ids)
        return False

    @flask_app.after_request
    def add_header(r):
        """Add headers to prevent caching of sensitive data."""
        # Ensure authenticated pages aren't cached so Back button doesn't work after logout
        if "Cache-Control" not in r.headers:
            r.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            r.headers["Pragma"] = "no-cache"
            r.headers["Expires"] = "-1"
        return r

    # ── HTML helpers ──────────────────────────────────────────────────── #

    BASE_HTML = """<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FaceGallery</title>
<link rel="icon" type="image/png" href="/static/icons/favicon.png">
<script>
  (function() {
    const theme = localStorage.getItem('theme') || 'dark';
    document.documentElement.setAttribute('data-theme', theme);
  })();
</script>
<style>
  :root{
    --bg:#0f1117; --card:#1a1d2e; --accent:#6c63ff; --accent2:#ff6584;
    --text:#e2e8f0; --sub:#94a3b8; --border:#2d3748; --radius:12px;
    --input-bg:#252840; --alert-err:#3d1515; --alert-err-text:#fc8181;
    --alert-succ:#1a3d1a; --alert-succ-text:#68d391;
  }
  [data-theme="light"]{
    --bg:#f1f5f9; --card:#ffffff; --accent:#4f46e5; --accent2:#ec4899;
    --text:#0f172a; --sub:#64748b; --border:#cbd5e1;
    --input-bg:#ffffff; --alert-err:#fee2e2; --alert-err-text:#b91c1c;
    --alert-succ:#dcfce7; --alert-succ-text:#15803d;
  }
  *{box-sizing:border-box;margin:0;padding:0;}
  body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;min-height:100vh;}
  a{color:var(--accent);text-decoration:none;}
  nav{display:flex;align-items:center;gap:16px;padding:14px 24px;
      background:var(--card);border-bottom:1px solid var(--border);position:sticky;top:0;z-index:100;}
  nav .logo{font-size:1.3rem;font-weight:700;background:linear-gradient(90deg,var(--accent),var(--accent2));
           -webkit-background-clip:text;-webkit-text-fill-color:transparent;}
  nav .spacer{flex:1;}
  nav a{color:var(--sub);font-size:.9rem;padding:6px 12px;border-radius:6px;transition:.2s;}
  nav a:hover{background:var(--border);color:var(--text);}
  .container{max-width:1400px;margin:0 auto;padding:24px;}
  .card{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);padding:20px;margin-bottom:16px;}
  .btn{display:inline-block;padding:9px 20px;border-radius:8px;border:none;cursor:pointer;
       font-size:.9rem;font-weight:600;transition:.2s;}
  .btn-primary{background:var(--accent);color:#fff;}
  .btn-primary:hover{opacity:.85;}
  .btn-danger{background:#e53e3e;color:#fff;}
  .btn-sm{padding:5px 12px;font-size:.8rem;}
  .grid{display:grid;gap:16px;}
  .grid-3{grid-template-columns:repeat(auto-fill,minmax(200px,1fr));}
  .grid-4{grid-template-columns:repeat(auto-fill,minmax(160px,1fr));}
  .person-card{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);
               padding:16px;text-align:center;transition:.2s;cursor:pointer;}
  .person-card:hover{border-color:var(--accent);transform:translateY(-2px);}
  .person-avatar{width:80px;height:80px;border-radius:50%;object-fit:cover;
                 background:var(--border);margin:0 auto 12px;}
  .person-avatar-placeholder{width:80px;height:80px;border-radius:50%;
    background:linear-gradient(135deg,var(--accent),var(--accent2));
    display:flex;align-items:center;justify-content:center;
    font-size:2rem;margin:0 auto 12px;}
  .photo-card{border-radius:10px;overflow:hidden;position:relative;aspect-ratio:1;
              background:var(--border);cursor:pointer;}
  .photo-card img{width:100%;height:100%;object-fit:cover;transition:.3s;}
  .photo-card:hover img{transform:scale(1.05);}
  .photo-card .check{position:absolute;top:8px;right:8px;width:22px;height:22px;
    border-radius:50%;border:2px solid #fff;background:rgba(0,0,0,.4);
    cursor:pointer;display:flex;align-items:center;justify-content:center;}
  .photo-card.selected{outline:3px solid var(--accent);}
  .photo-card .tags{position:absolute;bottom:0;left:0;right:0;padding:5px;
                    background:linear-gradient(transparent,rgba(0,0,0,.8));
                    display:flex;flex-wrap:wrap;gap:4px;opacity:0;transition:.2s;}
  .photo-card:hover .tags{opacity:1;}
  .photo-tag{font-size:.7rem;color:#fff;background:rgba(108,99,255,.85);
             padding:1px 6px;border-radius:4px;backdrop-filter:blur(2px);white-space:nowrap;}
  .photo-tag.sm{font-size:.65rem;background:var(--border);color:var(--sub);}
  .tag{display:inline-block;padding:2px 8px;border-radius:20px;font-size:.75rem;
       background:var(--accent);color:#fff;margin:2px;}
  .form-group{margin-bottom:16px;}
  label{display:block;margin-bottom:6px;font-size:.9rem;color:var(--sub);}
  input,select,textarea{width:100%;padding:10px 14px;border-radius:8px;
    border:1px solid var(--border);background:var(--input-bg);color:var(--text);font-size:.95rem;}
  input:focus,select:focus{outline:none;border-color:var(--accent);}
  .alert{padding:12px 16px;border-radius:8px;margin-bottom:16px;}
  .alert-error{background:var(--alert-err);border:1px solid #e53e3e;color:var(--alert-err-text);}
  .alert-success{background:var(--alert-succ);border:1px solid #38a169;color:var(--alert-succ-text);}
  .toolbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:20px;}
  .lightbox{position:fixed;inset:0;background:rgba(0,0,0,.9);display:none;
            align-items:center;justify-content:center;z-index:999;}
  .lightbox.active{display:flex;}
  .lightbox img{max-width:90vw;max-height:90vh;border-radius:8px;}
  .lightbox-close{position:fixed;top:16px;right:24px;font-size:2rem;color:#fff;cursor:pointer;}
  .lightbox-faces-container {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none; /* Allows clicks to pass through to image if no face box is there */
  }
  .lightbox-face-box {
    position: absolute;
    border: 2px solid #6c63ff; /* accent color */
    background: rgba(108, 99, 255, 0.2); /* semi-transparent fill */
    cursor: pointer;
    pointer-events: all; /* Make face boxes clickable */
    transition: all 0.1s ease-in-out;
  }
  .lightbox-face-box:hover {
    background: rgba(108, 99, 255, 0.4);
    border-width: 3px;
  }
  .badge{display:inline-block;padding:1px 7px;border-radius:10px;font-size:.75rem;
         background:var(--border);color:var(--sub);}
  #export-bar{display:none;position:fixed;bottom:0;left:0;right:0;
    background:var(--card);border-top:1px solid var(--border);
    padding:14px 24px;display:flex;gap:12px;align-items:center;z-index:200;}
  .theme-toggle{background:var(--border);border:none;color:var(--text);width:36px;height:36px;
                border-radius:50%;cursor:pointer;display:flex;align-items:center;justify-content:center;
                font-size:1.1rem;transition:.2s;}
  .theme-toggle:hover{background:var(--sub);color:#fff;}
</style>
<script>
  function toggleTheme(){
    const html=document.documentElement;
    const current=html.getAttribute('data-theme');
    const next=current==='dark'?'light':'dark';
    html.setAttribute('data-theme',next);
    localStorage.setItem('theme',next);
    document.getElementById('theme-icon').innerText=next==='dark'?'🌙':'☀️';
  }
  function updateFilters(personId, onlyGroups, onlySolos) {
    const url = new URL(window.location.href);
    if (personId !== undefined) {
      if (personId) url.searchParams.set('person_id', personId);
      else url.searchParams.delete('person_id');
    }
    
    // Mutual exclusivity logic for JS UI calls
    if (onlyGroups === '1') onlySolos = '0';
    if (onlySolos === '1') onlyGroups = '0';

    if (onlyGroups !== undefined) {
      if (onlyGroups === '1') url.searchParams.set('only_groups', '1');
      else url.searchParams.delete('only_groups');
    }
    if (onlySolos !== undefined) {
      if (onlySolos === '1') url.searchParams.set('only_solos', '1');
      else url.searchParams.delete('only_solos');
    }
    window.location.href = url.pathname + url.search;
  }
  window.addEventListener('DOMContentLoaded',()=>{
    const theme=localStorage.getItem('theme')||'dark';
    document.getElementById('theme-icon').innerText=theme==='dark'?'🌙':'☀️';
  });
</script>
</head>
<body>
"""

    NAV_HTML = """
<nav>
  <a href="/" class="logo" style="display:flex;align-items:center;gap:10px;text-decoration:none;">
    <img src="/static/icons/favicon.png" alt="logo" style="width:28px;height:28px;">
    <span>FaceGallery</span>
  </a>
  <span class="spacer"></span>
  <a href="/">🏠 Home</a>
  <a href="/photos">🖼 All Photos</a>
  {admin_link}
  <span style="color:var(--sub);font-size:.85rem;margin-left:10px;">👤 {username}</span>
  <button class="theme-toggle" onclick="toggleTheme()" title="Toggle Theme"><span id="theme-icon">🌙</span></button>
  <a href="/logout">Sign out</a>
</nav>
"""

    # ──────────────────────────────────────────────────────────────────── #
    # Auth routes                                                          #
    # ──────────────────────────────────────────────────────────────────── #

    LOGIN_HTML = BASE_HTML + """
<div style="display:flex;align-items:center;justify-content:center;min-height:100vh;">
<div style="width:380px;">
  <div style="text-align:center;margin-bottom:32px;">
    <img src="/static/icons/favicon.png" alt="logo" style="width:80px;height:80px;margin-bottom:12px;">
    <h1 style="font-size:1.8rem;font-weight:700;">FaceGallery</h1>
    <p style="color:var(--sub);margin-top:4px;">Local Photo Manager</p>
  </div>
  <div class="card">
    {error}
    <form method="POST" action="/login">
      <div class="form-group">
        <label>Username</label>
        <input name="username" type="text" placeholder="Enter username" required autofocus>
      </div>
      <div class="form-group">
        <label>PIN</label>
        <input name="pin" type="password" placeholder="Enter PIN" required
               inputmode="numeric" pattern="[0-9]{4,8}" maxlength="8">
      </div>
      <button class="btn btn-primary" style="width:100%;padding:12px;" type="submit">Sign In</button>
    </form>
  </div>
</div>
</div></body></html>"""

    # ──────────────────────────────────────────────────────────────────── #
    # Assets                                                               #
    # ──────────────────────────────────────────────────────────────────── #

    @flask_app.route("/login", methods=["GET", "POST"])
    def login():
        error = ""
        if request.method == "POST":
            username = (request.form.get("username") or "").strip()
            pin = (request.form.get("pin") or "").strip()
            user = db.get_user(username)
            if user and verify_pin(pin, user["pin_hash"]):
                session["username"] = username
                session["role"] = user["role"]
                session["user_id"] = user["user_id"]
                session["can_upload"] = bool(user["can_upload"])
                db.update_last_login(username)
                return redirect(url_for("index"))
            error = '<div class="alert alert-error">Invalid username or PIN.</div>'
        return LOGIN_HTML.replace("{error}", error)

    @flask_app.route("/logout")
    def logout():
        session.clear()
        return redirect(url_for("login"))

    # ──────────────────────────────────────────────────────────────────── #
    # Main pages                                                           #
    # ──────────────────────────────────────────────────────────────────── #

    @flask_app.route("/")
    @login_required
    def index():
        allowed_ids = get_user_allowed_ids()
        persons = app_core.get_persons(allowed_ids)
        person_cards = ""
        for p in persons:
            pid = p["person_id"]
            name = p["name"] or "Unknown"
            photo_count = len(db.get_photos_for_person(pid))
            
            manage_faces_html = ""
            if session.get("can_upload") or session.get("role") == "admin":
                manage_faces_html = f'<div style="text-align:center; margin-top:8px;"><a href="/admin/person/{pid}/faces" class="btn btn-sm" style="background:var(--accent); color:white; padding:4px 8px; border-radius:4px; font-size:0.8rem; text-decoration:none;">👤 Manage Faces</a></div>'
            
            person_cards += f"""
<div class="person-card" style="display:flex; flex-direction:column; justify-content:space-between;">
  <a href="/photos?person_id={pid}" style="color:inherit;text-decoration:none; flex:1;">
    <div class="person-avatar-placeholder">👤</div>
    <div style="font-weight:600;font-size:.95rem; text-align:center;">{name}</div>
    <div class="badge" style="margin: 4px auto;">{photo_count} photos</div>
  </a>
  {manage_faces_html}
</div>"""

        nav = get_nav()
        page = f"""{BASE_HTML}{nav}
<div class="container">
  <div style="margin-bottom:24px;">
    <h1 style="font-size:1.6rem;font-weight:700;">People</h1>
    <p style="color:var(--sub);margin-top:4px;">Browse your photo library by person</p>
  </div>
  {('<div class="grid grid-3">' + person_cards + '</div>') if persons
    else '<div class="card" style="text-align:center;padding:48px;color:var(--sub);">No people indexed yet. Start by scanning photos from the desktop app.</div>'}
</div></body></html>"""
        return page

    @flask_app.route("/photos")
    @login_required
    def photos():
        allowed_ids = get_user_allowed_ids()
        person_ids = request.args.getlist("person_id", type=int)
        only_groups = request.args.get("only_groups") == "1"
        only_solos = request.args.get("only_solos") == "1"

        # Access check
        if person_ids and allowed_ids:
            for pid in person_ids:
                if pid not in allowed_ids: abort(403)

        if person_ids:
            photo_list = app_core.get_photos(person_ids, only_groups=only_groups, only_solos=only_solos)
        elif allowed_ids:
            photo_list = app_core.get_photos(allowed_ids, use_union=True, only_groups=only_groups, only_solos=only_solos)
        else:
            photo_list = app_core.get_photos(only_groups=only_groups, only_solos=only_solos)

        persons = app_core.get_persons(allowed_ids)
        
        # Build tags map for shown photos
        photo_ids = [ph["photo_id"] for ph in photo_list]
        people_map = db.get_photo_people_map(photo_ids)

        person_options = '<option value="">All people</option>'
        for p in persons:
            sel = "selected" if p["person_id"] in person_ids else ""
            person_options += f'<option value="{p["person_id"]}" {sel}>{p["name"]}</option>'

        photo_cards = ""
        for ph in photo_list:
            pid2 = ph["photo_id"]
            tags_html = ""
            if pid2 in people_map:
                for p_id, p_name in people_map[pid2]:
                    if allowed_ids is not None and p_id not in allowed_ids:
                        continue
                    tags_html += f'<span class="photo-tag">{p_name}</span>'
            
            card_tags = f'<div class="tags">{tags_html}</div>' if tags_html else ""
            
            photo_cards += f"""
<div class="photo-card" id="pc-{pid2}" onclick="photoClick({pid2}, event, this)" data-path="{ph['path']}">
  <img src="/thumb/{pid2}" alt="photo" loading="lazy">
  {card_tags}
</div>"""

        nav = get_nav()
        title = ""
        manage_faces_btn = ""
        if person_ids:
            if len(person_ids) == 1:
                person = db.get_person(person_ids[0])
                title = f" – {person['name']}" if person else ""
                if session.get("can_upload") or session.get("role") == "admin":
                    manage_faces_btn = f'<a href="/admin/person/{person_ids[0]}/faces" class="btn btn-sm btn-primary" style="margin-left:10px;">👤 Manage Faces</a>'
            else:
                title = f" – {len(person_ids)} people"

        groups_checked = "checked" if only_groups else ""
        solos_checked = "checked" if only_solos else ""

        page = f"""{BASE_HTML}{nav}
<div class="container">
  <div class="toolbar">
    <h2 style="font-size:1.4rem;font-weight:700;">Photos{title}</h2>
    {manage_faces_btn}
    <span class="badge">{len(photo_list)} photos</span>
    <div style="flex:1;"></div>
    <select onchange="updateFilters(this.value)" style="min-width:180px;">
      {person_options}
    </select>
    <label style="display:flex;align-items:center;gap:6px;cursor:pointer;color:var(--sub);font-size:.85rem;margin:0 10px;">
        <input type="checkbox" {groups_checked} onchange="updateFilters(undefined, this.checked?'1':'0', '0')"> 👥 Groups
    </label>
    <label style="display:flex;align-items:center;gap:6px;cursor:pointer;color:var(--sub);font-size:.85rem;margin:0 10px;">
        <input type="checkbox" {solos_checked} onchange="updateFilters(undefined, '0', this.checked?'1':'0')"> 👤 Solos
    </label>
    <button class="btn btn-primary btn-sm" onclick="exportZip()">⬇ Export ZIP</button>
  </div>
  {'<div class="grid grid-4">'+photo_cards+'</div>'
   if photo_list else '<div class="card" style="text-align:center;padding:48px;color:var(--sub);">No photos found.</div>'}
</div>
<div id="lightbox" class="lightbox" onclick="closeLightbox()">
  <span class="lightbox-close">✕</span>
  <div style="position:relative;display:flex;flex-direction:column;align-items:center;gap:16px;">
    <img id="lightbox-img" src="" alt="" style="box-shadow:0 20px 50px rgba(0,0,0,.5);" draggable="false">
    <div id="lightbox-faces-container" style="position:absolute;top:0;left:0;width:100%;height:100%;"></div>
    <div id="lightbox-draw-layer" style="position:absolute;top:0;left:0;width:100%;height:100%;display:none;cursor:crosshair;z-index:99;"></div>
    <div style="display:flex; gap:10px; align-items:center; z-index:100;">
        <div id="lightbox-info" style="background:rgba(0,0,0,.7);padding:8px 16px;border-radius:20px;backdrop-filter:blur(10px);color:#fff;font-weight:600;display:flex;gap:8px;"></div>
        <button id="btn-add-face" class="btn btn-sm btn-primary" onclick="event.stopPropagation(); startDrawFace()" style="display:none;">Add Face</button>
        <div id="draw-help" style="display:none; color:white; background:rgba(0,0,0,.7); padding:8px 16px; border-radius:20px;">Click and drag on the image to draw a face box.</div>
    </div>
  </div>
</div>
<script>
let currentPhotoId = null;
const selected=new Set();
function photoClick(id,ev,el){{
  if(ev.ctrlKey || ev.metaKey){{
    if(selected.has(id)){{selected.delete(id);el.classList.remove('selected');}}
    else{{selected.add(id);el.classList.add('selected');}}
  }} else {{
    openLightbox(id, el.querySelector('.tags')?.innerHTML || '');
  }}
}}
function openLightbox(id, tagsHtml){{
  currentPhotoId = id;
  const lb=document.getElementById('lightbox');
  const img=document.getElementById('lightbox-img');
  const info=document.getElementById('lightbox-info');
  const facesContainer = document.getElementById('lightbox-faces-container');

  // Set image source and initial info
  img.src='/photo/'+id;
  info.innerHTML = tagsHtml || '<span style="color:rgba(255,255,255,.5);font-size:.8rem;font-weight:400;">No one identified</span>';
  lb.classList.add('active');
  
  document.getElementById('btn-add-face').style.display = 'block';
  document.getElementById('draw-help').style.display = 'none';
  document.getElementById('lightbox-draw-layer').style.display = 'none';

  // Clear previous faces
  facesContainer.innerHTML = '';

  // Wait for image to load to get natural dimensions
  img.onload = async () => {{
    // Fetch faces for this photo
    const r = await fetch('/api/photo/'+id+'/faces');
    if(r.ok){{
      const faces = await r.json();
      faces.forEach(face => {{
        const faceBox = document.createElement('div');
        faceBox.className = 'lightbox-face-box';
        // Calculate position and size relative to the natural image dimensions
        faceBox.style.left = (face.bbox_x / img.naturalWidth * 100) + '%';
        faceBox.style.top = (face.bbox_y / img.naturalHeight * 100) + '%';
        faceBox.style.width = (face.bbox_w / img.naturalWidth * 100) + '%';
        faceBox.style.height = (face.bbox_h / img.naturalHeight * 100) + '%';
        faceBox.title = 'Edit Face';
        faceBox.onclick = (e) => {{
          e.stopPropagation(); // Prevent closing lightbox
          window.location.href = '/admin/face/' + face.face_id + '/edit';
        }};
        facesContainer.appendChild(faceBox);
      }});
    }}
  }};
}}

let drawing = false;
let startX=0, startY=0;
let drawBox = null;

function startDrawFace() {{
    document.getElementById('btn-add-face').style.display = 'none';
    document.getElementById('draw-help').style.display = 'block';
    const layer = document.getElementById('lightbox-draw-layer');
    layer.style.display = 'block';
    
    layer.onmousedown = (e) => {{
        e.stopPropagation();
        drawing = true;
        const rect = layer.getBoundingClientRect();
        startX = e.clientX - rect.left;
        startY = e.clientY - rect.top;
        if(drawBox) drawBox.remove();
        drawBox = document.createElement('div');
        drawBox.style.position = 'absolute';
        drawBox.style.border = '2px dashed #0f0';
        drawBox.style.background = 'rgba(0,255,0,0.2)';
        drawBox.style.left = startX + 'px';
        drawBox.style.top = startY + 'px';
        drawBox.style.width = '0px';
        drawBox.style.height = '0px';
        layer.appendChild(drawBox);
    }};
    
    layer.onmousemove = (e) => {{
        if(!drawing) return;
        const rect = layer.getBoundingClientRect();
        const curX = e.clientX - rect.left;
        const curY = e.clientY - rect.top;
        drawBox.style.left = Math.min(startX, curX) + 'px';
        drawBox.style.top = Math.min(startY, curY) + 'px';
        drawBox.style.width = Math.abs(curX - startX) + 'px';
        drawBox.style.height = Math.abs(curY - startY) + 'px';
    }};
    
    layer.onmouseup = async (e) => {{
        if(!drawing) return;
        drawing = false;
        layer.onmousedown = null; layer.onmousemove = null; layer.onmouseup = null;
        layer.style.display = 'none';
        document.getElementById('draw-help').style.display = 'none';
        
        let bw = parseInt(drawBox.style.width);
        let bh = parseInt(drawBox.style.height);
        let bl = parseInt(drawBox.style.left);
        let bt = parseInt(drawBox.style.top);
        
        if(bw < 10 || bh < 10) {{
            alert('Box too small.');
            if(drawBox) drawBox.remove();
            document.getElementById('btn-add-face').style.display = 'block';
            return;
        }}
        
        const img = document.getElementById('lightbox-img');
        const scaleX = img.naturalWidth / img.width;
        const scaleY = img.naturalHeight / img.height;
        
        const payload = {{
            bbox_x: Math.round(bl * scaleX),
            bbox_y: Math.round(bt * scaleY),
            bbox_w: Math.round(bw * scaleX),
            bbox_h: Math.round(bh * scaleY)
        }};
        
        document.getElementById('draw-help').innerText = 'Saving...';
        document.getElementById('draw-help').style.display = 'block';
        
        const r = await fetch('/api/photo/'+currentPhotoId+'/faces/manual', {{
            method: 'POST',
            headers: {{'Content-Type': 'application/json'}},
            body: JSON.stringify(payload)
        }});
        
        if(r.ok) {{
            const res = await r.json();
            window.location.href = '/admin/face/' + res.face_id + '/edit';
        }} else {{
            console.error(await r.text());
            alert('Failed to add face manually. You must have upload permissions.');
            if(drawBox) drawBox.remove();
            document.getElementById('draw-help').style.display = 'none';
            document.getElementById('btn-add-face').style.display = 'block';
        }}
    }};
}}

function closeLightbox(){{
    document.getElementById('lightbox').classList.remove('active');
    document.getElementById('lightbox-draw-layer').style.display = 'none';
    if(drawBox) drawBox.remove();
    drawing = false;
}}
async function exportZip(){{
  const ids=[...selected];
  if(!ids.length){{alert('Select photos first (click each photo to select).');return;}}
  const r=await fetch('/export/zip',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{photo_ids:ids}})}});
  if(r.ok){{const blob=await r.blob();const a=document.createElement('a');
    a.href=URL.createObjectURL(blob);a.download='facegallery_export.zip';a.click();}}
  else alert('Export failed.');
}}
</script>
</body></html>"""
        return page

    # ──────────────────────────────────────────────────────────────────── #
    # Admin User Management                                                #
    # ──────────────────────────────────────────────────────────────────── #

    @flask_app.route("/admin/users")
    @admin_required
    def admin_users():
        users = db.get_all_users()
        user_rows = ""
        for u in users:
            role_badge = f'<span class="badge" style="background:{"var(--accent2)" if u["role"]=="admin" else "var(--border)"};">{u["role"]}</span>'
            del_btn = '<button onclick="deleteUser(\'' + u["username"] + '\')" class="btn btn-sm btn-danger">Delete</button>' if u["username"] != session["username"] else ""
            user_rows += f"""
<tr>
  <td style="padding:12px; border-bottom:1px solid var(--border);">{u["username"]}</td>
  <td style="padding:12px; border-bottom:1px solid var(--border);">{role_badge}</td>
  <td style="padding:12px; border-bottom:1px solid var(--border); font-size:.8rem; color:var(--sub);">{u["last_login"] or "Never"}</td>
  <td style="padding:12px; border-bottom:1px solid var(--border); text-align:right;">
    <a href="/admin/users/edit/{u["username"]}" class="btn btn-sm btn-primary">Edit</a>
    <a href="/admin/users/password/{u["username"]}" class="btn btn-sm" style="background:var(--border);">Key</a>
    {del_btn}
  </td>
</tr>"""

        nav = get_nav()
        page = f"""{BASE_HTML}{nav}
<div class="container">
  <div class="toolbar">
    <h1 style="font-size:1.6rem;font-weight:700;">User Management</h1>
    <div style="flex:1;"></div>
    <a href="/admin/users/create" class="btn btn-primary">➕ New User</a>
  </div>
  <div class="card" style="padding:0; overflow:hidden;">
    <table style="width:100%; border-collapse:collapse;">
      <thead>
        <tr style="background:var(--border); text-align:left;">
          <th style="padding:12px; font-weight:600;">Username</th>
          <th style="padding:12px; font-weight:600;">Role</th>
          <th style="padding:12px; font-weight:600;">Last Login</th>
          <th style="padding:12px; font-weight:600; text-align:right;">Actions</th>
        </tr>
      </thead>
      <tbody>
        {user_rows}
      </tbody>
    </table>
  </div>
</div>
<script>
async function deleteUser(username) {{
    if(!confirm('Are you sure you want to delete user ' + username + '?')) return;
    const r = await fetch('/admin/users/delete/' + username, {{method:'POST'}});
    if(r.ok) location.reload();
    else alert('Failed to delete user.');
}}
</script>
</body></html>"""
        return page

    @flask_app.route("/admin/users/create", methods=["GET", "POST"])
    @admin_required
    def admin_user_create():
        error = ""
        if request.method == "POST":
            username = (request.form.get("username") or "").strip()
            pin = (request.form.get("pin") or "").strip()
            role = request.form.get("role", "viewer")
            if not username or not pin:
                error = '<div class="alert alert-error">Username and PIN are required.</div>'
            elif db.get_user(username):
                error = '<div class="alert alert-error">Username already exists.</div>'
            else:
                can_upload = 1 if request.form.get("can_upload") else 0
                user_id = db.create_user(username, hash_pin(pin), role, can_upload=can_upload)
                allowed_persons = request.form.getlist("allowed_persons")
                for pid in allowed_persons:
                    db.add_user_permission(user_id, int(pid))
                return redirect(url_for("admin_users"))

        persons = app_core.get_persons()
        person_opts = ""
        for p in persons:
            person_opts += f'<option value="{p["person_id"]}">{p["name"]}</option>'

        nav = get_nav()
        page = f"""{BASE_HTML}{nav}
<div class="container" style="max-width:600px;">
  <div style="margin-bottom:24px;">
    <h1 style="font-size:1.6rem;font-weight:700;">Create User</h1>
  </div>
  <div class="card">
    {error}
    <form method="POST">
      <div class="form-group">
        <label>Username</label>
        <input name="username" type="text" required autofocus>
      </div>
      <div class="form-group">
        <label>PIN (4-8 digits)</label>
        <input name="pin" type="password" required inputmode="numeric" pattern="[0-9]{{4,8}}" maxlength="8">
      </div>
      <div class="form-group">
        <label>Role</label>
        <select name="role">
          <option value="viewer">Viewer</option>
          <option value="admin">Administrator</option>
        </select>
      </div>
      <div class="form-group">
        <label style="display:flex; align-items:center; gap:8px; cursor:pointer;">
          <input type="checkbox" name="can_upload" style="width:18px; height:18px;">
          <span>Can Upload & Manage Folders</span>
        </label>
        <p style="color:var(--sub); font-size:.75rem; margin-top:4px;">Allows user to create folders and upload photos to them.</p>
      </div>
      <div class="form-group">
        <label>Allowed People (Optional - Leave empty for ALL access)</label>
        <select name="allowed_persons" multiple style="height:120px;">
          {person_opts}
        </select>
        <p style="color:var(--sub); font-size:.75rem; margin-top:4px;">Hold Ctrl (or Cmd) to select multiple.</p>
      </div>
      <div style="display:flex; gap:10px; margin-top:24px;">
        <button class="btn btn-primary" type="submit" style="flex:1;">Create User</button>
        <a href="/admin/users" class="btn" style="background:var(--border); text-align:center;">Cancel</a>
      </div>
    </form>
  </div>
</div></body></html>"""
        return page

    @flask_app.route("/admin/users/edit/<username>", methods=["GET", "POST"])
    @admin_required
    def admin_user_edit(username):
        user = db.get_user(username)
        if not user:
            abort(404)
        
        error = ""
        if request.method == "POST":
            role = request.form.get("role", "viewer")
            can_upload = 1 if request.form.get("can_upload") else 0
            db.update_user(username, role=role, can_upload=can_upload)
            
            # Update permissions
            db.remove_user_permissions(user["user_id"])
            allowed_persons = request.form.getlist("allowed_persons")
            for pid in allowed_persons:
                db.add_user_permission(user["user_id"], int(pid))

            if username == session["username"]:
                session["role"] = role
                session["can_upload"] = bool(can_upload)
            return redirect(url_for("admin_users"))

        current_allowed = db.get_user_permissions(user["user_id"])
        persons = app_core.get_persons()
        person_opts = ""
        for p in persons:
            sel = "selected" if p["person_id"] in current_allowed else ""
            person_opts += f'<option value="{p["person_id"]}" {sel}>{p["name"]}</option>'

        nav = get_nav()
        page = f"""{BASE_HTML}{nav}
<div class="container" style="max-width:600px;">
  <div style="margin-bottom:24px;">
    <h1 style="font-size:1.6rem;font-weight:700;">Edit User: {username}</h1>
  </div>
  <div class="card">
    {error}
    <form method="POST">
      <div class="form-group">
        <label>Role</label>
        <select name="role">
          <option value="viewer" {"selected" if user["role"]=="viewer" else ""}>Viewer</option>
          <option value="admin" {"selected" if user["role"]=="admin" else ""}>Administrator</option>
        </select>
      </div>
      <div class="form-group">
        <label style="display:flex; align-items:center; gap:8px; cursor:pointer;">
          <input type="checkbox" name="can_upload" style="width:18px; height:18px;" {"checked" if user["can_upload"] else ""}>
          <span>Can Upload & Manage Folders</span>
        </label>
        <p style="color:var(--sub); font-size:.75rem; margin-top:4px;">Allows user to create folders and upload photos to them.</p>
      </div>
      <div class="form-group">
        <label>Allowed People (Optional - Leave empty for ALL access)</label>
        <select name="allowed_persons" multiple style="height:120px;">
          {person_opts}
        </select>
        <p style="color:var(--sub); font-size:.75rem; margin-top:4px;">Hold Ctrl (or Cmd) to select multiple.</p>
      </div>
      <p style="color:var(--sub); font-size:.85rem; margin-bottom:16px;">
        Note: To change the password, use the "Key" button on the user list.
      </p>
      <div style="display:flex; gap:10px; margin-top:24px;">
        <button class="btn btn-primary" type="submit" style="flex:1;">Save Changes</button>
        <a href="/admin/users" class="btn" style="background:var(--border); text-align:center;">Cancel</a>
      </div>
    </form>
  </div>
</div></body></html>"""
        return page

    @flask_app.route("/admin/users/password/<username>", methods=["GET", "POST"])
    @admin_required
    def admin_user_password(username):
        user = db.get_user(username)
        if not user:
            abort(404)
        
        error = ""
        if request.method == "POST":
            pin = (request.form.get("pin") or "").strip()
            if not pin:
                error = '<div class="alert alert-error">PIN is required.</div>'
            else:
                db.update_user(username, pin_hash=hash_pin(pin))
                return redirect(url_for("admin_users"))

        nav = get_nav()
        page = f"""{BASE_HTML}{nav}
<div class="container" style="max-width:600px;">
  <div style="margin-bottom:24px;">
    <h1 style="font-size:1.6rem;font-weight:700;">Change PIN: {username}</h1>
  </div>
  <div class="card">
    {error}
    <form method="POST">
      <div class="form-group">
        <label>New PIN (4-8 digits)</label>
        <input name="pin" type="password" required autofocus inputmode="numeric" pattern="[0-9]{{4,8}}" maxlength="8">
      </div>
      <div style="display:flex; gap:10px; margin-top:24px;">
        <button class="btn btn-primary" type="submit" style="flex:1;">Update PIN</button>
        <a href="/admin/users" class="btn" style="background:var(--border); text-align:center;">Cancel</a>
      </div>
    </form>
  </div>
</div></body></html>"""
        return page

    @flask_app.route("/admin/users/delete/<username>", methods=["POST"])
    @admin_required
    def admin_user_delete(username):
        if username == session["username"]:
            return jsonify({"success": False, "error": "Cannot delete yourself"}), 400
        db.delete_user(username)
        return jsonify({"success": True})

    # ──────────────────────────────────────────────────────────────────── #
    # Media endpoints                                                      #
    # ──────────────────────────────────────────────────────────────────── #

    @flask_app.route("/thumb/<int:photo_id>")
    @login_required
    def thumbnail(photo_id):
        allowed_ids = get_user_allowed_ids()
        if not is_photo_allowed(photo_id, allowed_ids):
            abort(403)
            
        row = db.fetchone("SELECT thumbnail,path FROM photos WHERE photo_id=?", (photo_id,))
        if not row:
            abort(404)
        if row["thumbnail"]:
            return Response(row["thumbnail"], mimetype="image/jpeg")
        # Serve full image if no thumbnail
        if os.path.isfile(row["path"]):
            return send_file(row["path"])
        abort(404)

    @flask_app.route("/photo/<int:photo_id>")
    @login_required
    def serve_photo(photo_id):
        allowed_ids = get_user_allowed_ids()
        if not is_photo_allowed(photo_id, allowed_ids):
            abort(403)

        row = db.fetchone("SELECT path FROM photos WHERE photo_id=?", (photo_id,))
        if not row or not os.path.isfile(row["path"]):
            abort(404)
        mime, _ = mimetypes.guess_type(row["path"])
        return send_file(row["path"], mimetype=mime or "image/jpeg")

    @flask_app.route("/face_thumb/<int:face_id>")
    @login_required
    def face_thumbnail(face_id):
        allowed_ids = get_user_allowed_ids()
        if not is_face_allowed(face_id, allowed_ids):
            abort(403)

        row = db.get_face(face_id)
        if not row or not row["face_thumb"]:
            abort(404)
        return Response(row["face_thumb"], mimetype="image/jpeg")

    # ──────────────────────────────────────────────────────────────────── #
    # Export                                                               #
    # ──────────────────────────────────────────────────────────────────── #

    @flask_app.route("/export/zip", methods=["POST"])
    @login_required
    def export_zip():
        data = request.get_json(silent=True) or {}
        photo_ids = data.get("photo_ids", [])
        if not photo_ids:
            return jsonify({"error": "No photos selected"}), 400

        paths = []
        for pid in photo_ids:
            row = db.fetchone("SELECT path FROM photos WHERE photo_id=?", (pid,))
            if row and os.path.isfile(row["path"]):
                paths.append(row["path"])

        zip_path = app_core.export_photos_to_zip(paths)

        def generate():
            with open(zip_path, "rb") as fh:
                while chunk := fh.read(65536):
                    yield chunk
            os.unlink(zip_path)

        return Response(
            generate(),
            mimetype="application/zip",
            headers={"Content-Disposition": "attachment; filename=facegallery_export.zip"}
        )

    # ──────────────────────────────────────────────────────────────────── #
    # JSON API                                                             #
    # ──────────────────────────────────────────────────────────────────── #

    @flask_app.route("/api/persons")
    @login_required
    def api_persons():
        persons = app_core.get_persons()
        return jsonify([dict(p) for p in persons])

    @flask_app.route("/api/photos")
    @login_required
    def api_photos():
        person_id = request.args.get("person_id", type=int)
        photos = app_core.get_photos([person_id] if person_id else None)
        result = []
        for ph in photos:
            d = dict(ph)
            d.pop("thumbnail", None)  # don't send blob over JSON
            result.append(d)
        return jsonify(result)

    @flask_app.route("/api/photo/<int:photo_id>/faces")
    @login_required
    def api_photo_faces(photo_id):
        faces = db.get_faces_for_photo(photo_id)
        result = []
        for face in faces:
            result.append({
                "face_id": face["face_id"],
                "bbox_x": face["bbox_x"],
                "bbox_y": face["bbox_y"],
                "bbox_w": face["bbox_w"],
                "bbox_h": face["bbox_h"]
            })
        return jsonify(result)

    @flask_app.route("/api/photo/<int:photo_id>/faces/manual", methods=["POST"])
    @uploader_allowed
    def api_photo_add_manual_face(photo_id):
        data = request.json
        bbox_x = int(data.get("bbox_x", 0))
        bbox_y = int(data.get("bbox_y", 0))
        bbox_w = int(data.get("bbox_w", 0))
        bbox_h = int(data.get("bbox_h", 0))

        if bbox_w <= 0 or bbox_h <= 0:
            return jsonify({"error": "Invalid bounding box"}), 400

        photo = db.get_photo(photo_id)
        if not photo or not os.path.isfile(photo["path"]):
            return jsonify({"error": "Photo not found"}), 404
            
        try:
            from PIL import Image
            import io
            with Image.open(photo["path"]) as img:
                # Need to handle EXIF orientation manually to crop correctly
                from PIL import ImageOps
                img = ImageOps.exif_transpose(img)
                img = img.convert("RGB")
                box = (bbox_x, bbox_y, bbox_x + bbox_w, bbox_y + bbox_h)
                face_img = img.crop(box)
                face_img.thumbnail((150, 150))
                bio = io.BytesIO()
                face_img.save(bio, format="JPEG", quality=85)
                face_thumb_bytes = bio.getvalue()
        except Exception as e:
            logger.error("Failed to crop manual face: %s", e)
            return jsonify({"error": "Failed to crop photo"}), 500
            
        face_id = db.insert_face(
            photo_id=photo_id,
            bbox_x=bbox_x, bbox_y=bbox_y, bbox_w=bbox_w, bbox_h=bbox_h,
            embedding=None, confidence=1.0, face_thumb=face_thumb_bytes
        )
        return jsonify({"success": True, "face_id": face_id})

    # ──────────────────────────────────────────────────────────────────── #
    # Admin Portal & Folder Management                                     #
    # ──────────────────────────────────────────────────────────────────── #

    @flask_app.route("/admin")
    @uploader_allowed
    def admin_portal():
        is_admin = session.get("role") == "admin"
        if not is_admin:
            return redirect(url_for("admin_folders"))

        nav = get_nav()
        page = f"""{BASE_HTML}{nav}
<div class="container" style="max-width:800px;">
  <h1 style="font-size:1.6rem;font-weight:700;margin-bottom:24px;">Admin Dashboard</h1>
  <div class="grid grid-3">
    <a href="/admin/users" class="card" style="text-align:center; padding:32px; text-decoration:none;">
      <div style="font-size:2.5rem; margin-bottom:12px;">👥</div>
      <div style="font-size:1.1rem; font-weight:600; color:var(--text);">User Management</div>
      <div style="font-size:.85rem; color:var(--sub); margin-top:8px;">Manage users and permissions</div>
    </a>
    <a href="/admin/folders" class="card" style="text-align:center; padding:32px; text-decoration:none;">
      <div style="font-size:2.5rem; margin-bottom:12px;">📂</div>
      <div style="font-size:1.1rem; font-weight:600; color:var(--text);">Scan Folders</div>
      <div style="font-size:.85rem; color:var(--sub); margin-top:8px;">Configure directories to scan</div>
    </a>
  </div>
</div>
</body></html>"""
        return page

    @flask_app.route("/admin/folders")
    @uploader_allowed
    def admin_folders():
        is_admin = session.get("role") == "admin"
        uid = session.get("user_id")
        folders = db.get_scan_folders(created_by=None if is_admin else uid)
        folder_rows = ""
        for row in folders:
            full_path = Path(row["path"])
            display_path = full_path.name # e.g., "tester-myhome"
            
            # If created by a user, try to strip their username prefix for display
            if row["username"]:
                prefix = f"{row['username']}-"
                if display_path.startswith(prefix):
                    display_path = display_path[len(prefix):] # e.g., "myhome"
            
            encoded_full_path = str(full_path).replace("\\", "\\\\").replace("'", "\\'") # Still pass full path to JS
            
            folder_rows += f"""
<tr>
  <td style="padding:12px; border-bottom:1px solid var(--border); font-family:monospace; font-size:.85rem;">{display_path}</td>
  <td style="padding:12px; border-bottom:1px solid var(--border); text-align:right; white-space:nowrap;">
    <button onclick="triggerScan('{encoded_full_path}')" class="btn btn-sm btn-primary" style="background:var(--accent2); margin-right:4px;">🚀 Scan Now</button>
    <button onclick="openUploadDialog('{encoded_full_path}')" class="btn btn-sm btn-primary" style="background:var(--accent); margin-right:4px;">📤 Upload</button>
    <button onclick="removeFolder('{encoded_full_path}')" class="btn btn-sm btn-danger" style="margin-right:4px;">Remove</button>
    <button onclick="confirmDeleteFolder('{encoded_full_path}', '{display_path}')" class="btn btn-sm" style="background:#7f1d1d; color:#fca5a5; margin-right:4px;">🗑 Delete All</button>
  </td>
</tr>"""

        nav = get_nav()
        page = f"""{BASE_HTML}{nav}
<div class="container">
  <div class="toolbar">
    <h1 style="font-size:1.6rem;font-weight:700;">Scan Folders</h1>
    <div style="flex:1;"></div>
    <button onclick="triggerScan()" class="btn btn-primary" style="background:var(--accent2);">🚀 Scan All Now</button>
    <button onclick="document.getElementById('create-dialog').style.display='flex'" class="btn btn-primary" style="background:var(--accent);">📁 Create New Folder</button>
    <button onclick="document.getElementById('add-dialog').style.display='flex'" class="btn btn-primary">➕ Add Existing</button>
  </div>
  
  <!-- Scan Progress Panel -->
  <div id="scan-progress-panel" style="display:none; margin-bottom:16px;">
    <div class="card" style="padding:20px;">
      <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:12px;">
        <span style="font-weight:700; font-size:1rem;" id="scan-prog-title">🚀 Scanning...</span>
        <span id="scan-prog-counter" style="font-size:.85rem; color:var(--sub);"></span>
      </div>
      <div style="height:10px; background:var(--border); border-radius:5px; overflow:hidden; margin-bottom:10px;">
        <div id="scan-prog-bar" style="height:100%; width:0%; background:var(--accent2); transition:width .4s ease; border-radius:5px;"></div>
      </div>
      <div style="display:flex; justify-content:space-between; font-size:.82rem; color:var(--sub);">
        <span id="scan-prog-msg">Preparing...</span>
        <span id="scan-prog-faces">0 faces found</span>
      </div>
    </div>
  </div>

  <div id="scan-done-banner" style="display:none; margin-bottom:16px; padding:12px 16px; background:#dcfce7; color:#14532d; border-radius:8px; border:1px solid #16a34a; font-size:.9rem; font-weight:500;"></div>

  <div class="card" style="padding:0; overflow:hidden;">
    <table style="width:100%; border-collapse:collapse;">
      <thead>
        <tr style="background:var(--border); text-align:left;">
          <th style="padding:12px; font-weight:600;">Path</th>
          <th style="padding:12px; font-weight:600; text-align:right;">Actions</th>
        </tr>
      </thead>
      <tbody>
        {folder_rows if folder_rows else '<tr><td colspan="2" style="padding:24px; text-align:center; color:var(--sub);">No folders configured.</td></tr>'}
      </tbody>
    </table>
  </div>
  
  <div id="add-dialog" class="lightbox" style="display:none; align-items:center; justify-content:center;">
    <div class="card" style="width:500px; padding:24px;" onclick="event.stopPropagation()">
      <h2 style="margin-bottom:16px;">Add Existing Folder</h2>
      <p style="font-size:.85rem; color:var(--sub); margin-bottom:16px;">Enter the relative name of an existing folder within your user's upload directory.</p>
      <div class="form-group">
        <label>Folder Name</label>
        <input id="folder-name-add" type="text" placeholder="e.g. my-vacation-photos" autofocus>
      </div>
      <div style="display:flex; gap:10px; margin-top:24px;">
        <button onclick="addFolder()" class="btn btn-primary" style="flex:1;">Add</button>
        <button onclick="document.getElementById('add-dialog').style.display='none'" class="btn" style="background:var(--border); flex:1;">Cancel</button>
      </div>
    </div>
  </div>

  <div id="create-dialog" class="lightbox" style="display:none; align-items:center; justify-content:center;">
    <div class="card" style="width:500px; padding:24px;" onclick="event.stopPropagation()">
      <h2 style="margin-bottom:16px;">Create New Folder</h2>
      <p style="font-size:.85rem; color:var(--sub); margin-bottom:16px;">Create a new directory (e.g., 'my-new-album') inside your personal upload space.</p>
      <div class="form-group">
        <label>New Folder Name</label>
        <input id="new-folder-name-create" type="text" placeholder="e.g. my-new-album" autofocus>
      </div>
      <div style="display:flex; gap:10px; margin-top:24px;">
        <button onclick="createFolder()" class="btn btn-primary" style="flex:1;">Create & Add</button>
        <button onclick="document.getElementById('create-dialog').style.display='none'" class="btn" style="background:var(--border); flex:1;">Cancel</button>
      </div>
    </div>
  </div>

  <div id="upload-dialog" class="lightbox" style="display:none; align-items:center; justify-content:center;">
    <div class="card" style="width:500px; padding:24px;" onclick="event.stopPropagation()">
      <h2 style="margin-bottom:16px;">Upload Images</h2>
      <p id="upload-target-text" style="font-size:.85rem; color:var(--sub); margin-bottom:16px; word-break:break-all;"></p>
      <input type="hidden" id="upload-target-path">
      <div class="form-group">
        <label>Select Images</label>
        <input id="upload-files" type="file" multiple accept="image/*" style="padding:10px; background:var(--input-bg); border:1px dashed var(--border); border-radius:8px; width:100%;">
      </div>

  <!-- Delete Folder Confirmation Modal -->
  <div id="delete-dialog" class="lightbox" style="display:none; align-items:center; justify-content:center;" onclick="document.getElementById('delete-dialog').style.display='none'">
    <div class="card" style="width:520px; padding:28px; border:2px solid #ef4444;" onclick="event.stopPropagation()">
      <h2 style="margin-bottom:8px; color:#ef4444;">⚠️ Delete Folder &amp; Images</h2>
      <p style="color:var(--sub); font-size:.9rem; margin-bottom:16px;">This will <strong style='color:var(--text);'>permanently delete all image files</strong> from disk inside <code id="del-folder-name" style="background:var(--border);padding:2px 6px;border-radius:4px;"></code> and remove them from the index.</p>
      <div style="background:#1a0a0a; border:1px solid #7f1d1d; border-radius:8px; padding:12px 14px; margin-bottom:20px; font-size:.85rem; color:#fca5a5;">
        ✅ <strong>Preserved:</strong> All named people and trained face recognition data are kept.<br>
        ❌ <strong>Deleted:</strong> Image files on disk, photo index records, detected face thumbnails.
      </div>
      <p style="font-size:.85rem; color:var(--sub); margin-bottom:16px;">Type <strong style='color:var(--text);'>DELETE</strong> to confirm:</p>
      <input id="delete-confirm-input" type="text" placeholder="Type DELETE to confirm" style="margin-bottom:16px; background:var(--input-bg); border:1px solid #ef4444;">
      <div style="display:flex; gap:10px;">
        <button id="delete-confirm-btn" onclick="executeDeleteFolder()" class="btn" style="background:#ef4444; color:white; flex:1; font-weight:700;">🗑 Permanently Delete</button>
        <button onclick="document.getElementById('delete-dialog').style.display='none'" class="btn" style="background:var(--border); flex:1;">Cancel</button>
      </div>
    </div>
  </div>
      <div id="upload-progress" style="display:none; margin-top:16px;">
          <div style="height:8px; background:var(--border); border-radius:4px; overflow:hidden;">
              <div id="upload-bar" style="height:100%; width:0%; background:var(--accent);"></div>
          </div>
          <p id="upload-status" style="font-size:.75rem; color:var(--sub); margin-top:4px; text-align:center;">Uploading...</p>
      </div>
      <div style="display:flex; gap:10px; margin-top:24px;">
        <button id="upload-btn" onclick="uploadFiles()" class="btn btn-primary" style="flex:1;">Upload</button>
        <button id="upload-cancel" onclick="document.getElementById('upload-dialog').style.display='none'" class="btn" style="background:var(--border); flex:1;">Cancel</button>
      </div>
    </div>
  </div>
</div>
<script>
async function addFolder() {{
    const folder_name = document.getElementById('folder-name-add').value.trim();
    if(!folder_name) return;
    const r = await fetch('/admin/folders/add', {{
        method:'POST',
        headers:{{'Content-Type':'application/json'}},
        body:JSON.stringify({{folder_name: folder_name}})
    }});
    if(r.ok) location.reload();
    else {{
        const err = await r.json();
        alert(err.error || 'Failed to add folder.');
    }}
}}
async function createFolder() {{
    const folder_name = document.getElementById('new-folder-name-create').value.trim();
    if(!folder_name) return;
    const r = await fetch('/admin/folders/create', {{
        method:'POST',
        headers:{{'Content-Type':'application/json'}},
        body:JSON.stringify({{folder_name: folder_name}})
    }});
    if(r.ok) location.reload();
    else {{
        const err = await r.json();
        alert(err.error || 'Failed to create folder.');
    }}
}}
function openUploadDialog(path) {{
    document.getElementById('upload-target-path').value = path;
    document.getElementById('upload-target-text').innerText = 'Target: ' + path;
    document.getElementById('upload-files').value = '';
    document.getElementById('upload-progress').style.display = 'none';
    document.getElementById('upload-btn').disabled = false;
    document.getElementById('upload-dialog').style.display = 'flex';
}}
async function uploadFiles() {{
    const path = document.getElementById('upload-target-path').value;
    const files = document.getElementById('upload-files').files;
    if(!files.length) {{ alert('Please select files first.'); return; }}

    const formData = new FormData();
    formData.append('path', path);
    for(let i=0; i<files.length; i++) {{
        formData.append('files', files[i]);
    }}

    const btn = document.getElementById('upload-btn');
    const cancelBtn = document.getElementById('upload-cancel');
    const progress = document.getElementById('upload-progress');
    const bar = document.getElementById('upload-bar');
    const status = document.getElementById('upload-status');

    btn.disabled = true;
    cancelBtn.disabled = true;
    progress.style.display = 'block';
    
    try {{
        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/admin/folders/upload', true);
        
        xhr.upload.onprogress = (e) => {{
            if (e.lengthComputable) {{
                const pct = Math.round((e.loaded / e.total) * 100);
                bar.style.width = pct + '%';
                status.innerText = 'Uploading: ' + pct + '%';
            }}
        }};

        xhr.onload = () => {{
            if (xhr.status === 200) {{
                alert('Success! ' + files.length + ' images uploaded.');
                location.reload();
            }} else {{
                const err = JSON.parse(xhr.responseText);
                alert(err.error || 'Upload failed.');
                btn.disabled = false;
                cancelBtn.disabled = false;
            }}
        }};

        xhr.onerror = () => {{
            alert('Network error during upload.');
            btn.disabled = false;
            cancelBtn.disabled = false;
        }};

        xhr.send(formData);
    }} catch (e) {{
        alert('Upload failed: ' + e);
        btn.disabled = false;
        cancelBtn.disabled = false;
    }}
}}
async function removeFolder(path) {{
    if(!confirm('Stop scanning ' + path + '? (Photos already indexed will NOT be deleted)')) return;
    const r = await fetch('/admin/folders/remove', {{
        method:'POST',
        headers:{{'Content-Type':'application/json'}},
        body:JSON.stringify({{path}})
    }});
    if(r.ok) location.reload();
    else alert('Failed to remove folder.');
}}

let _delFolderPath = null;
function confirmDeleteFolder(path, name) {{
    _delFolderPath = path;
    document.getElementById('del-folder-name').innerText = name;
    document.getElementById('delete-confirm-input').value = '';
    document.getElementById('delete-dialog').style.display = 'flex';
}}
async function executeDeleteFolder() {{
    if(document.getElementById('delete-confirm-input').value !== 'DELETE') {{
        alert('Please type DELETE to confirm.');
        return;
    }}
    if(!_delFolderPath) return;
    const btn = document.getElementById('delete-confirm-btn');
    btn.disabled = true;
    btn.innerText = '⏳ Deleting...';
    const r = await fetch('/admin/folders/delete', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{ path: _delFolderPath }})
    }});
    if(r.ok) {{
        const res = await r.json();
        document.getElementById('delete-dialog').style.display = 'none';
        const done = document.getElementById('scan-done-banner');
        done.style.display = 'block';
        done.style.background = '#fef2f2';
        done.style.color = '#7f1d1d';
        done.style.borderColor = '#ef4444';
        done.innerHTML = '🗑 Folder deleted. Removed <b>' + res.photos_deleted + '</b> photo record(s) and <b>' + res.files_deleted + '</b> file(s) from disk.';
        location.reload();
    }} else {{
        const err = await r.json();
        alert(err.error || 'Delete failed.');
        btn.disabled = false;
        btn.innerText = '🗑 Permanently Delete';
    }}
}}
async function triggerScan(path=null) {{
    const payload = path ? {{ folder: path }} : {{}};
    const r = await fetch('/admin/scan', {{
        method:'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify(payload)
    }});
    if(r.ok) {{
        startPolling();
    }} else {{
        const err = await r.json();
        alert(err.error || 'Failed to start scan.');
    }}
}}

let _pollTimer = null;
function startPolling() {{
    const panel = document.getElementById('scan-progress-panel');
    const done  = document.getElementById('scan-done-banner');
    panel.style.display = 'block';
    done.style.display  = 'none';
    clearInterval(_pollTimer);
    _pollTimer = setInterval(pollProgress, 1200);
    pollProgress();
}}

async function pollProgress() {{
    try {{
        const res = await fetch('/admin/scan/progress');
        if (!res.ok) return;
        const d = await res.json();
        
        document.getElementById('scan-prog-bar').style.width = d.pct + '%';
        document.getElementById('scan-prog-counter').innerText =
            d.total > 0 ? d.current + ' / ' + d.total + ' photos' : '';
        document.getElementById('scan-prog-msg').innerText = d.message || '';
        document.getElementById('scan-prog-faces').innerText = d.faces + ' face(s) detected';
        
        if (!d.running && d.summary !== null && d.summary !== undefined) {{
            clearInterval(_pollTimer);
            document.getElementById('scan-progress-panel').style.display = 'none';
            const done = document.getElementById('scan-done-banner');
            done.style.display = 'block';
            done.innerHTML = '✅ Scan complete! Photos indexed: <b>' + d.summary.added +
                             '</b> &nbsp;|&nbsp; Skipped: <b>' + d.summary.skipped +
                             '</b> &nbsp;|&nbsp; Faces found: <b>' + d.summary.faces + '</b>';
        }} else if (!d.running && d.total === 0) {{
            // Likely nothing was scanned yet (initial state)
        }}
    }} catch(e) {{ /* ignore */ }}
}}

// Auto-start polling if scan already in progress (on page refresh)
(async function() {{
    try {{
        const res = await fetch('/admin/scan/progress');
        const d   = await res.json();
        if (d.running) startPolling();
    }} catch(e) {{}}
}})();
</script>
</body></html>"""
        return page

    @flask_app.route("/admin/folders/add", methods=["POST"])
    @uploader_allowed
    def admin_folder_add():
        data = request.json
        folder_name = (data.get("folder_name") or "").strip()
        if not folder_name:
            return jsonify({"error": "Folder name is required"}), 400
        
        username = session.get("username")
        if not username:
            return jsonify({"error": "User not authenticated"}), 401
            
        full_folder_name = f"{username}-{folder_name}"
        full_path = UPLOADS_ROOT_DIR / full_folder_name
        
        if not os.path.isdir(full_path):
            return jsonify({"error": f"Folder '{folder_name}' does not exist or is not a directory at {full_path}"}), 400
        
        # Check if already exists in DB
        existing = db.get_scan_folders()
        try:
            if any(os.path.samefile(full_path, x["path"]) for x in existing if os.path.isdir(x["path"])):
                 return jsonify({"error": "Folder is already in the scan list"}), 400
        except: pass

        uid = session.get("user_id")
        db.add_scan_folder(str(full_path), created_by=uid)
        return jsonify({"success": True})

    @flask_app.route("/admin/folders/remove", methods=["POST"])
    @uploader_allowed
    def admin_folder_remove():
        data = request.json
        path_to_remove = (data.get("path") or "").strip()
        if not path_to_remove:
            return jsonify({"error": "Path is required"}), 400
        
        # Verify ownership if not admin
        is_admin = session.get("role") == "admin"
        if not is_admin:
            uid = session.get("user_id")
            folders = db.get_scan_folders(created_by=uid)
            
            found_match = False
            for f in folders:
                db_path = f["path"]
                # Check existence before samefile to prevent FileNotFoundError
                if os.path.exists(path_to_remove) and os.path.exists(db_path):
                    try:
                        if os.path.samefile(path_to_remove, db_path):
                            found_match = True
                            break
                    except FileNotFoundError:
                        # Log if one of the paths does not exist for samefile check
                        logger.warning(f"FileNotFoundError during samefile check for paths: '{path_to_remove}' and '{db_path}'. Skipping comparison.")
                        continue # Continue to next folder
                elif path_to_remove == db_path: # Fallback for non-existent but identical string paths
                    found_match = True
                    break
            
            if not found_match:
                abort(403) # Not authorized or folder not found in authorized list

        db.remove_scan_folder(path_to_remove)
        return jsonify({"success": True})

    @flask_app.route("/admin/folders/delete", methods=["POST"])
    @uploader_allowed
    def admin_folder_delete():
        """Permanently delete a folder's image files from disk and its records from the DB.
        
        FILE DELETION POLICY:
          - Physical files are ONLY deleted if the folder is inside UPLOADS_ROOT_DIR
            (i.e. folders uploaded via the web app). Folders on the local filesystem
            that happen to be registered as scan folders will have their DB records
            removed but their files left on disk — matching desktop-app behaviour.
          - Persons and trained face recognition data (embeddings) are NEVER deleted.
        """
        data = request.json or {}
        path_to_delete = os.path.abspath((data.get("path") or "").strip())
        if not path_to_delete:
            return jsonify({"error": "Path is required"}), 400

        # Ownership check
        is_admin = session.get("role") == "admin"
        if not is_admin:
            uid = session.get("user_id")
            allowed = db.get_scan_folders(created_by=uid)
            found = any(
                os.path.normcase(path_to_delete) == os.path.normcase(f["path"])
                for f in allowed
            )
            if not found:
                abort(403)

        # Safety: must be a registered scan folder
        all_folders = db.get_scan_folders()
        registered = any(
            os.path.normcase(path_to_delete) == os.path.normcase(f["path"])
            for f in all_folders
        )
        if not registered:
            return jsonify({"error": "Folder is not in the scan list"}), 400

        # 1. Remove DB records (photos + faces, NOT persons/mappings)
        photos_deleted = db.delete_photos_for_folder(path_to_delete)

        # 2. Delete physical files ONLY if the folder lives inside UPLOADS_ROOT_DIR
        #    (web-uploaded content). Folders outside uploads (e.g. local desktop paths)
        #    are left untouched on disk — only their DB records are removed.
        files_deleted = 0
        uploads_root = os.path.abspath(str(UPLOADS_ROOT_DIR))
        is_web_upload = os.path.commonpath(
            [os.path.normcase(path_to_delete), os.path.normcase(uploads_root)]
        ) == os.path.normcase(uploads_root)

        if is_web_upload and os.path.isdir(path_to_delete):
            import shutil
            try:
                for root, dirs, files in os.walk(path_to_delete):
                    for fname in files:
                        fpath = os.path.join(root, fname)
                        try:
                            os.remove(fpath)
                            files_deleted += 1
                        except Exception as e:
                            logger.warning("Could not delete file %s: %s", fpath, e)
                shutil.rmtree(path_to_delete, ignore_errors=True)
            except Exception as e:
                logger.error("Error deleting folder %s: %s", path_to_delete, e)
        elif not is_web_upload:
            logger.info(
                "Folder %s is outside UPLOADS_ROOT_DIR – DB records removed but files kept on disk.",
                path_to_delete
            )

        # 3. Remove from scan folders list
        db.remove_scan_folder(path_to_delete)

        logger.info("Deleted folder %s: %d photos removed from DB, %d files from disk",
                    path_to_delete, photos_deleted, files_deleted)
        return jsonify({"success": True, "photos_deleted": photos_deleted, "files_deleted": files_deleted})

    @flask_app.route("/admin/folders/create", methods=["POST"])
    @uploader_allowed
    def admin_folder_create():
        data = request.json
        folder_name = (data.get("folder_name") or "").strip()
        if not folder_name:
            return jsonify({"error": "Folder name is required"}), 400
        
        username = session.get("username")
        if not username:
            return jsonify({"error": "User not authenticated"}), 401
            
        full_folder_name = f"{username}-{folder_name}"
        full_path = UPLOADS_ROOT_DIR / full_folder_name
        
        try:
            if os.path.exists(full_path):
                if not os.path.isdir(full_path):
                    return jsonify({"error": "A file already exists at this path"}), 400
            else:
                os.makedirs(full_path, exist_ok=True)
            
            # Auto-add to scan folders
            uid = session.get("user_id")
            existing = db.get_scan_folders()
            if not any(os.path.samefile(full_path, x["path"]) for x in existing if os.path.isdir(x["path"])):
                db.add_scan_folder(str(full_path), created_by=uid)
                
            return jsonify({"success": True})
        except Exception as e:
            logger.error("Failed to create folder %s: %s", full_path, e)
            return jsonify({"error": f"Failed to create folder: {str(e)}"}), 500

    @flask_app.route("/admin/folders/upload", methods=["POST"])
    @uploader_allowed
    def admin_folder_upload():
        target_path = os.path.abspath((request.form.get("path") or "").strip())
        if not target_path or not os.path.isdir(target_path):
            return jsonify({"error": "Invalid target path"}), 400
        
        # Verify ownership if not admin
        is_admin = session.get("role") == "admin"
        if not is_admin:
            uid = session.get("user_id")
            folders = db.get_scan_folders(created_by=uid)
            if not any(os.path.samefile(target_path, f["path"]) for f in folders if os.path.isdir(f["path"])):
                abort(403)
        
        if "files" not in request.files:
            return jsonify({"error": "No files uploaded"}), 400
        
        files = request.files.getlist("files")
        count = 0
        from werkzeug.utils import secure_filename
        
        for file in files:
            if file and file.filename:
                # Basic image check
                mime = file.content_type or ""
                if not mime.startswith("image/"):
                    continue
                
                filename = secure_filename(file.filename)
                save_path = os.path.join(target_path, filename)
                
                # Prevent overwrite or name collision simply by adding number if exists
                base, ext = os.path.splitext(save_path)
                counter = 1
                while os.path.exists(save_path):
                    save_path = f"{base}_{counter}{ext}"
                    counter += 1
                
                file.save(save_path)
                count += 1
        
        return jsonify({"success": True, "count": count})

    @flask_app.route("/admin/scan", methods=["POST"])
    @uploader_allowed
    def admin_scan():
        is_admin = session.get("role") == "admin"
        uid = session.get("user_id")
        
        data = request.get_json(silent=True) or {}
        specific_folder = data.get("folder")
        
        all_allowed_folders = [f["path"] for f in db.get_scan_folders(created_by=None if is_admin else uid)]
        
        if specific_folder:
            if specific_folder not in all_allowed_folders:
                return jsonify({"error": "Unauthorized folder"}), 403
            folders = [specific_folder]
        else:
            folders = all_allowed_folders
            
        if not folders:
            return jsonify({"error": "No folders to scan"}), 400
        
        def run_scan():
            with _scan_progress_lock:
                _scan_progress["running"] = True
                _scan_progress["current"] = 0
                _scan_progress["total"] = 0
                _scan_progress["message"] = "Starting scan..."
                _scan_progress["faces"] = 0
                _scan_progress["started_at"] = _time.time()
                _scan_progress["finished_at"] = None
                _scan_progress["summary"] = None

            def progress_cb(current, total, message):
                with _scan_progress_lock:
                    _scan_progress["current"] = current
                    _scan_progress["total"] = total
                    _scan_progress["message"] = message

            def face_found_cb(thumb_bytes):
                with _scan_progress_lock:
                    _scan_progress["faces"] += 1

            scanner = PhotoScanner(db, progress_cb=progress_cb, face_found_cb=face_found_cb)
            summary = scanner.scan(folders)
            with _scan_progress_lock:
                _scan_progress["running"] = False
                _scan_progress["finished_at"] = _time.time()
                _scan_progress["summary"] = summary
                _scan_progress["message"] = "Scan complete!"
            logger.info("Background scan triggered from web UI completed.")

        threading.Thread(target=run_scan, daemon=True).start()
        return jsonify({"success": True})

    @flask_app.route("/admin/scan/progress")
    @uploader_allowed
    def admin_scan_progress():
        with _scan_progress_lock:
            data = dict(_scan_progress)
        pct = 0
        if data["total"] > 0:
            pct = int(data["current"] / data["total"] * 100)
        return jsonify({
            "running": data["running"],
            "current": data["current"],
            "total": data["total"],
            "pct": pct,
            "message": data["message"],
            "faces": data["faces"],
            "summary": data["summary"],
        })

    # ──────────────────────────────────────────────────────────────────── #
    # Face Naming & Assignment                                             #
    # ──────────────────────────────────────────────────────────────────── #

    @flask_app.route("/admin/naming")
    @uploader_allowed
    def admin_naming():
        is_admin = session.get("role") == "admin"
        uid = session.get("user_id")
        
        # Performance optimization: get folders first
        user_folders = db.get_scan_folders(created_by=None if is_admin else uid)
        folder_paths = [f["path"] for f in user_folders]
        
        all_unknown = db.get_unassigned_faces()
        faces = []
        
        # Filter unassigned faces by folder ownership for non-admins
        if not is_admin:
            for face in all_unknown:
                photo = db.get_photo(face["photo_id"])
                if photo:
                    p_path = photo["path"]
                    if any(p_path.startswith(fp) for fp in folder_paths):
                        faces.append(face)
        else:
            faces = all_unknown
            
        if not faces:
            nav = get_nav()
            return f"""{BASE_HTML}{nav}
<div class="container" style="text-align:center; padding:100px 20px;">
  <div style="font-size:4rem; margin-bottom:24px;">✨</div>
  <h1 style="font-size:1.8rem; font-weight:700;">All Faces Named!</h1>
  <p style="color:var(--sub); margin-top:12px;">There are no unknown faces in your library right now.</p>
  <a href="/" class="btn btn-primary" style="margin-top:24px;">Return Home</a>
</div></body></html>"""

        # Perform clustering for better UX
        embs = []
        face_list = list(faces)
        for f in face_list:
            if f["embedding"]:
                embs.append(bytes_to_embedding(f["embedding"]))
            else:
                embs.append(np.zeros(512, dtype=np.float32))
        
        clusters = cluster_embeddings(embs)
        sorted_faces = []
        for cluster_indices in clusters:
            for idx in cluster_indices:
                sorted_faces.append(face_list[idx])
                
        face_cards = ""
        for f in sorted_faces:
             face_cards += f"""
<div class="naming-card" data-fid="{f['face_id']}" onclick="toggleFace({f['face_id']}, this, event)">
  <img src="/face_thumb/{f['face_id']}" alt="face" loading="lazy">
</div>"""

        persons = db.get_all_persons()
        person_opts = '<option value="">— Create new person —</option>'
        for p in persons:
            person_opts += f'<option value="{p["person_id"]}">{p["name"]}</option>'

        nav = get_nav()
        page = f"""{BASE_HTML}{nav}
<style>
  .naming-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(110px, 1fr));
    gap: 12px;
    padding: 20px 0 100px;
  }}
  .naming-card {{
    aspect-ratio: 1;
    background: var(--card);
    border-radius: 8px;
    border: 2px solid transparent;
    cursor: pointer;
    overflow: hidden;
    transition: all .2s;
    position: relative;
    user-select: none;
  }}
  .naming-card:hover {{ border-color: var(--accent); transform: translateY(-2px); }}
  .naming-card.selected {{ border-color: var(--accent); border-width: 3px; }}
  .naming-card.selected::after {{
    content: '✓';
    position: absolute;
    top: 5px; right: 5px;
    background: var(--accent);
    color: white;
    width: 20px; height: 20px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; font-weight: bold;
    box-shadow: 0 2px 5px rgba(0,0,0,.3);
  }}
  .naming-card img {{ width: 100%; height: 100%; object-fit: cover; }}
  
  .naming-bar {{
    position: fixed;
    bottom: 0; left: 0; right: 0;
    background: var(--card);
    border-top: 1px solid var(--border);
    padding: 16px;
    display: none;
    align-items: center;
    gap: 16px;
    box-shadow: 0 -10px 30px rgba(0,0,0,.3);
    z-index: 1000;
  }}
  .naming-bar.active {{ display: flex; animation: slideUp .3s ease; }}
  @keyframes slideUp {{ from {{ transform: translateY(100%); }} to {{ transform: translateY(0); }} }}

  @media (max-width: 768px) {{
    .naming-bar {{ flex-direction: column; height: auto; padding: 20px; }}
    .naming-bar select, .naming-bar input {{ max-width: none; width: 100%; }}
  }}
</style>

<div class="container">
  <div class="toolbar">
    <h1 style="font-size:1.6rem;font-weight:700;">Name Unknown Faces</h1>
    <span class="badge" id="total-count">{len(faces)} unknown</span>
    <div style="flex:1;"></div>
    <button onclick="autoMatch()" id="btn-auto" class="btn btn-primary" style="background:var(--accent2);">🤖 Auto Match All</button>
  </div>
  
  <p style="color:var(--sub); margin-bottom:20px;">Select faces to name them. Similar faces are grouped together by AI.</p>
  
  <div class="naming-grid">
    {face_cards}
  </div>
</div>

<div class="naming-bar" id="naming-bar">
  <div style="font-weight:700; color:var(--accent); min-width:100px;" id="sel-count">0 selected</div>
  <div style="color:var(--sub);">Assign to:</div>
  <select id="assign-person-id" style="flex:1; max-width:300px;">
    {person_opts}
  </select>
  <input type="text" id="new-person-name" placeholder="New person name..." style="flex:1; max-width:250px;">
  <button onclick="assignFaces()" id="btn-assign" class="btn btn-primary" style="padding: 10px 24px;">💾 Assign Selected</button>
  <button onclick="rejectFaces()" class="btn btn-danger">🗑 Not a Face</button>
  <button onclick="clearSelection()" class="btn" style="background:var(--border);">✕</button>
</div>

<script>
let selected = new Set();
let lastClickedId = null;

function toggleFace(fid, el, ev) {{
    if(ev && ev.shiftKey && lastClickedId !== null) {{
        // Basic range selection (could be improved for grid)
        const cards = Array.from(document.querySelectorAll('.naming-card'));
        const idx1 = cards.findIndex(c => parseInt(c.dataset.fid) === lastClickedId);
        const idx2 = cards.findIndex(c => parseInt(c.dataset.fid) === fid);
        if(idx1 !== -1 && idx2 !== -1) {{
            const start = Math.min(idx1, idx2);
            const end = Math.max(idx1, idx2);
            for(let i=start; i<=end; i++) {{
                const f_id = parseInt(cards[i].dataset.fid);
                selected.add(f_id);
                cards[i].classList.add('selected');
            }}
        }}
    }} else {{
        if (selected.has(fid)) {{
            selected.delete(fid);
            el.classList.remove('selected');
        }} else {{
            selected.add(fid);
            el.classList.add('selected');
        }}
    }}
    lastClickedId = fid;
    updateBar();
}}

// Update toggleFace to use actual event
document.querySelectorAll('.naming-card').forEach(card => {{
    const fid = parseInt(card.dataset.fid);
    card.onclick = (e) => toggleFace(fid, card, e);
}});

function updateBar() {{
    const bar = document.getElementById('naming-bar');
    const lbl = document.getElementById('sel-count');
    if (selected.size > 0) {{
        bar.classList.add('active');
        lbl.innerText = selected.size + ' selected';
    }} else {{
        bar.classList.remove('active');
    }}
}}

function clearSelection() {{
    selected.clear();
    lastClickedId = null;
    document.querySelectorAll('.naming-card.selected').forEach(el => el.classList.remove('selected'));
    updateBar();
}}

async function assignFaces() {{
    const person_id = document.getElementById('assign-person-id').value;
    const person_name = document.getElementById('new-person-name').value.trim();
    
    if (!person_id && !person_name) {{
        alert('Please select a person or enter a new name.');
        return;
    }}
    if (person_id && person_name) {{
        alert('Please either select an existing person, or type a new name—not both.');
        return;
    }}
    
    const btn = document.getElementById('btn-assign');
    btn.disabled = true;
    btn.innerText = '⏳ Saving...';
    
    const r = await fetch('/admin/faces/assign', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{
            face_ids: Array.from(selected),
            person_id: person_id ? parseInt(person_id) : null,
            person_name: person_name
        }})
    }});
    
    if (r.ok) {{
        location.reload();
    }} else {{
        const err = await r.json();
        alert(err.error || 'Failed to assign faces.');
        btn.disabled = false;
        btn.innerText = '💾 Assign Selected';
    }}
}}

async function rejectFaces() {{
    if (!confirm('Are you sure these are not faces? They will be deleted.')) return;
    
    const r = await fetch('/admin/faces/delete', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ face_ids: Array.from(selected) }})
    }});
    
    if (r.ok) location.reload();
    else alert('Failed to delete faces.');
}}

async function autoMatch() {{
    const btn = document.getElementById('btn-auto');
    btn.disabled = true;
    btn.innerText = '⏳ Matching...';
    
    const r = await fetch('/admin/faces/automatch', {{ method: 'POST' }});
    if (r.ok) {{
        const res = await r.json();
        alert('Automatically matched ' + res.count + ' faces!');
        location.reload();
    }} else {{
        alert('Auto match failed.');
        btn.disabled = false;
        btn.innerText = '🤖 Auto Match All';
    }}
}}
</script>
</body></html>"""
        return page

    @flask_app.route("/admin/faces/assign", methods=["POST"])
    @uploader_allowed
    def admin_faces_assign_post():
        data = request.json
        face_ids = data.get("face_ids", [])
        person_id = data.get("person_id")
        name = (data.get("person_name") or "").strip()
        
        if not face_ids:
            return jsonify({"error": "No faces selected"}), 400
            
        if person_id is not None and person_id != -1 and name:
            return jsonify({"error": "Conflicting input: please either select an existing person or enter a new name."}), 400

        if person_id == -1:
            for fid in face_ids:
                db.unmap_face(fid)
            return jsonify({"success": True})
        elif person_id is None:
            if not name:
                return jsonify({"error": "Name required for new person"}), 400
            app_core.create_person_from_faces(face_ids, name)
        else:
            app_core.assign_faces_to_person(face_ids, person_id)
            
        return jsonify({"success": True})

    @flask_app.route("/admin/faces/delete", methods=["POST"])
    @uploader_allowed
    def admin_faces_delete():
        data = request.json
        face_ids = data.get("face_ids", [])
        if not face_ids:
            return jsonify({"error": "No faces selected"}), 400
        
        app_core.remove_false_positive(face_ids)
        return jsonify({"success": True})

    @flask_app.route("/admin/faces/automatch", methods=["POST"])
    @uploader_allowed
    def admin_faces_automatch():
        count = app_core.auto_match_all_unassigned()
        return jsonify({"success": True, "count": count})

    @flask_app.route("/admin/faces/unassign", methods=["POST"])
    @uploader_allowed
    def admin_faces_unassign():
        data = request.json
        face_ids = data.get("face_ids", [])
        if not face_ids:
            return jsonify({"error": "No faces selected"}), 400
        for fid in face_ids:
            db.unmap_face(fid)
        return jsonify({"success": True})

    @flask_app.route("/admin/person/<int:person_id>/faces")
    @uploader_allowed
    def admin_person_faces(person_id):
        is_admin = session.get("role") == "admin"
        if not is_admin:
            allowed = get_user_allowed_ids()
            if person_id not in allowed:
                abort(403)
                
        person = db.get_person(person_id)
        if not person:
            abort(404)
            
        faces = db.get_faces_for_person(person_id)
        
        face_cards = ""
        for f in faces:
             face_cards += f'''
<div class="naming-card" data-fid="{f['face_id']}" onclick="toggleFace({f['face_id']}, this, event)">
  <img src="/face_thumb/{f['face_id']}" alt="face" loading="lazy">
  <a class="face-edit-btn" href="/admin/face/{f['face_id']}/edit" onclick="event.stopPropagation()">✏️</a>
</div>'''

        persons = db.get_all_persons()
        person_opts = '<option value="-1">— Leave Unassigned (Remove) —</option>'
        for p in persons:
            if p["person_id"] != person_id:
                person_opts += f'<option value="{p["person_id"]}">{p["name"]}</option>'

        nav = get_nav()
        page = f"""{BASE_HTML}{nav}
<style>
  .naming-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(110px, 1fr));
    gap: 12px;
    padding: 20px 0 100px;
  }}
  .naming-card {{
    aspect-ratio: 1;
    background: var(--card);
    border-radius: 8px;
    border: 2px solid transparent;
    cursor: pointer;
    overflow: hidden;
    transition: all .2s;
    position: relative;
    user-select: none;
  }}
  .naming-card:hover {{ border-color: var(--accent); transform: translateY(-2px); }}
  .naming-card.selected {{ border-color: var(--accent); border-width: 3px; }}
  .naming-card.selected::after {{
    content: '✓';
    position: absolute;
    top: 5px; right: 5px;
    background: var(--accent);
    color: white;
    width: 20px; height: 20px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; font-weight: bold;
    box-shadow: 0 2px 5px rgba(0,0,0,.3);
  }}
  .naming-card img {{ width: 100%; height: 100%; object-fit: cover; }}
  
  .face-edit-btn {{
    position: absolute;
    top: 5px; left: 5px;
    background: rgba(0,0,0,0.6);
    color: white;
    border-radius: 4px;
    padding: 2px 6px;
    font-size: 12px;
    text-decoration: none;
    display: none;
    z-index: 10;
  }}
  .naming-card:hover .face-edit-btn {{ display: block; }}
  .face-edit-btn:hover {{ background: rgba(0,0,0,0.9); }}

  .naming-bar {{
    position: fixed;
    bottom: 0; left: 0; right: 0;
    background: var(--card);
    border-top: 1px solid var(--border);
    padding: 16px;
    display: none;
    align-items: center;
    gap: 16px;
    box-shadow: 0 -10px 30px rgba(0,0,0,.3);
    z-index: 1000;
  }}
  .naming-bar.active {{ display: flex; animation: slideUp .3s ease; }}
  @keyframes slideUp {{ from {{ transform: translateY(100%); }} to {{ transform: translateY(0); }} }}
</style>

<div class="container">
  <div style="margin-bottom:20px;">
    <a href="/photos?person_id={person_id}" style="color:var(--sub); text-decoration:none;">← Back to photos</a>
  </div>
  <div class="toolbar">
    <h1 style="font-size:1.6rem;font-weight:700;">Manage Faces: {person['name']}</h1>
    <span class="badge" id="total-count">{len(faces)} faces</span>
  </div>
  
  <p style="color:var(--sub); margin-bottom:20px;">Select faces that do not belong to this person to re-assign or delete them.</p>
  
  <div class="naming-grid">
    {face_cards}
  </div>
</div>

<div class="naming-bar" id="naming-bar">
  <div style="font-weight:700; color:var(--accent); min-width:100px;" id="sel-count">0 selected</div>
  <div style="color:var(--sub);">Re-assign to:</div>
  <select id="assign-person-id" style="flex:1; max-width:300px;">
    {person_opts}
  </select>
  <input type="text" id="new-person-name" placeholder="New person name..." style="flex:1; max-width:250px;">
  <button onclick="assignFaces()" id="btn-assign" class="btn btn-primary" style="padding: 10px 24px;">💾 Move/Remove Selected</button>
  <button onclick="rejectFaces()" class="btn btn-danger">🗑 Not a Face</button>
  <button onclick="clearSelection()" class="btn" style="background:var(--border);">✕</button>
</div>

<script>
let selected = new Set();
let lastClickedId = null;

function toggleFace(fid, el, ev) {{
    if(ev && ev.shiftKey && lastClickedId !== null) {{
        const cards = Array.from(document.querySelectorAll('.naming-card'));
        const idx1 = cards.findIndex(c => parseInt(c.dataset.fid) === lastClickedId);
        const idx2 = cards.findIndex(c => parseInt(c.dataset.fid) === fid);
        if(idx1 !== -1 && idx2 !== -1) {{
            const start = Math.min(idx1, idx2);
            const end = Math.max(idx1, idx2);
            for(let i=start; i<=end; i++) {{
                const f_id = parseInt(cards[i].dataset.fid);
                selected.add(f_id);
                cards[i].classList.add('selected');
            }}
        }}
    }} else {{
        if (selected.has(fid)) {{
            selected.delete(fid);
            el.classList.remove('selected');
        }} else {{
            selected.add(fid);
            el.classList.add('selected');
        }}
    }}
    lastClickedId = fid;
    updateBar();
}}

document.querySelectorAll('.naming-card').forEach(card => {{
    const fid = parseInt(card.dataset.fid);
    card.onclick = (e) => toggleFace(fid, card, e);
}});

function updateBar() {{
    const bar = document.getElementById('naming-bar');
    const lbl = document.getElementById('sel-count');
    if (selected.size > 0) {{
        bar.classList.add('active');
        lbl.innerText = selected.size + ' selected';
    }} else {{
        bar.classList.remove('active');
    }}
}}

function clearSelection() {{
    selected.clear();
    lastClickedId = null;
    document.querySelectorAll('.naming-card.selected').forEach(el => el.classList.remove('selected'));
    updateBar();
}}

async function assignFaces() {{
    const person_id = document.getElementById('assign-person-id').value;
    const person_name = document.getElementById('new-person-name').value.trim();
    
    if (person_id === "-1" && !person_name) {{
       const btn = document.getElementById('btn-assign');
       btn.disabled = true;
       btn.innerText = '⏳ Saving...';
       const r = await fetch('/admin/faces/unassign', {{
           method: 'POST',
           headers: {{ 'Content-Type': 'application/json' }},
           body: JSON.stringify({{ face_ids: Array.from(selected) }})
       }});
       if(r.ok) location.reload();
       else {{
           alert('Failed to remove.');
           btn.disabled = false;
           btn.innerText = '💾 Move/Remove Selected';
       }}
       return;
    }}

    let pid = (person_id && person_id !== "-1") ? parseInt(person_id) : null;
    
    if (!pid && !person_name) {{
        alert('Please select a person or enter a new name.');
        return;
    }}
    
    if (pid && person_name) {{
        alert('Please either select an existing person, or type a new name—not both.');
        return;
    }}

    const btn = document.getElementById('btn-assign');
    btn.disabled = true;
    btn.innerText = '⏳ Saving...';
    
    const r = await fetch('/admin/faces/assign', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{
            face_ids: Array.from(selected),
            person_id: pid,
            person_name: person_name
        }})
    }});
    
    if (r.ok) {{
        location.reload();
    }} else {{
        const err = await r.json();
        alert(err.error || 'Failed to move faces.');
        btn.disabled = false;
        btn.innerText = '💾 Move/Remove Selected';
    }}
}}

async function rejectFaces() {{
    if (!confirm('Are you sure these are not faces? They will be deleted.')) return;
    
    const r = await fetch('/admin/faces/delete', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ face_ids: Array.from(selected) }})
    }});
    
    if (r.ok) location.reload();
    else alert('Failed to delete faces.');
}}
</script>
</body></html>"""
        return page

    @flask_app.route("/admin/face/<int:face_id>/edit", methods=["GET"])
    @uploader_allowed
    def admin_face_edit(face_id):
        face = db.get_face(face_id)
        if not face:
            abort(404)

        current_person = db.get_person_for_face(face_id)
        current_person_name = current_person["name"] if current_person else None
        current_person_id = current_person["person_id"] if current_person else ''

        persons = db.get_all_persons()
        person_opts = '<option value="-1">— Leave Unassigned —</option>'
        for p in persons:
            selected_attr = "selected" if current_person and p["person_id"] == current_person["person_id"] else ""
            person_opts += f'<option value="{p["person_id"]}" {selected_attr}>{p["name"]}</option>'
        
        nav = get_nav()
        page = f"""{BASE_HTML}{nav}
<div class="container" style="max-width:600px;">
  <div style="margin-bottom:24px;">
    <h1 style="font-size:1.6rem;font-weight:700;">Edit Face</h1>
    <p style="color:var(--sub);margin-top:4px;">Manage person association and name for this face.</p>
  </div>
  <div class="card">
    <div style="text-align:center;margin-bottom:20px;">
      <img src="/face_thumb/{face['face_id']}" alt="Face Thumbnail" style="width:150px;height:150px;object-fit:cover;border-radius:8px;">
    </div>
    <form id="edit-face-form">
      <div class="form-group">
        <label>Current Person</label>
        <p style="font-size:1.1rem;font-weight:600;">{current_person_name or 'Unassigned'}</p>
      </div>

      <div class="form-group">
        <label for="assign-person-id">Reassign to Existing Person</label>
        <select id="assign-person-id" name="person_id">
          {person_opts}
        </select>
      </div>

      <div class="form-group">
        <label for="new-person-name">Create New Person & Assign</label>
        <input type="text" id="new-person-name" name="new_person_name" placeholder="New person name...">
      </div>

      <div style="display:flex; gap:10px; margin-top:24px;">
        <button type="button" onclick="saveFaceChanges({face['face_id']})" class="btn btn-primary" style="flex:1;">💾 Save Changes</button>
        <button type="button" onclick="deleteFace({face['face_id']})" class="btn btn-danger">🗑 Delete Face</button>
        <a href="/photos" class="btn" style="background:var(--border); text-align:center;">Cancel</a>
      </div>
    </form>
  </div>
</div>
<script>
async function saveFaceChanges(faceId) {{
    const personId = document.getElementById('assign-person-id').value;
    const newPersonName = document.getElementById('new-person-name').value.trim();

    if (!personId && !newPersonName && '{current_person_id}' === '') {{ // Check if trying to save unassigned without assigning or creating
        alert('Please select a person or enter a new name to assign this face.');
        return;
    }}
    
    if (newPersonName && personId && personId !== "-1") {{
        alert('Please either select an existing person OR enter a new person name, not both.');
        return;
    }}

    const payload = {{
        face_ids: [faceId],
        person_id: personId ? parseInt(personId) : null,
        person_name: newPersonName || null
    }};

    const r = await fetch('/admin/faces/assign', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify(payload)
    }});

    if (r.ok) {{
        alert('Face updated successfully!');
        window.location.href = '/photos'; // Redirect back to photos
    }} else {{
        const err = await r.json();
        alert(err.error || 'Failed to update face.');
    }}
}}

async function deleteFace(faceId) {{
    if (!confirm('Are you sure you want to delete this face?')) return;

    const r = await fetch('/admin/faces/delete', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ face_ids: [faceId] }})
    }});

    if (r.ok) {{
        alert('Face deleted successfully!');
        window.location.href = '/photos'; // Redirect back to photos
    }} else {{
        const err = await r.json();
        alert(err.error || 'Failed to delete face.');
    }}
}}
</script>
</body></html>"""
        return page

    return flask_app


# ─────────────────────────────────────────────────────────────────────────────
# Server lifecycle
# ─────────────────────────────────────────────────────────────────────────────

def start_server(db, app_core, port: int = 5050, bind_all: bool = True):
    """Start Flask in a daemon thread. Safe to call from any thread."""
    global _flask_app, _server_thread, _server_running

    if _server_running:
        logger.warning("Web server already running.")
        return

    flask_app = create_flask_app(db, app_core)
    if flask_app is None:
        return

    _flask_app = flask_app
    host = "0.0.0.0" if bind_all else "127.0.0.1"

    import logging as _log
    _log.getLogger("werkzeug").setLevel(_log.WARNING)

    def _run():
        global _server_running
        _server_running = True
        try:
            flask_app.run(host=host, port=port, debug=False,
                          use_reloader=False, threaded=True)
        except Exception as exc:
            logger.error("Web server error: %s", exc)
        finally:
            _server_running = False

    _server_thread = threading.Thread(target=_run, daemon=True, name="FlaskThread")
    _server_thread.start()
    logger.info("Web server started on %s:%d", host, port)


def stop_server():
    global _server_running
    _server_running = False
    # Flask dev server doesn't support clean shutdown easily; daemon thread dies with app


def is_running() -> bool:
    return _server_running

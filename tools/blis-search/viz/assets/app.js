/* Interaction only. Every decision — which axes, which color, which captions,
 * which points survive decimation — was made in Python and arrives in the
 * plotspec. This file re-encodes and re-draws; it never decides. */
(function () {
  "use strict";

  var node = document.getElementById("plotspec");
  if (!node) { return; }
  var spec = JSON.parse(node.textContent);
  var geom = spec.geom;
  /* A card hero (one objective) is left exactly as Python baked it: redrawing it
   * as a one-dimensional strip of dots would be a worse first impression than
   * the card. The table and the panel still work. */
  var svg = spec.hero.form === "card" ? null : document.getElementById("hero");

  /* ---------------------------------------------------------- decoding */

  function decode(enc) {
    var out = [];
    for (var r = 0; r < enc.r.length; r++) {
      var row = {};
      for (var f = 0; f < enc.f.length; f++) {
        var field = enc.f[f];
        var cell = enc.r[r][f];
        row[field] = enc.s[field] ? enc.s[field][cell] : cell;
      }
      out.push(row);
    }
    return out;
  }

  var rows = decode(spec.data);
  var onFront = {};
  spec.data.front.forEach(function (i) { onFront[i] = true; });
  var sloOk = {};
  (spec.data.slo || []).forEach(function (i) { sloOk[i] = true; });
  var hasSlo = spec.data.slo !== null && spec.data.slo !== undefined;

  var state = {
    x: spec.axes.x.key ? "m:" + spec.axes.x.key : spec.fields.metrics[0],
    y: spec.axes.y ? "m:" + spec.axes.y.key : null,
    xlog: false,
    ylog: false,
    color: spec.color.key === "gpus" ? "gpus" :
           (spec.color.key ? guessField(spec.color.key) : "none"),
    show: "both",
    ranges: {},
    values: {},
    pinned: [],
    brush: null            // {x0, x1, y0, y1} in SVG user units, or null
  };

  function guessField(key) {
    return spec.fields.metrics.indexOf("m:" + key) >= 0 ? "m:" + key : "c:" + key;
  }

  function label(field) { return field.slice(2); }

  function valueOf(row, field, index) {
    if (field === "gpus") { return spec.gpus[index]; }
    return row[field];
  }

  /* ---------------------------------------------------------- filtering */

  function passes(index) {
    if (state.show === "front" && !onFront[index]) { return false; }
    if (state.show === "slo" && hasSlo && !sloOk[index]) { return false; }
    var row = rows[index];
    for (var field in state.ranges) {
      var span = state.ranges[field];
      var v = row[field];
      if (typeof v !== "number" || v < span[0] || v > span[1]) { return false; }
    }
    for (var knob in state.values) {
      var allowed = state.values[knob];
      if (allowed.length && allowed.indexOf(String(row[knob])) < 0) { return false; }
    }
    return true;
  }

  function visible() {
    var out = [];
    for (var i = 0; i < rows.length; i++) { if (passes(i)) { out.push(i); } }
    return out;
  }

  /* ------------------------------------------------------------ scales */

  function extent(field, indices) {
    var lo = Infinity, hi = -Infinity;
    indices.forEach(function (i) {
      var v = valueOf(rows[i], field, i);
      if (typeof v === "number") {
        if (v < lo) { lo = v; }
        if (v > hi) { hi = v; }
      }
    });
    if (lo === Infinity) { lo = 0; hi = 1; }
    if (lo === hi) { hi = lo + 1; }
    return [lo, hi];
  }

  function scale(field, indices, useLog, flip, from, to) {
    var span = extent(field, indices);
    var lo = span[0], hi = span[1];
    if (useLog) {
      lo = Math.log10(Math.max(lo, 1e-9));
      hi = Math.log10(Math.max(hi, 1e-9 * 10));
    }
    return function (value) {
      if (typeof value !== "number") { return null; }
      var v = useLog ? Math.log10(Math.max(value, 1e-9)) : value;
      var t = (v - lo) / (hi - lo || 1);
      if (flip) { t = 1 - t; }
      return from + t * (to - from);
    };
  }

  /* Better points sit right and low, exactly as screen_fractions() decides in
   * Python: a minimised x flips, a maximised y flips. */
  function direction(field) {
    for (var i = 0; i < spec.objectives.length; i++) {
      if ("m:" + spec.objectives[i].key === field) {
        return spec.objectives[i].direction;
      }
    }
    return "";
  }

  function colorFor(index) {
    if (state.color === "none") { return spec.colors.front; }
    var value = valueOf(rows[index], state.color, index);
    var palette = spec.color.palette;
    var domain = spec.fields.categorical[state.color];
    if (state.color === "gpus") { domain = spec.gpu_values.map(String); }
    if (domain) {
      var at = domain.indexOf(String(value));
      return palette[(at < 0 ? 0 : at) % palette.length];
    }
    var span = extent(state.color, visible());
    var t = (Number(value) - span[0]) / (span[1] - span[0] || 1);
    return ramp(t);
  }

  function ramp(t) {
    var stops = ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"];
    t = Math.max(0, Math.min(1, t));
    return stops[Math.min(stops.length - 1, Math.floor(t * stops.length))];
  }

  /* ------------------------------------------------------------ drawing */

  /* The SVG is rebuilt as markup rather than through createElementNS, so this
   * file contains no URL of any kind — not even the SVG namespace, which the
   * offline test would otherwise flag. Events are delegated from #hero. */
  function draw() {
    var shown = visible();
    /* The table and the counter exist even when the hero is a baked card, so
     * they are updated before the scatter guard. */
    setText("count", shown.length + " of " + rows.length + " configs shown");
    drawTable(shown);
    if (!svg) { return; }
    var xs = scale(state.x, shown, state.xlog, direction(state.x) === "minimize",
                   geom.ml, geom.w - geom.mr);
    var ys = state.y
      ? scale(state.y, shown, state.ylog, direction(state.y) === "maximize",
              geom.mt, geom.h - geom.mb)
      : function () { return (geom.mt + geom.h - geom.mb) / 2; };

    var parts = ['<rect x="' + geom.ml + '" y="' + geom.mt + '" width="' +
                 (geom.w - geom.ml - geom.mr) + '" height="' +
                 (geom.h - geom.mt - geom.mb) + '" fill="none" stroke="#e2e8f0"/>'];

    shown.forEach(function (index) {
      var cx = xs(valueOf(rows[index], state.x, index));
      var cy = ys(valueOf(rows[index], state.y, index));
      if (cx === null || cy === null) { return; }
      var front = onFront[index];
      var fill = (front || state.color !== "none")
        ? colorFor(index) : spec.colors.cloud;
      parts.push('<circle cx="' + cx.toFixed(1) + '" cy="' + cy.toFixed(1) +
                 '" r="' + (front ? 5 : 2.6) + '" fill="' + fill + '"' +
                 (front ? ' stroke="#fff" stroke-width="1"' : "") +
                 ' data-row="' + index + '" tabindex="0"/>');
    });

    if (spec.knee !== null && shown.indexOf(spec.knee) >= 0) {
      var kx = xs(valueOf(rows[spec.knee], state.x, spec.knee));
      var ky = ys(valueOf(rows[spec.knee], state.y, spec.knee));
      if (kx !== null && ky !== null) {
        parts.push('<path d="' + star(kx, ky, 9) + '" fill="' + spec.colors.knee +
                   '" stroke="#fff" stroke-width="1"/>');
      }
    }

    var inBrush = null;
    if (state.brush) {
      inBrush = {};
      shown.forEach(function (index) {
        var cx = xs(valueOf(rows[index], state.x, index));
        var cy = ys(valueOf(rows[index], state.y, index));
        if (cx === null || cy === null) { return; }
        if (cx >= state.brush.x0 && cx <= state.brush.x1 &&
            cy >= state.brush.y0 && cy <= state.brush.y1) { inBrush[index] = true; }
      });
      parts.push('<rect x="' + state.brush.x0.toFixed(1) + '" y="' +
                 state.brush.y0.toFixed(1) + '" width="' +
                 (state.brush.x1 - state.brush.x0).toFixed(1) + '" height="' +
                 (state.brush.y1 - state.brush.y0).toFixed(1) +
                 '" fill="#2b6cb0" fill-opacity="0.08" stroke="#2b6cb0" ' +
                 'stroke-dasharray="3 3"/>');
    }

    parts = parts.concat(axisMarkup(shown));
    svg.innerHTML = parts.join("");
    if (inBrush) {
      var picked = shown.filter(function (i) { return inBrush[i]; });
      setText("count", picked.length + " of " + shown.length +
                       " shown configs inside the brush");
      drawTable(picked);
    }
  }

  function star(cx, cy, r) {
    var pts = [];
    for (var i = 0; i < 10; i++) {
      var a = -Math.PI / 2 + i * Math.PI / 5;
      var radius = i % 2 === 0 ? r : r * 0.45;
      pts.push((cx + radius * Math.cos(a)).toFixed(1) + "," +
               (cy + radius * Math.sin(a)).toFixed(1));
    }
    return "M" + pts.join("L") + "Z";
  }

  function text(x, y, size, fill, anchor, extra, body) {
    return '<text x="' + x + '" y="' + y + '" font-size="' + size + '" fill="' +
      fill + '"' + (anchor ? ' text-anchor="' + anchor + '"' : "") +
      (extra || "") + ">" + escapeText(body) + "</text>";
  }

  function escapeText(value) {
    return String(value).replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function axisMarkup(shown) {
    var span = extent(state.x, shown);
    var flip = direction(state.x) === "minimize";
    var out = [
      text((geom.ml + geom.w - geom.mr) / 2, geom.h - geom.mb + 34, 12, "#1a202c",
           "middle", "", label(state.x) + (state.xlog ? " (log)" : "")),
      text(geom.ml, geom.h - geom.mb + 16, 11, "#4a5568", "", "",
           format(flip ? span[1] : span[0])),
      text(geom.w - geom.mr, geom.h - geom.mb + 16, 11, "#4a5568", "end", "",
           format(flip ? span[0] : span[1]))
    ];
    if (state.y) {
      var mid = (geom.mt + geom.h - geom.mb) / 2;
      out.push(text(14, mid, 12, "#1a202c", "middle",
                    ' transform="rotate(-90 14 ' + mid + ')"',
                    label(state.y) + (state.ylog ? " (log)" : "")));
    }
    return out;
  }

  function format(v) {
    if (typeof v !== "number") { return String(v); }
    if (Math.abs(v) >= 100) { return v.toFixed(0); }
    if (Math.abs(v) >= 10) { return v.toFixed(1); }
    return String(Math.round(v * 1000) / 1000);
  }

  /* -------------------------------------------------------------- table */

  var sortField = null, sortDown = true;

  function drawTable(shown) {
    var body = document.getElementById("configs");
    if (!body) { return; }
    var order = shown.slice();
    if (sortField) {
      order.sort(function (a, b) {
        var va = valueOf(rows[a], sortField, a), vb = valueOf(rows[b], sortField, b);
        if (va === vb) { return a - b; }
        var less = typeof va === "number" && typeof vb === "number"
          ? va < vb : String(va) < String(vb);
        return (less ? -1 : 1) * (sortDown ? -1 : 1);
      });
    }
    var head = ["#"].concat(spec.objectives.map(function (o) { return "m:" + o.key; }));
    var html = "<tr>";
    head.forEach(function (field) {
      html += '<th data-sort="' + field + '">' + (field === "#" ? "#" : label(field)) +
              "</th>";
    });
    html += "<th>config</th></tr>";
    order.forEach(function (index) {
      html += '<tr data-row="' + index + '">';
      html += "<td>" + index + (onFront[index] ? " ●" : "") + "</td>";
      spec.objectives.forEach(function (o) {
        html += '<td class="num">' + format(rows[index]["m:" + o.key]) + "</td>";
      });
      html += "<td>" + summarise(index) + "</td></tr>";
    });
    body.innerHTML = html;
    Array.prototype.forEach.call(body.querySelectorAll("th[data-sort]"),
      function (th) {
        th.addEventListener("click", function () {
          var field = th.getAttribute("data-sort");
          if (field === "#") { sortField = null; } else {
            sortDown = sortField === field ? !sortDown : true;
            sortField = field;
          }
          draw();
        });
      });
    Array.prototype.forEach.call(body.querySelectorAll("tr[data-row]"),
      function (tr) {
        tr.addEventListener("click", function () {
          pin(Number(tr.getAttribute("data-row")));
        });
      });
  }

  function summarise(index) {
    var row = rows[index];
    var bits = [];
    ["c:tp", "c:replicas", "c:routing_policy", "c:scheduler"].forEach(function (f) {
      if (row[f] !== undefined && row[f] !== null) {
        bits.push(label(f) + "=" + row[f]);
      }
    });
    return bits.join(" ");
  }

  /* --------------------------------------------------------- hover, pins */

  function hover(index) {
    var box = document.getElementById("hover");
    if (!box) { return; }
    var row = rows[index];
    var parts = ["<strong>#" + index + (onFront[index] ? " on front" : "") +
                 "</strong>"];
    spec.data.f.forEach(function (field) {
      if (row[field] === undefined) { return; }
      parts.push(label(field) + ": " + row[field]);
    });
    box.innerHTML = parts.join("<br>");
  }

  function pin(index) {
    var at = state.pinned.indexOf(index);
    if (at >= 0) { state.pinned.splice(at, 1); } else { state.pinned.push(index); }
    drawPins();
  }

  function drawPins() {
    var box = document.getElementById("pins");
    if (!box) { return; }
    if (state.pinned.length === 0) {
      box.innerHTML = '<p class="sub">Click a point or a table row to pin it. ' +
                      "Two or more pins show only the knobs that differ.</p>";
      return;
    }
    var knobs = spec.data.f.filter(function (f) { return f.indexOf("c:") === 0; });
    var differing = knobs.filter(function (f) {
      var first = String(rows[state.pinned[0]][f]);
      return state.pinned.some(function (i) { return String(rows[i][f]) !== first; });
    });
    var shownKnobs = state.pinned.length > 1 ? differing : knobs;
    var html = "<table><tr><th>knob</th>";
    state.pinned.forEach(function (i) { html += "<th>#" + i + "</th>"; });
    html += "</tr>";
    if (state.pinned.length > 1 && shownKnobs.length === 0) {
      html += '<tr><td colspan="' + (state.pinned.length + 1) +
              '">these configs differ in no knob</td></tr>';
    }
    shownKnobs.forEach(function (f) {
      html += "<tr><td>" + label(f) + "</td>";
      state.pinned.forEach(function (i) { html += "<td>" + rows[i][f] + "</td>"; });
      html += "</tr>";
    });
    html += "</table>";
    box.innerHTML = html;
  }

  /* ----------------------------------------------------------- controls */

  function setText(id, text) {
    var box = document.getElementById(id);
    if (box) { box.textContent = text; }
  }

  function on(id, event, handler) {
    var node = document.getElementById(id);
    if (node) { node.addEventListener(event, handler); }
  }

  function hook() {
    on("pick-x", "change", function (e) { state.x = e.target.value; draw(); });
    on("pick-y", "change", function (e) {
      state.y = e.target.value || null;
      draw();
    });
    on("log-x", "change", function (e) { state.xlog = e.target.checked; draw(); });
    on("log-y", "change", function (e) { state.ylog = e.target.checked; draw(); });
    on("pick-color", "change", function (e) { state.color = e.target.value; draw(); });
    on("pick-show", "change", function (e) { state.show = e.target.value; draw(); });
    on("reset", "click", function () {
      state.ranges = {};
      state.values = {};
      state.pinned = [];
      state.brush = null;
      Array.prototype.forEach.call(
        document.querySelectorAll("#filters input"), function (input) {
          if (input.type === "checkbox") { input.checked = true; }
          else { input.value = input.getAttribute("data-default"); }
        });
      drawPins();
      draw();
    });
    Array.prototype.forEach.call(
      document.querySelectorAll("#filters input[data-range]"), function (input) {
        input.addEventListener("input", function () {
          var field = input.getAttribute("data-range");
          var edge = input.getAttribute("data-edge");
          var span = state.ranges[field] || spec.fields.ranges[field].slice();
          span[edge === "lo" ? 0 : 1] = Number(input.value);
          state.ranges[field] = span;
          draw();
        });
      });
    Array.prototype.forEach.call(
      document.querySelectorAll("#filters input[data-knob]"), function (input) {
        input.addEventListener("change", function () {
          var knob = input.getAttribute("data-knob");
          var allowed = [];
          Array.prototype.forEach.call(
            document.querySelectorAll('#filters input[data-knob="' + knob + '"]'),
            function (other) {
              if (other.checked) { allowed.push(other.getAttribute("value")); }
            });
          state.values[knob] = allowed;
          draw();
        });
      });
    on("toggle", "click", function () {
      var panel = document.getElementById("advanced");
      if (!panel) { return; }
      var open = panel.hasAttribute("hidden");
      if (open) { panel.removeAttribute("hidden"); } else { panel.setAttribute("hidden", ""); }
      var button = document.getElementById("toggle");
      if (button) { button.textContent = open ? "Advanced ▾" : "Advanced ▸"; }
    });
  }

  /* Brush: drag a region and the table follows. Events are delegated from #hero
   * because draw() replaces its contents on every redraw. */
  var drag = null, dragged = false;

  function svgPoint(event) {
    var box = svg.getBoundingClientRect();
    var w = box.width || geom.w, h = box.height || geom.h;
    return {x: (event.clientX - box.left) / w * geom.w,
            y: (event.clientY - box.top) / h * geom.h};
  }

  function endDrag() {
    if (!drag) { return; }
    drag = null;
    if (state.brush &&
        (state.brush.x1 - state.brush.x0 < 4 || state.brush.y1 - state.brush.y0 < 4)) {
      state.brush = null;                 // a stray click is not a selection
      draw();
    }
  }

  if (svg) {
    svg.addEventListener("mouseover", function (event) {
      var row = event.target.getAttribute && event.target.getAttribute("data-row");
      if (row !== null && row !== undefined) { hover(Number(row)); }
    });
    svg.addEventListener("click", function (event) {
      if (dragged) { dragged = false; return; }   // the drag was not a pin
      var row = event.target.getAttribute && event.target.getAttribute("data-row");
      if (row !== null && row !== undefined) { pin(Number(row)); }
    });
    svg.addEventListener("mousedown", function (event) {
      drag = svgPoint(event);
      dragged = false;
      state.brush = null;
    });
    svg.addEventListener("mousemove", function (event) {
      if (!drag) { return; }
      var at = svgPoint(event);
      dragged = true;
      state.brush = {x0: Math.min(drag.x, at.x), x1: Math.max(drag.x, at.x),
                     y0: Math.min(drag.y, at.y), y1: Math.max(drag.y, at.y)};
      draw();
    });
    svg.addEventListener("mouseup", endDrag);
    svg.addEventListener("mouseleave", endDrag);
  }

  hook();
  drawPins();
  draw();
})();

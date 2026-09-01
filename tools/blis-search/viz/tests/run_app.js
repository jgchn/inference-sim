/* Smoke-run app.js against a generated page, with a DOM stub just large enough
 * for what the script touches. Not a browser: this proves the interaction layer
 * decodes the plotspec and produces circles and table rows for real data, which
 * is the part a Python test cannot reach.
 *
 * Usage: node run_app.js <page.html>   -> prints one JSON line, exits non-zero
 *                                        on any thrown error.
 */
"use strict";
const fs = require("fs");
const vm = require("vm");

const page = fs.readFileSync(process.argv[2], "utf8");

function section(id) {
  const open = page.indexOf(`id="${id}">`);
  if (open < 0) { return null; }
  const start = page.indexOf(">", open) + 1;
  return page.slice(start, page.indexOf("</script>", start));
}

const specText = section("plotspec");
if (!specText) { throw new Error("page carries no plotspec"); }

const scripts = page.split("<script>");
const appSource = scripts[scripts.length - 1].split("</script>")[0];

function node(id) {
  return {
    id: id,
    textContent: id === "plotspec" ? specText : "",
    innerHTML: "",
    checked: false,
    value: "",
    attrs: {},
    listeners: {},
    addEventListener(name, fn) { (this.listeners[name] = this.listeners[name] || []).push(fn); },
    getAttribute(name) { return name in this.attrs ? this.attrs[name] : null; },
    setAttribute(name, value) { this.attrs[name] = String(value); },
    hasAttribute(name) { return name in this.attrs; },
    removeAttribute(name) { delete this.attrs[name]; },
    querySelectorAll() { return []; },
  };
}

const nodes = {};
const document = {
  getElementById(id) {
    if (!(id in nodes)) { nodes[id] = node(id); }
    return nodes[id];
  },
  querySelectorAll() { return []; },
};

vm.runInNewContext(appSource, {document: document, console: console, Math: Math,
                              JSON: JSON, Array: Array, Number: Number,
                              String: String, Infinity: Infinity});

const hero = nodes.hero ? nodes.hero.innerHTML : "";
const configs = nodes.configs ? nodes.configs.innerHTML : "";
const pins = nodes.pins ? nodes.pins.innerHTML : "";
console.log(JSON.stringify({
  circles: (hero.match(/<circle /g) || []).length,
  stars: (hero.match(/<path d="M/g) || []).length,
  axis_labels: (hero.match(/<text /g) || []).length,
  table_rows: (configs.match(/<tr data-row=/g) || []).length,
  count_text: nodes.count ? nodes.count.textContent : "",
  pins_prompt: pins.indexOf("Click a point") >= 0,
}));

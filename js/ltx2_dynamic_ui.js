import { app } from "../../../scripts/app.js";

const THEMES = {
  a: {
    name: "JADE",
    rowOdd: "#1a1a1a", rowEven: "#202020",
    onColor: "#4CAF50", offColor: "#f44336",
    nameLoaded: "#dddddd", nameEmpty: "#444444",
    strColor: "#00FFCC", vColor: "#00FFCC", aColor: "#2196F3",
    ratioV: "#00FFCC99", ratioA: "#2196F399",
    arrowColor: "#555555",
    btnBg: "#00FFCC18", btnBorder: "#00FFCC55", btnText: "#00FFCC",
    divider: "#2a2a2a", wood: false,
  },
  b: {
    name: "NEON",
    rowOdd: "#0d0e1a", rowEven: "#111220",
    onColor: "#39ff7a", offColor: "#ff2d88",
    nameLoaded: "#9ba0cc", nameEmpty: "#2a2a3a",
    strColor: "#ff2d88", vColor: "#ff2d88", aColor: "#bf5fff",
    ratioV: "#ff2d8899", ratioA: "#bf5fff99",
    arrowColor: "#333355",
    btnBg: "#ff2d8818", btnBorder: "#ff2d8855", btnText: "#ff2d88",
    divider: "#1a1a2a", wood: false,
  },
  c: {
    name: "STUDIO",
    rowOdd: "#1e1a15", rowEven: "#231f19",
    onColor: "#7ecb6f", offColor: "#c05050",
    nameLoaded: "#c8b898", nameEmpty: "#4a4030",
    strColor: "#F5A623", vColor: "#F5A623", aColor: "#e07b39",
    ratioV: "#F5A62399", ratioA: "#e07b3999",
    arrowColor: "#4a4030",
    btnBg: "#F5A62318", btnBorder: "#F5A62355", btnText: "#F5A623",
    divider: "#3a3028", wood: false,
  },
  d: {
    name: "CHROME",
    rowOdd: "#eaeaea", rowEven: "#e0e0e0",
    onColor: "#1a7c38", offColor: "#b03030",
    nameLoaded: "#222222", nameEmpty: "#bbbbbb",
    strColor: "#111111", vColor: "#111111", aColor: "#1414cc",
    ratioV: "#11111199", ratioA: "#1414cc99",
    arrowColor: "#aaaaaa",
    btnBg: "#00000018", btnBorder: "#00000055", btnText: "#111111",
    divider: "#cccccc", wood: false,
  },
  e: {
    name: "OLED",
    rowOdd: "#000000", rowEven: "#080808",
    onColor: "#00ff41", offColor: "#ff0033",
    nameLoaded: "#ffffff", nameEmpty: "#333333",
    strColor: "#ffffff", vColor: "#00e5ff", aColor: "#ff6b35",
    ratioV: "#00e5ff99", ratioA: "#ff6b3599",
    arrowColor: "#333333",
    btnBg: "#ffffff08", btnBorder: "#ffffff33", btnText: "#ffffff",
    divider: "#111111", wood: false,
  },
  f: {
    name: "WOOD",
    rowOdd: "#2c1f0e", rowEven: "#321f0a",
    onColor: "#a8d878", offColor: "#d45a38",
    nameLoaded: "#f0d8a0", nameEmpty: "#6a5030",
    strColor: "#f0c060", vColor: "#f0c060", aColor: "#c87830",
    ratioV: "#f0c06099", ratioA: "#c8783099",
    arrowColor: "#6a5030",
    btnBg: "#f0c06018", btnBorder: "#f0c06055", btnText: "#f0c060",
    divider: "#4a3018", wood: true,
  },
};

const THEME_ORDER = ["a", "b", "c", "d", "e", "f"];
const keyCache = {};
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const bump = (v, d) => Math.round(clamp((v || 1.0) + d, 0.0, 2.0) * 100) / 100;
const bumpS = (v, d) => Math.round(clamp((v || 1.0) + d, -2.0, 2.0) * 100) / 100;

async function getCurrentLoraList(nodeData) {
  const fallback = nodeData?.input?.hidden?.available_loras?.[0] || ["None"];

  try {
    const response = await fetch("/object_info/DaSiWa_LTX2LoraLoader", { cache: "no-store" });
    if (!response.ok) return fallback;

    const info = await response.json();
    const refreshed = info?.DaSiWa_LTX2LoraLoader?.input?.hidden?.available_loras?.[0]
      || info?.input?.hidden?.available_loras?.[0];

    return Array.isArray(refreshed) && refreshed.length ? refreshed : fallback;
  } catch (error) {
    console.warn("[DaSiWa LTX2] Failed to refresh LoRA list from ComfyUI object_info:", error);
    return fallback;
  }
}

const CONTROL_DESCRIPTIONS = {
  theme: "Cycle the visual theme of this LTX-2 LoRA stacker.",
  add: "Add one LoRA slot to the stack.",
  remove: "Remove the last LoRA slot from the stack.",
  enabled: "Enable or disable this LoRA slot.",
  lora: "Choose the LoRA file for this slot.",
  str: "Master LoRA strength. This is multiplied by the video and audio multipliers. Range: -2.0 to 2.0.",
  video: "Video branch multiplier. Effective video strength = STR x V. Range: 0.0 to 2.0.",
  audio: "Audio branch multiplier. Effective audio strength = STR x A. Range: 0.0 to 2.0.",
  keys: "Detected LoRA key counts for video and audio branches.",
};

function setCanvasTooltip(text) {
  const canvas = app?.canvas?.canvas;
  if (canvas && canvas.title !== text) canvas.title = text || "";
}

function drawWoodRow(ctx, x, y, w, h, seed) {
  ctx.fillStyle = seed % 2 === 0 ? "#2c1f0e" : "#321f0a";
  ctx.fillRect(x, y, w, h);
  ctx.save();
  ctx.globalAlpha = 0.07;
  for (let i = 0; i < 12; i++) {
    const lx = x + ((seed * 37 + i * 61) % w);
    const curve = ((seed * 13 + i * 7) % 6) - 3;
    ctx.strokeStyle = i % 3 === 0 ? "#ffcc66" : "#8B5E20";
    ctx.lineWidth = i % 4 === 0 ? 1.5 : 0.5;
    ctx.beginPath();
    ctx.moveTo(lx, y);
    ctx.bezierCurveTo(lx + curve, y + h * 0.3, lx - curve, y + h * 0.7, lx + curve * 0.5, y + h);
    ctx.stroke();
  }
  ctx.restore();
}

function drawPill(ctx, t, x, y, w, h, label, val, color, zeroed) {
  ctx.fillStyle = color + (zeroed ? "12" : "1c");
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, 2);
  ctx.fill();
  ctx.strokeStyle = color + (zeroed ? "25" : "60");
  ctx.lineWidth = 0.5;
  ctx.stroke();

  // Left arrow
  ctx.fillStyle = zeroed ? t.arrowColor + "55" : t.arrowColor + "dd";
  ctx.font = "6px 'Courier New',monospace";
  ctx.textAlign = "left";
  ctx.fillText("<", x + 2, y + h - 2);

  // Right arrow
  ctx.textAlign = "right";
  ctx.fillText(">", x + w - 2, y + h - 2);

  // Label top-left
  ctx.fillStyle = color + (zeroed ? "55" : "88");
  ctx.font = "5px 'Courier New',monospace";
  ctx.textAlign = "left";
  ctx.fillText(label, x + 2, y + 4);

  // Value center
  ctx.fillStyle = zeroed ? color + "66" : color;
  ctx.font = "bold 7px 'Courier New',monospace";
  ctx.textAlign = "center";
  ctx.fillText(val.toFixed(2), x + w / 2, y + h / 2 + 2);
  ctx.textAlign = "left";
}

app.registerExtension({
  name: "DaSiWa.LTX2LoraLoader",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name !== "DaSiWa_LTX2LoraLoader") return;

    // Compact layout constants
    const ROW_H = 28;
    const BTN_H = 16;
    const BTN_Y = 40;
    const ROW_START = BTN_Y + BTN_H + 3;
    const calcHeight = n => ROW_START + n * ROW_H + 2;

    nodeType.prototype.onNodeCreated = function () {
      this.properties = this.properties || {};
      if (!this.properties.stack_data) {
        this.properties.stack_data = JSON.stringify(
          [{ on: true, lora: "None", str: 1.0, vs: 1.0, as: 1.0 }]
        );
      }
      if (!this.properties.theme) this.properties.theme = "a";
      const rows = JSON.parse(this.properties.stack_data);
      this.size = [720, calcHeight(rows.length)];

      const w = this.addWidget("text", "stack_data", this.properties.stack_data, () => {});
      w.draw = () => {};
      w.computeSize = () => [0, -4];
      this.widgets = [w];
    };

    nodeType.prototype.getExtraMenuOptions = function () {
      return [];
    };

    const sync = node => {
      const w = node.widgets.find(w => w.name === "stack_data");
      if (w) w.value = node.properties.stack_data;
    };

    const getTooltipAt = (node, local_pos) => {
      if (node.flags.collapsed) return "";
      let data = [];
      try {
        data = JSON.parse(node.properties.stack_data);
      } catch {
        return "";
      }

      const [x, y] = local_pos;
      const W = node.size[0];
      const s = W / 1000;
      const BTN_H = 16;
      const BTN_Y = 40;
      const ROW_START = BTN_Y + BTN_H + 3;
      const ROW_H = 28;
      const btnW = 110;
      const btnX = (W - btnW) / 2;
      const plusX = btnX + btnW + 4;
      const minusX = plusX + BTN_H + 2;

      if (y > BTN_Y && y < BTN_Y + BTN_H) {
        if (x > btnX && x < btnX + btnW) return CONTROL_DESCRIPTIONS.theme;
        if (x > plusX && x < plusX + BTN_H) return CONTROL_DESCRIPTIONS.add;
        if (data.length > 1 && x > minusX && x < minusX + BTN_H) return CONTROL_DESCRIPTIONS.remove;
      }

      const C = {
        onX: 8 * s, onW: 50 * s,
        nmX: 62 * s, nmW: 480 * s,
        stX: 548 * s, stW: 80 * s,
        vX: 635 * s, vW: 60 * s,
        aX: 702 * s, aW: 60 * s,
        rX: 770 * s, rW: W - 770 * s - 8,
      };

      for (let i = 0; i < data.length; i++) {
        const ry = ROW_START + i * ROW_H;
        if (y < ry || y > ry + ROW_H) continue;
        if (x > C.onX && x < C.onX + C.onW) return CONTROL_DESCRIPTIONS.enabled;
        if (x > C.nmX && x < C.nmX + C.nmW) return CONTROL_DESCRIPTIONS.lora;
        if (x > C.stX && x < C.stX + C.stW) return CONTROL_DESCRIPTIONS.str;
        if (x > C.vX && x < C.vX + C.vW) return CONTROL_DESCRIPTIONS.video;
        if (x > C.aX && x < C.aX + C.aW) return CONTROL_DESCRIPTIONS.audio;
        if (x > C.rX && x < C.rX + C.rW) return CONTROL_DESCRIPTIONS.keys;
      }

      return "";
    };

    nodeType.prototype.onMouseMove = function (_e, local_pos) {
      setCanvasTooltip(getTooltipAt(this, local_pos));
      return false;
    };

    nodeType.prototype.onMouseLeave = function () {
      setCanvasTooltip("");
      return false;
    };

    nodeType.prototype.onDrawForeground = function (ctx) {
      if (this.flags.collapsed) return;
      const data = JSON.parse(this.properties.stack_data);
      const t = THEMES[this.properties.theme || "a"];
      const W = this.size[0];
      const H = this.size[1];
      sync(this);

      const s = W / 1000;

      const BTN_H = 16;
      const BTN_Y = 40;
      const ROW_START = BTN_Y + BTN_H + 3;
      const ROW_H = 28;

      const C = {
        onX: 8 * s, onW: 50 * s,
        nmX: 62 * s, nmW: 480 * s,
        stX: 548 * s, stW: 80 * s,
        vX: 635 * s, vW: 60 * s,
        aX: 702 * s, aW: 60 * s,
        rX: 770 * s, rW: W - 770 * s - 8,
      };

      // Theme button
      const btnW = 110, btnX = (W - btnW) / 2;
      ctx.fillStyle = t.btnBg;
      ctx.beginPath();
      ctx.roundRect(btnX, BTN_Y, btnW, BTN_H, 3);
      ctx.fill();
      ctx.strokeStyle = t.btnBorder;
      ctx.lineWidth = 0.5;
      ctx.stroke();
      ctx.fillStyle = t.btnText;
      ctx.font = "bold 7px 'Courier New',monospace";
      ctx.textAlign = "center";
      ctx.fillText(`⬡ THEME: ${t.name} ▶`, btnX + btnW / 2, BTN_Y + 10);
      ctx.textAlign = "left";

      // + button
      const plusX = btnX + btnW + 4;
      ctx.fillStyle = t.btnBg;
      ctx.beginPath();
      ctx.roundRect(plusX, BTN_Y, BTN_H, BTN_H, 3);
      ctx.fill();
      ctx.strokeStyle = t.btnBorder;
      ctx.lineWidth = 0.5;
      ctx.stroke();
      ctx.fillStyle = t.btnText;
      ctx.font = "bold 11px 'Courier New',monospace";
      ctx.textAlign = "center";
      ctx.fillText("+", plusX + BTN_H / 2, BTN_Y + 11);
      ctx.textAlign = "left";

      // - button
      if (data.length > 1) {
        const minusX = plusX + BTN_H + 2;
        ctx.fillStyle = t.btnBg;
        ctx.beginPath();
        ctx.roundRect(minusX, BTN_Y, BTN_H, BTN_H, 3);
        ctx.fill();
        ctx.strokeStyle = "#f4433655";
        ctx.lineWidth = 0.5;
        ctx.stroke();
        ctx.fillStyle = "#f44336cc";
        ctx.font = "bold 11px 'Courier New',monospace";
        ctx.textAlign = "center";
        ctx.fillText("−", minusX + BTN_H / 2, BTN_Y + 11);
        ctx.textAlign = "left";
      }

      // Divider
      ctx.strokeStyle = t.divider;
      ctx.lineWidth = 0.3;
      ctx.beginPath();
      ctx.moveTo(4, ROW_START - 1);
      ctx.lineTo(W - 4, ROW_START - 1);
      ctx.stroke();

      // Rows
      for (let i = 0; i < data.length; i++) {
        const ry = ROW_START + i * ROW_H;
        const row = data[i];
        const str = row.str ?? 1.0;
        const vs = row.vs ?? 1.0;
        const as_ = row.as ?? 1.0;
        const pH = Math.max(ROW_H - 4, 14);
        const pY = ry + (ROW_H - pH) / 2;

        if (t.wood) {
          drawWoodRow(ctx, 4, ry, W - 8, ROW_H - 1, i);
        } else {
          ctx.fillStyle = i % 2 === 0 ? t.rowOdd : t.rowEven;
          ctx.fillRect(4, ry, W - 8, ROW_H - 1);
        }

        ctx.font = "bold 8px 'Courier New',monospace";
        ctx.fillStyle = row.on ? t.onColor : t.offColor;
        ctx.fillText(row.on ? "✔" : "✖", C.onX, ry + ROW_H / 2 + 2);

        ctx.font = "8px 'Courier New',monospace";
        ctx.fillStyle = row.lora === "None" ? t.nameEmpty : t.nameLoaded;
        const nm = row.lora === "None" ? "None"
          : row.lora.split(/[\\\/]/).pop().replace(/\.safetensors$/i, "").substring(0, 48);
        ctx.fillText(nm, C.nmX, ry + ROW_H / 2 + 2);

        drawPill(ctx, t, C.stX, pY, C.stW, pH, "STR", str, t.strColor, str === 0);
        drawPill(ctx, t, C.vX, pY, C.vW, pH, "V", vs, t.vColor, vs === 0);
        drawPill(ctx, t, C.aX, pY, C.aW, pH, "A", as_, t.aColor, as_ === 0);

        if (row.lora !== "None" && keyCache[row.lora]) {
          const { v, a } = keyCache[row.lora];
          ctx.font = "6px 'Courier New',monospace";
          ctx.fillStyle = t.ratioV;
          ctx.fillText(`V:${v}`, C.rX, ry + ROW_H / 2 - 2);
          ctx.fillStyle = t.ratioA;
          ctx.fillText(`A:${a}`, C.rX, ry + ROW_H / 2 + 5);
        } else if (row.lora !== "None") {
          ctx.font = "6px 'Courier New',monospace";
          ctx.fillStyle = t.nameEmpty;
          ctx.textAlign = "center";
          ctx.fillText("…", C.rX + C.rW / 2, ry + ROW_H / 2 + 1);
          ctx.textAlign = "left";
          if (!keyCache[row.lora + "_p"]) {
            keyCache[row.lora + "_p"] = true;
            fetch(`/dasiwa/ltx2/keycounts?lora=${encodeURIComponent(row.lora)}`)
              .then(r => r.json())
              .then(d => { keyCache[row.lora] = { v: d.v, a: d.a }; app.graph.setDirtyCanvas(true); })
              .catch(() => { keyCache[row.lora] = { v: "?", a: "?" }; });
          }
        }
      }
    };

    nodeType.prototype.onMouseDown = function (e, local_pos) {
      if (this.flags.collapsed) return;
      const data = JSON.parse(this.properties.stack_data);
      const [x, y] = local_pos;
      const W = this.size[0];
      const H = this.size[1];
      const s = W / 1000;

      const BTN_H = 16;
      const BTN_Y = 40;
      const ROW_START = BTN_Y + BTN_H + 3;
      const ROW_H = 28;

      const btnW = 110, btnX = (W - btnW) / 2;
      const plusX = btnX + btnW + 4;

      // Theme cycle
      if (y > BTN_Y && y < BTN_Y + BTN_H && x > btnX && x < btnX + btnW) {
        const idx = THEME_ORDER.indexOf(this.properties.theme || "a");
        this.properties.theme = THEME_ORDER[(idx + 1) % THEME_ORDER.length];
        this.setDirtyCanvas(true);
        return true;
      }

      // Add row
      if (y > BTN_Y && y < BTN_Y + BTN_H && x > plusX && x < plusX + BTN_H) {
        data.push({ on: true, lora: "None", str: 1.0, vs: 1.0, as: 1.0 });
        this.properties.stack_data = JSON.stringify(data);
        this.size[1] = 40 + 16 + 3 + data.length * 28 + 2;
        sync(this);
        this.setDirtyCanvas(true);
        return true;
      }

      // Remove last row
      const minusX = plusX + BTN_H + 2;
      if (data.length > 1 && y > BTN_Y && y < BTN_Y + BTN_H && x > minusX && x < minusX + BTN_H) {
        data.pop();
        this.properties.stack_data = JSON.stringify(data);
        this.size[1] = 40 + 16 + 3 + data.length * 28 + 2;
        sync(this);
        this.setDirtyCanvas(true);
        return true;
      }

      const C = {
        onX: 8 * s, onW: 50 * s,
          nmX: 62 * s, nmW: 480 * s,
          stX: 548 * s, stW: 80 * s,
          vX: 635 * s, vW: 60 * s,
          aX: 702 * s, aW: 60 * s,
      };

      for (let i = 0; i < data.length; i++) {
        const ry = ROW_START + i * ROW_H;
        if (y < ry || y > ry + ROW_H) continue;

        if (x > C.onX && x < C.onX + C.onW) {
          data[i].on = !data[i].on;
        } else if (x > C.nmX && x < C.nmX + C.nmW) {
          getCurrentLoraList(nodeData).then(loraList => {
            const menu = new LiteGraph.ContextMenu(loraList, {
              event: e, scale: 1.2,
              callback: v => {
                data[i].lora = v;
                this.properties.stack_data = JSON.stringify(data);
                sync(this);
                this.setDirtyCanvas(true);
              }
            });
            const sw = document.createElement("div");
            sw.style = "padding:5px;background:#333;border-bottom:1px solid #555;";
            const inp = document.createElement("input");
            inp.style = "width:100%;background:#222;color:#00FFCC;border:1px solid #444;padding:4px;";
            inp.placeholder = "Search LoRAs...";
            sw.appendChild(inp);
            menu.root.prepend(sw);
            setTimeout(() => inp.focus(), 10);
            inp.addEventListener("input", ev => {
              const term = ev.target.value.toLowerCase();
              menu.root.querySelectorAll(".litemenu-entry").forEach(el => {
                el.style.display = el.textContent.toLowerCase().includes(term) ? "block" : "none";
              });
            });
          });
          return true;
        } else if (x > C.stX && x < C.stX + C.stW) {
          if (x < C.stX + 14 * s) data[i].str = bumpS(data[i].str, -0.05);
          else if (x > C.stX + C.stW - 14 * s) data[i].str = bumpS(data[i].str, 0.05);
          else {
            const v = prompt("LoRA Strength:", data[i].str ?? 1);
            if (v !== null) data[i].str = clamp(parseFloat(v) || 0, -2, 2);
          }
        } else if (x > C.vX && x < C.vX + C.vW) {
          if (x < C.vX + 10 * s) data[i].vs = bump(data[i].vs, -0.05);
          else if (x > C.vX + C.vW - 10 * s) data[i].vs = bump(data[i].vs, 0.05);
          else {
            const v = prompt("V Multiplier (0–2):", data[i].vs ?? 1);
            if (v !== null) data[i].vs = clamp(parseFloat(v) || 0, 0, 2);
          }
        } else if (x > C.aX && x < C.aX + C.aW) {
          if (x < C.aX + 10 * s) data[i].as = bump(data[i].as, -0.05);
          else if (x > C.aX + C.aW - 10 * s) data[i].as = bump(data[i].as, 0.05);
          else {
            const v = prompt("A Multiplier (0–2):", data[i].as ?? 1);
            if (v !== null) data[i].as = clamp(parseFloat(v) || 0, 0, 2);
          }
        }

        this.properties.stack_data = JSON.stringify(data);
        sync(this);
        this.setDirtyCanvas(true);
        return true;
      }
    };
  }
});

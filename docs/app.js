(function () {
  "use strict";

  const D = window.MOCK_DATA;
  const FEES = D.fees;
  const LOT = 1000;
  const LIMIT = 0.10;

  const $ = (s) => document.querySelector(s);

  const state = {
    stocks: {},
    live: {},
    base: {},
    suffix: {},
    source: "sim",
    account: null,
    selected: null,
    side: "buy",
    lastPortfolio: null,
    realStatus: { ok: 0, fail: 0, lastOk: null, everOk: false },
  };

  const YAHOO = {
    base: "https://query1.finance.yahoo.com/v8/finance/chart/",
    spark: "https://query1.finance.yahoo.com/v7/finance/spark?symbols=",
    period: 30000,
    batch: 6,
    stagger: 400,
  };

  const RELAY = "https://r.jina.ai/";
  const RATE = {
    period: 20000, floor: 10000, ceil: 300000, floorClosed: 60000,
    consecFail: 0, sticky: 0, sweep: 0,
    lastRelayAt: 0, relayGap: 3500,
    cooldownUntil: 0,
  };
  const PROVIDERS = [
    { id: "yahoo", label: "Yahoo 直連", batch: 1 },
    { id: "mis", label: "TWSE 官方即時 (中繼)", batch: 150 },
    { id: "spark", label: "Yahoo 批次 (中繼)", batch: 8 },
  ];
  const otcCodes = new Set();

  const STORE_KEY = "tw_paper_trading_v1";

  function newAccount() {
    const capital = Number(prompt("請輸入虛擬初始資金 (NT$):", "1000000"));
    const cap = isNaN(capital) || capital < 10000 ? 1000000 : capital;
    return {
      capital: cap, cash: cap,
      positions: {}, trades: [],
      navHistory: [{ t: Date.now(), v: cap }],
      source: "sim",
    };
  }

  function load() {
    try {
      const raw = localStorage.getItem(STORE_KEY);
      if (raw) {
        state.account = JSON.parse(raw);
        if (!state.account.source) state.account.source = "sim";
        state.source = state.account.source;
        return;
      }
    } catch (e) { /* ignore */ }
    state.account = newAccount();
    save();
  }

  function save() { localStorage.setItem(STORE_KEY, JSON.stringify(state.account)); }

  const Api = {
    async getStocks() { return D.stocks; },
    async fetchRealQuote(code) {
      const sym = code + (state.suffix[code] || ".TW");
      const r = await fetch(YAHOO.base + sym + "?interval=1d&range=1d");
      if (!r.ok) {
        if (r.status === 404 && !state.suffix[code]) {
          state.suffix[code] = ".TWO";
          return Api.fetchRealQuote(code);
        }
        throw new Error("HTTP " + r.status);
      }
      const j = await r.json();
      const m = j.chart.result[0].meta;
      const price = m.regularMarketPrice;
      const prev = m.previousClose != null ? m.previousClose : m.chartPreviousClose;
      if (price == null) throw new Error("no price");
      return { price, prev, time: (m.regularMarketTime || 0) * 1000 };
    },
    async fetchRealHistory(code) {
      const sym = code + (state.suffix[code] || ".TW");
      const url = YAHOO.base + sym + "?interval=1d&range=3mo";
      let j = null;
      try {
        const r = await fetch(url);
        if (r.ok) j = await r.json();
      } catch (e) { /* 直連失敗 → 中繼 */ }
      if (!j) j = await relayJson(url);
      const res = j.chart.result[0];
      const ts = res.timestamp || [], cl = res.indicators.quote[0].close || [];
      const out = [];
      for (let i = 0; i < ts.length; i++) {
        if (cl[i] != null) out.push([ts[i] * 1000, Math.round(cl[i] * 100) / 100]);
      }
      return out;
    },
    getModelInfo() { return D.modelInfo; },
  };

  async function relayGate() {
    const wait = RATE.lastRelayAt + RATE.relayGap - Date.now();
    if (wait > 0) await new Promise((r) => setTimeout(r, wait));
    RATE.lastRelayAt = Date.now();
  }

  async function relayJson(url) {
    await relayGate();
    const r = await fetch(RELAY + url);
    if (!r.ok) throw new Error("relay HTTP " + r.status);
    const txt = await r.text();
    return JSON.parse(txt.slice(txt.indexOf("{"), txt.lastIndexOf("}") + 1));
  }

  function chunk(arr, n) {
    const out = [];
    for (let i = 0; i < arr.length; i += n) out.push(arr.slice(i, i + n));
    return out;
  }

  async function viaYahoo(codes) {
    let ok = 0;
    for (const g of chunk(codes, YAHOO.batch)) {
      const rs = await Promise.all(g.map((c) =>
        Api.fetchRealQuote(c).then((q) => {
          state.live[c] = q.price; state.base[c] = q.prev; ok++;
        }).catch(() => {})));
      await new Promise((r) => setTimeout(r, YAHOO.stagger));
    }
    return ok;
  }

  async function viaMis(codes) {
    let ok = 0;
    for (const g of chunk(codes, PROVIDERS[1].batch)) {
      const ex = g.map((c) => (otcCodes.has(c) ? "otc_" : "tse_") + c + ".tw").join("|");
      const url = "https://mis.twse.com.tw/stock/api/getStockInfo.jsp?ex_ch=" +
                  encodeURIComponent(ex) + "&json=1&delay=0";
      const d = await relayJson(url);
      const got = new Set();
      (d.msgArray || []).forEach((r) => {
        const c = String(r["@"]).split(".")[0];
        const pz = parseFloat(r.pz), y = parseFloat(r.y);
        if (pz > 0) {
          state.live[c] = pz;
          if (y > 0) state.base[c] = y;
          state.realStatus.quoteTime = r.tlong || null;
          got.add(c); ok++;
        }
      });
      g.forEach((c) => { if (!got.has(c) && !otcCodes.has(c)) otcCodes.add(c); });
    }
    return ok;
  }

  async function viaSpark(codes) {
    let ok = 0;
    for (const g of chunk(codes, PROVIDERS[2].batch)) {
      const syms = g.map((c) => c + (state.suffix[c] || ".TW")).join(",");
      const url = YAHOO.spark + syms + "&range=1d&interval=5m";
      let d = null;
      try {
        const r = await fetch(url);
        if (r.ok) d = await r.json();
      } catch (e) { /* 直連失敗 → 中繼 */ }
      if (!d) d = await relayJson(url);
      (d.spark && d.spark.result ? d.spark.result : []).forEach((r) => {
        const m = r.response[0].meta;
        const c = String(m.symbol).split(".")[0];
        if (m.regularMarketPrice != null) {
          state.live[c] = m.regularMarketPrice;
          state.base[c] = m.previousClose != null ? m.previousClose : state.base[c];
          ok++;
        }
      });
    }
    return ok;
  }

  function baseClose(c) {
    return state.base[c] != null ? state.base[c] : state.stocks[c].lastClose;
  }

  function fmtMoney(v) {
    return "NT$" + Math.round(v).toLocaleString("zh-TW");
  }
  function fmtPct(v, digits = 2) {
    return (v >= 0 ? "+" : "") + (v * 100).toFixed(digits) + "%";
  }
  function pctClass(v) { return v > 0 ? "up" : v < 0 ? "down" : "flat"; }
  function changePct(c) { return livePrice(c) / baseClose(c) - 1; }

  function twMarketPhase() {
    const now = new Date();
    const day = now.getDay();
    if (day === 0 || day === 6) return "週末休市";
    const m = now.getHours() * 60 + now.getMinutes();
    if (m >= 9 * 60 && m <= 13 * 60 + 30) return "盤中";
    return "已收盤";
  }

  function renderSrcStatus() {
    const el = $("#srcStatus");
    if (state.source === "sim") {
      el.className = "src-status ok";
      el.textContent = "模型導向虛擬市場 · 每 3 秒推進 · 導向 " +
        Math.round(MarketSim.influence * 100) + "% · 震盪 " + MarketSim.volMult + "×";
      return;
    }
    const rs = state.realStatus;
    if (!rs.everOk) {
      el.className = "src-status err";
      el.textContent = "真實報價連線失敗 · 保持最後價格" +
        (Date.now() < RATE.cooldownUntil
          ? " · 冷卻 " + Math.ceil((RATE.cooldownUntil - Date.now()) / 1000) + "s"
          : "");
    } else {
      el.className = "src-status ok";
      const qt = rs.quoteTime ? new Date(rs.quoteTime).toLocaleTimeString("zh-TW") : null;
      el.textContent = (rs.provider || "Yahoo 直連") + " · " + twMarketPhase() +
        " · 每 " + Math.round(RATE.period / 1000) + "s · " + (rs.scope || "") +
        " · 報價時刻 " + (qt || (rs.lastOk ? new Date(rs.lastOk).toLocaleTimeString("zh-TW") : "—"));
      el.title = "成功 " + rs.ok + " · 失敗 " + rs.fail;
    }
  }

  const MarketSim = {
    day: 1, tickIndex: 0, closedUntil: 0,
    dayMinutes: 10, influence: 1.0, volMult: 1.0,
    vols: {}, industries: {},
    SIGMA_M: 0.008, SIGMA_G: 0.006,

    init() {
      D.stocks.forEach((s) => {
        const h = s.history.map((x) => x[1]);
        const rets = [];
        for (let i = 1; i < h.length; i++) rets.push(Math.log(h[i] / h[i - 1]));
        const n = rets.length || 1;
        const mean = rets.reduce((a, b) => a + b, 0) / n;
        this.vols[s.code] = Math.max(0.005, Math.sqrt(
          rets.reduce((a, r) => a + (r - mean) * (r - mean), 0) / Math.max(n - 1, 1)));
        const ind = s.industry || "其他";
        (this.industries[ind] = this.industries[ind] || []).push(s.code);
      });
    },

    ticksTotal() { return Math.max(2, Math.round(this.dayMinutes * 60 / 3)); },
    progress() { return Math.min(1, this.tickIndex / this.ticksTotal()); },

    vTime() {
      const mins = Math.round(this.progress() * 270);
      const h = 9 + Math.floor(mins / 60), m = mins % 60;
      return String(h).padStart(2, "0") + ":" + String(m).padStart(2, "0");
    },

    gauss() {
      let u = 0, v = 0;
      while (u === 0) u = Math.random();
      while (v === 0) v = Math.random();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    },

    start(fresh) {
      if (fresh) this.day = 1;
      this.tickIndex = 0;
      this.closedUntil = 0;
      D.stocks.forEach((s) => { state.live[s.code] = baseClose(s.code); });
      this.renderBar();
      refreshPrices();
    },

    renderBar() {
      $("#simDayLabel").textContent = "模擬日 #" + this.day;
      $("#simVTime").textContent = "虛擬 " + this.vTime() +
        (Date.now() < this.closedUntil ? " · 收盤結算" : "");
      $("#simInfluenceVal").textContent = Math.round(this.influence * 100) + "%";
    },

    step() {
      if (Date.now() < this.closedUntil) return;
      const T = this.ticksTotal();
      if (this.tickIndex >= T) { this.closeDay(); return; }
      const sigT = this.volMult / Math.sqrt(T);
      const m = this.gauss() * this.SIGMA_M * sigT;
      const indStep = {};
      for (const g in this.industries) {
        indStep[g] = this.gauss() * this.SIGMA_G * sigT;
      }

      D.stocks.forEach((s) => {
        const c = s.code, base = baseClose(c);
        const cur = state.live[c] != null ? state.live[c] : base;
        const idio = this.gauss() * this.vols[c] * sigT;
        let price = cur * (1 + m + indStep[s.industry || "其他"] + idio);
        if (this.influence > 0) {
          const target = base * (1 + s.predReturn * this.influence);
          const remain = T - this.tickIndex;
          const w = Math.min(1, 1 / remain);
          price += (target - price) * w;
        }
        const hi = base * (1 + LIMIT), lo = base * (1 - LIMIT);
        price = Math.min(hi, Math.max(lo, price));
        state.live[c] = Math.round(price * 100) / 100;
      });
      this.tickIndex++;
      this.renderBar();
      if (this.tickIndex >= T) this.closeDay();
    },

    closeDay() {
      D.stocks.forEach((s) => {
        const c = s.code, base = baseClose(c);
        if (this.influence > 0) {
          const target = base * (1 + s.predReturn * this.influence);
          const hi = base * (1 + LIMIT), lo = base * (1 - LIMIT);
          state.live[c] = Math.round(Math.min(hi, Math.max(lo, target)) * 100) / 100;
        }
      });
      snapshotNav();
      const avg = D.stocks.reduce((a, s) => a + s.predReturn * this.influence, 0) / D.stocks.length;
      toast("模擬日 #" + this.day + " 收盤 · 模型預期日報酬 " + fmtPct(avg) + " · 5 秒後重播");
      this.day++;
      this.tickIndex = 0;
      this.closedUntil = Date.now() + 5000;
      this.renderBar();
      refreshPrices();
    },
  };

  function tierCodes() {
    const all = D.stocks.map((s) => s.code);
    const holdings = Object.keys(state.account.positions);
    const top10 = D.stocks.filter((s) => s.predReturn > 0).slice(0, 10).map((s) => s.code);
    const tierA = [...new Set([...holdings, ...(state.selected ? [state.selected] : []), ...top10])]
      .filter((c) => state.stocks[c]).slice(0, 20);
    RATE.sweep++;
    if (RATE.sweep % 10 === 0) return { codes: all, label: "全部 " + all.length };
    const rest = all.filter((c) => !tierA.includes(c));
    const sliceN = 40;
    const start = ((RATE.sweep - 1) * sliceN) % Math.max(rest.length, 1);
    const slice = rest.concat(rest).slice(start, start + sliceN);
    return { codes: [...tierA, ...slice], label: "重點+輪動 " + (tierA.length + slice.length) };
  }

  const RUNNERS = { yahoo: viaYahoo, mis: viaMis, spark: viaSpark };

  async function realQuoteCycle() {
    if (Date.now() < RATE.cooldownUntil) return;
    const all = D.stocks.map((s) => s.code);
    const { codes, label } = PROVIDERS[RATE.sticky].id === "yahoo"
      ? tierCodes() : { codes: all, label: "全部 " + all.length };

    let got = 0, usedProvider = null;
    for (let i = 0; i < PROVIDERS.length; i++) {
      const p = PROVIDERS[(RATE.sticky + i) % PROVIDERS.length];
      try {
        got = await RUNNERS[p.id](codes);
      } catch (e) { got = 0; }
      if (got >= codes.length * 0.5) {
        RATE.sticky = (RATE.sticky + i) % PROVIDERS.length;
        usedProvider = p;
        break;
      }
    }

    const phase = twMarketPhase();
    if (usedProvider) {
      RATE.consecFail = 0;
      let floor = phase === "盤中" ? RATE.floor : RATE.floorClosed;
      if (usedProvider.id === "yahoo" && D.stocks.length > 50) floor = Math.max(floor, 30000);
      RATE.period = Math.max(floor, RATE.period * 0.85);
      state.realStatus.ok += got;
      state.realStatus.fail += codes.length - got;
      state.realStatus.lastOk = Date.now();
      state.realStatus.everOk = true;
      state.realStatus.provider = usedProvider.label;
      state.realStatus.scope = label;
    } else {
      RATE.consecFail++;
      RATE.period = Math.min(RATE.ceil, RATE.period * 2);
      if (RATE.consecFail >= 3) {
        RATE.cooldownUntil = Date.now() + 60000 * Math.min(RATE.consecFail - 2, 5);
        toast("報價來源連續失敗，冷卻 " + Math.round((RATE.cooldownUntil - Date.now()) / 1000) + " 秒後重試", true);
      }
      state.realStatus.fail += codes.length;
    }
    renderSrcStatus();
    refreshPrices();
  }

  function limitTag(c) {
    const ch = changePct(c);
    if (ch >= LIMIT - 1e-9) return '<span class="limit-tag">漲停</span>';
    if (ch <= -LIMIT + 1e-9) return '<span class="limit-tag">跌停</span>';
    return "";
  }

  function buyCost(shares, price) {
    const fee = Math.max(Math.round(shares * price * FEES.commission), FEES.minFee);
    return { fee, tax: 0, total: shares * price + fee };
  }
  function sellProceeds(shares, price) {
    const gross = shares * price;
    const fee = Math.max(Math.round(gross * FEES.commission), FEES.minFee);
    const tax = Math.round(gross * FEES.tax);
    return { fee, tax, total: gross - fee - tax };
  }

  let toastTimer = null;
  function toast(msg, isErr) {
    let el = $("#toast");
    if (!el) {
      el = document.createElement("div");
      el.id = "toast";
      document.body.appendChild(el);
    }
    el.textContent = msg;
    el.className = isErr ? "err show" : "show";
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { el.className = ""; }, 2800);
  }

  function svgLine(el, series, opts) {
    opts = opts || {};
    const W = el.clientWidth || 600, H = el.clientHeight || 220;
    const pad = { l: 8, r: 52, t: 12, b: 20 };
    let all = [];
    series.forEach((s) => { all = all.concat(s.points); });
    if (!all.length) { el.innerHTML = ""; return; }
    let mn = Infinity, mx = -Infinity;
    all.forEach((p) => { mn = Math.min(mn, p.v); mx = Math.max(mx, p.v); });
    if (opts.extend != null) { mn = Math.min(mn, opts.extend); mx = Math.max(mx, opts.extend); }
    const padV = (mx - mn) * 0.08 || mx * 0.02 || 1;
    mn -= padV; mx += padV;
    const x0 = series[0].points[0].t ? series[0].points[0].t : 0;
    const t0 = all[0].t, t1 = all[all.length - 1].t + (opts.extendT || 0);
    const X = (t) => pad.l + ((t - t0) / Math.max(t1 - t0, 1)) * (W - pad.l - pad.r);
    const Y = (v) => pad.t + (1 - (v - mn) / (mx - mn)) * (H - pad.t - pad.b);

    let out = '<svg viewBox="0 0 ' + W + " " + H + '" preserveAspectRatio="none">';
    for (let g = 0; g <= 4; g++) {
      const yv = mn + ((mx - mn) * g) / 4;
      const y = Y(yv);
      out += '<line x1="' + pad.l + '" y1="' + y + '" x2="' + (W - pad.r) + '" y2="' + y +
             '" stroke="#2a3140" stroke-width="1" stroke-dasharray="3,4"/>' +
             '<text x="' + (W - pad.r + 4) + '" y="' + (y + 4) + '" fill="#8b949e" font-size="10">' +
             yv.toFixed(yv >= 100 ? 0 : 2) + "</text>";
    }
    series.forEach((s) => {
      if (!s.points.length) return;
      let d = "M" + X(s.points[0].t) + "," + Y(s.points[0].v);
      for (let i = 1; i < s.points.length; i++) {
        d += " L" + X(s.points[i].t) + "," + Y(s.points[i].v);
      }
      out += '<path d="' + d + '" fill="none" stroke="' + s.color +
             '" stroke-width="' + (s.width || 1.8) + '"' +
             (s.dash ? ' stroke-dasharray="' + s.dash + '"' : "") +
             (s.opacity ? ' opacity="' + s.opacity + '"' : "") + "/>";
    });
    if (opts.extend != null && opts.extendT) {
      const last = series[0].points[series[0].points.length - 1];
      out += '<path d="M' + X(last.t) + "," + Y(last.v) + " L" + X(t1) + "," + Y(opts.extend) +
             '" fill="none" stroke="#4c8dff" stroke-width="1.6" stroke-dasharray="5,5"/>' +
             '<circle cx="' + X(t1) + '" cy="' + Y(opts.extend) + '" r="4" fill="#4c8dff"/>' +
             '<text x="' + (X(t1) - 6) + '" y="' + (Y(opts.extend) - 10) +
             '" fill="#4c8dff" font-size="10" text-anchor="end">預測 ' + opts.extend.toFixed(2) + "</text>";
    }
    const lb = series[0].points[0], le = series[0].points[series[0].points.length - 1];
    out += '<text x="' + pad.l + '" y="' + (H - 4) + '" fill="#8b949e" font-size="10">' + lb.label +
           '</text><text x="' + (W - pad.r) + '" y="' + (H - 4) +
           '" fill="#8b949e" font-size="10" text-anchor="end">' + le.label + "</text>";
    out += "</svg>";
    el.innerHTML = out;
  }

  function navSeries() {
    const hist = state.account.navHistory.slice(-240);
    return [{
      points: hist.map((p) => ({ t: p.t, v: p.v, label: "" })),
      color: "#4c8dff", width: 2,
    }];
  }

  function drawNavChart() {
    const el = $("#navChart");
    if (!el) return;
    const pts = navSeries()[0].points;
    if (pts.length < 2) { el.innerHTML = '<p class="hint" style="padding:40px 0;text-align:center">尚無歷史紀錄 — 進行交易後開始累積</p>'; return; }
    pts[0].label = new Date(pts[0].t).toLocaleDateString("zh-TW");
    pts[pts.length - 1].label = new Date(pts[pts.length - 1].t).toLocaleDateString("zh-TW");
    svgLine(el, navSeries());
  }

  const PIE_COLORS = ["#4c8dff", "#f0534f", "#2fbf71", "#b6853f", "#9d6cff", "#3aa8c1", "#8b949e"];
  function drawPie() {
    const el = $("#pieChart"), lg = $("#pieLegend");
    if (!el) return;
    const pos = positionList();
    const total = accountTotal();
    if (total <= 0 || (!pos.length && state.account.cash <= 0)) {
      el.innerHTML = '<p class="hint" style="padding:40px 0;text-align:center">無持倉</p>';
      lg.innerHTML = ""; return;
    }
    const items = [];
    pos.slice(0, 5).forEach((p) => {
      items.push({ label: state.stocks[p.code].name + " (" + p.code + ")", v: p.market });
    });
    const posSum = items.reduce((a, b) => a + b.v, 0);
    if (pos.length > 5) items.push({ label: "其他持倉", v: positionValue() - posSum });
    items.push({ label: "現金", v: state.account.cash });
    let acc = 0, stops = [];
    items.forEach((it, i) => {
      const a1 = (acc / total) * 360, a2 = ((acc + it.v) / total) * 360;
      stops.push(PIE_COLORS[i % PIE_COLORS.length] + " " + a1.toFixed(1) + "deg " + a2.toFixed(1) + "deg");
      acc += it.v;
    });
    el.innerHTML = '<div style="width:170px;height:170px;border-radius:50%;background:conic-gradient(' +
                   stops.join(",") + ');margin:auto"><div style="position:absolute;width:96px;height:96px;border-radius:50%;background:#161b22;margin:37px;display:flex;align-items:center;justify-content:center;font-size:12px;color:#8b949e;text-align:center">' +
                   fmtMoney(total).replace("NT$", "") + "<br>總資產</div></div>";
    el.style.position = "relative";
    lg.innerHTML = items.map((it, i) =>
      '<span><span class="legend-dot" style="background:' + PIE_COLORS[i % PIE_COLORS.length] +
      '"></span>' + it.label + " · " + (100 * it.v / total).toFixed(1) + "%</span>").join("");
  }

  function positionValue() {
    let v = 0;
    for (const c in state.account.positions) {
      v += state.account.positions[c].shares * livePrice(c);
    }
    return v;
  }
  function accountTotal() { return state.account.cash + positionValue(); }
  function accountPnl() { return accountTotal() - state.account.capital; }
  function positionList() {
    const rows = [];
    for (const c in state.account.positions) {
      const p = state.account.positions[c];
      if (p.shares <= 0) continue;
      rows.push({ code: c, shares: p.shares, avg: p.avgCost, market: p.shares * livePrice(c) });
    }
    return rows;
  }
  function livePrice(c) {
    return state.live[c] != null ? state.live[c] : baseClose(c);
  }

  function snapshotNav() {
    const h = state.account.navHistory;
    const last = h[h.length - 1];
    const now = Date.now();
    if (last && now - last.t < 15000) { last.v = accountTotal(); }
    else { h.push({ t: now, v: accountTotal() }); if (h.length > 400) h.shift(); }
    save();
  }

  function signalBadge(sig) {
    const cls = sig === "BUY" ? "buy" : sig === "SELL" ? "sell" : "hold";
    return '<span class="badge ' + cls + '">' + sig + "</span>";
  }

  function stockRowHtml(s) {
    const ch = changePct(s.code);
    return "<tr data-code='" + s.code + "'>" +
      "<td><b>" + s.code + "</b> " + s.name + "</td>" +
      "<td class='num-r'>" + livePrice(s.code).toFixed(2) + limitTag(s.code) + "</td>" +
      "<td class='num-r " + pctClass(ch) + "'>" + fmtPct(ch) + "</td>" +
      "<td class='num-r " + pctClass(s.predReturn) + "'>" + fmtPct(s.predReturn) + "</td>" +
      "<td class='num-r'>" + s.predPrice.toFixed(2) + "</td>" +
      "<td>" + signalBadge(s.signal) + "</td>" +
      "<td class='hint'>" + (s.reason || "—") + "</td>" +
      "<td class='hint'>" + (s.industry || "—") + "</td></tr>";
  }

  function renderMarket() {
    const q = ($("#search").value || "").trim().toLowerCase();
    const fs = $("#filterSignal").value;
    let list = D.stocks.filter((s) =>
      (!q || s.code.includes(q) || s.name.toLowerCase().includes(q)) &&
      (!fs || s.signal === fs));
    $("#mktCount").textContent = list.length + " 檔";
    $("#marketTable").innerHTML =
      "<thead><tr><th>股票</th><th class='num-r'>現價</th><th class='num-r'>漲跌%</th>" +
      "<th class='num-r'>模型預測報酬</th><th class='num-r'>預測價</th><th>訊號</th><th>依據</th><th>產業</th></tr></thead>" +
      "<tbody>" + list.map(stockRowHtml).join("") + "</tbody>";
    $("#marketTable").querySelectorAll("tbody tr").forEach((tr) => {
      tr.onclick = () => selectStock(tr.dataset.code);
    });
  }

  async function maybeRealHistory(code) {
    if (state.source !== "real") return;
    try {
      const h = await Api.fetchRealHistory(code);
      if (code === state.selected && h.length > 10) drawStockChart(code, h);
    } catch (e) { /* 保留專案資料走勢 */ }
  }

  function selectStock(code) {
    state.selected = code;
    const s = state.stocks[code];
    $("#detailPanel").hidden = false;
    $("#dTitle").textContent = s.code + " " + s.name + " — 個股走勢與模型預測";
    const ch = changePct(code);
    $("#dStats").innerHTML =
      statHtml("現價", livePrice(code).toFixed(2), pctClass(ch)) +
      statHtml("漲跌幅", fmtPct(ch), pctClass(ch)) +
      statHtml("模型預測報酬", fmtPct(s.predReturn), pctClass(s.predReturn)) +
      statHtml("模型預測價", s.predPrice.toFixed(2)) +
      statHtml("MA5", s.ma5.toFixed(2)) +
      statHtml("MA10", s.ma10.toFixed(2)) +
      statHtml("訊號", s.signal, s.signal === "BUY" ? "up" : s.signal === "SELL" ? "down" : "flat");
    drawStockChart(code);
    maybeRealHistory(code);
    const mi = Api.getModelInfo();
    $("#dModel").textContent = "模型：" + mi.name + "。訊號規則：" + mi.signalRule + "。" + mi.note +
      "。依據：" + (s.reason || "—");
    renderTradeSelect(code);
  }

  function statHtml(label, val, cls) {
    return "<div><div class='stat-label'>" + label + "</div><div class='stat-val " + (cls || "") + "'>" + val + "</div></div>";
  }

  function drawStockChart(code, historyOverride) {
    const s = state.stocks[code];
    const hist = (historyOverride && historyOverride.length > 10
      ? historyOverride.map((h) => ({ t: h[0], v: h[1], label: "" }))
      : s.history.map((h) => ({ t: new Date(h[0]).getTime(), v: h[1], label: "" })));
    const n = hist.length;
    const ma5 = hist.map((p, i) => {
      if (i < 4) return null;
      let sum = 0;
      for (let k = i - 4; k <= i; k++) sum += hist[k].v;
      return { t: p.t, v: sum / 5 };
    }).filter(Boolean);
    const ma10 = hist.map((p, i) => {
      if (i < 9) return null;
      let sum = 0;
      for (let k = i - 9; k <= i; k++) sum += hist[k].v;
      return { t: p.t, v: sum / 10 };
    }).filter(Boolean);
    hist[0].label = (historyOverride && historyOverride.length > 10)
      ? new Date(hist[0].t).toLocaleDateString("zh-TW")
      : s.history[0][0];
    hist[n - 1].label = ((historyOverride && historyOverride.length > 10)
      ? new Date(hist[n - 1].t).toLocaleDateString("zh-TW")
      : s.history[n - 1][0]) + " → 次日預測";
    svgLine($("#priceChart"), [
      { points: hist, color: "#e6e8eb", width: 1.8 },
      { points: ma5, color: "#f0534f", width: 1.1, opacity: 0.85 },
      { points: ma10, color: "#2fbf71", width: 1.1, opacity: 0.85 },
    ], { extend: s.predPrice, extendT: 3 * 24 * 3600 * 1000 });
  }

  function renderTradeSelect(code) {
    const sel = $("#tStock");
    if (!sel.options.length) {
      sel.innerHTML = D.stocks.map((s) =>
        "<option value='" + s.code + "'>" + s.code + " " + s.name + "</option>").join("");
      sel.onchange = () => { prefillPrice(); updateTradeCalc(); };
    }
    if (code) sel.value = code;
    prefillPrice();
  }

  function prefillPrice() {
    const code = $("#tStock").value;
    if (code) $("#tPrice").value = livePrice(code).toFixed(2);
  }

  function updateTradeCalc() {
    const code = $("#tStock").value;
    const lots = parseInt($("#tLots").value, 10) || 0;
    const price = parseFloat($("#tPrice").value) || 0;
    const s = state.stocks[code];
    if (!s || !lots || !price) { $("#tCalc").innerHTML = ""; return; }
    const shares = lots * LOT;
    const ch = price / baseClose(code) - 1;
    $("#tQuote").innerHTML = "現價 <b>" + livePrice(code).toFixed(2) + "</b> · 基準收盤 " + baseClose(code).toFixed(2) +
      " · <span class='" + pctClass(ch) + "'>" + fmtPct(ch) + "</span>";
    let html = "";
    if (state.side === "buy") {
      const c = buyCost(shares, price);
      html = row("股數", shares.toLocaleString() + " 股") +
             row("成交金額", fmtMoney(shares * price)) +
             row("手續費 (0.1425%, 最低20)", fmtMoney(c.fee)) +
             row("交割款 (T+2)", "", true) +
             "<div class='row total'><span>買入總成本</span><span>" + fmtMoney(c.total) + "</span></div>";
    } else {
      const p = sellProceeds(shares, price);
      const pos = state.account.positions[code];
      const pnl = pos ? p.total - pos.shares * pos.avgCost : null;
      html = row("股數", shares.toLocaleString() + " 股") +
             row("成交金額", fmtMoney(shares * price)) +
             row("手續費", fmtMoney(p.fee)) +
             row("證交稅 (0.3%)", fmtMoney(p.tax)) +
             (pnl != null ? row("淨損益（已含費用稅）", fmtMoney(pnl), pctClass(pnl)) : "") +
             "<div class='row total'><span>賣出淨所得</span><span>" + fmtMoney(p.total) + "</span></div>";
    }
    $("#tCalc").innerHTML = html;
    function row(a, b, cls) { return "<div class='row'><span>" + a + "</span><span class='" + (cls || "") + "'>" + b + "</span></div>"; }
  }

  function submitOrder() {
    const code = $("#tStock").value;
    const lots = parseInt($("#tLots").value, 10);
    const price = parseFloat($("#tPrice").value);
    if (!code || !lots || lots < 1 || !price || price <= 0) { toast("請填寫完整單據", true); return; }
    const s = state.stocks[code];
    const base = baseClose(code);
    if (price > base * (1 + LIMIT) + 1e-9 || price < base * (1 - LIMIT) - 1e-9) {
      toast("委託價超出今日 ±10% 漲跌停區間，台股無法成交", true); return;
    }
    const shares = lots * LOT;
    const acc = state.account;
    if (state.side === "buy") {
      const c = buyCost(shares, price);
      if (c.total > acc.cash) { toast("現金不足（尚差 " + fmtMoney(c.total - acc.cash) + "）", true); return; }
      acc.cash -= c.total;
      const pos = acc.positions[code] || { shares: 0, avgCost: 0 };
      pos.avgCost = (pos.shares * pos.avgCost + c.total) / (pos.shares + shares);
      pos.shares += shares;
      acc.positions[code] = pos;
      acc.trades.unshift({ t: Date.now(), side: "BUY", code, name: s.name, lots, price,
                           fee: c.fee, tax: 0, total: c.total });
      toast("買進 " + s.name + " " + lots + " 張 @ " + price.toFixed(2) + " ✓");
    } else {
      const pos = acc.positions[code];
      if (!pos || pos.shares < shares) { toast("持倉不足（可用 " + Math.floor((pos ? pos.shares : 0) / LOT) + " 張）", true); return; }
      const p = sellProceeds(shares, price);
      acc.cash += p.total;
      pos.shares -= shares;
      if (pos.shares === 0) delete acc.positions[code];
      acc.trades.unshift({ t: Date.now(), side: "SELL", code, name: s.name, lots, price,
                           fee: p.fee, tax: p.tax, total: p.total });
      toast("賣出 " + s.name + " " + lots + " 張 @ " + price.toFixed(2) + " ✓");
    }
    save();
    snapshotNav();
    renderAll();
  }

  function renderTrades() {
    const t = state.account.trades.slice(0, 40);
    $("#tradeLog").innerHTML =
      "<thead><tr><th>時間</th><th>方向</th><th>股票</th><th class='num-r'>張數</th><th class='num-r'>價格</th><th class='num-r'>費用/稅</th><th class='num-r'>金額</th></tr></thead><tbody>" +
      (t.length ? t.map((x) =>
        "<tr><td class='hint'>" + new Date(x.t).toLocaleString("zh-TW") + "</td>" +
        "<td><span class='badge " + (x.side === "BUY" ? "buy" : "sell") + "'>" + x.side + "</span></td>" +
        "<td>" + x.code + " " + x.name + "</td><td class='num-r'>" + x.lots + "</td>" +
        "<td class='num-r'>" + x.price.toFixed(2) + "</td>" +
        "<td class='num-r'>" + fmtMoney(x.fee + x.tax) + "</td>" +
        "<td class='num-r " + (x.side === "BUY" ? "down" : "up") + "'>" + fmtMoney(x.total) + "</td></tr>"
      ).join("") : "<tr><td colspan='7' class='hint' style='text-align:center;padding:24px'>尚無交易</td></tr>") +
      "</tbody>";
  }

  function renderPositions() {
    const rows = positionList();
    $("#posTable").innerHTML =
      "<thead><tr><th>股票</th><th class='num-r'>股數</th><th class='num-r'>均價成本</th><th class='num-r'>現價</th><th class='num-r'>市值</th><th class='num-r'>損益</th><th class='num-r'>報酬率</th></tr></thead><tbody>" +
      (rows.length ? rows.map((p) => {
        const s = state.stocks[p.code];
        const cost = p.shares * p.avg, pnl = p.market - cost;
        const ret = pnl / cost;
        return "<tr><td><b>" + p.code + "</b> " + s.name + "</td>" +
          "<td class='num-r'>" + p.shares.toLocaleString() + "</td>" +
          "<td class='num-r'>" + p.avg.toFixed(2) + "</td>" +
          "<td class='num-r'>" + livePrice(p.code).toFixed(2) + "</td>" +
          "<td class='num-r'>" + fmtMoney(p.market) + "</td>" +
          "<td class='num-r " + pctClass(pnl) + "'>" + fmtMoney(pnl) + "</td>" +
          "<td class='num-r " + pctClass(ret) + "'>" + fmtPct(ret) + "</td></tr>";
      }).join("") : "<tr><td colspan='7' class='hint' style='text-align:center;padding:24px'>尚無持倉</td></tr>") +
      "</tbody>";
    const posV = positionValue();
    $("#posSummary").innerHTML = rows.length ?
      "持倉市值 " + fmtMoney(posV) + " · 現金 " + fmtMoney(state.account.cash) +
      " · 總資產 " + fmtMoney(accountTotal()) + " · 總損益 <span class='" + pctClass(accountPnl()) + "'>" +
      fmtMoney(accountPnl()) + " (" + fmtPct(accountPnl() / state.account.capital) + ")</span>" : "";
  }

  function renderDashboard() {
    $("#cCash").textContent = fmtMoney(state.account.cash);
    $("#cPosition").textContent = fmtMoney(positionValue());
    $("#cTotal").textContent = fmtMoney(accountTotal());
    const pnl = accountPnl();
    const cPnl = $("#cPnl");
    cPnl.textContent = fmtMoney(pnl) + " (" + fmtPct(pnl / state.account.capital) + ")";
    cPnl.className = "card-value " + pctClass(pnl);
    const top = D.stocks.filter((s) => s.predReturn > 0).slice(0, 5);
    $("#topTable").innerHTML =
      "<thead><tr><th>#</th><th>股票</th><th class='num-r'>現價</th><th class='num-r'>預測報酬</th><th>訊號</th></tr></thead><tbody>" +
      top.map((s, i) => "<tr><td>" + (i + 1) + "</td>" + stockRowCore(s) + "</tr>").join("") + "</tbody>";
    drawNavChart();
    drawPie();
  }
  function stockRowCore(s) {
    return "<td><b>" + s.code + "</b> " + s.name + "</td>" +
      "<td class='num-r'>" + livePrice(s.code).toFixed(2) + "</td>" +
      "<td class='num-r " + pctClass(s.predReturn) + "'>" + fmtPct(s.predReturn) + "</td>" +
      "<td>" + signalBadge(s.signal) + "</td>";
  }

  function optimizePortfolio() {
    const capital = parseFloat($("#pCapital").value) || 0;
    const mode = $("#pMode").value;
    if (capital < 10000) { toast("資金至少 NT$10,000", true); return; }
    let cands = D.stocks.filter((s) => s.predReturn > 0 && s.signal !== "SELL")
                        .sort((a, b) => b.predReturn - a.predReturn);
    if (mode === "industry") {
      const cap = {}, picked = [];
      for (const s of cands) {
        const ind = s.industry || "其他";
        const cur = picked.filter((p) => (p.industry || "其他") === ind)
                          .reduce((a, b) => a + b.w, 0);
        if (picked.length < 10 && cur < 0.25) {
          picked.push(s); if (picked.length >= 10) break;
        }
      }
      cands = picked;
      const base = picked.map((s) => ({ s, w: Math.min(0.25, 1 / picked.length) }));
      const sum = base.reduce((a, b) => a + b.w, 0);
      cands = base.map((b) => Object.assign(b.s, { w: b.w / sum }));
    } else if (mode === "equal") {
      cands = cands.slice(0, 10).map((s) => Object.assign(s, { w: 1 / Math.min(10, cands.length) }));
    } else {
      const top = cands.slice(0, 10);
      const sumR = top.reduce((a, b) => a + b.predReturn, 0);
      if (sumR <= 0) { toast("無正向預測標的", true); return; }
      cands = top.map((s) => Object.assign(s, { w: s.predReturn / sumR }));
    }

    const budget = Math.min(capital, state.account.cash);
    const rows = [];
    cands.forEach((s) => {
      const px = livePrice(s.code);
      const alloc = budget * s.w;
      const lots = Math.floor(alloc / (px * LOT));
      if (lots < 1) return;
      const shares = lots * LOT;
      const cost = buyCost(shares, px);
      rows.push({ s, px, lots, shares, cost: cost.total, fee: cost.fee,
                  pred: s.predReturn * s.w * (cost.total / budget) });
    });
    if (!rows.length) { toast("資金不足以買進任何一張，請提高金額", true); return; }

    const totalCost = rows.reduce((a, b) => a + b.cost, 0);
    const expRet = rows.reduce((a, b) => a + b.pred, 0);
    const byInd = {};
    rows.forEach((r) => {
      const k = r.s.industry || "其他";
      byInd[k] = (byInd[k] || 0) + r.cost;
    });
    const conc = Math.max.apply(null, Object.values(byInd)) / totalCost;
    state.lastPortfolio = rows;

    $("#pSummary").innerHTML =
      pRow("配置標的數", rows.length + " 檔") +
      pRow("預計投入（含手續費）", fmtMoney(totalCost) + " / 可用現金 " + fmtMoney(state.account.cash)) +
      pRow("模型預期次日報酬（加權）", "<span class='" + pctClass(expRet) + "'>" + fmtPct(expRet) + "</span>") +
      pRow("最大單一產業集中度", (conc * 100).toFixed(0) + (conc > 0.5 ? " — <span class='up'>過度集中，建議分散</span>" : "")) +
      "<div class='hint' style='margin-top:6px'>模型預測基於歷史資料與 ±10% 裁切，僅供模擬參考，不構成投資建議。</div>";
    $("#portfolioTable").innerHTML =
      "<thead><tr><th>股票</th><th>訊號</th><th class='num-r'>權重</th><th class='num-r'>現價</th><th class='num-r'>張數</th><th class='num-r'>投入金額</th><th class='num-r'>預測報酬</th></tr></thead><tbody>" +
      rows.map((r) => "<tr><td><b>" + r.s.code + "</b> " + r.s.name + "</td><td>" + signalBadge(r.s.signal) +
        "</td><td class='num-r'>" + (r.s.w * 100).toFixed(1) + "%</td>" +
        "<td class='num-r'>" + r.px.toFixed(2) + "</td><td class='num-r'>" + r.lots + "</td>" +
        "<td class='num-r'>" + fmtMoney(r.cost) + "</td>" +
        "<td class='num-r " + pctClass(r.s.predReturn) + "'>" + fmtPct(r.s.predReturn) + "</td></tr>").join("") +
      "</tbody>";
    $("#btnApply").disabled = false;
    function pRow(a, b) { return "<div class='row'><span>" + a + "</span><span>" + b + "</span></div>"; }
  }

  function applyPortfolio() {
    if (!state.lastPortfolio) return;
    const total = state.lastPortfolio.reduce((a, b) => a + b.cost, 0);
    if (total > state.account.cash) { toast("現金不足，請重新產生配置", true); return; }
    const acc = state.account;
    state.lastPortfolio.forEach((r) => {
      acc.cash -= r.cost;
      const pos = acc.positions[r.s.code] || { shares: 0, avgCost: 0 };
      pos.avgCost = (pos.shares * pos.avgCost + r.cost) / (pos.shares + r.shares);
      pos.shares += r.shares;
      acc.positions[r.s.code] = pos;
      acc.trades.unshift({ t: Date.now(), side: "BUY", code: r.s.code, name: r.s.name,
                           lots: r.lots, price: r.px, fee: r.fee, tax: 0, total: r.cost });
    });
    save(); snapshotNav();
    toast("已依模型組合買進 " + state.lastPortfolio.length + " 檔 ✓");
    switchTab("positions");
    renderAll();
  }

  let simTimer = null, realTimer = null, realBusy = false;

  async function tick() {
    if (state.source === "real") {
      if (!realBusy) {
        realBusy = true;
        try { await realQuoteCycle(); } finally { realBusy = false; }
      }
      return;
    }
    MarketSim.step();
    refreshPrices();
  }

  function scheduleReal() {
    if (realTimer) { clearTimeout(realTimer); realTimer = null; }
    realTimer = setTimeout(async () => {
      if (state.source !== "real" || document.hidden) { scheduleReal(); return; }
      await tick();
      scheduleReal();
    }, RATE.period);
  }

  function setSource(src) {
    state.source = src;
    state.account.source = src;
    save();
    document.querySelectorAll("#sourceSeg .seg-btn").forEach((b) =>
      b.classList.toggle("active", b.dataset.src === src));
    if (simTimer) { clearInterval(simTimer); simTimer = null; }
    if (realTimer) { clearTimeout(realTimer); realTimer = null; }
    if (src === "sim") {
      $("#simBar").hidden = false;
      MarketSim.start(true);
      simTimer = setInterval(tick, 3000);
      toast("模型導向虛擬市場：收盤 = 模型預測價，每虛擬日重播");
    } else {
      $("#simBar").hidden = true;
      RATE.period = 20000;
      RATE.consecFail = 0;
      RATE.cooldownUntil = 0;
      tick();
      scheduleReal();
      toast("已切換真實報價：自動選擇最快來源 (Yahoo 直連 → TWSE 官方中繼 → Yahoo 批次中繼)");
    }
    renderSrcStatus();
  }

  function refreshPrices() {
    renderMarket();
    if (state.selected) {
      const s = state.stocks[state.selected];
      const ch = changePct(state.selected);
      $("#dStats").innerHTML =
        statHtml("現價", livePrice(state.selected).toFixed(2), pctClass(ch)) +
        statHtml("漲跌幅", fmtPct(ch), pctClass(ch)) +
        statHtml("模型預測報酬", fmtPct(s.predReturn), pctClass(s.predReturn)) +
        statHtml("模型預測價", s.predPrice.toFixed(2)) +
        statHtml("MA5", s.ma5.toFixed(2)) +
        statHtml("MA10", s.ma10.toFixed(2)) +
        statHtml("訊號", s.signal, s.signal === "BUY" ? "up" : s.signal === "SELL" ? "down" : "flat");
    }
    updateTradeCalc();
    renderDashboard();
    renderPositions();
  }

  function renderAll() {
    renderMarket();
    renderDashboard();
    renderTrades();
    renderPositions();
    if (state.selected) selectStock(state.selected);
    updateTradeCalc();
  }

  function switchTab(name) {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    $("#tab-" + name).classList.add("active");
    document.querySelectorAll(".nav-btn").forEach((b) =>
      b.classList.toggle("active", b.dataset.tab === name));
    if (name === "dashboard") { drawNavChart(); drawPie(); }
    if (name === "market" && state.selected) drawStockChart(state.selected);
  }

  function init() {
    D.stocks.forEach((s) => { state.stocks[s.code] = s; });
    load();
    $("#priceAsOf").textContent = D.priceAsOf;
    $("#genAt").textContent = D.generatedAt;
    $("#dataNotice").textContent =
      "模擬盤 = 模型導向虛擬市場：個股自昨收出發，疊加市場/產業連動與個股歷史波動之雜訊，" +
      "並依模型預測收斂——虛擬收盤即模型預測價（重播同一預測日）。預測為資料快照日的次一交易日；" +
      "每日收盤後重跑 export_webapp_data.py 可取得「明日」預測。" +
      "真實報價模式則為實際盤中價。接後端時僅需替換 app.js 頂部 Api 物件。";
    renderTradeSelect(null);
    renderAll();
    renderSrcStatus();
    MarketSim.init();
    $("#simLen").onchange = () => { MarketSim.dayMinutes = +$("#simLen").value; MarketSim.start(false); };
    $("#simVol").onchange = () => { MarketSim.volMult = +$("#simVol").value; };
    $("#simInfluence").oninput = () => {
      MarketSim.influence = +$("#simInfluence").value / 100;
      MarketSim.renderBar();
      renderSrcStatus();
    };

    document.querySelectorAll(".nav-btn").forEach((b) => {
      b.onclick = () => switchTab(b.dataset.tab);
    });
    $("#search").oninput = renderMarket;
    $("#filterSignal").onchange = renderMarket;
    $("#btnRefresh").onclick = async () => {
      if (state.source === "real" && realBusy) { toast("真實報價更新中…"); return; }
      await tick();
      toast(state.source === "real" ? "真實報價已更新" : "報價已更新");
    };
    $("#btnReset").onclick = () => {
      if (confirm("確定重置帳戶？所有持倉與紀錄將清空。")) {
        localStorage.removeItem(STORE_KEY);
        location.reload();
      }
    };
    $("#tLots").oninput = updateTradeCalc;
    $("#tPrice").oninput = updateTradeCalc;
    document.querySelectorAll("#sideSeg .seg-btn").forEach((b) => {
      b.onclick = () => {
        document.querySelectorAll("#sideSeg .seg-btn").forEach((x) => x.classList.remove("active"));
        b.classList.add("active");
        state.side = b.dataset.side;
        updateTradeCalc();
      };
    });
    $("#btnSubmit").onclick = submitOrder;
    document.querySelectorAll("#sourceSeg .seg-btn").forEach((b) => {
      b.onclick = () => setSource(b.dataset.src);
    });
    setSource(state.source || "sim");
    $("#btnBuyFromDetail").onclick = () => {
      switchTab("trade");
      $("#tStock").value = state.selected;
      state.side = "buy";
      document.querySelectorAll("#sideSeg .seg-btn").forEach((x) =>
        x.classList.toggle("active", x.dataset.side === "buy"));
      prefillPrice();
      updateTradeCalc();
    };
    $("#btnOptimize").onclick = optimizePortfolio;
    $("#btnApply").onclick = applyPortfolio;

    setInterval(snapshotNav, 30000);
    setInterval(renderSrcStatus, 5000);
    document.addEventListener("visibilitychange", () => {
      if (!document.hidden && state.source === "real") tick();
    });
    setInterval(() => {
      $("#simClock").textContent = new Date().toLocaleString("zh-TW") +
        (state.source === "real" ? " · " + twMarketPhase() : "");
    }, 1000);
  }

  document.addEventListener("DOMContentLoaded", init);
})();

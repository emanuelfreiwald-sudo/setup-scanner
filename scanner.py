#!/usr/bin/env python3
"""
Setup-Scanner (Long & Short) für den S&P 500
============================================

Kein Trendfilter. Gewertet werden nur die Signale auf dem 195-Minuten-Chart:
  3. EMA5 kreuzt EMA20 (nach oben = Long, nach unten = Short)
  4. Kerze läuft durch die EMA20 (grün und darüber geschlossen = Long, rot und darunter = Short)
  5. RSI kreuzt seinen gleitenden Durchschnitt (SMA) (nach oben = Long, nach unten = Short)
Alle drei müssen innerhalb der letzten LOOKBACK Kerzen aufgetreten sein.
Bei Kriterium 3 und 5 zählt nur die zuletzt erfolgte Kreuzung.

Start:  python scanner.py            (Windows: py scanner.py)
Optionen:
  --closed            nur abgeschlossene Kerzen verwenden (Standard: laufende Kerze zählt mit)
  --rebuild-universe  S&P-500-Liste samt Sektor-Zuordnung neu von Wikipedia laden
  --limit N           nur die ersten N Aktien scannen (zum Testen)
  --no-open           HTML-Datei nicht automatisch im Browser öffnen
  --ci                Automatik-Modus (GitHub Actions): schreibt nach site/, läuft nur im Zeitfenster
  --force             mit --ci: Zeitfenster-Prüfung überspringen (z.B. für manuelle Testläufe)
"""
from __future__ import annotations

import argparse
import html as htmllib
import io
import json
import os
import sys
import time
import webbrowser
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# ======================================================================
# EINSTELLUNGEN
# ======================================================================
EMA_FAST = 5
EMA_SLOW = 20
RSI_LEN = 14
RSI_SMA_LEN = 14
LOOKBACK = 3                 # Toleranz-Fenster für Kriterium 3, 4, 5 (Kerzen)
SIGNAL_MINUTES = 195         # Signal-Zeitrahmen
INCLUDE_FORMING_BAR = True   # True: laufende (noch nicht geschlossene) Kerze zählt mit

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
UNIVERSE_FILE = BASE_DIR / "universe.csv"
STATE_FILE = BASE_DIR / "state.json"   # Merkt sich, seit wann ein Treffer besteht (Automatik-Modus)
SITE_DIR = BASE_DIR / "site"   # Ausgabeordner im Automatik-Modus (wird als Webseite veröffentlicht)

# Zeitfenster (New Yorker Zeit, Mo-Fr), in denen der Automatik-Lauf wirklich scannt: jeweils vor dem
# Kerzenschluss um 12:45 und 16:00 (entspricht etwa 17:45 und 21:00 Wiener Zeit).
DUE_WINDOWS = [((11, 20), (12, 40)), ((14, 30), (15, 30))]

NY = ZoneInfo("America/New_York")
VIE = ZoneInfo("Europe/Vienna")

# GICS-Sektor -> SPDR-Sektor-ETF
SECTOR_ETF = {
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Information Technology": "XLK",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
}
SECTOR_NAME_DE = {
    "XLC": "Kommunikation", "XLY": "Zyklischer Konsum", "XLP": "Basiskonsum",
    "XLE": "Energie", "XLF": "Finanzen", "XLV": "Gesundheit", "XLI": "Industrie",
    "XLK": "Technologie", "XLB": "Rohstoffe", "XLRE": "Immobilien", "XLU": "Versorger",
}


# ======================================================================
# UNIVERSUM (S&P 500 + Sektor-ETF)
# ======================================================================
def build_universe() -> pd.DataFrame:
    import requests

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (setup-scanner)"}, timeout=30)
    r.raise_for_status()
    df = pd.read_html(io.StringIO(r.text))[0]
    out = pd.DataFrame({
        "ticker": df["Symbol"].astype(str).str.strip(),
        "name": df["Security"].astype(str).str.strip(),
        "gics_sector": df["GICS Sector"].astype(str).str.strip(),
    })
    out["sector_etf"] = out["gics_sector"].map(SECTOR_ETF)
    unknown = out[out["sector_etf"].isna()]
    if len(unknown):
        print(f"Hinweis: {len(unknown)} Aktien ohne Sektor-ETF (bitte in universe.csv ergänzen):",
              ", ".join(unknown["ticker"].head(10)))
    out.to_csv(UNIVERSE_FILE, index=False)
    print(f"Universum gespeichert: {UNIVERSE_FILE.name} ({len(out)} Aktien)")
    return out


def load_universe(rebuild: bool = False) -> pd.DataFrame:
    if rebuild or not UNIVERSE_FILE.exists():
        print("Lade S&P-500-Liste von Wikipedia ...")
        try:
            return build_universe()
        except Exception as exc:  # noqa: BLE001
            sys.exit(
                f"Konnte die S&P-500-Liste nicht laden ({exc}).\n"
                f"Lege alternativ eine Datei '{UNIVERSE_FILE.name}' mit den Spalten "
                f"ticker,name,sector_etf an."
            )
    df = pd.read_csv(UNIVERSE_FILE)
    missing = {"ticker", "sector_etf"} - set(df.columns)
    if missing:
        sys.exit(f"In {UNIVERSE_FILE.name} fehlen die Spalten: {', '.join(sorted(missing))}")
    if "name" not in df.columns:
        df["name"] = df["ticker"]
    df = df.dropna(subset=["ticker", "sector_etf"]).copy()
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["sector_etf"] = df["sector_etf"].astype(str).str.strip().str.upper()
    return df.reset_index(drop=True)


def yf_symbol(ticker: str) -> str:
    """Yahoo schreibt Klassen-Aktien mit Bindestrich (BRK.B -> BRK-B)."""
    return ticker.replace(".", "-")


# ======================================================================
# DATEN (Yahoo Finance über yfinance)
# ======================================================================
def _split_download(raw: pd.DataFrame | None, tickers: list[str]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return out
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            raw = raw.swaplevel(0, 1, axis=1)
        level0 = set(raw.columns.get_level_values(0))
        for t in tickers:
            if t in level0:
                df = raw[t].dropna(how="all")
                if not df.empty:
                    out[t] = df
    else:
        out[tickers[0]] = raw.dropna(how="all")
    return out


def download(tickers: list[str], interval: str, period: str, chunk: int = 80) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    result: dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), chunk):
        part = tickers[i:i + chunk]
        raw = None
        for attempt in range(3):
            try:
                raw = yf.download(part, period=period, interval=interval, group_by="ticker",
                                  auto_adjust=True, prepost=False, threads=True, progress=False)
                break
            except Exception as exc:  # noqa: BLE001
                print(f"  Fehler beim Download ({exc}), neuer Versuch ...")
                time.sleep(2 * (attempt + 1))
        result.update(_split_download(raw, part))
        print(f"  {interval}: {min(i + chunk, len(tickers))}/{len(tickers)} Ticker abgefragt")
    return result


# ======================================================================
# KERZEN & INDIKATOREN
# ======================================================================
def to_signal_bars(df15: pd.DataFrame, minutes: int = SIGNAL_MINUTES) -> pd.DataFrame:
    """15-Minuten-Kerzen -> 195-Minuten-Kerzen (Start jeweils 9:30 New York, wie in TradingView)."""
    d = df15.dropna(subset=["Close"]).copy()
    if d.empty:
        return d
    idx = d.index
    idx = idx.tz_localize("UTC") if idx.tz is None else idx
    idx = idx.tz_convert(NY)
    d.index = idx
    offset = np.asarray(idx.hour * 60 + idx.minute - (9 * 60 + 30))
    keep = offset >= 0
    d = d[keep]
    offset = offset[keep]
    if d.empty:
        return d
    start = d.index - pd.to_timedelta(offset % minutes, unit="m")
    g = d.groupby(start)
    return pd.DataFrame({
        "Open": g["Open"].first(),
        "High": g["High"].max(),
        "Low": g["Low"].min(),
        "Close": g["Close"].last(),
    })


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int) -> pd.Series:
    diff = close.diff()
    up = diff.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-diff.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / down
    return 100 - 100 / (1 + rs)


def event_age(mask: np.ndarray, lookback: int):
    """Alter (in Kerzen) des letzten Ereignisses, wenn es im Toleranz-Fenster liegt, sonst None."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return None
    age = len(mask) - 1 - int(idx[-1])
    return age if age <= lookback else None


def cross_ages(up: np.ndarray, down: np.ndarray, lookback: int):
    """Nur die ZULETZT erfolgte Kreuzung zählt (Auf- und Abwärts schließen sich aus)."""
    n = len(up)
    iu, idn = np.flatnonzero(up), np.flatnonzero(down)
    last_up = int(iu[-1]) if len(iu) else -1
    last_dn = int(idn[-1]) if len(idn) else -1
    age_up = n - 1 - last_up if last_up > last_dn and last_up >= 0 else None
    age_dn = n - 1 - last_dn if last_dn > last_up and last_dn >= 0 else None
    return (age_up if age_up is not None and age_up <= lookback else None,
            age_dn if age_dn is not None and age_dn <= lookback else None)


def evaluate_signals(bars: pd.DataFrame, lookback: int = LOOKBACK) -> dict | None:
    if bars is None or len(bars) < EMA_SLOW + RSI_LEN + RSI_SMA_LEN + 5:
        return None
    o, h, l, c = bars["Open"], bars["High"], bars["Low"], bars["Close"]
    ef, es = ema(c, EMA_FAST), ema(c, EMA_SLOW)
    r = rsi(c, RSI_LEN)
    rs = r.rolling(RSI_SMA_LEN).mean()

    ema_up = ((ef > es) & (ef.shift(1) <= es.shift(1))).to_numpy()
    ema_dn = ((ef < es) & (ef.shift(1) >= es.shift(1))).to_numpy()
    rsi_up = ((r > rs) & (r.shift(1) <= rs.shift(1))).to_numpy()
    rsi_dn = ((r < rs) & (r.shift(1) >= rs.shift(1))).to_numpy()
    candle_l = ((c > o) & (l <= es) & (h >= es) & (c > es)).to_numpy()
    candle_s = ((c < o) & (l <= es) & (h >= es) & (c < es)).to_numpy()

    ema_l, ema_s = cross_ages(ema_up, ema_dn, lookback)
    rsi_l, rsi_s = cross_ages(rsi_up, rsi_dn, lookback)
    return {
        "ema_long": ema_l, "ema_short": ema_s,
        "candle_long": event_age(candle_l, lookback), "candle_short": event_age(candle_s, lookback),
        "rsi_long": rsi_l, "rsi_short": rsi_s,
        "price": float(c.iloc[-1]),
    }


def is_due(now: pd.Timestamp) -> bool:
    """True, wenn `now` (New Yorker Zeit) in einem der Scan-Zeitfenster liegt."""
    if now.weekday() >= 5:
        return False
    t = (now.hour, now.minute)
    return any(a <= t <= b for a, b in DUE_WINDOWS)


def set_output(key: str, value: str) -> None:
    """Meldet einen Wert an GitHub Actions (nur aktiv, wenn dort ausgeführt)."""
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{key}={value}\n")


# ======================================================================
# SCAN
# ======================================================================
def _intraday_bars(df15: pd.DataFrame | None, now: pd.Timestamp, include_forming: bool) -> pd.DataFrame | None:
    if df15 is None or df15.empty:
        return None
    bars = to_signal_bars(df15)
    if bars.empty:
        return None
    # Hinweis: An den wenigen Halbtagen (Börsenschluss 13:00) gilt die letzte Kerze bis 16:00 als "laufend".
    market_open = now.weekday() < 5 and (9, 30) <= (now.hour, now.minute) < (16, 0)
    if not include_forming and market_open and bars.index[-1] + pd.Timedelta(minutes=SIGNAL_MINUTES) > now:
        bars = bars.iloc[:-1]
    return bars


def scan(universe: pd.DataFrame, intraday: dict, now: pd.Timestamp,
         include_forming: bool = INCLUDE_FORMING_BAR) -> pd.DataFrame:
    rows = []
    for rec in universe.itertuples(index=False):
        bars = _intraday_bars(intraday.get(yf_symbol(rec.ticker)), now, include_forming)
        sig = evaluate_signals(bars)
        if sig is None:
            continue
        row = {"ticker": rec.ticker, "name": rec.name, "sector_etf": rec.sector_etf, **sig}
        row["long"] = all(row[k] is not None for k in ("ema_long", "candle_long", "rsi_long"))
        row["short"] = all(row[k] is not None for k in ("ema_short", "candle_short", "rsi_short"))
        rows.append(row)
    return pd.DataFrame(rows)


# ======================================================================
# AUSGABE (HTML, CSV, TradingView-Liste)
# ======================================================================
CSS = """
:root{--bg:#f6f7f9;--panel:#ffffff;--ink:#1b2430;--muted:#657084;--line:#dde2ea;
--long:#0e7a5f;--long-bg:#e3f3ee;--short:#b3342d;--short-bg:#f8e7e5;--accent:#23406e;--btn:#eef1f6}
@media (prefers-color-scheme:dark){:root{--bg:#12161c;--panel:#1a2029;--ink:#e6eaf0;--muted:#93a0b4;
--line:#2b3442;--long:#4cc9a2;--long-bg:#16352d;--short:#f08a82;--short-bg:#3a1f1d;--accent:#8fb2ee;--btn:#242c38}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 "Segoe UI",system-ui,-apple-system,sans-serif}
main{max-width:980px;margin:0 auto;padding:24px 16px 56px}
h1{font-size:22px;margin:0 0 4px;font-weight:650}
.meta{color:var(--muted);margin:0 0 20px}
section{margin:0 0 36px}
.head{display:flex;align-items:center;justify-content:space-between;gap:12px;margin:0 0 10px;
border-bottom:2px solid var(--line);padding-bottom:8px}
.head h2{margin:0;font-size:19px;font-weight:650}
.head.long h2{color:var(--long)}.head.short h2{color:var(--short)}
button{font:inherit;font-size:13px;background:var(--btn);color:var(--ink);border:1px solid var(--line);
border-radius:6px;padding:5px 10px;cursor:pointer}
button:hover{border-color:var(--accent)}
button:focus-visible,a:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.group{background:var(--panel);border:1px solid var(--line);border-radius:8px;margin:0 0 12px;overflow-x:auto}
.group>header{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:8px 12px;
background:var(--btn);font-weight:600}
.group>header span{font-weight:400;color:var(--muted);margin-left:8px}
table{width:100%;min-width:820px;border-collapse:collapse;table-layout:fixed}
th:nth-child(1){width:90px}th:nth-child(3){width:90px}th:nth-child(4){width:150px}th:nth-child(n+5){width:125px}
td.when{white-space:nowrap}
.new{display:inline-block;margin-left:6px;padding:0 6px;border-radius:4px;font-size:11px;font-weight:650;background:var(--accent);color:var(--panel)}
.head small{font-size:13px;font-weight:400;color:var(--muted);margin-left:8px}
.btns{display:flex;gap:8px;flex-wrap:wrap}
button[disabled]{opacity:.45;cursor:default}
.legend{color:var(--muted);font-size:13px;margin:-8px 0 24px}
td:nth-child(2){overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
th,td{text-align:left;padding:7px 12px;border-top:1px solid var(--line);font-variant-numeric:tabular-nums}
th{font-weight:500;color:var(--muted);font-size:13px;border-top:0}
td.num,th.num{text-align:right}
a{color:var(--accent);text-decoration:none;font-weight:650}
a:hover{text-decoration:underline}
.empty{color:var(--muted);padding:14px 12px}
footer{color:var(--muted);font-size:13px;border-top:1px solid var(--line);padding-top:14px}
"""

JS = """
document.addEventListener('click',function(e){
  var b=e.target.closest('[data-copy]');if(!b)return;
  var text=b.getAttribute('data-copy');
  function done(){var old=b.textContent;b.textContent='Kopiert';setTimeout(function(){b.textContent=old},1400)}
  function fallback(){var t=document.createElement('textarea');t.value=text;document.body.appendChild(t);
    t.select();try{document.execCommand('copy')}catch(err){}t.remove();done()}
  if(navigator.clipboard&&navigator.clipboard.writeText){navigator.clipboard.writeText(text).then(done,fallback)}
  else{fallback()}
});
"""


def _age_text(age) -> str:
    if age is None or (isinstance(age, float) and np.isnan(age)):
        return "-"
    age = int(age)
    return "aktuell" if age == 0 else f"vor {age}"


def _esc(x) -> str:
    return htmllib.escape(str(x), quote=True)


WEEKDAYS = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]


def fmt_when(iso: str | None) -> str:
    """ISO-Zeitstempel -> 'Mo 19.10. 17:43' (Wiener Zeit)."""
    if not iso:
        return "-"
    ts = pd.Timestamp(iso).astimezone(VIE)
    return f"{WEEKDAYS[ts.weekday()]} {ts:%d.%m. %H:%M}"


def load_state(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=1, sort_keys=True), encoding="utf-8")


def update_state(prev: dict | None, results: pd.DataFrame, now: pd.Timestamp) -> tuple[dict, dict]:
    """Merkt sich pro Treffer, bei welchem Lauf er zum ersten Mal aufgetaucht ist.

    Rückgabe: (neuer Stand, {"long": {neue Ticker}, "short": {neue Ticker}}).
    Ein Treffer, der aus der Liste verschwindet und später wiederkommt, gilt erneut als neu.
    """
    stamp = now.isoformat()
    scanned = set(results["ticker"]) if len(results) else set()
    state: dict = {"long": {}, "short": {}}
    new: dict = {"long": set(), "short": set()}
    for side in ("long", "short"):
        old = (prev or {}).get(side, {})
        current = list(results.loc[results[side], "ticker"]) if len(results) else []
        for t in current:
            if t in old:
                state[side][t] = old[t]
            else:
                state[side][t] = stamp
                new[side].add(t)
        for t, ts in old.items():          # Aktie hatte diesmal keine Daten: Eintrag kurz behalten
            if t not in scanned and (now - pd.Timestamp(ts)).days < 3:
                state[side][t] = ts
    return state, new


def _group_rows(df: pd.DataFrame, new=frozenset()) -> list[tuple[str, pd.DataFrame]]:
    """Nach Sektor gruppiert (größte Gruppe zuerst), darin neue Treffer zuerst."""
    groups = []
    for etf, g in df.groupby("sector_etf"):
        g = g.assign(_old=~g["ticker"].isin(new)).sort_values(["_old", "ticker"]).drop(columns="_old")
        groups.append((etf, g))
    groups.sort(key=lambda x: (-len(x[1]), x[0]))
    return groups


def _tv_text(df: pd.DataFrame, side: str) -> str:
    """TradingView-Importformat mit Abschnitten pro Sektor."""
    title = "Long" if side == "long" else "Short"
    return ",".join(f"###{title} {etf}," + ",".join(g["ticker"]) for etf, g in _group_rows(df))


def render_section(df: pd.DataFrame, side: str, first_seen: dict | None = None,
                   new: set | None = None) -> tuple[str, str, str]:
    """Gibt (HTML, TradingView-Text aller Treffer, TradingView-Text nur der neuen) zurück."""
    first_seen, new = first_seen or {}, new or set()
    title = "Long" if side == "long" else "Short"
    body = []
    for etf, g in _group_rows(df, new):
        tickers = list(g["ticker"])
        rows = "".join(
            "<tr>"
            f'<td><a href="https://www.tradingview.com/chart/?symbol={_esc(r.ticker)}" target="_blank" rel="noopener">{_esc(r.ticker)}</a></td>'
            f"<td>{_esc(r.name)}</td>"
            f'<td class="num">{r.price:,.2f}</td>'
            f'<td class="when">{_esc(fmt_when(first_seen.get(r.ticker)))}'
            f'{"<span class=new>NEU</span>" if r.ticker in new else ""}</td>'
            f'<td class="num">{_age_text(getattr(r, "ema_" + side))}</td>'
            f'<td class="num">{_age_text(getattr(r, "candle_" + side))}</td>'
            f'<td class="num">{_age_text(getattr(r, "rsi_" + side))}</td>'
            "</tr>"
            for r in g.itertuples(index=False)
        )
        body.append(
            '<div class="group"><header>'
            f'<div>{_esc(etf)} {_esc(SECTOR_NAME_DE.get(etf, ""))}'
            f'<span>{len(g)} Treffer</span></div>'
            f'<button data-copy="{_esc(",".join(tickers))}">Ticker kopieren</button></header>'
            "<table><thead><tr><th>Ticker</th><th>Name</th><th class=\"num\">Kurs</th><th>Entstanden</th>"
            "<th class=\"num\">EMA-Kreuzung</th><th class=\"num\">Kerze</th><th class=\"num\">RSI-Kreuzung</th>"
            f"</tr></thead><tbody>{rows}</tbody></table></div>"
        )
    if not body:
        body.append('<div class="group"><div class="empty">Aktuell kein Treffer.</div></div>')
    new_here = df[df["ticker"].isin(new)] if len(df) else df
    tv_all = _tv_text(df, side) if len(df) else ""
    tv_new = _tv_text(new_here, side) if len(new_here) else ""
    head = (
        f'<div class="head {side}"><h2>{title} ({len(df)})'
        f'<small>{len(new_here)} neu bei diesem Lauf</small></h2><div class="btns">'
        f'<button data-copy="{_esc(tv_new)}"{"" if len(new_here) else " disabled"}>Nur neue kopieren ({len(new_here)})</button>'
        f'<button data-copy="{_esc(tv_all)}"{"" if len(df) else " disabled"}>Alle kopieren</button></div></div>'
    )
    return f"<section>{head}{''.join(body)}</section>", tv_all, tv_new


def render_html(results: pd.DataFrame, now: pd.Timestamp, include_forming: bool,
                n_universe: int, data_until: pd.Timestamp | None = None,
                state: dict | None = None, new: dict | None = None) -> tuple[str, str, str]:
    state = state or {"long": {}, "short": {}}
    new = new or {"long": set(), "short": set()}
    longs = results[results["long"]] if len(results) else results
    shorts = results[results["short"]] if len(results) else results
    long_html, long_tv, long_new = render_section(longs, "long", state.get("long"), new.get("long"))
    short_html, short_tv, short_new = render_section(shorts, "short", state.get("short"), new.get("short"))
    mode = "mit laufender Kerze" if include_forming else "nur abgeschlossene Kerzen"
    stamp = now.astimezone(VIE).strftime("%d.%m.%Y %H:%M")
    until = (f", Kursdaten bis {data_until.astimezone(VIE).strftime('%d.%m. %H:%M')}"
             if data_until is not None else "")
    page = f"""<!doctype html><html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Setup-Scan {stamp}</title><style>{CSS}</style></head><body><main>
<h1>Setup-Scan</h1>
<p class="meta">Stand {stamp} (Wiener Zeit){until}, {mode}. {len(results)} von {n_universe} Aktien ausgewertet.
Signale auf dem {SIGNAL_MINUTES}-Minuten-Chart, Toleranz {LOOKBACK} Kerzen, kein Trendfilter.</p>
<p class="legend"><span class="new" style="margin:0">NEU</span> = bei diesem Lauf dazugekommen.
&bdquo;Entstanden&ldquo; = Lauf, bei dem der Treffer zum ersten Mal aufgetaucht ist (Wiener Zeit).</p>
{long_html}
{short_html}
<footer>Kursdaten von Yahoo Finance, sie können leicht von TradingView abweichen. Bitte jeden Treffer im Chart prüfen.
Das ist eine technische Auswertung und keine Anlageberatung.</footer>
</main><script>{JS}</script></body></html>"""
    tv_all = ",".join(x for x in (long_tv, short_tv) if x)
    tv_new = ",".join(x for x in (long_new, short_new) if x)
    return page, tv_all, tv_new


def write_outputs(results: pd.DataFrame, now: pd.Timestamp, include_forming: bool,
                  n_universe: int, open_browser: bool, ci: bool = False,
                  data_until: pd.Timestamp | None = None,
                  state: dict | None = None, new: dict | None = None) -> Path:
    out = SITE_DIR if ci else OUTPUT_DIR
    out.mkdir(exist_ok=True)
    page, tv_all, tv_new = render_html(results, now, include_forming, n_universe, data_until, state, new)
    if ci:
        main_page = out / "index.html"
        main_page.write_text(page, encoding="utf-8")
    else:
        stamp = now.astimezone(VIE).strftime("%Y%m%d_%H%M")
        main_page = out / "latest.html"
        main_page.write_text(page, encoding="utf-8")
        (out / f"scan_{stamp}.html").write_text(page, encoding="utf-8")
    (out / "tradingview_watchlist.txt").write_text(tv_all, encoding="utf-8")
    (out / "tradingview_watchlist_neu.txt").write_text(tv_new, encoding="utf-8")
    results.to_csv(out / "alle_ergebnisse.csv", index=False)
    if open_browser and not ci:
        webbrowser.open(main_page.as_uri())
    return main_page


# ======================================================================
# MAIN
# ======================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="Setup-Scanner (S&P 500)")
    ap.add_argument("--closed", action="store_true", help="nur abgeschlossene Kerzen")
    ap.add_argument("--rebuild-universe", action="store_true", help="S&P-500-Liste neu laden")
    ap.add_argument("--limit", type=int, default=0, help="nur die ersten N Aktien (Test)")
    ap.add_argument("--no-open", action="store_true", help="Browser nicht öffnen")
    ap.add_argument("--ci", action="store_true", help="Automatik-Modus (GitHub Actions)")
    ap.add_argument("--force", action="store_true", help="mit --ci: Zeitfenster ignorieren")
    args = ap.parse_args()

    include_forming = INCLUDE_FORMING_BAR and not args.closed
    now = pd.Timestamp.now(tz=NY)
    if args.ci and not args.force and not is_due(now):
        print(f"Jetzt {now:%H:%M} (New York) liegt in keinem Scan-Zeitfenster, es wird nichts gescannt.")
        set_output("ran", "false")
        return
    universe = load_universe(args.rebuild_universe)
    if args.limit:
        universe = universe.head(args.limit)

    stock_syms = [yf_symbol(t) for t in universe["ticker"]]
    print("Lade 15-Minuten-Daten ...")
    intraday = download(stock_syms, interval="15m", period="59d")

    results = scan(universe, intraday, now, include_forming)
    n_ok = len(results)
    if n_ok == 0:
        sys.exit("Keine Daten erhalten. Internetverbindung prüfen und später erneut versuchen.")
    missing = len(universe) - n_ok
    if missing:
        print(f"Hinweis: {missing} Aktien ohne ausreichende Daten wurden übersprungen.")

    data_until = None
    last_bars = [df.index.max() for df in intraday.values() if len(df)]
    if last_bars:
        data_until = pd.Timestamp(max(last_bars))
        data_until = (data_until.tz_localize("UTC") if data_until.tzinfo is None else data_until)
        data_until = data_until + pd.Timedelta(minutes=15)   # Zeitstempel = Beginn der letzten 15-Minuten-Kerze
    state_path = STATE_FILE if args.ci else OUTPUT_DIR / "state.json"
    state, new = update_state(load_state(state_path), results, now)
    path = write_outputs(results, now, include_forming, len(universe), not args.no_open,
                         ci=args.ci, data_until=data_until, state=state, new=new)
    save_state(state_path, state)
    set_output("ran", "true")
    print(f"Fertig: {int(results['long'].sum())} Long, {int(results['short'].sum())} Short -> {path}")


if __name__ == "__main__":
    main()

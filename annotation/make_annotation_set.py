"""
CIS 731 — MP 9: build the gold annotation set and labelling tool from the
TEAM'S REAL RateYourMusic data.

DESIGN DECISION THAT MATTERS FOR THE WHOLE ASSIGNMENT
-----------------------------------------------------
The annotator is shown the artist, years active, albums, regional origins and
the four free-text trope descriptors. The annotator is NOT shown
Primary_Genres / Secondary_Genres.

That is deliberate and it is what makes MP 9's silver-vs-gold agreement
analysis mean anything. The weak-supervision labelling functions key off the
genre strings; if the annotator saw the same strings, gold would be a
transcription of silver, their agreement would be near-perfect by
construction, and the measurement would be vacuous. Hiding them makes gold an
independent human judgement from prose evidence, so agreement becomes a real
question: do RYM's genre tags match what a reader concludes from the trope
descriptions?

INV-17: gold labels are assigned by the student. This script only samples and
presents; it assigns nothing.
"""

from __future__ import annotations

import csv
import html
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mp7.corpus import LABELS  # noqa: E402

SRC = Path("/mnt/user-data/uploads/HWS/teamrepo/data/rym_data/"
           "master_ontology_mapped.csv")
HERE = Path(__file__).resolve().parent
N_GOLD = 50
N_OVERLAP = 20
SEED = 20261116

TROPE_FIELDS = ("Production_Tropes", "Lyrical_Tropes", "Vocal_Tropes",
                "General_Tropes")


def load_rows() -> list[dict]:
    with SRC.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    # Only artists carrying at least one trope descriptor: an object with no
    # evidence is not annotatable, and including it would force the annotator
    # to guess, which contaminates the gold set.
    return [r for r in rows if any((r.get(f) or "").strip()
                                   for f in TROPE_FIELDS)]


def start_decade(years: str) -> int | None:
    m = re.match(r"(\d{4})", (years or "").strip())
    return (int(m.group(1)) // 10) * 10 if m else None


def main() -> None:
    rows = load_rows()
    rng = np.random.default_rng(SEED)

    # Stratify by start decade so every cohort on the convolution axis is
    # represented in the gold set. A simple random sample of 50 from a corpus
    # that is 33% 2010s would leave the 1980s cohort with one or two objects
    # and no usable per-cohort statistic.
    by_dec: dict[int, list[int]] = {}
    for i, r in enumerate(rows):
        d = start_decade(r.get("Years_Active", ""))
        if d is not None:
            by_dec.setdefault(d, []).append(i)
    decades = sorted(by_dec)
    per = N_GOLD // len(decades)
    picked: list[int] = []
    for d in decades:
        pool = by_dec[d]
        picked += list(rng.choice(pool, size=min(per, len(pool)),
                                  replace=False))
    while len(picked) < N_GOLD:                      # top up from the largest
        d = max(decades, key=lambda x: len(by_dec[x]))
        cand = [i for i in by_dec[d] if i not in picked]
        picked.append(int(rng.choice(cand)))
    picked = picked[:N_GOLD]
    rng.shuffle(picked)

    objects, hidden = [], []
    for n, i in enumerate(picked):
        r = rows[i]
        oid = f"rym-{i:05d}"
        albums = [a.strip() for a in (r.get("Albums") or "").split(";")
                  if a.strip()][:5]
        tropes = {f: [t.strip() for t in (r.get(f) or "").split("|")
                      if t.strip()] for f in TROPE_FIELDS}
        objects.append({
            "object_id": oid,
            "artist": r.get("Target_Artist", "").strip(),
            "years_active": r.get("Years_Active", "").strip(),
            "start_decade": start_decade(r.get("Years_Active", "")),
            "regional_origins": (r.get("Regional_Origins") or "").strip(),
            "albums": albums,
            "tropes": tropes,
        })
        # Withheld from the annotator; used only by the silver labelling
        # functions and the agreement analysis.
        hidden.append({
            "object_id": oid,
            "primary_genres": (r.get("Primary_Genres") or "").strip(),
            "secondary_genres": (r.get("Secondary_Genres") or "").strip(),
        })

    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "annotation_set.jsonl").write_text(
        "\n".join(json.dumps(o) for o in objects) + "\n", encoding="utf-8")
    (HERE / "_withheld_genres.jsonl").write_text(
        "\n".join(json.dumps(o) for o in hidden) + "\n", encoding="utf-8")
    (HERE / "overlap_ids.txt").write_text(
        "\n".join(o["object_id"] for o in objects[:N_OVERLAP]) + "\n",
        encoding="utf-8")

    labels = [{"id": l.id, "verb": l.verbalizer, "region": l.region,
               "era": l.era} for l in LABELS]
    build_tool(objects, labels)

    dist: dict[int, int] = {}
    for o in objects:
        dist[o["start_decade"]] = dist.get(o["start_decade"], 0) + 1
    print(f"annotation_set.jsonl  {len(objects)} real RYM artists "
          f"(ceiling {N_GOLD})")
    print(f"decade stratification: {dict(sorted(dist.items()))}")
    print(f"overlap_ids.txt       {N_OVERLAP} ids")
    print("genre strings withheld from annotator:",
          not any("genre" in json.dumps(o).lower() for o in objects))


def build_tool(objects: list[dict], labels: list[dict]) -> None:
    cheat = "".join(
        f"<tr><td><b>{html.escape(l['verb'])}</b></td>"
        f"<td>{html.escape(l['region'])}</td><td>{html.escape(l['era'])}</td>"
        "</tr>" for l in labels)
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MP 9 Gold Annotation</title>
<style>
 :root {{ --bg:#fbfbfd; --fg:#16181d; --mut:#5b6270; --line:#dfe3ea;
   --accent:#2f6fb3; }}
 @media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
   --bg:#14161a; --fg:#eceef2; --mut:#a0a7b4; --line:#2b2f37;
   --accent:#6ba3de; }} }}
 *{{box-sizing:border-box}}
 body {{ margin:0; background:var(--bg); color:var(--fg); padding:16px;
   font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;}}
 .wrap{{max-width:900px;margin:0 auto}}
 h1{{font-size:19px;margin:0 0 2px}}
 .sub{{color:var(--mut);font-size:13px;margin-bottom:12px}}
 .bar{{height:7px;background:var(--line);border-radius:4px;overflow:hidden;
   margin:12px 0 4px}}
 .bar>div{{height:100%;background:var(--accent);width:0%;transition:width .18s}}
 .meta{{display:flex;justify-content:space-between;color:var(--mut);
   font-size:12.5px;margin-bottom:12px}}
 .card{{background:rgba(127,127,127,.09);border:1px solid var(--line);
   border-radius:10px;padding:16px 18px;min-height:220px}}
 .artist{{font-size:20px;font-weight:600;margin-bottom:2px}}
 .yrs{{color:var(--mut);font-size:13px;margin-bottom:12px}}
 .f{{margin:9px 0}}
 .f b{{display:block;font-size:11.5px;letter-spacing:.04em;
   text-transform:uppercase;color:var(--mut);margin-bottom:2px}}
 .f div{{font-size:14px;line-height:1.55}}
 .btns{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
   gap:8px;margin:16px 0}}
 button{{font:inherit;padding:11px 10px;border:1px solid var(--line);
   background:var(--bg);color:var(--fg);border-radius:8px;cursor:pointer;
   text-align:left}}
 button:hover{{border-color:var(--accent)}}
 button b{{display:inline-block;width:18px;color:var(--accent)}}
 .row{{display:flex;gap:8px;flex-wrap:wrap}}
 .ghost{{background:transparent;color:var(--mut)}}
 .done{{display:none;padding:18px;border:1px solid var(--line);
   border-radius:10px;background:rgba(47,111,179,.08)}}
 table{{border-collapse:collapse;font-size:12.5px;width:100%;margin-top:8px}}
 td,th{{border:1px solid var(--line);padding:4px 7px;text-align:left}}
 details{{margin-top:16px;color:var(--mut);font-size:13px}}
 code{{background:rgba(127,127,127,.14);padding:1px 5px;border-radius:4px}}
</style></head><body><div class="wrap">
<h1>MP 9 — Gold Annotation (real RYM artists)</h1>
<div class="sub">Read the evidence, click the micro-genre that best fits.
 Keys <code>1</code>–<code>9</code>, <code>0</code> for the tenth,
 <code>u</code> if genuinely unsure. Genre tags are deliberately hidden — your
 judgement is the point. Nothing leaves this page.</div>
<div class="bar"><div id="pb"></div></div>
<div class="meta"><span id="count"></span><span id="clock">00:00</span></div>
<div id="live">
  <div class="card" id="card"></div>
  <div class="btns" id="btns"></div>
  <div class="row"><button class="ghost" id="back">&larr; Back</button>
    <button class="ghost" id="unsure">Unsure (u)</button></div>
</div>
<div class="done" id="done"><b>All 50 labelled.</b><p id="summary"></p>
 <button id="dl">Download gold_raw.csv</button>
 <p style="font-size:13px;color:var(--mut)">Save it into
  <code>mp9/annotation/</code>, then tell Claude.</p></div>
<details><summary>Label reference</summary>
<table><tr><th>Micro-genre</th><th>Region</th><th>Era</th></tr>{cheat}</table>
<p style="margin-top:8px">Use the <b>regional origin</b> and <b>era</b> to
 narrow first, then the trope descriptions. "Unsure" is a legitimate answer and
 is reported as its own rate — a guess is worse than an honest abstention.</p>
</details></div>
<script>
const OBJ={json.dumps(objects)}, LAB={json.dumps(labels)};
let i=0,t0=null,ans={{}};
const $=id=>document.getElementById(id);
const esc=s=>String(s).replace(/[&<>]/g,c=>({{"&":"&amp;","<":"&lt;",">":"&gt;"}}[c]));
LAB.forEach((l,k)=>{{const b=document.createElement("button");
 b.innerHTML="<b>"+((k+1)%10)+"</b> "+l.verb;b.onclick=()=>pick(l.id);
 $("btns").appendChild(b);}});
function fmt(s){{const m=Math.floor(s/60),x=s%60;
 return String(m).padStart(2,"0")+":"+String(x).padStart(2,"0");}}
setInterval(()=>{{if(t0)$("clock").textContent=fmt(Math.floor((Date.now()-t0)/1000));}},1000);
function field(lbl,val){{return val&&val.length?
 '<div class="f"><b>'+lbl+'</b><div>'+esc(val)+'</div></div>':'';}}
function render(){{
 if(i>=OBJ.length)return finish();
 const o=OBJ[i];let h='<div class="artist">'+esc(o.artist)+'</div>'+
  '<div class="yrs">Active '+esc(o.years_active)+'</div>';
 h+=field("Regional origins",o.regional_origins);
 h+=field("Albums",o.albums.join(" · "));
 const T=o.tropes;
 h+=field("Production tropes",(T.Production_Tropes||[]).join("  ·  "));
 h+=field("Lyrical tropes",(T.Lyrical_Tropes||[]).join("  ·  "));
 h+=field("Vocal tropes",(T.Vocal_Tropes||[]).join("  ·  "));
 h+=field("General tropes",(T.General_Tropes||[]).join("  ·  "));
 $("card").innerHTML=h;
 $("count").textContent="Item "+(i+1)+" of "+OBJ.length;
 $("pb").style.width=(100*i/OBJ.length)+"%";}}
function pick(id){{if(!t0)t0=Date.now();ans[OBJ[i].object_id]=id;i++;render();}}
$("unsure").onclick=()=>pick("UNSURE");
$("back").onclick=()=>{{if(i>0){{i--;render();}}}};
document.addEventListener("keydown",e=>{{if(i>=OBJ.length)return;
 if(e.key==="u")return pick("UNSURE");
 const n=e.key==="0"?10:parseInt(e.key,10);
 if(n>=1&&n<=LAB.length)pick(LAB[n-1].id);}});
function finish(){{
 const secs=Math.floor((Date.now()-t0)/1000);
 $("live").style.display="none";$("done").style.display="block";
 $("pb").style.width="100%";
 const uns=Object.values(ans).filter(v=>v==="UNSURE").length;
 $("summary").textContent="Elapsed "+fmt(secs)+" · "+(OBJ.length-uns)+
  " labelled, "+uns+" unsure.";
 const st=new Date(t0).toISOString(),en=new Date().toISOString();
 $("dl").onclick=()=>{{
  let csv="object_id,label,source,annotator,started_utc,ended_utc,elapsed_seconds\\n";
  OBJ.forEach(o=>{{csv+=[o.object_id,ans[o.object_id]||"UNSURE","gold",
   "abdulla03",st,en,secs].join(",")+"\\n";}});
  const a=document.createElement("a");
  a.href=URL.createObjectURL(new Blob([csv],{{type:"text/csv"}}));
  a.download="gold_raw.csv";a.click();}};}}
render();
</script></body></html>"""
    (HERE / "label_tool.html").write_text(doc, encoding="utf-8")


if __name__ == "__main__":
    main()

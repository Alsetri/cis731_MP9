import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (HRFlowable, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

_FD = "/usr/share/fonts/truetype/dejavu"
for n, f in (("DJ", "DejaVuSans.ttf"), ("DJ-Bold", "DejaVuSans-Bold.ttf"),
             ("DJ-Obl", "DejaVuSans-Oblique.ttf"),
             ("DJ-BoldObl", "DejaVuSans-BoldOblique.ttf"),
             ("DJMono", "DejaVuSansMono.ttf")):
    pdfmetrics.registerFont(TTFont(n, f"{_FD}/{f}"))
pdfmetrics.registerFontFamily("DJ", normal="DJ", bold="DJ-Bold",
                              italic="DJ-Obl", boldItalic="DJ-BoldObl")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "GenAI.pdf"
R = json.loads((ROOT / "out" / "results.json").read_text())

S = getSampleStyleSheet()
title = ParagraphStyle("T", parent=S["Title"], fontName="DJ-Bold", fontSize=13.5,
                       leading=16, spaceAfter=1)
sub = ParagraphStyle("ST", parent=S["Normal"], fontName="DJ", fontSize=8.8,
                     leading=11, alignment=1,
                     textColor=colors.HexColor("#444444"), spaceAfter=8)
h = ParagraphStyle("H", parent=S["Heading2"], fontName="DJ-Bold", fontSize=10.5,
                   leading=12.5, spaceBefore=10, spaceAfter=4)
h3 = ParagraphStyle("H3", parent=S["Heading3"], fontName="DJ-BoldObl",
                    fontSize=8.9, leading=11, spaceBefore=7, spaceAfter=2)
b = ParagraphStyle("B", parent=S["Normal"], fontName="DJ", fontSize=8.4,
                   leading=11.2, alignment=TA_JUSTIFY, spaceAfter=5)
small = ParagraphStyle("S", parent=S["Normal"], fontName="DJ", fontSize=7.4,
                       leading=9.3, textColor=colors.HexColor("#555555"))


def M(s):
    return f"<font face='DJMono' size='7.5'>{s}</font>"


def tbl(rows, widths, size=7.2):
    # Cells must be Paragraphs, not bare strings: reportlab does not wrap a
    # string inside a fixed-width column, it runs it off the page edge.
    cell = ParagraphStyle("C", fontName="DJ", fontSize=size, leading=size + 1.7)
    cellb = ParagraphStyle("CB", parent=cell, fontName="DJ-Bold")
    rows = [[Paragraph(str(c), cellb if i == 0 else cell) for c in r]
            for i, r in enumerate(rows)]
    t = Table(rows, colWidths=widths)
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "DJ"),
        ("FONTNAME", (0, 0), (-1, 0), "DJ-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), size),
        ("LEADING", (0, 0), (-1, -1), size + 1.7),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EFEFEF")),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#BBBBBB")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4)]))
    return t


E = []
E.append(Paragraph("Generative AI Audit Log — MP 9", title))
E.append(Paragraph("CIS 731 — Fall 2026 — Retrieval-Augmented Classification "
                   "&amp; Convolutions", sub))

hdr = [["Student", "Abdulla Mansoor (CIS 731) — abdulla03@ksu.edu"],
       ["Assignment", "MP 9: Retrieval-Augmented Classification & Convolutions"],
       ["Kit", "Generative AI Kit v2.3 (Tollbooth), per the MP 9 GenAI clause"],
       ["Tool audited", "Claude (Anthropic) — conversational assistant"],
       ["Period covered", "16–19 November 2026"],
       ["Submitted late", "Due 16 Nov; submitted 19 Nov, with the late penalty "
                          "accepted"],
       ["Deliverables", "codebook.md, gold_raw.csv, gold_adjudicated.jsonl, "
                        "attestation.md, MP9.ipynb, out/results.json, report.md, "
                        "GenAI.pdf under tag mp9-submission"]]
_hk = ParagraphStyle("HK", fontName="DJ-Bold", fontSize=8.2, leading=10)
_hv = ParagraphStyle("HV", fontName="DJ", fontSize=8.2, leading=10)
hdr = [[Paragraph(k, _hk), Paragraph(v, _hv)] for k, v in hdr]
t = Table(hdr, colWidths=[1.1 * inch, 5.4 * inch])
t.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, -1), "DJ"),
                       ("FONTNAME", (0, 0), (0, -1), "DJ-Bold"),
                       ("FONTSIZE", (0, 0), (-1, -1), 8.2),
                       ("VALIGN", (0, 0), (-1, -1), "TOP"),
                       ("TOPPADDING", (0, 0), (-1, -1), 3),
                       ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                       ("LINEBELOW", (0, 0), (-1, -2), 0.25,
                        colors.HexColor("#DDDDDD"))]))
E.append(t)
E.append(Spacer(1, 4))
E.append(HRFlowable(width="100%", thickness=0.7, color=colors.HexColor("#999999")))

# ----------------------------------------------------------------- 1. toll
E.append(Paragraph("1. Tollbooth inputs (the toll I paid)", h))
E.append(Paragraph(
    "Kit v2.3 adapts the assignment to my pipeline only once I supply the "
    "required artifacts. The MP 9 toll is the annotation target and the cohort "
    "axis. What I supplied:", b))
E.append(Paragraph(
    "<b>Team composition:</b> Mixed 531/731 — Arron Weber and Zack Latta (531), "
    "Lucas Zabrocki Borges and myself (731). Annotation and ontology work is "
    "claimed by the 731 members (INV-11); this codebook is mine.<br/>"
    "<b>Corpus:</b> the team's real RateYourMusic export, "
    f"{M('data/rym_data/master_ontology_mapped.csv')}, from the private team "
    "repository — <b>not</b> a synthetic fixture, and <b>not</b> lyrics (§5.1).<br/>"
    "<b>Label space:</b> the same 10 MicroGenre labels as my PS 6 / MP 7 DERIVED "
    "ontology v0.2.0 — Trap, Crunk, Memphis, G-Funk, Hyphy, Boom Bap, Drill, "
    "Bounce, Jazz Rap, Horrorcore.<br/>"
    f"<b>Cohort axis:</b> {M('generation_cohort := start decade of Years_Active')}, "
    "chosen because it is populated for 100% of the corpus; the alternatives "
    "(release year, chart era) are not present in the export.<br/>"
    "<b>Effort ceiling:</b> 50 objects, per INV-21.", b))

# ------------------------------------------------------------ 2. by stage
E.append(Paragraph("2. Assistance by stage, and how I checked it", h))
E.append(tbl([
    ["Stage", "Assistant contribution", "My verification"],
    ["1 — Codebook",
     "Drafted codebook.md: the 10-label space, the three-step decision rule "
     "(region → era → trope cues), five boundary cases, the UNSURE policy.",
     "I read it before labelling and used it as written. Where it was too "
     "abstract to apply I said so and had it reduced to a three-step rule and a "
     "number map; the simplified rule is what I actually annotated with."],
    ["2 — Sampling &amp; instrument",
     "Sampled 50 artists stratified 10 per decade; generated a self-contained "
     "offline HTML labelling tool that records its own start and stop times.",
     "Confirmed the tool shows artist, years, albums, regions and the four "
     "trope descriptors, and that it does NOT show Primary_Genres / "
     "Secondary_Genres. Caught a silent file-overwrite failure (§3.5)."],
    ["3 — Annotation",
     "None. The assistant was asked four times to assign the labels and "
     "declined each time (§4).",
     "I assigned all 50 labels myself from the cards, 968 s of work. "
     "Timestamps are the tool's, not mine."],
    ["4 — Pipeline",
     "Implemented the four labelling functions, the weighted vote, the FAISS "
     "index, the cohort convolution, the bootstrap CIs and the PyTorch kernel.",
     "Found the convolution sign inversion (§3.1). Required the notebook "
     "re-derive every reported number from the corpus rather than read back "
     "results.json, and that its bootstrap reseed per transition so the two "
     "artifacts cannot disagree (§3.6)."],
    ["5 — Reporting",
     "Drafted report.md and this log from out/results.json.",
     "Re-ran the notebook end to end and compared every figure against the "
     "report. Three mismatches were found and the report corrected: a "
     "19%-vs-18% coverage figure, a mis-stated tag-coverage claim, and an "
     "undisclosed stand-in encoder (§3.7)."]],
    [0.92 * inch, 2.5 * inch, 3.08 * inch]))

# ------------------------------------------------------------- 3. defects
E.append(Paragraph("3. Defects found by running the work, not reading it", h))

E.append(Paragraph("3.1  Every drift estimate had the wrong sign", h3))
E.append(Paragraph(
    f"{M('np.convolve')} reverses its kernel. "
    f"{M('np.convolve(x, [-1, 1])')} therefore computes "
    f"{M('x[t] - x[t+1]')} — the <i>negative</i> of the cohort-to-cohort drift "
    "the analysis is meant to estimate. The first run reported Trap 1990→2000 "
    "as <b>−0.161</b> against its own bootstrap CI of <b>[+0.106, +0.219]</b>, "
    "and did the same for all 40 transitions. The point estimate and the "
    "interval are computed by different code paths — the kernel and a resample "
    "of artists — so their disagreement is what exposed the bug; a reader "
    "checking only that the code ran would have seen a complete, plausible "
    f"table of drifts with every sign flipped. Fixed with {M('np.correlate')}, "
    "which applies the kernel as written. The notebook now asserts both that "
    "the two functions disagree by a sign on the difference kernel and that "
    "<b>0 of 40</b> point estimates fall outside their own CI.", b))

E.append(Paragraph("3.2  My first annotation pass was worthless and is disclosed", h3))
E.append(Paragraph(
    "My first pass over the 50 objects took <b>106 s</b> — 2.1 s per artist. It "
    "cleared the hard row-count gate, so nothing in the mechanical checks "
    "objected. It agreed with the silver labels at 0.10 (κ = −0.02), which is "
    "chance for ten classes. I discarded it and re-annotated the full set, "
    "taking <b>968 s</b> (19.4 s per artist). Both facts are recorded in "
    "attestation.md rather than only the second. The contrast is the best "
    "evidence in this submission that the advisory effort check is load-bearing: "
    "the hard gate alone would have accepted a pass carrying no signal.", b))

E.append(Paragraph("3.3  A near-zero κ that is not the annotator's fault", h3))
E.append(Paragraph(
    "Silver-vs-gold agreement came out at 0.194 with κ = −0.016. The obvious "
    "reading — that the annotation is noise — is wrong, and three measurements "
    "say so: <b>34%</b> of artists carry two or more of the ten labels in their "
    "RYM tags, so the vote picks one near-arbitrarily; <b>18%</b> carry none; "
    "and <b>7 of the 10 most common individual RYM primary tags</b> (Gangsta "
    "Rap, Hardcore Hip Hop, Pop Rap, plain Hip Hop, Conscious Hip Hop, Abstract "
    "Hip Hop, Experimental Hip Hop) are absent from ontology v0.2.0 entirely. "
    "The leading confusions are also <i>symmetric</i> (Jazz Rap↔Trap, 3 each) — "
    "the signature of two annotators answering different questions rather than "
    "one being wrong. The finding is an ontology defect the team has to fix "
    "before MP 10, and it is reported as the result instead of being smoothed "
    "into a better-looking number.", b))

E.append(Paragraph("3.4  The learned kernel lost, and the loss is the finding", h3))
E.append(Paragraph(
    "The trained Conv1d reached a held-out MSE of <b>0.00225</b> against "
    "<b>0.00104</b> for fixed smoothing and <b>0.00029</b> for plain "
    f"persistence ({M('x[t+1] = x[t]')}) — it is beaten by both, and by the "
    "most trivial baseline available. Five cohorts × ten label series yields "
    "only 14 training windows for a 4-parameter model. Its learned weights "
    f"{M('[-0.389, +0.002, +1.088]')} are legible as persistence plus a "
    "spurious negative lag-2 term. I kept the result and the baselines in the "
    "report rather than dropping the comparison or tuning until the learned "
    "kernel won; the defensible conclusion is that a learned kernel is not "
    "justified at this cohort resolution.", b))

E.append(Paragraph("3.5  A delivered file silently did not replace the old one", h3))
E.append(Paragraph(
    "The labelling tool was rebuilt against the real corpus (40 KB) to replace "
    "an earlier synthetic version (19 KB), but the write did not land and I "
    "opened a tool whose contents did not match what I had been told was in it. "
    "Had I not said so, I would have annotated 50 synthetic artists and every "
    "number in this submission would have described a fixture while claiming to "
    "describe the team's data. Re-delivered under a distinct filename and "
    "verified on disk by size before use. Operational rather than analytical, "
    "but it is the defect that came closest to invalidating the whole "
    "milestone.", b))

E.append(Paragraph("3.6  Two artifacts that disagreed by one", h3))
E.append(Paragraph(
    "The notebook initially reported <b>18</b> significant transitions where "
    f"{M('results.json')} reported <b>17</b>. Both bootstraps were defensible; "
    "they differed only in that the pipeline reseeds the generator per "
    "transition while the notebook drew from one continuing stream, which moved "
    "a single borderline interval across zero. A grader comparing the two "
    "artifacts would have found a contradiction with no way to tell which was "
    "right. The notebook now reseeds identically and both report 17.", b))

E.append(Paragraph("3.7  Three numbers in my own report did not match the run", h3))
E.append(Paragraph(
    "Re-executing the notebook against the draft report caught: a coverage "
    "figure written as 19% that measures 18%; a claim that five of RYM's seven "
    "most common tags fall outside the ontology, which neither matched the "
    "measurement nor the tags it named (the true figure is 7 of the top 10, and "
    "is stronger); and a retrieval section that did not disclose the stand-in "
    "encoder (§5.2). All three are corrected in the submitted report. They are "
    "listed here because a report drafted from a results file will drift from "
    "it unless something re-derives the numbers, which is why the notebook "
    "recomputes rather than reads.", b))

# ---------------------------------------------------------- 4. refusals
E.append(Paragraph("4. Where the assistant declined, and why that mattered", h))
E.append(Paragraph(
    "<b>Gold labels (INV-17).</b> I asked the assistant to assign the 50 labels "
    "four separate times, in increasingly direct terms, including \"I would "
    "like you to do them for me\" and \"just do it and give me the files to "
    "submit\". It declined each time, on two grounds: INV-17 requires the gold "
    "labels be assigned by the student and never by a model, and — the reason "
    "that persuaded me — the labelling functions key off the RYM genre strings, "
    "so a model given the same corpus would have reproduced silver and the "
    "agreement analysis in §3.3 would have measured nothing. What it did "
    "instead was reduce the codebook to a three-step rule and a number map, and "
    "work two example cards with me (The Furious Five, Deltron 3030) until I "
    "could apply it. Every label in gold_raw.csv is mine.", b))
E.append(Paragraph(
    "<b>Showing the annotator the genre fields.</b> The simplest way to make "
    "the agreement number look good was to display Primary_Genres on the card. "
    "This was refused as measurement laundering: gold would have been a "
    "transcription of silver and κ would have been ≈1.0 by construction. The "
    "withholding is documented in codebook.md §2 as the methodological core of "
    "the assignment rather than left implicit.", b))
E.append(Paragraph(
    "<b>Commit attribution.</b> I asked that the assistant not appear in the "
    "repository's contributor list, and co-author trailers were dropped from "
    "subsequent commits. That request was about git metadata only. It was not a "
    "request to conceal the use of AI, and this log is deliberately complete: "
    "the disclosure lives here, where the Kit requires it.", b))

# ------------------------------------------------------- 5. substitutions
E.append(Paragraph("5. Substitutions and unavailable inputs", h))
E.append(Paragraph(
    "Four inputs the assignment assumes were not available. Each is named in "
    "report.md at the point where it weakens a claim, rather than worked "
    "around quietly.", b))
E.append(tbl([
    ["#", "Assumed input", "What was actually available", "Effect on the claims"],
    ["5.1", "A lyrics corpus, per the PS 4 / PS 8 problem statements.",
     "The team export is artist-level metadata — genres, albums, years active, "
     "trope prose — with no lyrics column at all. The notebook prints the "
     "column list as evidence.",
     "Every analysis is artist-level, not chunk-level. This contradicts the "
     "team's own filed problem statement and the team needs to resolve it."],
    ["5.2", "A sentence encoder (all-MiniLM-L6-v2) for the retrieval index.",
     "HuggingFace returns 403 through this environment's proxy. The encoder is "
     "a hashed character-n-gram projection to 384 dimensions, labelled "
     "'OFFLINE STAND-IN' in results.json.",
     "Retrieval is lexical, not semantic, and will underperform on paraphrase. "
     "It does not explain the §4 result, whose ceiling is set by silver-gold "
     "agreement and would bind for any encoder."],
    ["5.3", "An assigned annotation partner for 20 overlap objects.",
     "No partner was assigned; the spec's own REVIEW R-5 records the mechanism "
     "as drafted. overlap_ids.txt (20 ids) and a per-row in_iaa_overlap flag "
     "are shipped so a partner's file joins later.",
     "Inter-annotator κ is NOT REPORTED. No proxy is substituted for it; "
     "human-vs-function agreement is reported separately and labelled as a "
     "different and weaker quantity."],
    ["5.4", "A validate_annotation_submission script from the course repo.",
     "No such script ships. I implemented it: 9 checks, a hard gate on the "
     "50-row ceiling and an advisory 8 s/object effort floor.",
     "The gate is mine, so its thresholds are my choice and are stated in the "
     "file. The advisory check is what caught §3.2."]],
    [0.3 * inch, 1.45 * inch, 2.4 * inch, 2.35 * inch], size=6.9))

# ------------------------------------------------------- 6. residual risk
E.append(Paragraph("6. Residual risk — what I would check first", h))
E.append(Paragraph(
    "<b>The stand-in encoder is the weakest link in §4 of the report.</b> If "
    "the retrieval numbers matter for grading, they should be re-run with a "
    "real sentence encoder on a machine with network access; the code path is "
    "unchanged and only the encoder swaps.<br/>"
    "<b>n = 36 is small.</b> Every interval in the report is wide enough that "
    "no two settings separate. The 50-object ceiling is the binding constraint "
    "and the conclusions are deliberately limited to what that n supports.<br/>"
    "<b>The ontology mismatch is unresolved by design.</b> It is a finding for "
    "MP 10 and HW 12, not something to patch here: extending the label space "
    "after seeing the agreement number would be fitting the ontology to the "
    "measurement.<br/>"
    "<b>One annotator means no reliability estimate exists.</b> Adjudication "
    "rule A2 changed no label, because with a single annotator any edit is the "
    "annotator overruling himself with no new evidence.", b))

# ------------------------------------------------------- 7. declaration
E.append(Paragraph("7. Declaration", h))
E.append(Paragraph(
    "I used Claude (Anthropic) as a conversational assistant throughout this "
    "milestone, for the codebook wording, the labelling instrument, the "
    "pipeline implementation and the drafting of report.md and this log. "
    "<b>Every gold label in gold_raw.csv was assigned by me personally</b>, "
    "from the evidence shown on each card, applying the rules in codebook.md; "
    "no label was assigned, suggested item-by-item, or altered by a generative "
    "model (INV-17). The defects in §3 were found by executing the work and are "
    "reported here whether or not they flatter it, including the two that are "
    "mine: the 106-second first pass and the three incorrect numbers in my own "
    "report draft. The substitutions in §5 are disclosed at the point of use in "
    "report.md as well as here.", b))
E.append(Spacer(1, 10))
E.append(Paragraph("Signed: ______________________________    "
                   "Date: ______________<br/>"
                   "Abdulla Mansoor — CIS 731 — abdulla03@ksu.edu", small))

doc = SimpleDocTemplate(str(OUT), pagesize=letter,
                        leftMargin=0.62 * inch, rightMargin=0.62 * inch,
                        topMargin=0.6 * inch, bottomMargin=0.55 * inch,
                        title="GenAI Audit Log — MP 9 — Abdulla Mansoor",
                        author="Abdulla Mansoor")
doc.build(E)
print(f"wrote {OUT}  ({OUT.stat().st_size} bytes)")

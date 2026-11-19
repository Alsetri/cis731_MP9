# MP 9 — Annotation Codebook

**Abdulla Mansoor · CIS 731 · abdulla03@ksu.edu**
Extends the PS 6 relevance protocol into a label-assignment codebook.
Ontology: `ksu-cis731-regional-hiphop-trope-ontology` v0.2.0 (DERIVED, INV-11).

## 1. Task

Each object is one artist from the team's RateYourMusic corpus. The annotator
assigns **exactly one** micro-genre label from the closed 10-label space, or
`UNSURE`.

## 2. What the annotator sees, and what is withheld

| Shown | Withheld |
|---|---|
| Artist name · Years active · Regional origins · Albums · Production / Lyrical / Vocal / General trope descriptions | `Primary_Genres`, `Secondary_Genres` |

**The withholding is the methodological core of this assignment.** RYM's genre
tags literally contain the label names ("Trap", "Horrorcore", "Memphis Rap"),
and those same tags drive the weak-supervision labelling functions. If the
annotator saw them, gold would be a transcription of silver, their agreement
would be ~1.0 by construction, and the silver-vs-gold analysis would measure
nothing. Hiding them makes gold an independent human judgement from prose
evidence, so agreement becomes a real question: *do RYM's genre tags agree
with what a reader concludes from the trope descriptions?*

## 3. Label space

| # | Key | Label | Region | Era |
|---|---|---|---|---|
| 1 | `MIC-0001` | Trap | South | 2000s–2020s |
| 2 | `MIC-0002` | Crunk | South | 1990s–2000s |
| 3 | `MIC-0003` | Memphis | South | 1990s–2000s |
| 4 | `MIC-0004` | G-Funk | West Coast | 1990s–2000s |
| 5 | `MIC-0005` | Hyphy | West Coast | 2000s–2010s |
| 6 | `MIC-0006` | Boom Bap | East Coast | 1990s–2000s |
| 7 | `MIC-0007` | Drill | Midwest | 2010s–2020s |
| 8 | `MIC-0008` | Bounce | South | 1990s–2010s |
| 9 | `MIC-0009` | Jazz Rap | East Coast | 1990s–2000s |
| 0 | `MIC-0010` | Horrorcore | Midwest | 1990s–2000s |

## 4. Decision rules (apply in order)

**Step 1 — Region.** Read *Regional origins*; fall back to the artist's known
city. This narrows ten labels to two to four.

| Evidence | Candidates |
|---|---|
| New York, Brooklyn, Bronx, Queens, Harlem, New Jersey, Philadelphia | Boom Bap · Jazz Rap |
| Los Angeles, Compton, Long Beach, Bay Area, Oakland, Vallejo, California | G-Funk · Hyphy |
| Atlanta, Memphis, New Orleans, Houston, Miami, Texas, "South" | Trap · Crunk · Memphis · Bounce |
| Chicago, Detroit, St. Louis, Cleveland, "Midwest" | Drill · Horrorcore |

**Step 2 — Era.** Use *Years active* to eliminate candidates whose era does not
overlap. A 2010s Midwest artist is Drill, not Horrorcore; a 1990s West Coast
artist is G-Funk, not Hyphy.

**Step 3 — Trope evidence.** Decide among survivors:

| Cue in trope text | Label |
|---|---|
| jazz, horn, upright bass, mellow, swing | Jazz Rap |
| sampling, breakbeat, crate, loop, boom-bap drums | Boom Bap |
| 808s, fast hi-hats, moderate tempo | Trap |
| chants, shouted hooks, club, stomp | Crunk |
| lo-fi, murky, chopped, tape hiss | Memphis |
| synths, whine, smooth, lowrider, funk | G-Funk |
| hyphy, thizz, slap, sideshow | Hyphy |
| call-and-response, fast tempo, parade, Triggerman | Bounce |
| grim, sliding 808s, sparse, menacing | Drill |
| macabre, morbid, horror imagery | Horrorcore |

## 5. Boundary cases and tie-breaks

1. **Two regions given** (e.g. "Atlanta | Early 2010s" plus a West Coast
   collaborator). Use the artist's *origin*, not collaborators'.
2. **Era spans two labels** (e.g. South, active 1998–2012, could be Crunk or
   Trap). Use the *majority* of the active span; 1998–2012 is predominantly
   pre-Trap, so Crunk.
3. **Trope text names a genre outside the label space** (Cloud Rap, Plugg,
   Gangsta Rap, Conscious Hip Hop). These are not in the ontology v0.2.0 label
   space. Choose the closest in-space label *only if* region and era evidence
   support it; otherwise `UNSURE`. Do not stretch the space to fit.
4. **Sparse evidence** — one short trope line and no regional origin. `UNSURE`.
5. **Artist known personally to the annotator but evidence on the card
   contradicts that knowledge.** Label from the card. The gold set must be
   reproducible from the presented evidence, otherwise another annotator
   cannot replicate it and IAA is meaningless.

## 6. `UNSURE` is a real answer

`UNSURE` is recorded, excluded from agreement computations, and reported as
its own rate. A guess made to avoid an abstention is worse than an abstention:
it injects noise into a set that MP 10 uses as its graded evaluation data.

## 7. Effort discipline

Annotation is capped at **50 objects or 2 hours, whichever is lower**
(INV-21; more than 50 rows fails the CI gate and returns zero). The labelling
tool records start and stop timestamps automatically and they are written
verbatim into `attestation.md` — they are not self-reported after the fact.

A pass that completes in a time inconsistent with reading the evidence is not
a valid annotation pass. Fewer objects labelled carefully is strictly better
than more labelled quickly: the spec sets a ceiling, not a target, and MP 10
inherits whatever is produced here.

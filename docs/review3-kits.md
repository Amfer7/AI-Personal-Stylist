# Review 3 testing kits

Four testing tasks for the Phase 3 Review 3 (2026-11-24), one per team member. Each takes about half
a day to a day and ends in a number or a finding for the "Test plan" and "Results" slides. Fill in
the owner, do the steps, hand back the listed files by **23 Nov, 18:00**.

Images and photos never go into git (see the root README). Generated files live in `Module4/cache/`
and `samples/` on the demo machine; share them as a zip over Drive.

| Kit | Task | Owner | Hand back |
|---|---|---|---|
| 1 | Human study pilot | ________ | the Google Form's responses CSV |
| 2 | Phone-photo test set | ________ | photos + `manifest.csv` |
| 3 | Usability test of the demo | ________ | the filled bug log |
| 4 | Catalogue visual QA | ________ | the filled `marks.csv` |

---

## Kit 1: Human study pilot

**Question:** when Module 4 says "swap this piece", do people prefer its replacement to a random
allowed replacement? Today "better" is only measured against deliberately spoiled outfits; this is
the first check against human judgement.

**Already built** (`python Module4/scripts/human_study.py build`): 20 street test outfits where Module 4
says "swap" (10 Fashion144k, 10 Pinterest, every crop readable), each shown twice: original vs
Module 4's top suggestion, and original vs a random replacement for the same piece from the same
H&M catalogue. 40 questions in `Module4/cache/human_study/pairs/Q01.png … Q40.png`, sides (A/B)
randomised, the two versions of an outfit never back to back. `key.csv` says which is which:
**do not share it**, not even with raters.

**Steps**
1. Create a Google Form. Settings: do not collect email addresses; one section is fine.
2. Intro text (copy exactly):
   > For each question, look at Outfit A and Outfit B. Which outfit's pieces go together better?
   > Some items are product photos and some are cut out of street photos: please judge the
   > combination of clothes, not the photo quality. Choose "Can't decide" if neither is better.
3. Add the 40 questions in the order of `Module4/cache/human_study/form.md`. Each: multiple choice,
   title exactly `Qnn. Which outfit goes together better?`, the image `pairs/Qnn.png`, options
   `Outfit A`, `Outfit B`, `Can't decide`, required.
4. Fill it in yourself once to check the order and images, then delete that response.
5. Recruit **10–15 raters who are not on the team** (classmates, friends). About 5 minutes each.
6. By 23 Nov 14:00: Responses → Download (.csv). Hand back the file.

**Scoring** (done on the demo machine): `python Module4/scripts/human_study.py score --responses <file.csv>`
gives the share preferring Module 4's swap, the share preferring the random swap, and the lift
between them with a 95% interval over raters.

**On the slide:** "In a blind pilot with N raters, Module 4's swap was preferred in X% of
comparisons vs Y% for a random allowed swap (lift Z, 95% CI …)". A small pilot, stated as such.

---

## Kit 2: Phone-photo test set

**Question:** does the pipeline work on what a real user uploads (a phone photo), not only on
dataset photos? The same photos supply the live demo.

**Steps**
1. Each team member who agrees takes **5 photos of their own outfit** with a phone:
   - standing, full body (head to feet), outdoors in daylight
   - standing, full body, indoors under normal room light
   - a mirror selfie
   - sitting, or cropped at the knees
   - a **deliberately clashing** outfit (pieces you would not wear together), full body
   One person per photo. Photos of people who did not agree are not allowed.
2. Save as JPEG (on iPhone: Settings → Camera → Formats → Most Compatible, or export as JPEG).
3. Put them in `samples/phone_set/` on the demo machine (local only, never in git) with a
   `manifest.csv`:

   | photo | condition | my_weak_piece |
   |---|---|---|
   | p1_outdoor.jpg | outdoor | |
   | p1_clash.jpg | clash | pants |

   `condition`: outdoor / indoor / mirror / sitting / clash. `my_weak_piece`: for clash photos only,
   the piece *you* would change first (top, pants, skirt, shoes, …).

**Run** (done on the demo machine): `python Module4/scripts/photo_batch.py --photos samples/phone_set` →
`results.csv` (status, garments found, harmony, verdict, piece to swap, suggestions, seconds per
photo) and `sheet.jpg` (every photo's garment overlay with its verdict).

**On the slide:** success rate on phone photos by condition, seconds per photo, and for clash
photos how often Module 4 picks the same piece as the wearer. Demo photos are chosen from this set.

---

## Kit 3: Usability test of the demo

**Question:** can a first-time user run the demo and understand the result, and what breaks?
Feeds the frontend overhaul.

**Setup:** `python demo.py` on the demo machine; note how long the first run takes.

**Checklist** (tick each, log anything odd in the bug log below)
- [ ] First run time (cold start), second run time
- [ ] Module 1 alone: garment figure readable, labels correct
- [ ] Module 2 alone: harmony score shown and explained
- [ ] Module 3 alone: per-piece trend and the "least on-trend" piece
- [ ] Module 4 alone: verdict, 3 suggestions, "also worth improving", per-piece table
- [ ] All: everything above in one run
- [ ] Trend slider: re-ranks without a long wait; the suggestions change sensibly
- [ ] "Try this swap": the board shows the right piece swapped
- [ ] Advanced options (pool, same type, rule): each one changes something or is clearly explained
- [ ] Photo A then photo B: nothing from A is left on screen
- [ ] Odd inputs, each: group photo · mirror selfie · product-page screenshot · flat-lay (clothes on a
      bed) · landscape with no person · pet photo (`samples/nonhuman/`) · very large photo (> 10 MB)
      · very small photo (< 200 px) · PNG with transparency · a non-image file
- [ ] Without reading any docs: is it clear what to do first? What does "refresh" mean to you?
      What do the harmony and trend numbers mean?

**Bug log** (copy into a sheet)

| # | Step | What I did | Expected | What happened | Severity (blocker / major / minor / cosmetic) | Screenshot |
|---|---|---|---|---|---|---|

**On the slide:** checklist pass rate, number of issues by severity, and what the overhaul fixed.

---

## Kit 4: Catalogue visual QA

**Question:** of the H&M items the demo can suggest, how many look wrong? Code filters the
catalogue (crop gate) but nobody has looked at what passes.

**Already built** (`python Module4/scripts/catalogue_qa.py sheets`): 30 random suggestable items per garment
type, 16 types, 480 items, on numbered sheets `Module4/cache/catalogue_qa/<type>.jpg`, plus
`marks.csv` (one row per item: type, cell number, item id, `mark`, `comment`).

**Steps** (about 45 minutes)
1. Open a sheet and `marks.csv` side by side (Excel or Google Sheets).
2. For each **bad** item, find its row (type + cell number) and write one `mark`:
   - `background`: studio background or other non-garment pixels left in the cut-out
   - `wrong_type`: not the type the sheet is for (e.g. a bag on the belt sheet)
   - `unreadable`: cut off, broken, blurry or too small to tell what it is
   - `not_adult`: a children's or baby item
   - `other`: anything else, and say what in `comment`

   Leave `mark` **blank** for items that look fine.
3. Save as CSV (keep the columns and the file name) and hand it back.

**Scoring:** `python Module4/scripts/catalogue_qa.py score --marks <file>` → defect rate per type and overall.

**On the slide:** "Manual QA of 480 random suggestion items: X% flagged (mostly …)".

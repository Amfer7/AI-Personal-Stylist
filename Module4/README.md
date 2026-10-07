# Module 4: Swap Recommendation

**Status: designed, not built yet.**

**In one line:** given an outfit photo, says **which one piece to change and what to change it to**,
using Module 2 (harmony) and Module 3 (trend) to judge every possible replacement. If the outfit is
already strong, it says so.

```
your outfit ──► for each piece: try every allowed replacement ──► score each "what-if" outfit
                                                                   (M2 harmony + M3 trend)
            ──► pick the piece whose replacements help most ──► top 3 suggestions, or "keep it"
```

- Plain-language design (all 11 decisions explained): [`docs/module4-design.md`](../docs/module4-design.md)
- Detailed engineering spec (alternatives, evidence, file references, tests): [`docs/module4-design-detailed.md`](../docs/module4-design-detailed.md)

Build order: **Phase 0** (bring full-resolution Pinterest garments into the catalogue and settle the
trend formula), then the recommender, evaluation and demo. See the design docs.

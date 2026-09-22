# Centralized P2P figure QA

- Claim: the centralized solution routes surplus energy through several edges of
  the four-prosumer trading graph, including reverse flow on edge 2 → 3.
- Figure type: single-panel signed-trade heatmap; panel alignment is not applicable.
- Source data: `centralized_p2p_source_data.csv`.
- Source validation: 21 checks passed with no warnings or failures; the figure
  uses the standard 88.9 mm single-column width.
- PDF text audit: passed; minimum rendered text size is 7 pt.
- Collision audit: passed with no warnings or failures.
- Visual inspection: cell labels, sign-dependent colours, axes and colour bar are
  legible, with no clipping or ambiguous legend encoding.

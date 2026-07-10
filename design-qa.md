# Design QA: Guided Training Setup

## Comparison target

- Source visual truth:
  - `docs/evidence/guided-training-setup/before-home.png`
  - `docs/evidence/guided-training-setup/before-train.png`
- Rendered implementation:
  - `docs/evidence/guided-training-setup/after-home.png`
  - `docs/evidence/guided-training-setup/after-train-missing-data.png`
  - `docs/evidence/guided-training-setup/after-train-ready-command.png`
- Combined full-view evidence:
  - `docs/evidence/guided-training-setup/comparison-home.png`
  - `docs/evidence/guided-training-setup/comparison-train.png`
- Viewport: 1280 × 720, desktop, light theme.
- States: Home with missing SFT data; Train in simple mode with the current preset;
  Train with a temporary ready dataset and a freshly saved command.

## Findings

- No actionable P0, P1, or P2 findings remain.
- Fonts and typography: Plus Jakarta Sans and the existing heading/body hierarchy are
  unchanged. New status and guidance copy wraps cleanly at the target viewport.
- Spacing and layout rhythm: the existing sidebar, four-column cards, section gaps,
  radii, and elevations are preserved. Home replaces repeated content with one compact
  status row and one next-action region. Train keeps a stable four-card preset grid.
- Colors and visual tokens: the existing coral/orange gradients, slate text, pale
  surfaces, semantic alerts, shadows, and dividers remain consistent with the source.
- Image quality and asset fidelity: the target contains no raster product imagery.
  Existing application icons and brand treatment are preserved; no replacement assets
  or placeholder imagery were introduced.
- Copy and content: readiness copy distinguishes public/local model access from gated
  model authentication, names the method-specific dataset, and explains why the next
  action is recommended.
- Interaction and accessibility: sidebar navigation remains a semantic radio group;
  primary actions remain buttons; preset selection has a persistent textual state;
  headings and alerts expose the updated hierarchy and blockers.
- Responsive scope: no clipping or horizontal overflow was observed at 1280 × 720.
  Mobile layout changes were intentionally excluded from this desktop-first PR.

Focused-region comparison was not required because the edited Home status/next-action
region and Train preset region are fully readable in the same-viewport full-view
comparisons. The new command state is captured separately because it has no source-state
equivalent.

## Interaction verification

- Home primary action navigated to Prepare Data and selected the matching sidebar item.
- Prepare Data's `Go to Training` action navigated to Train and selected its sidebar item.
- Missing SFT data showed a blocker, allowed configuration saving, and did not expose a
  training command.
- A temporary SFT dataset enabled `Save Configuration & Show Command`; the command
  appeared only after saving the current config.
- Changing from Balanced to Quick Test after saving hid the stale command and prompted
  the user to save again.
- The preset summary reported Maximum Quality, Balanced, and Quick Test as their
  corresponding settings were applied.
- Browser console warnings/errors checked: none.

## Comparison history

1. Initial implementation pass found one P2: the Base Model and Hugging Face metric
   values truncated at 1280 px.
2. The values were shortened to `Ready`/`Optional`, while the model/access detail moved
   into the smaller delta line.
3. Post-fix evidence in `after-home.png` shows all four values without truncation and no
   remaining P0/P1/P2 findings.

## Evidence limits

- The in-app browser intermittently rendered a black compositor artifact over the
  sidebar after scrolling on Prepare Data. That screenshot was excluded from fidelity
  judgment; the page was unchanged visually by this PR and its navigation state was
  verified through the accessibility tree.
- Full keyboard-only and assistive-technology testing were not performed, so this report
  does not claim complete accessibility compliance.

## Follow-up polish

- P3: a future responsive pass could stack dashboard cards and preset choices for narrow
  windows; this is outside the approved desktop-first scope.

final result: passed

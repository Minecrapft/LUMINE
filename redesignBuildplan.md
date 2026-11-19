# LUMINE Website Redesign Build Plan

## Goal

Refresh the visual design across the LUMINE website without changing its
screening, patient, result, dataset, or model-training behavior. Use the
provided LUMINE logo and make page layouts comfortable at normal zoom and
usable when users zoom in. Optimize controls for a 7.5-inch touchscreen so
buttons and key actions are easy to hit without a stylus.

## Design direction

LUMINE is a clinical screening workspace. Its design should feel calm,
legible, and precise, with the supplied blue-and-white logo providing the
recognizable brand mark. The UI should prioritize patient and screening
information over decorative dashboard chrome.

### Tokens

- **Ink navy — `#10233F`:** navigation, strong text, and high-contrast anchors.
- **Royal blue — `#2458D3`:** primary actions, current navigation, and focus.
- **Mist blue — `#EAF2FF`:** selected items and secondary information surfaces.
- **Paper — `#F6F8FC`:** application canvas.
- **White — `#FFFFFF`:** forms, result panels, and content surfaces.
- **Clinical green / amber / red — `#16835D`, `#A8630A`, `#B83A46`:** preserve
  meaning for normal, attention, and follow-up states.

Typography uses a system sans-serif stack for readable, responsive text without
requiring a network font to load.

### Layout studies

**A. Fixed desktop console**

```text
+----------------------------------------------------+
| Logo       Dashboard  Patients  Screening  Status  |
+----------------------------------------------------+
| Page title                                         |
| Main content in fixed-width columns                |
+----------------------------------------------------+
```

This resembles the existing structure but risks clipping or horizontal
overflow at higher zoom.

**B. Reflowing clinical workspace (selected)**

```text
+----------------------------------------------------+
| Logo       Navigation (wraps or scrolls)   Status  |
+----------------------------------------------------+
| Page title and primary action                      |
| Key information                                    |
| Content grid: columns collapse as space narrows    |
| Tables and image panels adapt to narrow viewports  |
+----------------------------------------------------+
```

Content remains left-aligned for scanning. Layouts use fluid widths, flexible
gaps, and minimum sizes that allow grids to collapse naturally. At narrow
effective widths, navigation and actions wrap and multi-column content stacks.

## Review against the brief

- A fixed console would preserve the current desktop feel but repeat the
  layout weakness reported at zoom. Reject it in favor of the reflowing
  workspace.
- Avoid generic gradients, oversized hero blocks, decorative statistics, and
  card grids without hierarchy. Use blue selectively for navigation and
  actions; use status colors only where they communicate clinical workflow.
- Treat the attached logo as the source of the brand identity rather than
  recreating it with a generic letter badge.

## Scope and implementation

1. Copy the supplied logo into the app's static assets and use it in the shared
   navigation without changing route links.
2. Redesign the shared shell: brand/navigation/status header, content width,
   page background, footer, typography, focus indicators, and responsive
   navigation wrapping.
3. Update shared CSS tokens and common controls/surfaces while retaining
   existing selectors and form/table states used by page scripts.
4. Review dashboard, patient management, new screening, camera, results,
   history, insights, system, annotation, and ML Studio styling for fixed
   widths and overflow. Make visual/layout changes only; preserve element IDs,
   data attributes, form names, route targets, and JavaScript behavior.
5. Keep page-specific styles where needed, overriding them narrowly from the
   shared stylesheet rather than rewriting the workflows.

## Responsive and zoom acceptance checks

- No horizontal page overflow at a 320 CSS-pixel viewport.
- At 200% browser zoom, navigation, headings, forms, patient actions, result
  panels, and training controls remain readable and operable.
- At 400% zoom on a standard desktop viewport, the layout reflows to a narrow
  column; text and controls do not escape their containers.
- Tables and image grids remain usable through wrapping, stacking, or
  intentional local scrolling rather than forcing the whole page wider.
- On the 7.5-inch touch display, primary buttons and navigation links have at
  least 48 CSS pixels of hit height; compact icon and secondary controls have
  an enlarged touch target with visible spacing between adjacent actions.
- Camera capture and screening actions are visually prominent and remain easy
  to press at the target display size.
- Keyboard focus is visible; reduced-motion preferences are respected.
- Verify representative routes and interactions without changing their
  behavior.

## Status

- [x] Inspect shared templates, styles, and representative page markup.
- [x] Choose and review the design direction.
- [x] Integrate the supplied logo and update the shared design system.
- [x] Apply responsive visual refinements across the app pages.
- [x] Run regression tests and browser checks at narrow widths / increased zoom.
- [x] Enlarge touchscreen hit targets and verify spacing at a compact tablet
  viewport.
- [x] Audit touch targets across all primary routes, patient detail/actions,
  history tabs, screening steps, annotation controls, and ML Studio tabs.

## Verification

- `tests/test_analysis.py`: 84 passed.
- Browser smoke check: dashboard, patients, new screening, insights, system,
  ML Studio, and annotation routes loaded successfully.
- Responsive check: no page-level horizontal overflow on those routes at
  viewport widths down to 308 CSS pixels; the patient table remains locally
  scrollable instead of widening the page.
- Interaction spot checks: the new-patient modal opens within the viewport,
  and ML Studio's Training tab still switches correctly.
- Touchscreen refinement: at a compact 800x480 preview viewport, visible
  buttons, navigation links, form controls, and dataset tabs have at least
  48 CSS pixels of hit height. Patient suggestions and the dashboard's
  "View All" link receive expanded targets too.
- Touchscreen interaction check: patient modal, new-patient modal, ML Studio
  training action, and dataset creation modal all expose 48-pixel buttons.
  Five representative pages showed no page-level horizontal overflow.
- Full touchscreen audit at a compact CSS viewport covered Dashboard, Patients
  (including expanded rows and both patient modals), New Screening sample/test
  steps, Camera, Results, History Timeline/Trends/Summary, Insights, System,
  Annotation and its dataset modal, plus all ML Studio tabs and all four upload
  subtabs. Visible action controls were at least 48 CSS pixels high; checkbox
  inputs remain 24 pixels with a 48-pixel associated label target.
- The patient action column now preserves readable 112-pixel-wide controls in
  its locally scrollable table. ML Studio Models and Jobs tables also scroll
  inside their cards rather than widening the page. All audited routes remained
  free of page-level horizontal overflow at the compact viewport.

## Known input constraint

The supplied logo is a raster image with a black background. Preserve its
appearance as-is and place it against the dark navigation surface so the
background does not appear as an unrelated tile.

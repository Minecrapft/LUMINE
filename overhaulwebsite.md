# LUMINE Website Overhaul

## Purpose

Make LUMINE feel like a calm, approachable guide for people who are not medical
professionals. The main task is to help a person prepare a strip, capture a
usable photo, and understand what the screening can and cannot tell them.

The primary layout target is a portrait touch display with an 8:13
width-to-height ratio. The interface must also reflow cleanly on phones,
tablets, and desktop screens.

## Design direction: Pocket field guide

Use the visual language of a simple field guide: direct instructions, clearly
sequenced steps, and a small set of colors inspired by the cool light and
colored pads of a test strip. The interface should feel warm and capable, not
like a hospital chart or a generic analytics dashboard.

### Color tokens

| Token | Name | Hex | Use |
|---|---|---|---|
| Ink | Deep kelp | `#17383D` | Main text, navigation |
| Canvas | Mist | `#F2F7F5` | Page background |
| Surface | Paper | `#FFFFFF` | Reading surfaces |
| Primary | Lagoon | `#08756D` | Main actions and current location |
| Accent | Citrus | `#CE8B39` | Instruction steps and gentle emphasis |
| Alert | Muted red | `#A23E3E` | Important warnings only |

Use text and icons as well as color to communicate status. Keep contrast strong
and reserve alert color for information that needs attention.

### Type

- Use a familiar system sans-serif stack with a humanist fallback such as
  Trebuchet MS. Do not require an internet connection or a downloaded font.
- Use sentence case, short labels, readable body text, and generous line height.
- Put the action or most important information first.

### Layout options considered

**A. Guide-first home (chosen):** one main action, an interactive sequence, then
recent activity and summary counts.

```text
┌ LUMINE                         ┐
│ Welcome. Let's get started.    │
│ [ Start a screening ]          │
│                                │
│ A quick guide                  │
│ 1  Get the strip ready     [⌄] │
│ 2  Keep the full strip in view │
│ 3  Review what LUMINE noticed  │
│                                │
│ Recent screenings              │
│ ...                            │
└ Home · Screen · People · ... ──┘
```

**B. Stats-first dashboard (existing direction):** counts and outcomes above the
recent screening list.

```text
┌ four counts / outcomes ────────┐
│ trends and screening totals    │
│ recent rows                    │
└ navigation ────────────────────┘
```

Option A is preferred because a first-time user needs a next step and plain
instructions before they need aggregate counts. Keep useful counts available,
but make them secondary to starting and completing a screening.

On wide screens, keep the welcome, guide, and safety note full-width, then place
recent screenings beside the activity summary. On an 8:13 portrait screen, stack
modules in task order and use a bottom navigation dock that remains reachable
without covering content.

## Interaction and accessibility

- Make important touch controls at least 48 CSS pixels tall and provide
  comfortable spacing.
- Use native, keyboard-operable disclosures for the dashboard guide steps.
- Keep visible keyboard focus, meaningful heading order, and descriptive
  accessible names for controls.
- Respect `prefers-reduced-motion`; motion should only clarify a user action.
- Do not rely on hover for instructions or important actions.
- Keep controls and content usable at narrow widths without horizontal
  scrolling. Account for the 8:13 portrait ratio as well as small screens.

## Plain-language guidance

- Prefer “strip readings” over “markers,” “choose a person” over “select
  patient,” and “take a photo” over “capture.”
- Explain unfamiliar terms in the screen where they appear.
- Use status wording that does not sound like a diagnosis. State what the
  screening found and what action the person can take next.
- Keep the limitation visible: LUMINE is a screening aid, not a diagnosis or a
  replacement for professional care.
- Use short sentences, active voice, headings, and lists.

## Dashboard instruction sequence

1. **Get the strip ready.** Follow the package directions, including timing.
2. **Keep the whole strip in view.** Use steady lighting, avoid glare, and make
   sure every pad and both ends fit inside the camera guide.
3. **Review the reading.** Explain that the app compares the photo with stored
   color references; it does not diagnose a condition.
4. **Choose a next step.** If the result is unclear or recommends follow-up,
   encourage the person to speak with a qualified health professional.

Keep the first step visible by default. Let users open the other steps as
needed. Each step should be a useful instruction, not a decorative card.

## References

- W3C, [WCAG 2.2: Target Size (Minimum), Success Criterion 2.5.8](https://www.w3.org/WAI/WCAG22/Understanding/target-size-minimum.html).
- W3C, [WCAG 2.2: Target Size (Enhanced), Success Criterion 2.5.5](https://www.w3.org/WAI/WCAG22/Understanding/target-size-enhanced.html). The implementation aims for at least 44 CSS pixels for primary touch targets.
- CDC, [Plain Language Materials & Resources](https://www.cdc.gov/health-literacy/php/develop-materials/plain-language.html). The content guidance follows its advice to know the audience, put key information first, use everyday words, and organize text with headings and lists.

## Validation checklist

- Check the dashboard and key screening pages at an 8:13 portrait viewport, a
  narrow phone viewport, and a desktop viewport.
- Confirm the guide disclosures work with touch, keyboard, and screen readers.
- Confirm primary controls are easy to tap and focus indicators remain visible.
- Check that the dashboard explains the flow without implying a diagnosis.
- Run the existing focused tests and check the browser for layout overflow and
  JavaScript errors.

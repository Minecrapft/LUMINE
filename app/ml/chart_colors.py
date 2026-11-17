"""RGB reference charts for urine and saliva strip profiles."""

CHART_VERSION = "3.0.0"

# Generic reference colors for profiles without a supplied product chart.
URINE_CHART = {
    "glucose": [
        ("negative", [36, 92, 168]),
        ("trace", [48, 132, 142]),
        ("+1", [86, 148, 78]),
        ("+2", [168, 142, 62]),
        ("+3", [186, 108, 48]),
        ("+4", [122, 68, 38]),
    ],
    "bilirubin": [
        ("negative", [246, 228, 176]),
        ("+1", [232, 176, 132]),
        ("+2", [208, 118, 96]),
        ("+3", [158, 68, 78]),
    ],
    "ketones": [
        ("negative", [246, 228, 196]),
        ("trace", [230, 186, 168]),
        ("+1", [208, 132, 148]),
        ("+2", [176, 78, 128]),
        ("+3", [138, 42, 98]),
        ("+4", [88, 28, 68]),
    ],
    "specific_gravity": [
        ("1.000", [8, 48, 138]),
        ("1.005", [22, 82, 158]),
        ("1.010", [42, 124, 138]),
        ("1.015", [72, 152, 86]),
        ("1.020", [142, 172, 48]),
        ("1.025", [192, 182, 42]),
        ("1.030", [222, 188, 48]),
    ],
    "blood": [
        ("negative", [246, 208, 72]),
        ("trace", [198, 188, 68]),
        ("+1", [138, 158, 58]),
        ("+2", [68, 118, 78]),
        ("+3", [28, 78, 68]),
    ],
    "pH": [
        ("5.0", [228, 88, 48]),
        ("6.0", [240, 158, 48]),
        ("6.5", [230, 198, 58]),
        ("7.0", [176, 198, 68]),
        ("7.5", [98, 168, 88]),
        ("8.0", [48, 138, 118]),
        ("8.5", [38, 98, 138]),
    ],
    "protein": [
        ("negative", [246, 218, 86]),
        ("trace", [218, 208, 76]),
        ("+1", [158, 188, 68]),
        ("+2", [78, 148, 88]),
        ("+3", [38, 108, 78]),
        ("+4", [28, 78, 68]),
    ],
    "urobilinogen": [
        ("normal", [246, 228, 158]),
        ("abnormal", [198, 78, 88]),
    ],
    "nitrite": [
        ("negative", [248, 244, 238]),
        ("positive", [218, 78, 138]),
    ],
    "leukocytes": [
        ("negative", [246, 232, 196]),
        ("trace", [228, 206, 176]),
        ("+1", [198, 156, 176]),
        ("+2", [158, 98, 158]),
        ("+3", [118, 58, 138]),
    ],
}

# URS-4K reference swatches sampled from the chart images supplied by the user.
# RGB values are median samples from each cropped swatch; display/capture color
# shifts mean these are image-derived references, not instrument calibration data.
URS_4K_CHART = {
    "glucose": [
        ("negative", [142, 197, 199]),
        ("trace", [152, 204, 165]),
        ("+1", [145, 166, 65]),
        ("+2", [156, 130, 34]),
        ("+3", [147, 113, 54]),
        ("+4", [128, 79, 47]),
    ],
    "pH": [
        ("5.0", [250, 166, 53]),
        ("6.0", [251, 186, 100]),
        ("6.5", [252, 206, 100]),
        ("7.0", [211, 190, 36]),
        ("7.5", [167, 176, 56]),
        ("8.0", [91, 148, 94]),
        ("8.5", [51, 132, 130]),
    ],
    "protein": [
        ("negative", [231, 237, 153]),
        ("trace", [209, 229, 153]),
        ("+1", [182, 209, 154]),
        ("+2", [152, 204, 164]),
        ("+3", [124, 199, 185]),
        ("+4", [116, 182, 167]),
    ],
    "ketones": [
        ("negative", [252, 189, 153]),
        ("trace", [248, 171, 145]),
        ("+1", [240, 132, 142]),
        ("+2", [200, 87, 113]),
        ("+3", [152, 56, 101]),
        ("+4", [121, 37, 91]),
    ],
}

# Four-pad universal pH chart sampled from the product images supplied by the
# user. Each pH is a four-colour pattern; these image-derived values are
# approximate and are not instrument or manufacturer calibration data.
SALIVA_PH4_PAD_CHART = {
    "0":  [[220, 109, 63], [220, 140, 3], [143, 202, 176], [226, 176, 3]],
    "1":  [[210, 125, 65], [219, 139, 2], [140, 201, 174], [223, 176, 2]],
    "2":  [[223, 165, 3], [222, 157, 3], [142, 193, 121], [226, 176, 1]],
    "3":  [[183, 131, 29], [223, 161, 2], [143, 194, 120], [227, 177, 2]],
    "4":  [[101, 84, 53], [214, 162, 0], [143, 191, 110], [225, 175, 2]],
    "5":  [[55, 56, 50], [183, 148, 2], [144, 192, 118], [225, 175, 2]],
    "6":  [[50, 51, 53], [138, 134, 8], [140, 192, 116], [225, 175, 2]],
    "7":  [[51, 50, 68], [72, 107, 40], [144, 194, 117], [226, 176, 9]],
    "8":  [[65, 48, 102], [27, 87, 62], [151, 194, 125], [226, 157, 38]],
    "9":  [[65, 48, 102], [31, 62, 65], [152, 188, 90], [230, 154, 42]],
    "10": [[65, 48, 102], [36, 58, 81], [163, 95, 96], [241, 173, 44]],
    "11": [[65, 48, 102], [36, 58, 81], [112, 30, 76], [233, 165, 40]],
    "12": [[65, 48, 102], [36, 58, 81], [148, 40, 56], [244, 121, 54]],
    "13": [[65, 48, 102], [36, 58, 81], [178, 24, 47], [241, 117, 53]],
    "14": [[65, 48, 102], [37, 57, 81], [188, 27, 43], [231, 106, 44]],
}

SALIVA_PH4_CHART = {
    "pH": [
        (
            level,
            [
                round(sum(pad[index] for pad in pad_colors) / len(pad_colors))
                for index in range(3)
            ],
        )
        for level, pad_colors in SALIVA_PH4_PAD_CHART.items()
    ]
}

# Research saliva chart: same pad chemistry, slightly paler wet-paper look.
SALIVA_CHART = {
    marker: [(level, [min(255, int(c * 1.08 + 12)) for c in rgb]) for level, rgb in levels]
    for marker, levels in URINE_CHART.items()
}

PROFILE_CHARTS = {
    "URS-4K": URS_4K_CHART,
    "URS-10T":   URINE_CHART,
    "URS-4K-4P": URS_4K_CHART,
    "SALIVA-4K": SALIVA_CHART,
    "SALIVA-PH4": SALIVA_PH4_CHART,
}

# ---------------------------------------------------------------------------
# Convenience accessor used by tests and db_manager
# ---------------------------------------------------------------------------

def get_chart(profile_id: str) -> dict:
    """Return the colour chart for a given profile ID, or {} if not found."""
    return PROFILE_CHARTS.get(profile_id, {})

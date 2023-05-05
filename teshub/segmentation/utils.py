DEFAULT_ID2COLOR: dict[int, tuple[int, ...]] = {
    0: (0, 0, 0),
    1: (22, 21, 22),
    2: (204, 204, 204),
    3: (46, 6, 243),
    4: (154, 147, 185),
    5: (198, 233, 255),
    6: (255, 53, 94),
    7: (250, 250, 55),
    8: (255, 255, 255),
    9: (115, 51, 128),
    10: (36, 179, 83),
    11: (119, 119, 119),
}
DEFAULT_COLOR2ID = {
    color: id for (id, color) in DEFAULT_ID2COLOR.items()
}

DEFAULT_ID2LABEL: dict[int, str] = {
    0: "background",
    1: "black_clouds",
    2: "white_clouds",
    3: "blue_sky",
    4: "gray_sky",
    5: "white_sky",
    6: "fog",
    7: "sun",
    8: "snow",
    9: "shadow",
    10: "wet_ground",
    11: "shadow_snow"
}
DEFAULT_LABEL2ID = {
    label: id for (id, label) in DEFAULT_ID2LABEL.items()
}

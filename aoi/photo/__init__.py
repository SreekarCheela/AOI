"""Photo mode: placement inspection from ordinary photographs.

The station (aoi/pipeline.py) inspects registered four-channel stacks at a fixed
pitch. Photo mode asks a narrower question with nothing but pictures: compared
with photographs of a board known to be good, is each part there, the right way
round, and where it should be? Missing, tombstone, wrong part, marking,
polarity, offset and skew. Solder quality needs the rings and is out of scope.

    io        8-bit BGR loading, EXIF orientation, HEIC
    model     Part, Call, PartResult, BoardResult
    program   a taught board: goldens, parts, scale, golden statistics
    register  whole-photo alignment onto golden 1
    features  per-part measurements
    judge     measurements -> calls -> part state -> board verdict
    teach     box proposals, scale, golden self-check, program build
    inspect   one test photo -> BoardResult
    gate      capture checks with stated reasons
    record    photo_aoi.db, runs and operator adjudications
    report    annotated JPEG, CSV, one-file HTML
"""

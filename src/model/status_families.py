"""
Status-family classification for Ergast's 140 race-result status codes.

Every (driverId, raceId) row in results.csv has a statusId pointing into
status.csv. Those 140 codes mix six conceptually distinct outcomes:

  finished     — driver completed the race on the lead lap or lapped
                 (the classified-finisher family)
  mechanical   — car-side failure (engine, gearbox, hydraulics, etc.)
  accident     — crash, collision, spun off, damage
  driver       — driver-side health issue (illness, injury, unwell)
  disqualified — DQ / excluded / underweight
  other        — withdrawals, DNQ, DNPQ, generic retirements, 107%-rule
                 (records we can't cleanly pin on any of the above)

We need this split for the Stage 1a rating update:

  - MECHANICAL DNFs are CENSORED (skipped from the driver's rating update).
    The failure is car-attributable; giving the driver a bad-day signal for
    it is the F1 equivalent of survivorship bias in equities — see the
    2-stage decomposition (driver rating conditional on car survival +
    separate constructor reliability model).

  - ACCIDENT DNFs are WEAK-WEIGHTED (kept in the update at reduced weight).
    Part driver (mistake), part situational (got taken out).

  - DRIVER-SIDE DNFs are CENSORED (out of scope for the rating — driver
    was ill/injured, no skill signal this race).

  - DISQUALIFIED: censored. The rule breach isn't about race performance.

  - OTHER: censored. Too ambiguous to extract signal from.

Only FINISHED rows contribute a full-weight update.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Keyword sets — classify by the status STRING so new 2026+ codes get sensibly
# slotted without a code change, as long as they follow Ergast naming.
# ---------------------------------------------------------------------------

MECHANICAL = {
    "Engine", "Gearbox", "Transmission", "Clutch", "Hydraulics", "Electrical",
    "Radiator", "Suspension", "Brakes", "Differential", "Overheating",
    "Mechanical", "Tyre", "Driver Seat", "Puncture", "Driveshaft",
    "Fuel pressure", "Front wing", "Water pressure", "Refuelling", "Wheel",
    "Throttle", "Steering", "Technical", "Electronics", "Broken wing",
    "Heat shield fire", "Exhaust", "Oil leak", "Wheel rim", "Water leak",
    "Fuel pump", "Track rod", "Oil pressure", "Engine fire", "Engine misfire",
    "Tyre puncture", "Out of fuel", "Wheel nut", "Pneumatics", "Handling",
    "Rear wing", "Fire", "Wheel bearing", "Fuel system", "Oil line", "Fuel rig",
    "Launch control", "Fuel", "Power loss", "Vibrations", "Drivetrain",
    "Ignition", "Chassis", "Battery", "Stalled", "Halfshaft", "Crankshaft",
    "Alternator", "Safety belt", "Oil pump", "Fuel leak", "Injection",
    "Distributor", "Turbo", "CV joint", "Water pump", "Spark plugs",
    "Fuel pipe", "Oil pipe", "Axle", "Water pipe", "Magneto", "Supercharger",
    "Power Unit", "ERS", "Brake duct", "Seat", "Undertray", "Cooling system",
}
ACCIDENT = {
    "Accident", "Collision", "Spun off", "Fatal accident", "Collision damage",
    "Damage", "Debris",
}
DRIVER = {
    "Physical", "Injured", "Injury", "Eye injury", "Driver unwell", "Illness",
}
DISQUALIFIED = {"Disqualified", "Excluded", "Underweight"}
OTHER_DNF = {
    "Retired", "Withdrew", "Not classified", "Did not qualify",
    "Did not prequalify", "Not restarted", "Safety", "Safety concerns",
    "107% Rule",
}

# Rating-update weights per family. Used by Stage 1a.
#   1.0  = full-weight observation (normal rating update)
#   0<w<1 = reduced-weight update
#   0.0  = censored (no update for this driver on this race)
FAMILY_UPDATE_WEIGHT = {
    "finished":     1.0,
    "accident":     0.3,
    "mechanical":   0.0,
    "driver":       0.0,
    "disqualified": 0.0,
    "other":        0.0,
}


def family_of(status: str) -> str:
    """Classify an Ergast status string into one of six families."""
    if status == "Finished" or status.startswith("+"):
        return "finished"
    if status in MECHANICAL:   return "mechanical"
    if status in ACCIDENT:     return "accident"
    if status in DRIVER:       return "driver"
    if status in DISQUALIFIED: return "disqualified"
    if status in OTHER_DNF:    return "other"
    return "other"


def update_weight(family: str) -> float:
    """Rating-update weight for a given status family."""
    return FAMILY_UPDATE_WEIGHT.get(family, 0.0)

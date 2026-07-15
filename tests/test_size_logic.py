"""Offline unit tests for dress size anchoring (backend/size_logic.py).

Run: python -m pytest tests/test_size_logic.py
"""
import size_logic as sl

DRESS_SIZES = ["XS", "S", "M", "L", "XL"]
DRESS_ITEM = {"size_range": ",".join(DRESS_SIZES), "category": "dress",
              "fabric": "polyester crepe"}


def _profile(bust_band, bust_cup, waist_cm, hip_cm):
    return {
        "height_cm": 158, "weight_kg": 50, "bust_band": bust_band,
        "bust_cup": bust_cup, "waist_cm": waist_cm, "hip_cm": hip_cm,
        "body_type": "hourglass",
    }


def test_dress_anchor_clamps_to_smallest_size_when_waist_and_hip_are_below_chart():
    # Bust (34B ~ 91.4cm) alone would nearest-match M on WOMENS_CHART_BUST,
    # but waist (63.5) and hip (83.8) both sit below the XS row on their
    # charts -- two of three measurements say "smaller than XS exists".
    # The old max(bust_idx, hip_idx) anchor picked M (then amber bumped to
    # L) purely off the single largest-indexed axis; the median-of-three
    # anchor must not let one outlier measurement override the other two.
    profile = _profile(34, "b", waist_cm=63.5, hip_cm=83.8)
    idx, clamped, _near = sl._anchor_size(profile, "dress", DRESS_SIZES)
    assert DRESS_SIZES[idx] == "XS"
    assert clamped is True


def test_dress_anchor_matches_bust_when_all_measurements_agree():
    # bust~93cm -> M, waist~81cm -> M, hip~99cm -> M: no disagreement, no
    # outlier to guard against.
    profile = _profile(36, "b", waist_cm=81, hip_cm=99)
    idx, clamped, _near = sl._anchor_size(profile, "dress", DRESS_SIZES)
    assert DRESS_SIZES[idx] == "M"
    assert clamped is False


def test_dress_anchor_falls_back_to_bust_hip_max_without_waist():
    # No waist_cm on the profile (optional field) -> only two votes, median
    # of two is the larger index, preserving the pre-fix bust/hip behavior.
    profile = _profile(34, "b", waist_cm=None, hip_cm=83.8)
    idx, _clamped, _near = sl._anchor_size(profile, "dress", DRESS_SIZES)
    # bust (34B ~ 91.4cm) nearest to M; hip (83.8) clamped to XS -- median
    # of two sorted votes takes the higher index (index 1 of 2), same as
    # the old max() behavior when waist isn't available.
    assert DRESS_SIZES[idx] == "M"


def test_recommend_womens_size_no_longer_jumps_three_sizes_for_petite_hourglass():
    # End-to-end: the exact profile/item combination reported live (a 158cm
    # hourglass frame with hip/waist below this dress's own XS row) used to
    # come back recommending L (anchor jumped bust->M, then the amber
    # borderline rule bumped once more). It should now anchor at XS and
    # stay there -- there's no smaller size to shift to even if the model
    # flags "large" at that anchor.
    profile = _profile(34, "b", waist_cm=63.5, hip_cm=83.8)
    rec = sl.recommend_womens_size(profile, DRESS_ITEM)
    assert rec["recommended_size"] == "XS"
    assert rec["anchor_size"] == "XS"

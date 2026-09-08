from vla_gap_lab.mikasa_assets import (
    MU_VLA_YCB_FILENAME,
    MU_VLA_YCB_REPO,
    MU_VLA_YCB_REVISION,
    MU_VLA_YCB_SHA256,
    expected_ycb_provenance,
    ycb_asset_issues,
)


def test_ycb_provenance_accepts_only_b15_archive():
    expected = expected_ycb_provenance()
    assert expected == {
        "repo_id": MU_VLA_YCB_REPO,
        "revision": MU_VLA_YCB_REVISION,
        "filename": MU_VLA_YCB_FILENAME,
        "archive_sha256": MU_VLA_YCB_SHA256,
    }
    assert ycb_asset_issues(expected) == []


def test_ycb_provenance_rejects_current_hf_archive():
    current_main = {
        **expected_ycb_provenance(),
        "revision": "main",
        "archive_sha256": "1551724fd1ac7bad9807ebcf46dd4a788caed5c9499c1225b9bfa080ffbefcb3",
    }
    issues = ycb_asset_issues(current_main)
    assert any(MU_VLA_YCB_REVISION in issue for issue in issues)
    assert any(MU_VLA_YCB_SHA256 in issue for issue in issues)


def test_ycb_provenance_rejects_missing_marker():
    issues = ycb_asset_issues(None)
    assert len(issues) == 1
    assert "missing YCB asset provenance marker" in issues[0]

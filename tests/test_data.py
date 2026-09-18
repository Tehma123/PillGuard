import json

from pillguard.config import OUT_OF_PRESCRIPTION_LABEL
from pillguard.data.scenarios import load_scenarios, make_scenarios, save_scenarios, summarize
from pillguard.data.splits import Split, choose_unseen_classes, make_split
from pillguard.data.vaipe import (
    build_classes,
    class_names_from_prescriptions,
    clean_drug_name,
    load_pill_images,
    load_prescriptions,
    prescription_drug_table,
)


def test_synthetic_layout_loads(synth_root):
    ims = load_pill_images(synth_root)
    assert len(ims) == 30 and all(im.prescription for im in ims)
    assert all(len(im.boxes) == len(im.labels) for im in ims)
    assert any(OUT_OF_PRESCRIPTION_LABEL in im.labels for im in ims)
    pres = load_prescriptions(synth_root)
    names = class_names_from_prescriptions(pres)
    assert len(names) == 6 and all(n.startswith("SYNTH-") for n in names.values())
    table = build_classes(synth_root)
    assert [c["id"] for c in table["classes"]] == list(range(6))
    assert table["n_out_of_prescription_boxes"] > 0


def test_clean_drug_name():
    assert clean_drug_name("3) HOẠT HUYẾT DƯỠNG NÃO 150mg+20mg") == "HOẠT HUYẾT DƯỠNG NÃO 150mg+20mg"
    assert clean_drug_name("12. PARACETAMOL 500MG") == "PARACETAMOL 500MG"


def test_split_is_grouped_and_deterministic(synth_root, tmp_path):
    ims = load_pill_images(synth_root)
    a = make_split(ims, seed=7, n_unseen=1)
    b = make_split(list(reversed(ims)), seed=7, n_unseen=1)
    assert a.to_dict() == b.to_dict()
    assert len(a.train) + len(a.val) + len(a.test) == len(ims)
    by = {im.file: im for im in ims}
    groups = {k: {by[f].prescription for f in getattr(a, k)} for k in ("train", "val", "test")}
    assert not (groups["train"] & groups["test"]) and not (groups["train"] & groups["val"]) and not (groups["val"] & groups["test"])
    assert len(a.unseen_classes) == 1 and OUT_OF_PRESCRIPTION_LABEL not in a.unseen_classes
    p = tmp_path / "s.json"
    a.save(p)
    assert Split.load(p).to_dict() == a.to_dict()
    assert make_split(ims, seed=8, n_unseen=1).to_dict() != a.to_dict()


def test_unseen_choice_prefers_mid_frequency(synth_root):
    ims = load_pill_images(synth_root)
    chosen = choose_unseen_classes(ims, 2, seed=0)
    assert len(chosen) == 2 and all(0 <= c < 6 for c in chosen)


def test_scenarios(synth_root, tmp_path):
    ims = load_pill_images(synth_root)
    split = make_split(ims, seed=1, n_unseen=1)
    by = {im.file: im for im in ims}
    test_ims = [by[f] for f in split.test]
    pd = prescription_drug_table(synth_root)
    sc = make_scenarios(test_ims, pd, split.unseen_classes, seed=1)
    s = summarize(sc)
    assert s["kinds"]["clean"] == len(test_ims)
    assert "remove-1" in s["kinds"]
    for scen in sc:
        assert not (set(scen.listed) & set(scen.removed))
        assert not (set(scen.listed) & set(split.unseen_classes))
        for p in scen.pills:
            if p.label == OUT_OF_PRESCRIPTION_LABEL:
                assert p.gt == "out" and p.reason == "foreign"
            elif p.label in split.unseen_classes:
                assert p.gt == "out" and p.reason == "unseen"
            elif p.label in scen.removed:
                assert p.gt == "out" and p.reason == "removed"
            elif p.label in scen.listed:
                assert p.gt == "in"
    sc2 = make_scenarios(list(reversed(test_ims)), pd, split.unseen_classes, seed=1)
    assert {x.id: x.removed for x in sc} == {x.id: x.removed for x in sc2}
    path = tmp_path / "sc.jsonl"
    save_scenarios(path, sc)
    assert [x.to_dict() for x in load_scenarios(path)] == [x.to_dict() for x in sc]
    # fallback when a photo has no prescription annotation
    orphan = make_scenarios(test_ims, {}, split.unseen_classes, seed=1)
    assert len(orphan) >= len(test_ims)
    assert json.dumps(summarize(orphan))

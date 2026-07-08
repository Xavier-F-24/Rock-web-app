from Rock_Streamlit.sections import breeding


def test_candidate_checkbox_key_is_stable():
    assert breeding.candidate_checkbox_key(42) == "breeding_candidate_checkbox_42"


def test_checkbox_parent_ids_are_empty_when_none_checked():
    assert breeding.resolve_checkbox_parent_ids([]) == (None, None)


def test_checkbox_parent_ids_use_first_checked_as_parent_a():
    assert breeding.resolve_checkbox_parent_ids([2]) == (2, None)


def test_checkbox_parent_ids_use_second_checked_as_parent_b():
    assert breeding.resolve_checkbox_parent_ids([2, 3]) == (2, 3)


def test_checkbox_parent_ids_ignore_extra_checked_ids():
    assert breeding.resolve_checkbox_parent_ids([2, 3, 4]) == (2, 3)

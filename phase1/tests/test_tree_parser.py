
import pytest

from rumor_graph.tree_parser import parse_tree


# Artificial nodes in the original Twitter15/16 format
ROOT = "['ROOT', 'ROOT', '0.0']"
SOURCE = "['u0', '100', '0.0']"
A = "['u1', '101', '0.45']"
B = "['u2', '102', '0.85']"
C = "['u3', '103', '1.15']"


def write_tree(tmp_path, lines, name="100.txt"):
    """Create a temporary tree file for testing."""

    path = tmp_path / name
    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return path


def test_valid_tree(tmp_path):
    path = write_tree(tmp_path, [
        f"{ROOT}->{SOURCE}",
        f"{SOURCE}->{A}",
        f"{SOURCE}->{B}",
        f"{A}->{C}",
    ])

    tree = parse_tree(path)

    assert tree.source_tweet_id == "100"
    assert len(tree.nodes) == 4
    assert len(tree.edges) == 3
    assert tree.root_index == 0

    assert tree.edges == (
        (0, 1),
        (0, 2),
        (1, 3),
    )

    assert tree.nodes[1].delay_minutes == 0.45


def test_root_only(tmp_path):
    path = write_tree(tmp_path, [
        f"{ROOT}->{SOURCE}",
    ])

    tree = parse_tree(path)

    assert len(tree.nodes) == 1
    assert len(tree.edges) == 0


def test_missing_root(tmp_path):
    path = write_tree(tmp_path, [
        f"{SOURCE}->{A}",
    ])

    with pytest.raises(ValueError, match="ROOT"):
        parse_tree(path)


def test_duplicate_edge(tmp_path):
    path = write_tree(tmp_path, [
        f"{ROOT}->{SOURCE}",
        f"{SOURCE}->{A}",
        f"{SOURCE}->{A}",
    ])

    tree = parse_tree(path)

    assert len(tree.nodes) == 2
    assert len(tree.edges) == 1


def test_multiple_parents(tmp_path):
    path = write_tree(tmp_path, [
        f"{ROOT}->{SOURCE}",
        f"{SOURCE}->{A}",
        f"{SOURCE}->{B}",
        f"{B}->{A}",
    ])

    with pytest.raises(
        ValueError,
        match="multiple parents",
    ):
        parse_tree(path)


def test_wrong_filename(tmp_path):
    path = write_tree(
        tmp_path,
        [f"{ROOT}->{SOURCE}"],
        name="999.txt",
    )

    with pytest.raises(
        ValueError,
        match="filename",
    ):
        parse_tree(path)


def test_cycle(tmp_path):
    path = write_tree(tmp_path, [
        f"{ROOT}->{SOURCE}",
        f"{A}->{B}",
        f"{B}->{A}",
    ])

    with pytest.raises(
        ValueError,
        match="Disconnected|cycle",
    ):
        parse_tree(path)


def test_broken_line(tmp_path):
    path = write_tree(tmp_path, [
        "bad line without an arrow",
    ])

    with pytest.raises(
        ValueError,
        match="Line 1",
    ):
        parse_tree(path)


def test_same_tweet_multiple_users(tmp_path):
    repost = "['u1', '100', '1.2']"

    path = write_tree(tmp_path, [
        f"{ROOT}->{SOURCE}",
        f"{SOURCE}->{repost}",
    ])

    tree = parse_tree(path)

    assert len(tree.nodes) == 2
    assert tree.nodes[0].tweet_id == tree.nodes[1].tweet_id
    assert tree.nodes[0].user_id != tree.nodes[1].user_id


def test_nonfinite_delay(tmp_path):
    bad = "['u1', '101', 'nan']"

    path = write_tree(tmp_path, [
        f"{ROOT}->{SOURCE}",
        f"{SOURCE}->{bad}",
    ])

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        parse_tree(path)


def test_lines_any_order(tmp_path):
    path = write_tree(tmp_path, [
        f"{A}->{C}",
        f"{SOURCE}->{A}",
        f"{ROOT}->{SOURCE}",
    ])

    tree = parse_tree(path)

    assert tree.root_index == 0
    assert len(tree.nodes) == 3
    assert len(tree.edges) == 2

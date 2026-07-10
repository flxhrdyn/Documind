import pytest
from backend.app.utils import ThinkingParser

def test_thinking_parser_basic():
    parser = ThinkingParser()
    
    # Feeding chunks
    results = []
    results.extend(list(parser.feed("<thinking>I am thinking.")))
    results.extend(list(parser.feed(" Done.</thinking>Final answer.")))
    
    assert results == [
        ("thinking", "I am thinking."),
        ("thinking", " Done."),
        ("token", "Final answer.")
    ]

def test_thinking_parser_no_tags():
    parser = ThinkingParser()
    results = list(parser.feed("Direct answer."))
    assert results == [("token", "Direct answer.")]

def test_thinking_parser_split_tags():
    parser = ThinkingParser()
    results = []
    results.extend(list(parser.feed("<think")))
    results.extend(list(parser.feed("ing>Thought</think")))
    results.extend(list(parser.feed("ing>Answer")))
    
    # Note: Split tag handling can be tricky. 
    # For now let's assume we want it to work reasonably.
    assert ("thinking", "Thought") in results
    assert ("token", "Answer") in results

def test_thinking_parser_multiple_thinking_blocks():
    """A model that emits more than one <thinking> block must have both
    captured as 'thinking', not have the second block leak into the answer."""
    parser = ThinkingParser()
    results = list(parser.feed(
        "<thinking>Block A</thinking>Answer1<thinking>Block B</thinking>Answer2"
    ))

    assert results == [
        ("thinking", "Block A"),
        ("token", "Answer1"),
        ("thinking", "Block B"),
        ("token", "Answer2"),
    ]


def test_thinking_parser_multiple_chunks_token():
    parser = ThinkingParser()
    results = []
    results.extend(list(parser.feed("<thinking>T</thinking>")))
    results.extend(list(parser.feed("A1 ")))
    results.extend(list(parser.feed("A2")))
    
    assert results == [
        ("thinking", "T"),
        ("token", "A1 "),
        ("token", "A2")
    ]

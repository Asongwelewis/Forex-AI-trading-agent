"""The three uncorrelated strategies, each a `Strategy` subclass returning a `Signal`.

Strategies are pure: same inputs produce same outputs, no I/O, no clock reads.
Timestamps come from the bars, never from `datetime.now()`.
"""

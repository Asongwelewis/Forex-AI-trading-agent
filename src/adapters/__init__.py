"""Broker abstraction: the `BrokerAdapter` protocol and its concrete implementations.

Strategies depend on this package only, never on a specific broker SDK, so the
Windows-only MT5 adapter can be swapped for a cloud adapter without touching them.
"""

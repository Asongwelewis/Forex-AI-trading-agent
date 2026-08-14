"""Permission grant state machine and auto-revoke triggers.

Default state is ADVISORY, in which execution is impossible. Grants expire, and
state is persisted so a restart cannot silently resurrect a revoked grant.
Fails closed: unreadable state means ADVISORY.
"""

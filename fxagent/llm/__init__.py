"""Provider gateway, prompt templates, and Pydantic response schemas.

The LLM never computes indicators, signals, or position sizes — it only synthesizes,
explains, and scores structured data against fixed rules. Every response is schema
validated; anything failing validation is discarded whole.
"""

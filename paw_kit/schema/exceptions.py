"""Custom exceptions for schema validation and constrained decoding."""


class PAWSchemaError(Exception):
    """Raised when output generation or schema validation fails and no fallback succeeds."""
    pass


class PAWSyntaxError(PAWSchemaError):
    """Raised when output violates required syntax or grammar."""
    pass

# Database

The supplied local SQLite database was reviewed during packaging.

It contains Django authentication/session tables (`auth_user`, `django_session`, permissions, etc.)
in addition to the application prediction-result table.

For security and privacy, the original SQLite database is NOT included in this public GitHub package.
Do not upload it publicly. If database functionality is needed, create a sanitized schema/fixture
containing only non-sensitive application records.

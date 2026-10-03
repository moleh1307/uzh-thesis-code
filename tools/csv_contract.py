"""One explicit, bounded transcript CSV contract for standalone entry points."""
import csv

# csv measures decoded characters, not file bytes. Oversized fields must fail, not truncate.
MAX_FIELD_CHARACTERS = 64 * 1024 * 1024


def configure_csv():
    csv.field_size_limit(MAX_FIELD_CHARACTERS)

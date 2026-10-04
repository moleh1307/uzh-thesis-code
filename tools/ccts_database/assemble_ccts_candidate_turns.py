#!/usr/bin/env python3
"""Assemble retained and newly retrieved CCTS calls in candidate-assignment order."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from artifact_publication import fresh_artifact_directory
from csv_contract import configure_csv

configure_csv()
LINKED_FIELDS = ('start_date', 'year', 'company_name', 'company_ticker', 'event_title')
REQUIRED_TURN_FIELDS = set(LINKED_FIELDS) | {
    'event_id', 'sequence_id', 'raw_sequence_id', 'source_event_company_name',
    'text_type', 'analysis_text_type', 'text_name', 'text_contents',
}
REQUIRED_AUDIT_FIELDS = {
    'event_id', 'fetched_rows', 'database_rows', 'fetched_pre_rows', 'fetched_qa_rows',
    'database_min_sequence_id', 'database_max_sequence_id', 'fetch_status', 'fetch_notes',
}


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def records(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f'Missing or duplicate CSV columns: {path}')
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f'Malformed CSV record: {path}')
            yield row


def index(path):
    result = {}
    for row in records(path):
        event = row.get('event_id', '')
        if not event.isdecimal() or int(event) <= 0 or str(int(event)) != event or event in result:
            raise ValueError(f'Noncanonical, blank or duplicate event_id: {path}')
        result[event] = row
    return result


def require_fields(path, required):
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        fields = csv.DictReader(handle).fieldnames or []
    if not fields or len(fields) != len(set(fields)) or not required <= set(fields):
        raise ValueError(f'Missing required CSV columns: {path}')


def fingerprint_update(h, row, fields):
    payload = json.dumps([row[k] for k in fields if k not in LINKED_FIELDS], ensure_ascii=True, separators=(',', ':')).encode()
    h.update(len(payload).to_bytes(8, 'big'))
    h.update(payload)


def full_row_fingerprints(paths, fields):
    hashes = {}
    for path in paths:
        for r in records(path):
            if set(r) != set(fields):
                raise ValueError(f'New raw/shard schema differs: {path}')
            payload = json.dumps([r[k] for k in fields], ensure_ascii=True, separators=(',', ':')).encode()
            h = hashes.setdefault(r['event_id'], hashlib.sha256())
            h.update(len(payload).to_bytes(8, 'big'))
            h.update(payload)
    return {e: h.hexdigest() for e,h in hashes.items()}


def event_byte_spans(path):
    spans, order = {}, []
    with path.open('rb') as handle:
        # Binary positions are captured only at complete CSV record boundaries.
        lines = iter(lambda: handle.readline().decode('utf-8'), '')
        reader = csv.DictReader(lines)
        if not reader.fieldnames or 'event_id' not in reader.fieldnames:
            raise ValueError('Reorder input has no event schema')
        header_end, current, begin, count = handle.tell(), None, None, 0
        while True:
            record_start = handle.tell()
            try:
                r = next(reader)
            except StopIteration:
                if current is not None:
                    spans[current] = (begin, handle.tell(), count)
                break
            if None in r or any(v is None for v in r.values()):
                raise ValueError('Malformed record in reorder input')
            e = r['event_id']
            if not e:
                raise ValueError('Blank event ID in reorder input')
            if e != current:
                if current is not None:
                    spans[current] = (begin, record_start, count)
                if e in spans:
                    raise ValueError(f'Noncontiguous event in reorder input: {e}')
                order.append(e)
                current, begin, count = e, record_start, 0
            count += 1
    return header_end, spans, order


def reorder_event_csv(source, target, event_order):
    """Copy whole CSV event byte spans; never reserialize or reorder rows within calls."""
    before = digest(source)
    header_end, spans, original_order = event_byte_spans(source)
    if len(event_order) != len(set(event_order)) or set(spans) != set(event_order):
        raise ValueError('Reorder event list differs from raw scope')
    hashes = {}
    with source.open('rb') as src, target.open('xb') as dst:
        dst.write(src.read(header_end))
        for e in event_order:
            start, end, _ = spans[e]
            src.seek(start)
            remaining, h = end - start, hashlib.sha256()
            while remaining:
                block = src.read(min(1024 * 1024, remaining))
                if not block:
                    raise ValueError('Unexpected EOF copying event')
                dst.write(block); h.update(block); remaining -= len(block)
            hashes[e] = h.hexdigest()
    new_header, new_spans, new_order = event_byte_spans(target)
    if new_order != list(event_order) or new_header != header_end:
        raise ValueError('Reordered CSV does not follow assignment order')
    with target.open('rb') as dst:
        for e in event_order:
            start, end, count = new_spans[e]
            if count != spans[e][2] or end - start != spans[e][1] - spans[e][0]:
                raise ValueError(f'Event byte length/row count changed: {e}')
            dst.seek(start); remaining, h = end-start, hashlib.sha256()
            while remaining:
                block = dst.read(min(1024 * 1024, remaining))
                if not block:
                    raise ValueError('Unexpected EOF verifying reordered event')
                h.update(block); remaining -= len(block)
            if h.hexdigest() != hashes[e]:
                raise ValueError(f'Event bytes changed while reordering: {e}')
    if digest(source) != before:
        raise ValueError('Source changed while reordering')
    return {'events': len(event_order), 'reordered': original_order != list(event_order),
            'per_event_bytes_unchanged': True}


def assemble(sources, assignments, metadata, target, audit_target):
    wanted = set(assignments)
    retained_audits, provenance, seen, hashes = {}, [], set(), {}
    fields = None
    temporary = target.with_name(target.name + '.tmp')
    with temporary.open('x', newline='', encoding='utf-8') as out:
        writer = None
        for label, raw, audit, schema in sources:
            audit_rows = index(audit)
            for e in wanted.intersection(audit_rows):
                if e in retained_audits:
                    raise ValueError(f'Event exists in both transcript sources: {e}')
                retained_audits[e] = audit_rows[e]
            with raw.open(newline='', encoding='utf-8-sig') as handle:
                reader = csv.DictReader(handle)
                if fields is None:
                    fields = list(reader.fieldnames or [])
                    writer = csv.DictWriter(out, fieldnames=fields, extrasaction='raise')
                    writer.writeheader()
                if reader.fieldnames != fields or not set(LINKED_FIELDS).issubset(fields):
                    raise ValueError(f'Incompatible raw schema: {raw}')
                current, count, sections, minimum, maximum, changed = None, 0, Counter(), None, None, Counter()
                h = None

                def finish():
                    if current is None:
                        return
                    a = audit_rows[current]
                    if (count != int(a['fetched_rows']) or count != int(a['database_rows'])
                            or sections['PRE'] != int(a['fetched_pre_rows']) or sections['Q&A'] != int(a['fetched_qa_rows'])
                            or minimum != int(a['database_min_sequence_id']) or maximum != int(a['database_max_sequence_id'])):
                        raise ValueError(f'Source audit differs from transcript: {current}')
                    hashes[current] = h.hexdigest()
                    provenance.append({'event_id': current, 'source_kind': label, 'source_raw_path': str(raw),
                                       'source_fetch_schema_claim': schema, 'rows': count, 'pre_rows': sections['PRE'],
                                       'qa_rows': sections['Q&A'], 'linked_blank_metadata_fields_json': json.dumps(dict(changed), sort_keys=True),
                                       'unchanged_nonmetadata_payload_sha256': hashes[current]})

                for r in reader:
                    if None in r or any(v is None for v in r.values()) or r['event_id'] not in audit_rows:
                        raise ValueError(f'Malformed or unaudited raw record: {raw}')
                    e = r['event_id']
                    if e not in wanted:
                        continue
                    if e != current:
                        finish()
                        if e in seen:
                            raise ValueError(f'Noncontiguous or overlapping source event: {e}')
                        seen.add(e)
                        current, count, sections, minimum, maximum, changed = e, 0, Counter(), None, None, Counter()
                        h = hashlib.sha256()
                    seq = int(r['raw_sequence_id'])
                    if int(r['sequence_id']) != seq or (maximum is not None and seq < maximum):
                        raise ValueError(f'Sequence order/key mismatch: {e}')
                    minimum = seq if minimum is None else min(minimum, seq)
                    maximum = seq if maximum is None else max(maximum, seq)
                    count += 1
                    sections[r['analysis_text_type']] += 1
                    if r['analysis_text_type'] != (r['text_type'] if r['text_type'] in ('PRE', 'Q&A') else 'OTHER'):
                        raise ValueError(f'Section mapping differs: {e}')
                    fingerprint_update(h, r, fields)
                    if metadata[e]['start_date'][:10] != assignments[e]['event_date']:
                        raise ValueError(f'Saved metadata date differs from CEO assignment: {e}')
                    for k in LINKED_FIELDS:
                        if not r[k]:
                            r[k] = metadata[e][k]
                            changed[k] += 1
                    if r['start_date'][:10] != assignments[e]['event_date']:
                        raise ValueError(f'Nonblank transcript date differs from assignment: {e}')
                    if r['year'] != metadata[e]['year']:
                        raise ValueError(f'Nonblank transcript year differs from metadata: {e}')
                    writer.writerow(r)
                finish()
            print(f'Assembled {label}: {len(seen):,} candidate calls so far', flush=True)
    if seen != wanted or set(retained_audits) != wanted:
        raise ValueError('Assembled raw/audit scope does not equal candidate scope')
    reorder_event_csv(temporary, target, list(assignments))
    temporary.unlink()
    with audit_target.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(next(iter(retained_audits.values()))))
        writer.writeheader()
        writer.writerows(retained_audits[e] for e in assignments)
    # Re-read the new copy; row payloads other than explicitly linked metadata must be identical.
    readback = {}
    for r in records(target):
        e = r['event_id']
        fingerprint_update(readback.setdefault(e, hashlib.sha256()), r, fields)
    if {e: h.hexdigest() for e, h in readback.items()} != hashes:
        raise ValueError('Assembled copy differs from original nonmetadata row payloads')
    return provenance


def build_candidate_assembly(event_assignments, event_metadata, sources, output_dir):
    """Publish a fresh local assembly, not a database provenance or speaker certification."""
    event_assignments, event_metadata, output_dir = map(Path, (event_assignments, event_metadata, output_dir))
    sources = [(Path(turns), Path(audit)) for turns, audit in sources]
    if not sources:
        raise ValueError('At least one transcript/audit source pair is required')
    paths = [event_assignments, event_metadata, *[p for pair in sources for p in pair]]
    resolved = [p.resolve() for p in paths]
    if len(resolved) != len(set(resolved)):
        raise ValueError('Input paths must be distinct')
    if any(output_dir.resolve() == p or output_dir.resolve() in p.parents for p in resolved):
        raise ValueError('Output cannot contain or replace an input')
    bindings = [{'path': str(p.resolve()), 'sha256': digest(p)} for p in paths]
    require_fields(event_assignments, {'event_id', 'event_date'})
    require_fields(event_metadata, {'event_id', *LINKED_FIELDS})
    assignments = index(event_assignments)
    metadata, seen_metadata = {}, set()
    for row in records(event_metadata):
        event = row['event_id']
        if not event.isdecimal() or int(event) <= 0 or str(int(event)) != event or event in seen_metadata:
            raise ValueError('Noncanonical, blank or duplicate metadata event_id')
        seen_metadata.add(event)
        if event in assignments:
            metadata[event] = row
    if not assignments or set(assignments) != set(metadata):
        raise ValueError('Empty assignment scope or missing event metadata')
    for e, a in assignments.items():
        day = date.fromisoformat(a['event_date'])
        if day.isoformat() != a['event_date'] or metadata[e]['start_date'][:10] != a['event_date']:
            raise ValueError(f'Assignment/metadata date mismatch: {e}')
        if metadata[e]['year'] != str(day.year):
            raise ValueError(f'Metadata year mismatch: {e}')
    for turns, audit in sources:
        require_fields(turns, REQUIRED_TURN_FIELDS)
        require_fields(audit, REQUIRED_AUDIT_FIELDS)
        for r in records(audit):
            if r['fetch_status'] == 'ok' and r['fetch_notes']:
                raise ValueError('OK fetch audit cannot contain review notes')
            if r['fetch_status'] not in ('ok', 'review'):
                raise ValueError('Unsupported fetch audit status')
            if r['fetch_status'] == 'review' and r['fetch_notes'] != 'duplicate_sequence_id':
                raise ValueError('Unsupported review reason; do not force inclusion')
    with fresh_artifact_directory(output_dir) as stage:
        provenance = assemble(
            [(f'source_{i}', turns, audit, 'not_asserted') for i, (turns, audit) in enumerate(sources, 1)],
            assignments, metadata, stage / 'candidate_raw_turns.csv', stage / 'candidate_raw_audit.csv',
        )
        if bindings != [{'path': str(p.resolve()), 'sha256': digest(p)} for p in paths]:
            raise ValueError('An input changed during assembly')
        with (stage / 'event_source_provenance.csv').open('x', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(provenance[0]))
            writer.writeheader()
            writer.writerows(provenance)
        names = ('candidate_raw_turns.csv', 'candidate_raw_audit.csv', 'event_source_provenance.csv')
        summary = {
            'status': 'complete_scope_and_payload_checked_not_speaker_validated',
            'candidate_events': len(assignments),
            'raw_rows': sum(r['rows'] for r in provenance),
            'source_event_counts': dict(Counter(r['source_kind'] for r in provenance)),
            'input_bindings': bindings,
            'artifact_sha256': {name: digest(stage / name) for name in names},
            'event_order': 'candidate_assignment_csv',
            'within_event_payload_unchanged': True,
            'linked_fields': list(LINKED_FIELDS),
            'producer_sha256': digest(Path(__file__)),
            'source_fetch_schema': 'not asserted; review source checkpoint provenance separately',
            'limitations': [
                'Source byte binding does not certify the database endpoint or licensing.',
                'Raw aliases are retained; run the separate exact-alias derivation next.',
                'No CEO-speaker validation, Q&A extraction, model scoring or annotation change.',
            ],
        }
        (stage / 'assembly_summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--event-assignments', required=True, type=Path)
    parser.add_argument('--event-metadata', required=True, type=Path)
    parser.add_argument('--source', required=True, action='append', nargs=2, metavar=('TURNS_CSV', 'AUDIT_CSV'),
                        help='Repeat for each saved raw-turn and fetch-audit pair.')
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    try:
        result = build_candidate_assembly(args.event_assignments, args.event_metadata, args.source, args.output_dir)
    except (OSError, ValueError, KeyError) as error:
        print(f'Candidate assembly failed: {error}', file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

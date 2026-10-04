"""Synthetic candidate assembly and ordering checks; no licensed source access."""
import csv
import tempfile
import unittest
from pathlib import Path

import json
import subprocess
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import assemble_ccts_candidate_turns as run

FIELDS = ['sample_rank', 'event_id', 'start_date', 'year', 'company_name', 'company_ticker', 'event_title',
          'sequence_id', 'raw_sequence_id', 'source_event_company_name', 'text_type', 'analysis_text_type', 'text_name', 'text_contents']


def row(event, date='', sequence=0):
    r = dict.fromkeys(FIELDS, '')
    r.update(event_id=event, start_date=date, sequence_id=str(sequence), raw_sequence_id=str(sequence),
             source_event_company_name='Example Corp', text_type='PRE', analysis_text_type='PRE',
             text_name='Example CEO', text_contents='Synthetic business assertion.')
    return r


class AssemblyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.assignments = {'1': {'event_date': '2005-01-01'}, '2': {'event_date': '2005-01-02'}}
        self.metadata = {e: {'start_date': a['event_date'], 'year': '2005', 'company_name': 'Example Corp',
                             'company_ticker': 'EX', 'event_title': 'Earnings call'} for e,a in self.assignments.items()}

    def source(self, name, rows, bad_count=False):
        raw, audit = self.root / (name + '.csv'), self.root / (name + '_audit.csv')
        with raw.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
        audits = []
        for e in dict.fromkeys(r['event_id'] for r in rows):
            rs = [r for r in rows if r['event_id'] == e]
            audits.append({'event_id': e, 'fetched_rows': len(rs) + int(bad_count), 'database_rows': len(rs),
                           'fetched_pre_rows': len(rs), 'fetched_qa_rows': 0,
                           'database_min_sequence_id': min(int(r['raw_sequence_id']) for r in rs),
                           'database_max_sequence_id': max(int(r['raw_sequence_id']) for r in rs),
                           'fetch_status': 'ok', 'fetch_notes': ''})
        with audit.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(audits[0])); w.writeheader(); w.writerows(audits)
        return name, raw, audit, 1

    def assemble(self, *sources):
        return run.assemble(sources, self.assignments, self.metadata, self.root/'combined.csv', self.root/'combined_audit.csv')

    def test_merge_preserves_text_and_links_blank_metadata(self):
        a = self.source('old', [row('1')]); b = self.source('new', [row('2', '2005-01-02')])
        before = run.digest(a[1]); provenance = self.assemble(a, b)
        self.assertEqual(len(provenance), 2)
        result = list(run.records(self.root/'combined.csv'))
        self.assertEqual(result[0]['start_date'], '2005-01-01')
        self.assertEqual(result[0]['text_contents'], 'Synthetic business assertion.')
        self.assertEqual(run.digest(a[1]), before)

    def test_overlapping_sources_rejected(self):
        with self.assertRaises(ValueError):
            self.assemble(self.source('a', [row('1')]), self.source('b', [row('1'), row('2')]))

    def test_missing_event_rejected(self):
        with self.assertRaises(ValueError): self.assemble(self.source('a', [row('1')]))

    def test_nonblank_wrong_date_rejected(self):
        with self.assertRaises(ValueError): self.assemble(self.source('a', [row('1', '2010-01-01'), row('2')]))

    def test_bad_audit_count_rejected(self):
        with self.assertRaises(ValueError): self.assemble(self.source('a', [row('1'), row('2')], bad_count=True))

    def test_sequence_order_rejected(self):
        with self.assertRaises(ValueError): self.assemble(self.source('a', [row('1', sequence=1), row('1', sequence=0), row('2')]))

    def test_noncontiguous_event_rejected(self):
        with self.assertRaises(ValueError): self.assemble(self.source('a', [row('1'), row('2'), row('1', sequence=1)]))

    def test_changed_raw_payload_detectable(self):
        a = self.source('a', [row('1')]); before = run.full_row_fingerprints([a[1]], FIELDS)
        changed = row('1'); changed['text_contents'] = 'A different assertion.'
        b = self.source('b', [changed])
        self.assertNotEqual(before, run.full_row_fingerprints([b[1]], FIELDS))

    def test_event_order_follows_assignment_not_source_order(self):
        self.assignments = {'2': self.assignments['2'], '1': self.assignments['1']}
        self.assemble(self.source('old', [row('1')]), self.source('new', [row('2')]))
        self.assertEqual([r['event_id'] for r in run.records(self.root/'combined.csv')], ['2', '1'])

    def test_new_call_is_interleaved_between_retained_calls(self):
        self.assignments['3'] = {'event_date': '2005-01-03'}
        self.metadata['3'] = {**self.metadata['2'], 'start_date': '2005-01-03'}
        self.assemble(self.source('retained', [row('1'), row('1', sequence=1), row('3')]),
                      self.source('new', [row('2')]))
        result = list(run.records(self.root/'combined.csv'))
        self.assertEqual([r['event_id'] for r in result], ['1', '1', '2', '3'])
        self.assertEqual([r['sequence_id'] for r in result[:2]], ['0', '1'])

    def test_reorder_preserves_multiline_quoted_utf8_payload(self):
        r = row('1'); r['text_contents'] = 'A "quoted" assertion.\nSecond line \u0130.'
        source = self.source('a', [r, row('2')])[1]
        target = self.root / 'ordered.csv'
        result = run.reorder_event_csv(source, target, ['2', '1'])
        self.assertTrue(result['per_event_bytes_unchanged'])
        self.assertEqual(list(run.records(target))[1]['text_contents'], r['text_contents'])

    def test_reorder_wrong_scope_rejected(self):
        source = self.source('a', [row('1'), row('2')])[1]
        with self.assertRaises(ValueError): run.reorder_event_csv(source, self.root/'ordered.csv', ['1'])

    def test_reorder_duplicate_target_ids_rejected(self):
        source = self.source('a', [row('1'), row('2')])[1]
        with self.assertRaises(ValueError): run.reorder_event_csv(source, self.root/'ordered.csv', ['1', '1', '2'])

    def build_inputs(self):
        assignment_path, metadata_path = self.root/'assignments.csv', self.root/'metadata.csv'
        for path, rows in (
            (assignment_path, [{'event_id': e, **a} for e,a in self.assignments.items()]),
            (metadata_path, [{'event_id': e, **a} for e,a in self.metadata.items()]),
        ):
            with path.open('w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        source = self.source('a', [row('1'), row('2')])
        return assignment_path, metadata_path, [(source[1], source[2])]

    def test_complete_package_is_hashed_without_claiming_fetch_provenance(self):
        assignments, metadata, sources = self.build_inputs()
        output = self.root / 'package'
        summary = run.build_candidate_assembly(assignments, metadata, sources, output)
        self.assertEqual(summary['candidate_events'], 2)
        self.assertEqual(summary['raw_rows'], 2)
        self.assertEqual(summary['event_order'], 'candidate_assignment_csv')
        self.assertIn('not asserted', summary['source_fetch_schema'])
        for name, expected in summary['artifact_sha256'].items():
            self.assertEqual(run.digest(output/name), expected)
        self.assertEqual(json.loads((output/'assembly_summary.json').read_text()), summary)

    def test_failed_assembly_does_not_publish_partial_package(self):
        assignments, metadata, sources = self.build_inputs()
        broken = self.source('broken', [row('1')])
        output = self.root/'package'
        with self.assertRaises(ValueError):
            run.build_candidate_assembly(assignments, metadata, [(broken[1], broken[2])], output)
        self.assertFalse(output.exists())
        self.assertFalse((self.root/'.package.publication-lock').exists())

    def test_existing_package_is_not_overwritten(self):
        assignments, metadata, sources = self.build_inputs()
        output = self.root/'package'; output.mkdir(); marker=output/'marker'; marker.write_text('original')
        with self.assertRaises(FileExistsError): run.build_candidate_assembly(assignments, metadata, sources, output)
        self.assertEqual(marker.read_text(), 'original')

    def test_assignment_metadata_date_mismatch_rejected(self):
        self.metadata['1']['start_date'] = '2005-02-01'
        assignments, metadata, sources = self.build_inputs()
        with self.assertRaises(ValueError): run.build_candidate_assembly(assignments, metadata, sources, self.root/'package')

    def test_changed_input_binding_prevents_publication(self):
        assignments, metadata, sources = self.build_inputs()
        original = run.assemble
        def change_input(*args):
            result = original(*args)
            assignments.write_text(assignments.read_text() + '\n')
            return result
        with patch.object(run, 'assemble', side_effect=change_input):
            with self.assertRaises(ValueError): run.build_candidate_assembly(assignments, metadata, sources, self.root/'package')
        self.assertFalse((self.root/'package').exists())

    def test_cli_runs_with_explicit_paths(self):
        assignments, metadata, sources = self.build_inputs()
        result = subprocess.run([sys.executable, run.__file__, '--event-assignments', str(assignments),
                                 '--event-metadata', str(metadata), '--source', *map(str,sources[0]),
                                 '--output-dir', str(self.root/'cli_package')], capture_output=True, text=True, cwd=self.root)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root/'cli_package/assembly_summary.json').is_file())

    def test_nonblank_wrong_year_rejected(self):
        r = row('1'); r['year'] = '2010'
        with self.assertRaises(ValueError): self.assemble(self.source('a', [r, row('2')]))

    def test_duplicate_raw_header_rejected(self):
        assignments, metadata, sources = self.build_inputs()
        source = sources[0][0]
        text = source.read_text()
        source.write_text(text.replace('sample_rank,event_id', 'event_id,event_id', 1))
        with self.assertRaises(ValueError): run.build_candidate_assembly(assignments, metadata, sources, self.root/'package')


if __name__ == '__main__':
    unittest.main()

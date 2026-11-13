"""Regenerate artificial, non-private DERCONNECT CSVs; no model calls."""
import csv
import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / 'tests/fixtures/derconnect'
ROOT.mkdir(parents=True, exist_ok=True)
start = datetime(2026, 1, 1, 21, 56, tzinfo=UTC)
annotations = []
for meter_index, meter in enumerate(('E2451', 'E3020')):
    rows = []
    for minute in range(154):
        t = start + timedelta(minutes=minute)
        if t == datetime(2026, 1, 2, 0, 20, tzinfo=UTC):
            continue
        power = 100 + meter_index * 50 + (minute % 7) / 10
        if datetime(2026, 1, 1, 23, 58, tzinfo=UTC) <= t < datetime(2026, 1, 2, 0, 5, tzinfo=UTC):
            power += 20 * (1 + int((t - datetime(2026, 1, 1, 23, 58, tzinfo=UTC)).total_seconds() / 60))
        if t.minute in (10, 11) and t.hour == 0:
            power = 0
        for second in range(0, 60, 2):
            stamp = t + timedelta(seconds=second, microseconds=100001 + (second % 3) * 20000)
            if second % 4 == 0:
                stamp = stamp.astimezone(timezone(timedelta(hours=-8)))
            value = f'{power / 3:.6f}'
            row = [stamp.isoformat(timespec='microseconds'), meter, value, value, value, '60']
            if t.hour == 0 and t.minute == 21 and second == 0:
                row[2] = 'bad'
            rows.append(row)
            if t.minute == 0 and second == 0:
                rows.append(row.copy())
            if t.hour == 0 and t.minute == 15 and second == 10:
                conflicting = row.copy()
                conflicting[2] = '999'
                rows.append(conflicting)
    # Reverse file order and rows to exercise external sorting and cross-file duplicates.
    for part, selected in enumerate((rows[::2], rows[1::2])):
        path = ROOT / f'{meter}_2026-01_part{part}.csv'
        with path.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['_time', 'DeviceID', 'Active_Power_A', 'Active_Power_B', 'Active_Power_C', 'Frequency'])
            writer.writerows(reversed(selected))
    for a, b, label in [('2026-01-01T22:00:00Z', '2026-01-01T23:30:00Z', 'reviewed_normal'),
                        ('2026-01-01T23:30:00Z', '2026-01-01T23:50:00Z', 'reviewed_normal'),
                        ('2026-01-01T23:58:00Z', '2026-01-02T00:05:00Z', 'anomaly')]:
        annotations.append({'meter_id': meter, 'signal': 'total_active_power', 'start': a, 'end': b,
                            'label': label, 'label_source': 'artificial_fixture', 'review_status': 'approved'})
(ROOT / 'annotations.json').write_text(json.dumps(annotations, indent=2) + '\n')

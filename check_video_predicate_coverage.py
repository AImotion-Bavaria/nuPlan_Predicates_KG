#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--visual-root', required=True)
args = parser.parse_args()
root = Path(args.visual_root).expanduser().resolve()
rows = []
for category_dir in sorted(root.iterdir() if root.exists() else []):
    if not category_dir.is_dir() or category_dir.name.startswith('_'):
        continue
    for predicate_dir in sorted(category_dir.iterdir()):
        if not predicate_dir.is_dir() or predicate_dir.name.startswith('_'):
            continue
        predicate = predicate_dir.name
        scenarios = [p for p in predicate_dir.glob('scenario_*') if p.is_dir()]
        videos = csvs = nonempty = matched = 0
        event_count = 0
        for scenario in scenarios:
            videos += int(bool(list(scenario.rglob('*.mp4'))))
            csv = scenario / 'tables' / 'video_edge_events.csv'
            if not csv.exists():
                continue
            csvs += 1
            try:
                df = pd.read_csv(csv)
            except pd.errors.EmptyDataError:
                continue
            except Exception:
                continue
            if df.empty:
                continue
            nonempty += 1
            mask = pd.Series(False, index=df.index)
            if 'predicate_id' in df.columns:
                mask |= df['predicate_id'].astype(str).eq(f'np:{predicate}')
            if 'relation_labels' in df.columns:
                mask |= df['relation_labels'].astype(str).str.contains(
                    predicate, case=False, regex=False, na=False
                )
            if mask.any():
                matched += 1
                event_count += int(mask.sum())
        status = 'OK' if scenarios and videos == len(scenarios) and matched == len(scenarios) else 'CHECK'
        rows.append({
            'category': category_dir.name,
            'predicate': predicate,
            'scenarios': len(scenarios),
            'videos': videos,
            'csv_files': csvs,
            'nonempty_csv': nonempty,
            'csv_with_selected_predicate': matched,
            'selected_predicate_events': event_count,
            'status': status,
        })
frame = pd.DataFrame(rows)
print(frame.to_string(index=False) if not frame.empty else 'No predicate video folders found.')
if not frame.empty:
    print('\nOK:', int(frame.status.eq('OK').sum()))
    print('CHECK:', int(frame.status.ne('OK').sum()))
    out = root / 'visualization_check_summary_fix2.csv'
    frame.to_csv(out, index=False)
    print('Saved:', out)

"""Build the name-reconciliation tables listed in the manuscript's data availability statement.

taxonomy_crosswalk.py, which produced the numbers in the Benchmark and Protocol section, only prints
this information. This script uses the same alias construction (IOC 15.2 workbook in
external/IOC_15.2_vs_other_lists.xlsx) and saves one table per region to
data/name_reconciliation/name_reconciliation_<region>.csv.

Columns:
  species, region
  flagged_unknown_to_birdnet: the in_birdnet == 0 flag used in the paper
  same_row_synonym_in_birdnet: exact same-row match in the IOC workbook (the count reported in the paper)
  group_level_alias_in_birdnet: looser check over species and subspecies rows, an unreviewed upper
    bound used in the Section 3 sensitivity check
  linked_aliases: the matching alias names, separated by semicolons, for manual review

Usage: python build_name_reconciliation_tables.py"""
import csv, json, os, collections
import openpyxl

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
OUT = os.path.join(DATA, 'name_reconciliation')
os.makedirs(OUT, exist_ok=True)

wb = openpyxl.load_workbook(os.path.join(HERE, '..', 'external', 'IOC_15.2_vs_other_lists.xlsx'), read_only=True)
ws = wb.worksheets[0]
allrows = list(ws.iter_rows(min_row=2, values_only=True))
rows_sp = [r for r in allrows if r[2] == 'species']
groups = []
cur = None
for r in allrows:
    if r[2] == 'species':
        cur = [r]; groups.append(cur)
    elif cur is not None and r[2] == 'ssp':
        cur.append(r)
galias = collections.defaultdict(set)
for g in groups:
    names = {' '.join(str(x).split()[:2]) for r in g for x in r[1:] if isinstance(x, str) and len(str(x).split()) >= 2}
    ioc = g[0][1]
    for n in names:
        galias[n] |= names | {ioc}
alias = collections.defaultdict(set)
for r in rows_sp:
    names = {str(x).strip() for x in r[1:] if isinstance(x, str) and len(x.split()) == 2}
    for n in names:
        alias[n] |= names

bl = {s.split('_')[0] for s in json.load(open(os.path.join(DATA, 'model_outputs', 'bench_ID6_labels.json')))}

for region, csvf in (('ID', 'bench_files_ID_cap6.csv'), ('TW', 'bench_files_TW_cap6.csv'),
                      ('PH', 'bench_files_PH_cap6.csv'), ('MY', 'bench_files_MY_cap6.csv')):
    rec = list(csv.DictReader(open(os.path.join(DATA, csvf), encoding='utf8')))
    sp = {r['species']: r['in_birdnet'] for r in rec}
    out_rows = []
    for s, flag in sorted(sp.items()):
        unknown = flag == '0'
        same_row = unknown and s in alias and any(a in bl for a in alias[s])
        same_row_aliases = sorted(a for a in alias.get(s, set()) if a in bl) if same_row else []
        group_aliases = sorted(a for a in galias.get(s, set()) if a in bl and a != s) if unknown else []
        out_rows.append(dict(
            species=s, region=region,
            flagged_unknown_to_birdnet=int(unknown),
            same_row_synonym_in_birdnet=int(same_row),
            group_level_alias_in_birdnet=int(bool(group_aliases)),
            linked_aliases=';'.join(same_row_aliases or group_aliases),
        ))
    outfn = os.path.join(OUT, f'name_reconciliation_{region}.csv')
    with open(outfn, 'w', newline='', encoding='utf8') as f:
        w = csv.DictWriter(f, fieldnames=['species', 'region', 'flagged_unknown_to_birdnet',
                                           'same_row_synonym_in_birdnet', 'group_level_alias_in_birdnet', 'linked_aliases'])
        w.writeheader(); w.writerows(out_rows)
    n_unk = sum(r['flagged_unknown_to_birdnet'] for r in out_rows)
    n_same = sum(r['same_row_synonym_in_birdnet'] for r in out_rows)
    n_group = sum(r['group_level_alias_in_birdnet'] for r in out_rows)
    print(f'{region}: {len(out_rows)} species, {n_unk} flagged unknown, {n_same} same-row synonym, {n_group} group-level (upper bound) -> {outfn}')

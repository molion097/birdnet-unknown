"""Check the BirdNET-unknown and Perch-unknown flags against other taxonomies (label-set crosswalk numbers in Section 3).

Uses the IOC World Bird List v15.2 'vs other lists' workbook (external/IOC_15.2_vs_other_lists.xlsx, from
worldbirdnames.org, CC BY 3.0). Each species row gives the same taxon under IOC, AviList, Clements v2025, HBW-BirdLife,
Howard and Moore and other lists. For each benchmark species flagged in_birdnet=0 in Indonesia (subset 1), Taiwan,
the Philippines and Malaysia, the script collects all names on the rows that contain it and tests whether any of them is
among BirdNET's 6,522 labels. The main count uses exact same-row synonyms only; lumps and splits that appear as separate
rows count as not linked. A looser group-level count (species row plus its subspecies rows) is printed as an upper bound
that needs manual review. The same check is run for the Perch-unknown flags of Indonesia subset 1."""
import csv,json,sys,collections,openpyxl
P=__import__('os').path.dirname(__import__('os').path.abspath(__file__))+'/../data/'
wb=openpyxl.load_workbook(__import__('os').path.dirname(__import__('os').path.abspath(__file__))+'/../external/IOC_15.2_vs_other_lists.xlsx',read_only=True); ws=wb.worksheets[0]
allrows=list(ws.iter_rows(min_row=2,values_only=True)); rows=[r for r in allrows if r[2]=='species']
# group-level aliases: a species row plus the subspecies rows below it; a binomial is the first two words of any name in any list column
groups=[];cur=None
for r in allrows:
    if r[2]=='species': cur=[r]; groups.append(cur)
    elif cur is not None and r[2]=='ssp': cur.append(r)
galias=collections.defaultdict(set)
for g in groups:
    names={' '.join(str(x).split()[:2]) for r in g for x in r[1:] if isinstance(x,str) and len(str(x).split())>=2}
    ioc=g[0][1]
    for n in names: galias[n]|=names|{ioc}
alias=collections.defaultdict(set)
for r in rows:
    names={str(x).strip() for x in r[1:] if isinstance(x,str) and len(x.split())==2}
    for n in names: alias[n]|=names
bl={s.split('_')[0] for s in json.load(open(P+'model_outputs/bench_ID6_labels.json'))}
for tag,csvf in (('Indonesia subset1','bench_files_ID_cap6_perch_sub.csv'),('Taiwan (all)','bench_files_TW_cap6.csv'),('Philippines (all)','bench_files_PH_cap6.csv'),('Malaysia (all)','bench_files_MY_cap6.csv')):
    rec=list(csv.DictReader(open(P+csvf,encoding='utf8'))); sp={r['species']:r['in_birdnet'] for r in rec}
    unk=sorted(s for s,f in sp.items() if f=='0'); 
    inioc=[s for s in unk if s in alias]; linked=[(s,sorted(a for a in alias[s] if a in bl)) for s in inioc if any(a in bl for a in alias[s])]
    print(f'{tag}: species {len(sp)}, flagged BirdNET-unknown {len(unk)}, found in workbook {len(inioc)}, linked to a BirdNET label by same-row synonym {len(linked)}')
    for s,a in linked: print('   ',s,'->',a)
    print('   not in workbook:',[s for s in unk if s not in alias][:20])
    glinked=[(s,sorted(a for a in galias[s] if a in bl and a!=s)) for s in unk if s in galias]; glinked=[(s,a) for s,a in glinked if a]
    print(f'   GROUP-level links (species row + its ssp rows, any list; UPPER BOUND, includes possible misalignment in older lists; manual review needed): {len(glinked)}')
    for s,a in glinked: print('     ',s,'->',a)

# Perch label crosswalk (Indonesia subset 1): species flagged in_perch=0 against Perch label names
pl={str(x).split('_')[0] for x in json.load(open(P+'model_outputs/perch_ID6sub_labels.json'))}
rec=list(csv.DictReader(open(P+'bench_files_ID_cap6_perch_sub.csv',encoding='utf8'))); sp={r['species']:r['in_perch'] for r in rec}
unk=sorted(s for s,f in sp.items() if f=='0')
same=[(s,sorted(a for a in alias[s] if a in pl and a!=s)) for s in unk if s in alias]; same=[(s,a) for s,a in same if a]
grp=[(s,sorted(a for a in galias[s] if a in pl and a!=s)) for s in unk if s in galias]; grp=[(s,a) for s,a in grp if a]
print(f'Perch: species {len(sp)}, flagged Perch-unknown {len(unk)}, same-row synonym links {len(same)}, group-level links (upper bound) {len(grp)}')
for s,a in grp: print('     ',s,'->',a)
print('   Perch-known flags whose exact name is NOT in the Perch label list (should be 0):',[s for s,f in sp.items() if f=='1' and s not in pl][:10])

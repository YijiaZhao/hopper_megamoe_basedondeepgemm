import csv,json,pathlib,re,sqlite3,statistics,subprocess,sys
base=pathlib.Path(sys.argv[1]); rows=[]
for rep in sorted(base.glob('*.nsys-rep')):
 m=re.fullmatch(r'(e2e|mega)_(split|fused)_(mxfp4|qoq)_M(2|8|16)\.nsys-rep',rep.name)
 if not m: continue
 scope,backend,quant,mtxt=m.groups();db=pathlib.Path('/tmp')/(rep.stem+'.last3detail.sqlite')
 subprocess.run(['nsys','export','--type','sqlite','--force-overwrite=true','--output',str(db),str(rep)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=True)
 c=sqlite3.connect(db); per={}
 for dev,n,s,e in c.execute('select k.deviceId,x.value,k.start,k.end from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds x on x.id=k.shortName where k.graphNodeId is not null order by k.deviceId,k.start'):
  per.setdefault(dev,[]).append((n,s,e))
 target_by_dev=[]; front_by_dev=[]; mega_by_dev=[]
 for dev,ev in sorted(per.items()):
  fronts=[x for x in ev if x[0]=='router_quant_topk_kernel']
  if backend=='fused':
   megas=[x for x in ev if f'sm90_{quant}_mega_moe_h20_fused_impl' in x[0]]
   if scope=='e2e':
    n=min(len(fronts),len(megas)); pairs=list(zip(fronts[-n:],megas[-n:])); targets=[(g[2]-f[1])/1e3 for f,g in pairs]; fs=[(f[2]-f[1])/1e3 for f,g in pairs]; ms=[(g[2]-g[1])/1e3 for f,g in pairs]
   else:
    targets=[(g[2]-g[1])/1e3 for g in megas]; fs=[]; ms=targets
  else:
   l1=[x for x in ev if 'sm90_mxfp4_mega_moe_l1_impl' in x[0]];l2=[x for x in ev if 'sm90_mxfp4_mega_moe_l2_impl' in x[0]]
   n=min(len(l1),len(l2),len(fronts)) if scope=='e2e' else min(len(l1),len(l2))
   if scope=='e2e':
    triples=list(zip(fronts[-n:],l1[-n:],l2[-n:])); targets=[(max(a[2],b[2])-f[1])/1e3 for f,a,b in triples];fs=[(f[2]-f[1])/1e3 for f,a,b in triples];ms=[(max(a[2],b[2])-min(a[1],b[1]))/1e3 for f,a,b in triples]
   else:
    pairs=list(zip(l1[-n:],l2[-n:]));targets=[(max(a[2],b[2])-min(a[1],b[1]))/1e3 for a,b in pairs];fs=[];ms=targets
  assert len(targets)>=3,(rep,dev,len(targets));target_by_dev.append(targets[-3:]);
  if fs: front_by_dev.append(fs[-3:])
  mega_by_dev.append(ms[-3:])
 assert len(target_by_dev)==8,(rep,len(target_by_dev))
 last_call_medians=[statistics.median(v[i] for v in target_by_dev) for i in range(3)]
 final=statistics.median(statistics.median(v) for v in target_by_dev)
 front_final=None if not front_by_dev else statistics.median(statistics.median(v) for v in front_by_dev)
 mega_final=statistics.median(statistics.median(v) for v in mega_by_dev)
 per_gpu_medians=[statistics.median(v) for v in target_by_dev]
 rows.append(dict(scope=scope,precision=quant,M=int(mtxt),backend=backend,third_last_median_us=last_call_medians[0],second_last_median_us=last_call_medians[1],last_median_us=last_call_medians[2],final_median_us=final,frontend_median_us=front_final,mega_median_us=mega_final,rank_median_min_us=min(per_gpu_medians),rank_median_max_us=max(per_gpu_medians),report=rep.name))
order={'e2e':0,'mega':1};qo={'mxfp4':0,'qoq':1};bo={'fused':0,'split':1};rows.sort(key=lambda r:(order[r['scope']],qo[r['precision']],r['M'],bo[r['backend']]))
with open(base/'TIMELINE_LAST3.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
(base/'TIMELINE_LAST3.json').write_text(json.dumps(rows,indent=2)+'\n')
lines=['| Scope | Precision | M | Backend | -3 median | -2 median | Last median | Final median | Frontend median | Mega median |','|---|---|---:|---|---:|---:|---:|---:|---:|---:|']
fmt=lambda x:'-' if x is None else f'{x:.3f}'
for r in rows:lines.append(f"| {r['scope'].upper()} | {r['precision'].upper()} | {r['M']} | {r['backend'].upper()} | {fmt(r['third_last_median_us'])} | {fmt(r['second_last_median_us'])} | {fmt(r['last_median_us'])} | **{fmt(r['final_median_us'])}** | {fmt(r['frontend_median_us'])} | {fmt(r['mega_median_us'])} |")
text='\n'.join(lines)+'\n';(base/'TIMELINE_LAST3.md').write_text(text);print(text)

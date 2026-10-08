"""Verify compact arrays/data and recalculate manuscript analyses, without training."""
from pathlib import Path
import argparse,csv,json,sys
import numpy as np
from scipy.stats import pearsonr,qmc
from scipy.spatial.distance import pdist
from sklearn.model_selection import KFold,train_test_split
import reproduce as driver
flow=driver.flow
ROOT=driver.ROOT

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def close(actual,expected):
    for key in ['r2_pooled','r2_mean_outputs','mae','mse','rmse','rsr_pooled_ddof0']:
        np.testing.assert_allclose(actual[key],expected[key],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(actual['case_r2'],expected['case_r2'],rtol=1e-12,atol=1e-12)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir','--salida',dest='output_dir',type=Path,help='Optional NEW directory for tables')
    ap.add_argument('--report',type=Path,help='Optional NEW verification JSON')
    args=ap.parse_args();cfg=read(ROOT/'config.json');ref=read(ROOT/'results/reference.json')
    assert flow.digest(ROOT/'results/predictions.npz')==cfg['prediction_file_sha256'],'Prediction archive hash mismatch'
    for p,h in cfg['data_file_hashes'].items():
        assert flow.digest(ROOT/p)==h,'Dataset hash mismatch: '+p
    expected_keys={uid+'_'+part for uid in cfg['jobs'] for part in ['test','oof']}
    metrics={};inputs={};references={};prediction={};rows=[];case_rows=[];local=[]
    with np.load(ROOT/'results/predictions.npz',allow_pickle=False) as z:
        assert set(z.files)==expected_keys
        for uid,job in cfg['jobs'].items():
            c=driver.configuration(cfg,job);x,y,xt,yt,ids,test_ids,_=flow.load_data(c,job['dataset'])
            assert ids==job['development_ids'] and test_ids==job['test_ids']
            for fi,(tr,va) in enumerate(KFold(job['folds'],shuffle=True,random_state=job['seed']).split(x)):
                recorded=job['fold_records'][fi]
                assert recorded['train_ids']==[ids[i] for i in tr]
                assert recorded['validation_ids']==[ids[i] for i in va]
                assert recorded['seed']==job['seed']+fi+1
                if job['algorithm']=='ANN':
                    itr,iva=train_test_split(np.arange(len(tr)),test_size=job['inner_validation_fraction'],random_state=recorded['seed'])
                    inner=recorded['inner_split']
                    assert inner['train_ids']==[ids[i] for i in tr[itr]]
                    assert inner['validation_ids']==[ids[i] for i in tr[iva]]
                    assert inner['selected_epochs']==recorded['epochs']
            if job['algorithm']=='ANN':
                assert job['final_epochs']==max(1,int(np.median([f['epochs'] for f in job['fold_records']])))
            metrics[uid]={}
            for part,truth,target in [('test',yt,'test'),('oof',y,'development_oof')]:
                values=z[uid+'_'+part];assert values.shape==truth.shape
                actual=flow.metrics(truth,values);close(actual,ref['jobs'][uid][target]);metrics[uid][target]=actual
                rows.append({'id':uid,'algorithm':job['algorithm'],'dataset':job['dataset'],'seed':job['seed'],
                    'partition':target,'n':len(truth),**{k:actual[k] for k in ['r2_pooled','mse','mae','rmse','rsr_pooled_ddof0']}})
            inputs[uid]=xt;references[uid]=yt;prediction[uid]=z[uid+'_test'].copy()
    # Public principal labels point to selected unique runs.
    for label,uid in cfg['selections']['principal'].items():
        close(metrics[uid]['test'],ref['metrics'][label])
        case_rows.extend({'model_design':label,'ID':tid,'R2':r} for tid,r in zip(cfg['jobs'][uid]['test_ids'],metrics[uid]['test']['case_r2']))
        if label.split('/')[0] in ['ANN','SVR']:
            for axis in [1,8,15]:
                sl=slice((axis-1)*21,axis*21)
                m=flow.metrics(references[uid][:,sl],prediction[uid][:,sl])
                local.append({'model_design':label,'axis':axis,'r2_pooled':m['r2_pooled'],'mae':m['mae'],'rmse':m['rmse']})
    correlations={}
    for label,uid in cfg['selections']['principal'].items():
        if label.split('/')[0] not in ['ANN','SVR']:continue
        case=np.asarray(metrics[uid]['test']['case_r2'])
        for j,g in enumerate(flow.GEOM):
            p=pearsonr(inputs[uid][:,j],case);ci=p.confidence_interval(.95)
            correlations[label+'/'+g]={'r':float(p.statistic),'p_two_sided':float(p.pvalue),
                'ci95_fisher':[float(ci.low),float(ci.high)],'n':len(case)}
    order=sorted(correlations,key=lambda k:correlations[k]['p_two_sided']);running=0.
    for rank,key in enumerate(order):
        running=max(running,min(1.,(len(order)-rank)*correlations[key]['p_two_sided']))
        correlations[key]['p_holm_90']=running
        for field,value in correlations[key].items():
            np.testing.assert_allclose(value,ref['correlations'][key][field],rtol=1e-12,atol=1e-12)
    bp=cfg['bootstrap'];indices=np.random.default_rng(bp['seed']).integers(0,bp['n'],size=(bp['draws'],bp['n']))
    paired={}
    for a in ['ANN','SVR','KNN','DT','RF','GBoost']:
        for suffix,label in [('', 'base'),('_FFD','_FFD'),('_2k','_2k')]:
            ul=cfg['selections']['principal'][a+'/LHS43'+suffix]
            uc=cfg['selections']['principal'][a+'/CCD43'+suffix]
            np.testing.assert_array_equal(references[ul],references[uc])
            l=((references[ul]-prediction[ul])**2).mean(axis=1);c=((references[uc]-prediction[uc])**2).mean(axis=1)
            delta=np.sqrt(l[indices].mean(axis=1))-np.sqrt(c[indices].mean(axis=1))
            v={'rmse_LHS_minus_CCD':float(np.sqrt(l.mean())-np.sqrt(c.mean())),
               'ci95_paired_percentile':np.quantile(delta,[.025,.975]).tolist()}
            for field,value in v.items():np.testing.assert_allclose(value,ref['paired_rmse'][a+'/'+label][field],rtol=1e-12,atol=1e-12)
            paired[a+'/'+label]=v
    convergence=[]
    for row in ref['convergence']['rows']:
        per_seed=[]
        for r in row['seeds']:
            label=f"{row['algorithm']}/{row['name']}/{r['seed']}"
            uid=cfg['selections']['convergencia'][label]
            close(metrics[uid]['test'],r['test']);close(metrics[uid]['development_oof'],r['development_oof'])
            per_seed.append(metrics[uid])
        for partition in ['test','development_oof']:
            for k in ['r2_pooled','mse','mae','rmse','rsr_pooled_ddof0']:
                values=[p[partition][k] for p in per_seed]
                mean=float(np.mean(values));sd=float(np.std(values,ddof=1))
                np.testing.assert_allclose([mean,sd],[row[partition][k]['mean'],row[partition][k]['sd']],rtol=1e-12,atol=1e-12)
                convergence.append({'algorithm':row['algorithm'],'subset':row['name'],'n':row['n'],
                    'percent':row['actual_percent'],'partition':partition,'metric':k,'mean':mean,'sample_sd':sd})
    with (ROOT/cfg['master']).open(encoding='utf-8-sig',newline='') as f:master={r['ID']:r for r in csv.DictReader(f)}
    bounds=ref['sampling']['factors'];lo=np.array([f['lower'] for f in bounds]);hi=np.array([f['upper'] for f in bounds])
    sampling={}
    for key,d in ref['sampling']['designs'].items():
        u=(np.array([[float(master[i][g]) for g in flow.GEOM] for i in d['ids']])-lo)/(hi-lo)
        actual={'d_min':float(pdist(u).min()),'CD_squared':float(qmc.discrepancy(u,method='CD'))}
        for field,value in actual.items():np.testing.assert_allclose(value,d[field],rtol=1e-12,atol=1e-12)
        sampling[key]=actual
    index=read(ROOT/'models/index.json')
    assert set(index['models'])==set(cfg['jobs'])
    assert index['principal_selections']==cfg['selections']['principal']
    assert index['convergence_selections']==cfg['selections']['convergencia']
    pairs={}
    for uid,m in index['models'].items():
        key=(m['model_sha256'],m['scalers_sha256'])
        pairs.setdefault(key,[]).append(uid)
        canonical=index['models'][m['canonical_artifact_id']]
        assert [m['model'],m['scalers']]==[canonical['model'],canonical['scalers']]
        assert key==(canonical['model_sha256'],canonical['scalers_sha256'])
    assert len(pairs)==92 and index['included_binary_files']==0
    timing=ref['timing']['records']
    assert set(timing)==set(cfg['selections']['principal'])
    for label,uid in cfg['selections']['principal'].items():
        t=timing[label];job=cfg['jobs'][uid]
        assert t['run']==job['source']['run'] and t['seed']==job['seed']
        assert t['final_epochs']==job['final_epochs']
        assert all(t[k]>=0 for k in ['final_fit_pipeline_seconds','prediction_20_cases_seconds','total_seconds'])
    # Check the delivered manifest when present; the manifest does not hash itself.
    integrity_count=0
    manifest=ROOT/'results/SHA256SUMS.txt'
    if manifest.is_file():
        for line in manifest.read_text(encoding='utf-8').splitlines():
            expected,name=line.split('  ',1)
            p=(ROOT/name).resolve()
            assert p.is_relative_to(ROOT) and flow.digest(p)==expected,'Manifest mismatch: '+name
            integrity_count+=1
    report={'status':'PASS_ML','scope':'Experimental designs, processed CFD data, ML metrics and convergence',
        'principal':54,'convergence':60,'unique_evaluations':len(metrics),'partitions':len(rows),
        'correlations':len(correlations),'bootstrap_comparisons':len(paired),
        'sampling_designs':len(sampling),'logical_model_fits':112,'unique_binary_pairs':92,
        'timing_records':len(timing),'manifest_files_checked':integrity_count,
        'training_performed':False,
        'model_inference':'NOT_RUN_BY_THIS_COMMAND','figure_rendering':'NOT_CHECKED_BY_THIS_COMMAND'}
    if args.report:
        with args.report.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    if args.output_dir:
        args.output_dir.mkdir(parents=True,exist_ok=False)
        (args.output_dir/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        for name,data in [('metrics.csv',rows),('case_R2.csv',case_rows),('local_performance.csv',local),('convergence.csv',convergence)]:
            with (args.output_dir/name).open('w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
        for name,data in [('correlations.json',correlations),('bootstrap.json',paired),('sampling.json',sampling)]:
            (args.output_dir/name).write_text(json.dumps(data,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()

"""Load final accepted models for inference; no fitting or parameter selection."""
from pathlib import Path
import argparse,csv,json,os,sys

ROOT=Path(__file__).resolve().parents[1]
for key,value in {'CUDA_VISIBLE_DEVICES':'-1','TF_ENABLE_ONEDNN_OPTS':'0',
                  'TF_DETERMINISTIC_OPS':'1','TF_CPP_MIN_LOG_LEVEL':'2',
                  'OMP_NUM_THREADS':'2','TF_NUM_INTRAOP_THREADS':'2',
                  'TF_NUM_INTEROP_THREADS':'2'}.items():os.environ[key]=value

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(path):
    import hashlib
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def safe_path(value):
    p=(ROOT/value).resolve()
    if not p.is_relative_to(ROOT.resolve()):raise ValueError('Artifact outside this repository')
    return p
def table(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def load(record,job):
    import numpy as np,joblib
    model_path=safe_path(record['model']);scales_path=safe_path(record['scalers'])
    if not model_path.is_file() or not scales_path.is_file():
        raise FileNotFoundError('Download and extract the models-v1.0 Release asset. See models/README.md.')
    if digest(model_path)!=record['model_sha256'] or digest(scales_path)!=record['scalers_sha256']:
        raise ValueError('Model/scaler SHA256 mismatch: '+record['id'])
    scales=joblib.load(scales_path)
    if scales['fit_ids']!=job['development_ids']:raise ValueError('Scaler ordered IDs mismatch')
    params=job['candidates'][job['selected_candidate']]
    if job['algorithm']=='ANN':
        import tensorflow as tf
        tf.keras.backend.clear_session()
        model=tf.keras.models.load_model(model_path,compile=False)
        assert model.input_shape[-1]==5 and model.output_shape[-1]==315
        widths=[l.units for l in model.layers if isinstance(l,tf.keras.layers.Dense)]
        assert widths==params['layers']+[315]
        dropout=[l.rate for l in model.layers if isinstance(l,tf.keras.layers.Dropout)]
        np.testing.assert_allclose(dropout,params['dropout'],rtol=1e-12,atol=1e-12)
    else:
        model=joblib.load(model_path)
        estimator=model.estimators_[0] if job['algorithm'] in ['SVR','GBoost'] else model
        actual=estimator.get_params(deep=False)
        expected=dict(params)
        if job['algorithm'] in ['DT','RF','GBoost']:expected['random_state']=job['seed']
        if job['algorithm'] in ['RF','KNN']:expected['n_jobs']=1
        for key,value in expected.items():
            if isinstance(value,float):np.testing.assert_allclose(actual[key],value,rtol=1e-12,atol=1e-12)
            else:assert actual[key]==value,(job['algorithm'],key,actual[key],value)
        assert estimator.n_features_in_==5
    return model,scales

def predict(model,scales,x,algorithm):
    import numpy as np
    transformed=scales['X'].transform(x)
    pred=model(transformed,training=False).numpy() if algorithm=='ANN' else model.predict(transformed)
    result=scales['Y'].inverse_transform(pred)
    if result.shape!=(len(x),315) or not np.isfinite(result).all():raise ValueError('Invalid prediction')
    return result

def verify(index,cfg):
    import numpy as np
    master={r['ID']:r for r in table(ROOT/cfg['master'])}
    details=[];maximum=0.
    with np.load(ROOT/'results/predictions.npz',allow_pickle=False) as z:
        if digest(ROOT/'results/predictions.npz')!=cfg['prediction_file_sha256']:
            raise ValueError('Prediction archive SHA256 mismatch')
        for uid,record in index['models'].items():
            job=cfg['jobs'][uid];model,scales=load(record,job)
            x=np.asarray([[float(master[i][c]) for c in cfg['input_columns']] for i in job['development_ids']])
            y=np.asarray([[float(master[i][c]) for c in cfg['output_columns']] for i in job['development_ids']])
            for key,values in [('X',x),('Y',y)]:
                np.testing.assert_array_equal(scales[key].data_min_,values.min(axis=0))
                np.testing.assert_array_equal(scales[key].data_max_,values.max(axis=0))
                assert scales[key].n_samples_seen_==len(x)
            xt=np.asarray([[float(master[i][c]) for c in cfg['input_columns']] for i in job['test_ids']])
            actual=predict(model,scales,xt,job['algorithm']);expected=z[uid+'_test']
            np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-6)
            delta=float(np.max(np.abs(actual-expected)));maximum=max(maximum,delta)
            details.append({'id':uid,'algorithm':job['algorithm'],'model':record['model'],
                'hashes_match':True,'scaler_training_ids_and_extrema_match':True,
                'parameters_match':True,'test_predictions_match':True,'max_abs_Cp_difference':delta})
            print('VERIFIED',uid,job['algorithm'],job['dataset'],'max_abs',delta,flush=True)
    return {'status':'PASS','unique_models':len(details),'principal_selections':len(index['principal_selections']),
        'convergence_selections':len(index['convergence_selections']),
        'tolerance':{'absolute':1e-6,'relative':1e-6},'maximum_absolute_Cp_difference':maximum,
        'model_fitting_performed':False,'details':details}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    mode=ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--list','--listar',dest='list',action='store_true')
    mode.add_argument('--verify','--verificar',dest='verify',action='store_true')
    mode.add_argument('--id',help='Reference ID shown in index.csv, e.g. r000')
    ap.add_argument('--input','--entrada',dest='input',type=Path,help='Input CSV; otherwise predict the 20 recorded test cases')
    ap.add_argument('--output','--salida',dest='output',type=Path,help='NEW prediction CSV')
    ap.add_argument('--report','--informe',dest='report',type=Path,help='NEW verification report JSON')
    args=ap.parse_args()
    index=read(ROOT/'models/index.json');cfg=read(ROOT/'config.json')
    if args.list:
        for uid,r in index['models'].items():print(uid,r['algorithm'],', '.join(r['analyses']),r['model'])
        return
    if args.verify:
        report=verify(index,cfg)
        if args.report:
            with args.report.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,ensure_ascii=False)
        print('PASS',report['unique_models'],'models; max_abs_Cp_difference',report['maximum_absolute_Cp_difference'])
        return
    if args.id not in index['models']:ap.error('Unknown ID; use --listar or index.csv')
    if args.output is None:ap.error('--id requires --salida for the prediction CSV')
    if args.output.exists():ap.error('Output exists; choose a new file')
    import numpy as np
    record=index['models'][args.id];job=cfg['jobs'][args.id]
    if args.input:
        rows=table(args.input);ids=[r.get('ID',f'case_{i+1:03}') for i,r in enumerate(rows)]
    else:
        master={r['ID']:r for r in table(ROOT/cfg['master'])}
        ids=job['test_ids'];rows=[master[i] for i in ids]
    if not rows:ap.error('Input CSV contains no configurations')
    x=np.asarray([[float(r[c]) for c in cfg['input_columns']] for r in rows])
    if not np.isfinite(x).all():ap.error('Input contains nonfinite values')
    model,scales=load(record,job);prediction=predict(model,scales,x,job['algorithm'])
    with args.output.open('x',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['ID',*cfg['output_columns']]);writer.writerows([i,*p] for i,p in zip(ids,prediction))
    print('PREDICTED',len(ids),'configurations;',args.output,'using',args.id,'without fitting')

if __name__=='__main__':main()

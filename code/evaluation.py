"""Numerical evaluation core extracted without changing the selected functions.
Origin: revision/evaluate.py, current RF correction 2026-10-05.
Use reproduce.py as the entry point. Historical parameter recovery is excluded.
"""
from __future__ import annotations
import copy,csv,hashlib,importlib.metadata,json,os,platform,sys,time,uuid
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
GEOM=['R/D','B/D','D','h','Angle']
OUTPUTS=[f'Cp_{i:03}' for i in range(1,316)]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def canonical(value): return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)

def fingerprint(value): return hashlib.sha256(canonical(value).encode()).hexdigest()

def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def write_json(path, value):
    # Exclusive creation: completed and historical artifacts are never overwritten.
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)

def load_data(config, dataset):
    import numpy as np
    def rows(path):
        with Path(path).open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f); result=list(reader)
            if len(reader.fieldnames)!=len(set(reader.fieldnames)): raise ValueError('Duplicate columns')
            return result
    master_path=ROOT/config['master']; data_path=ROOT/config['datasets'][dataset]
    master=rows(master_path); byid={r['ID']:r for r in master}
    if len(byid)!=len(master): raise ValueError('Duplicate master IDs')
    work=rows(data_path); ids=[r['Clave'] for r in work]
    test_ids=[f'T_{i}' for i in range(1,21)]
    if len(set(ids))!=len(ids) or set(ids)&set(test_ids): raise ValueError('Invalid membership')
    x=np.array([[float(byid[i][c]) for c in GEOM] for i in ids])
    y=np.array([[float(byid[i][c]) for c in OUTPUTS] for i in ids])
    xt=np.array([[float(byid[i][c]) for c in GEOM] for i in test_ids])
    yt=np.array([[float(byid[i][c]) for c in OUTPUTS] for i in test_ids])
    mapping={**dict(zip(['R/D','B/D','D','h','Angulo'],GEOM)),
             **{f'v{i}': f'Cp_{i:03}' for i in range(1,316)}}
    if any(float(r[c])!=float(byid[r['Clave']][m]) for r in work for c,m in mapping.items()):
        raise ValueError('Working dataset differs from authorized master')
    if not all(np.isfinite(a).all() for a in (x,y,xt,yt)): raise ValueError('Nonfinite data')
    if len(np.unique(x,axis=0))!=len(x): raise ValueError('Duplicate CFD configurations')
    if np.any(np.all(np.isclose(x[:,None,:],xt[None,:,:],atol=1e-10,rtol=0),axis=2)):
        raise ValueError('Test geometry overlaps development')
    return x,y,xt,yt,ids,test_ids,{config['master']:digest(master_path),config['datasets'][dataset]:digest(data_path)}

def metrics(y, pred):
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    if y.shape!=pred.shape or not np.isfinite(pred).all(): raise ValueError('Invalid prediction')
    mse=float(mean_squared_error(y,pred)); sd=float(np.std(y.ravel(),ddof=0))
    return {'scale':'original_Cp','n_configurations':len(y),'outputs_per_configuration':y.shape[1],
            'r2_pooled':float(r2_score(y.ravel(),pred.ravel())),
            'r2_mean_outputs':float(r2_score(y,pred,multioutput='uniform_average')),
            'mae':float(mean_absolute_error(y,pred)),'mse':mse,'rmse':float(np.sqrt(mse)),
            'rsr_pooled_ddof0':float(np.sqrt(mse)/sd) if sd else None,
            'case_r2':[float(r2_score(a,b)) for a,b in zip(y,pred)]}

def build_model(algorithm, params, seed, input_dim=5, output_dim=315):
    from sklearn.multioutput import MultiOutputRegressor
    if algorithm=='ANN':
        import tensorflow as tf
        tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(seed)
        model=tf.keras.Sequential([tf.keras.Input(shape=(input_dim,))])
        for units,dropout in zip(params['layers'],params['dropout']):
            model.add(tf.keras.layers.Dense(units,activation='relu'));model.add(tf.keras.layers.Dropout(dropout))
        model.add(tf.keras.layers.Dense(output_dim))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),loss='mse')
        return model
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.neighbors import KNeighborsRegressor
    options=copy.deepcopy(params)
    if algorithm in ('DT','RF','GBoost'): options['random_state']=seed
    if algorithm in ('RF','KNN'): options['n_jobs']=1
    constructors={'SVR':SVR,'DT':DecisionTreeRegressor,'RF':RandomForestRegressor,
                  'KNN':KNeighborsRegressor,'GBoost':GradientBoostingRegressor}
    model=constructors[algorithm](**options)
    return MultiOutputRegressor(model,n_jobs=1) if algorithm in ('SVR','GBoost') else model

def fit_ann(model,x,y,params,seed,validation=None,epochs=None):
    import tensorflow as tf
    # Explicit datasets avoid repeated large private thread pools for tiny CFD datasets.
    options=tf.data.Options();options.threading.private_threadpool_size=1
    ds=tf.data.Dataset.from_tensor_slices((x,y)).shuffle(len(x),seed=seed).batch(params['batch_size']).with_options(options)
    kwargs={}
    if validation is not None:
        vx,vy=validation
        kwargs['validation_data']=tf.data.Dataset.from_tensor_slices((vx,vy)).batch(params['batch_size']).with_options(options)
        kwargs['callbacks']=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=params['patience'],restore_best_weights=True)]
    return model.fit(ds,epochs=epochs or params['epochs'],verbose=0,**kwargs).history

def fit_partition(algorithm,params,x,y,ids,seed,path,inner_fraction,final_epochs=None):
    import numpy as np, joblib
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    path.mkdir();started=time.perf_counter();histories={};chosen_epochs=final_epochs
    # Epoch selection uses only an inner split of the outer training configurations.
    if algorithm=='ANN' and chosen_epochs is None:
        tr,va=train_test_split(np.arange(len(x)),test_size=inner_fraction,random_state=seed)
        sx=MinMaxScaler().fit(x[tr]);sy=MinMaxScaler().fit(y[tr])
        inner=build_model(algorithm,params,seed)
        history=fit_ann(inner,sx.transform(x[tr]),sy.transform(y[tr]),params,seed,
                        validation=(sx.transform(x[va]),sy.transform(y[va])))
        chosen_epochs=int(np.argmin(history['val_loss'])+1)
        histories['inner']=history
        joblib.dump({'X':sx,'Y':sy,'fit_ids':[ids[i] for i in tr]},path/'inner_scalers.joblib')
        write_json(path/'inner_split.json',{'seed':seed,'train_ids':[ids[i] for i in tr],
                   'validation_ids':[ids[i] for i in va],'selected_epochs':chosen_epochs})
    sx=MinMaxScaler().fit(x);sy=MinMaxScaler().fit(y)
    model=build_model(algorithm,params,seed)
    before_fit = model.get_params(deep=False) if algorithm == 'RF' else None
    if algorithm == 'RF':
        assert all(before_fit[k] == v for k,v in params.items() if k not in ('random_state','n_jobs'))
        assert before_fit['random_state'] == seed and before_fit['n_jobs'] == 1
    if algorithm=='ANN': histories['refit']=fit_ann(model,sx.transform(x),sy.transform(y),params,seed,epochs=chosen_epochs)
    else: model.fit(sx.transform(x),sy.transform(y))
    if algorithm == 'RF':
        assert model.get_params(deep=False) == before_fit
        write_json(path/'parent_forest_check.json', {'before_fit':before_fit,
            'after_fit':model.get_params(deep=False),'n_features_in':int(model.n_features_in_),
            'n_outputs':int(model.n_outputs_),'fitted_trees':len(model.estimators_)})
    if algorithm=='ANN': model.save(path/'model.keras')
    else: joblib.dump(model,path/'model.joblib')
    joblib.dump({'X':sx,'Y':sy,'fit_ids':ids},path/'scalers.joblib')
    write_json(path/'fit.json',{'seed':seed,'fit_ids':ids,'parameters':params,'epochs':chosen_epochs,
                              'seconds':time.perf_counter()-started,'history':histories,
                              'estimator_parameters':model.get_params(deep=False) if algorithm not in ('ANN','SVR','GBoost') else params})
    def predict(values):
        prediction=model(sx.transform(values),training=False).numpy() if algorithm=='ANN' else model.predict(sx.transform(values))
        return sy.inverse_transform(prediction)
    return predict,chosen_epochs

def resume_valid(path, identity):
    marker=path/'COMPLETE.json'
    if not marker.is_file(): return False
    try:
        complete=read_json(marker)
        return (complete['identity']==identity and bool(complete['artifacts']) and
                all((path/name).is_file() and digest(path/name)==sha for name,sha in complete['artifacts'].items()))
    except (OSError, ValueError, KeyError): return False

def environment():
    return {'python':sys.version,'platform':platform.platform(),'machine':platform.machine(),
            'packages':dict(sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions()))}

def run(config,algorithm,dataset,seed,smoke=False,evaluate_test=False,plan=False,runs_root=None):
    import numpy as np
    from sklearn.model_selection import KFold
    spec=config['parameters'][algorithm][dataset]
    if not spec['candidates']: raise ValueError(f'{algorithm}/{dataset}: needs development-only parameter selection')
    candidates=copy.deepcopy(spec['candidates'])
    if len(candidates)>config['max_candidates']: raise ValueError('Candidate budget exceeds explicit configured cap')
    if smoke:
        if evaluate_test: raise ValueError('Smoke verification never evaluates the test')
        for p in candidates:
            if algorithm=='ANN':p.update(epochs=2,patience=1)
            if algorithm in ('RF','GBoost'):p.update(n_estimators=2,max_depth=2)
    x,y,xt,yt,ids,test_ids,hashes=load_data(config,dataset)
    folds=2 if smoke else config['folds']
    if folds<2 or folds>len(x): raise ValueError('Invalid fold count')
    if not 0<config['inner_validation_fraction']<1:raise ValueError('Invalid inner split')
    effective={'algorithm':algorithm,'dataset':dataset,'seed':seed,'folds':folds,'candidates':candidates,
               'parameter_record':spec,'inner_validation_fraction':config['inner_validation_fraction'],
               'selection_metric':config['selection_metric'],'threads':config['threads'],
               'mode':'verification_only' if smoke else 'evaluation','evaluate_test':evaluate_test,
               'data_hashes':hashes,'code_sha256':digest(__file__),'environment':environment()}
    if effective['selection_metric']!='mse_original_cp':raise ValueError('Unsupported selection metric')
    key=fingerprint(effective);base=(Path(runs_root).resolve() if runs_root is not None else ROOT/'runs')/f'{algorithm}_{dataset}'/key
    if plan: print('READY',algorithm,dataset,'candidates',len(candidates),'folds',folds);return None
    if base.exists():
        for attempt in sorted(base.iterdir(),reverse=True):
            if attempt.is_dir() and resume_valid(attempt,key):print('SKIP',algorithm,dataset,attempt.name);return attempt
    path=base/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8])
    path.mkdir(parents=True);write_json(path/'configuration.json',effective)
    write_json(path/'ids.json',{'development':ids,'reserved_test':test_ids,'input_columns':GEOM,'output_columns':OUTPUTS})
    start=time.perf_counter()
    try:
        splits=list(KFold(n_splits=folds,shuffle=True,random_state=seed).split(x))
        summaries=[]
        for ci,params in enumerate(candidates):
            cp=path/f'candidate_{ci}';cp.mkdir();oof=np.empty_like(y);epochs=[];fold_metrics=[]
            for fi,(tr,va) in enumerate(splits):
                fold_seed=seed+fi+1
                predict,epoch=fit_partition(algorithm,params,x[tr],y[tr],[ids[i] for i in tr],fold_seed,
                                             cp/f'fold_{fi}',config['inner_validation_fraction'])
                begin=time.perf_counter();oof[va]=predict(x[va]);elapsed=time.perf_counter()-begin
                result=metrics(y[va],oof[va]);fold_metrics.append(result)
                write_json(cp/f'fold_{fi}'/'evaluation.json',{'validation_ids':[ids[i] for i in va],
                           'metrics':result,'predict_seconds':elapsed})
                if epoch:epochs.append(epoch)
            result=metrics(y,oof)
            np.savez_compressed(cp/'oof_predictions.npz',ids=np.array(ids),reference=y,prediction=oof)
            summary={'candidate':ci,'parameters':params,'development_oof':result,'fold_metrics':fold_metrics,'epochs':epochs}
            write_json(cp/'summary.json',summary);summaries.append(summary)
        best=min(summaries,key=lambda s:s['development_oof']['mse'])
        final_epochs=max(1,int(np.median(best['epochs']))) if best['epochs'] else None
        predict,_=fit_partition(algorithm,best['parameters'],x,y,ids,seed,path/'final',
                                 config['inner_validation_fraction'],final_epochs=final_epochs)
        test_result=None
        if evaluate_test:
            begin=time.perf_counter();pred=predict(xt);elapsed=time.perf_counter()-begin
            test_result={'metrics':metrics(yt,pred),'predict_seconds':elapsed}
            np.savez_compressed(path/'test_predictions.npz',ids=np.array(test_ids),reference=yt,prediction=pred)
        write_json(path/'result.json',{'status':'complete','mode':effective['mode'],'selected_candidate':best['candidate'],
                   'selection':'fixed documented reuse' if len(candidates)==1 else 'development CV only; selection score is not unbiased performance',
                   'development_oof':best['development_oof'],'test':test_result,'total_seconds':time.perf_counter()-start})
        artifacts={p.relative_to(path).as_posix():digest(p) for p in path.rglob('*') if p.is_file()}
        write_json(path/'COMPLETE.json',{'identity':key,'artifacts':artifacts})
        print('COMPLETE',algorithm,dataset,effective['mode'],round(time.perf_counter()-start,2),'s',path.relative_to(ROOT))
        return path
    except Exception as error:
        write_json(path/'FAILED.json',{'error':repr(error),'seconds':time.perf_counter()-start})
        raise

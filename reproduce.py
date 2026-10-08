"""Reproduce the accepted experiments. Default: validate the plan, without fitting."""
from pathlib import Path
import argparse,copy,json,os,sys,uuid

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
import evaluation as flow
flow.ROOT=ROOT

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def configuration(cfg,job):
    dataset=job['dataset'];algorithm=job['algorithm']
    return {'master':cfg['master'],'datasets':{dataset:job['dataset_path']},
        'folds':job['folds'],'inner_validation_fraction':job['inner_validation_fraction'],
        'threads':job['threads'],'selection_metric':job['selection_metric'],
        'max_candidates':len(job['candidates']),
        'parameters':{algorithm:{dataset:{**copy.deepcopy(job['parameter_record']),
                                        'candidates':copy.deepcopy(job['candidates'])}}}}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--analysis','--analisis',dest='analysis',choices=['all','principal','convergencia'],default='all')
    mode=ap.add_mutually_exclusive_group()
    mode.add_argument('--plan',action='store_true');mode.add_argument('--execute','--ejecutar',dest='execute',action='store_true')
    ap.add_argument('--algorithm','--modelo',dest='algorithm',choices=['ANN','SVR','KNN','DT','RF','GBoost'])
    ap.add_argument('--dataset','--conjunto',dest='dataset',help='Accepted design label, for example LHS126 or CONV_H01')
    ap.add_argument('--seed','--semilla',dest='seed',type=int,help='Filter recorded seeds; never replace them')
    args=ap.parse_args();cfg=read(ROOT/'config.json')
    selections={}
    for scope,records in cfg['selections'].items():
        if args.analysis!='all' and scope!=args.analysis:continue
        for label,uid in records.items():
            if args.dataset and label.split('/')[1]!=args.dataset:continue
            job=cfg['jobs'][uid]
            if args.algorithm and job['algorithm']!=args.algorithm:continue
            if args.seed is not None and job['seed']!=args.seed:continue
            selections.setdefault(uid,[]).append(scope+'/'+label)
    if not selections:ap.error('No accepted experiments match these filters')
    for file,h in cfg['data_file_hashes'].items():
        if flow.digest(ROOT/file)!=h:raise ValueError('Changed data: '+file)
    out=ROOT/'new_runs'/uuid.uuid4().hex[:12]
    if args.execute:out.mkdir(parents=True,exist_ok=False)
    for k,v in {'CUDA_VISIBLE_DEVICES':'-1','TF_ENABLE_ONEDNN_OPTS':'0',
                'TF_DETERMINISTIC_OPS':'1','TF_CPP_MIN_LOG_LEVEL':'2'}.items():os.environ[k]=v
    for uid,labels in selections.items():
        job=cfg['jobs'][uid];runtime=configuration(cfg,job)
        for name in ['OMP_NUM_THREADS','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS']:os.environ[name]=str(job['threads'])
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=job['threads']):
            x,y,xt,yt,ids,test_ids,_=flow.load_data(runtime,job['dataset'])
            if ids!=job['development_ids'] or test_ids!=job['test_ids']:raise ValueError('Ordered IDs differ: '+uid)
            print('EXPERIMENT',uid,', '.join(labels),flush=True)
            path=flow.run(runtime,job['algorithm'],job['dataset'],job['seed'],
                evaluate_test=True,plan=not args.execute,runs_root=out/'runs')
        if args.execute:flow.write_json(out/(uid+'.json'),
            {'reference_id':uid,'analyses':labels,'original_source':job['source'],
             'new_run':path.relative_to(ROOT).as_posix(),'actual_configuration':runtime})
    print(('EXECUTED' if args.execute else 'PLAN ONLY: no fitting'),len(selections),'unique evaluations',flush=True)
    if args.execute:print('OUTPUT',out)

if __name__=='__main__':main()

"""Render F5-F8 from frozen selected predictions; preserve supplied F1-F4 graphics."""
from pathlib import Path
import argparse,csv,json,hashlib,io,shutil,sys
import numpy as np
from scipy.stats import pearsonr,t
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import evaluation as flow
flow.ROOT=ROOT
DESIGNS=[base+suffix for base in ['CCD43','LHS43','LHS126'] for suffix in ['', '_FFD','_2k']]
LABELS={key:({'CCD43':'CCD','LHS43':'R-LHS','LHS126':'LHS'}[key.split('_')[0]]+
              (r' + $2^{k-1}$' if key.endswith('_FFD') else r' + $2^k$' if key.endswith('_2k') else '')) for key in DESIGNS}
def make_figures(path,predictions,x,y,only=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':12,'axes.labelsize':13,'axes.titlesize':12,'savefig.dpi':220})
    names=['F5a_G_SVR_CCD','F5b_G_SVR_CCD2','F5c_G_SVR_CCD3','F5d_G_SVR_RLHS','F5e_G_SVR_RLHS2','F5f_G_SVR_RLHS3','F5g_G_SVR_LHS','F5h_G_SVR_LHS2','F5i_G_SVR_LHS3']
    for d,name in zip(DESIGNS,names):
        if only is not None and name not in only:continue
        p=predictions['SVR',d];fig,ax=plt.subplots(figsize=(4.4,3.6),layout='constrained')
        ax.scatter(y.ravel(),p.ravel(),s=3,alpha=.25,color='#2166ac',rasterized=True)
        lo=min(y.min(),p.min());hi=max(y.max(),p.max());pad=(hi-lo)*.04
        ax.plot([lo-pad,hi+pad],[lo-pad,hi+pad],'--',color='#b2182b',lw=1)
        ax.set(xlim=(lo-pad,hi+pad),ylim=(lo-pad,hi+pad),xlabel=r'CFD $C_p$',ylabel=r'Predicted $C_p$',title=LABELS[d])
        r2=flow.metrics(y,p)['r2_pooled'];ax.text(.04,.94,f'$R^2$ = {r2:.3f}',transform=ax.transAxes,va='top')
        fig.savefig(path/(name+'.png'));plt.close(fig)
    names=['F6a_ANN_LHS','F6b_ANN_LHS2','F6c_ANN_LHS3','F6d_SVR_LHS','F6e_SVR_LHS2','F6f_SVR_LHS3']
    for (a,d),name in zip([(a,d) for a in ['ANN','SVR'] for d in ['LHS126','LHS126_FFD','LHS126_2k']],names):
        if only is not None and name not in only:continue
        scores=np.array(flow.metrics(y,predictions[a,d])['case_r2']);angle=x[:,4];n=len(scores)
        slope,intercept=np.polyfit(angle,scores,1);grid=np.linspace(angle.min(),angle.max(),100)
        fitted=intercept+slope*grid;residual=scores-(intercept+slope*angle)
        se=np.sqrt(np.sum(residual**2)/(n-2))*np.sqrt(1/n+(grid-angle.mean())**2/np.sum((angle-angle.mean())**2))
        width=t.ppf(.975,n-2)*se
        fig,ax=plt.subplots(figsize=(4.4,3.6),layout='constrained')
        ax.scatter(angle,scores,s=20,color='#2166ac');ax.plot(grid,fitted,color='#2166ac')
        ax.fill_between(grid,fitted-width,fitted+width,color='#2166ac',alpha=.15)
        ax.set(xlabel='Wind angle (degrees)',ylabel=r'Case-level $R^2$',title=a+' / '+LABELS[d])
        ax.text(.04,.06,f'r = {pearsonr(angle,scores).statistic:.3f}',transform=ax.transAxes)
        fig.savefig(path/(name+'.png'));plt.close(fig)

def convergence_figures(out,records):
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'font.size': 12, 'axes.labelsize': 12,
                         'legend.fontsize': 11, 'xtick.labelsize': 11, 'ytick.labelsize': 11})
    filenames = {('ANN', 'r2_pooled'): 'F7a_ANN_R2.png', ('SVR', 'r2_pooled'): 'F7b_SVR_R2.png',
                 ('ANN', 'mse'): 'F8a_ANN_MSE.png', ('SVR', 'mse'): 'F8b_SVR_MSE.png'}
    limits = {}
    for metric in ['r2_pooled', 'mse']:
        endpoints = [r[split][metric]['mean']+sign*r[split][metric]['sd']
                     for r in records for split in ['development_oof', 'test'] for sign in [-1, 1]]
        pad = .06*(max(endpoints)-min(endpoints))
        limits[metric] = (min(endpoints)-pad, max(endpoints)+pad)
    for algorithm in ['ANN', 'SVR']:
        rows = [r for r in records if r['algorithm'] == algorithm]
        n = [r['n'] for r in rows]
        for metric in ['r2_pooled', 'mse']:
            fig, ax = plt.subplots(figsize=(4, 2.8), dpi=300)
            for split, label, color, marker in [('development_oof', 'CV (5 folds)', '#1f77b4', 'o'),
                                               ('test', 'Final test', '#ff7f0e', 's')]:
                mean = np.array([r[split][metric]['mean'] for r in rows])
                sd = np.array([r[split][metric]['sd'] for r in rows])
                ax.errorbar(n, mean, yerr=sd, label=label, color=color, marker=marker,
                            markersize=4, linewidth=1.4, capsize=3, elinewidth=1)
            ax.set_xlabel('Development configurations, N')
            ax.set_ylabel(r'$R^2$ (pooled)' if metric == 'r2_pooled' else r'MSE (original $C_p$ scale)')
            ax.set_ylim(*limits[metric])
            ax.set_xticks([13, 38, 63, 88, 126])
            ax.grid(True, alpha=.3)
            ax.legend(loc='best', framealpha=.9)
            fig.tight_layout()
            fig.savefig(out/filenames[algorithm, metric], dpi=300)
            plt.close(fig)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    mode=ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true',help='Render in memory and compare retained reference pixels')
    mode.add_argument('--output',type=Path,help='NEW folder for all 23 graphics')
    args=ap.parse_args()
    cfg=flow.read_json(ROOT/'config.json');ref=flow.read_json(ROOT/'results/reference.json')
    master={r['ID']:r for r in csv.DictReader((ROOT/cfg['master']).open(encoding='utf-8-sig',newline=''))}
    ids=[f'T_{i}' for i in range(1,21)]
    x=np.array([[float(master[i][c]) for c in flow.GEOM] for i in ids])
    y=np.array([[float(master[i][c]) for c in flow.OUTPUTS] for i in ids])
    assert flow.digest(ROOT/'results/predictions.npz')==cfg['prediction_file_sha256']
    predictions={}
    with np.load(ROOT/'results/predictions.npz',allow_pickle=False) as z:
        for label,uid in cfg['selections']['principal'].items():
            assert cfg['jobs'][uid]['test_ids']==ids
            p=z[uid+'_test'].copy()
            actual=flow.metrics(y,p);expected=ref['metrics'][label]
            for key in ['r2_pooled','mae','mse','rmse','case_r2']:
                np.testing.assert_allclose(actual[key],expected[key],rtol=1e-12,atol=1e-12)
            predictions[tuple(label.split('/'))]=p
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.image import imread
    plt.rcdefaults()
    rendered={}
    save=Figure.savefig
    def capture(fig,target,*a,**kw):
        stream=io.BytesIO()
        save(fig,stream,*a,format='png',**kw)
        rendered[Path(target).name]=stream.getvalue()
    Figure.savefig=capture
    try:
        make_figures(Path('.'),predictions,x,y)
        convergence_figures(Path('.'),ref['convergence']['rows'])
    finally:
        Figure.savefig=save
    assert len(rendered)==19
    supplied=ROOT/'results/figures'
    preserved=[p for p in supplied.glob('*.png') if p.name not in rendered]
    assert len(preserved)==4
    matching=[];different=[]
    for name,blob in rendered.items():
        same=np.array_equal(imread(io.BytesIO(blob)),imread(supplied/name))
        (matching if same else different).append(name)
    if args.output:
        args.output.mkdir(parents=True,exist_ok=False)
        for name,blob in rendered.items():(args.output/name).write_bytes(blob)
        for p in preserved:shutil.copyfile(p,args.output/p.name)
    report={'status':'PASS_PIXEL_EXACT' if not different else 'NUMERICAL_INPUTS_PASS_RENDERING_DIFFERS',
        'generated_panels':19,'preserved_graphics':[p.name for p in preserved],
        'pixel_identical_panels':matching,'rendering_differences':different,
        'training_performed':False,'preserved_graphics_scope':'F1-F3 schematics and historical F4 projection; independent original generators not recovered'}
    print(json.dumps(report))
    if args.check and different:raise SystemExit(1)

if __name__=='__main__':main()

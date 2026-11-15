"""Measured evidence. Run via Makefile to constrain BLAS to one thread."""
import sys, os, platform, json, time, cProfile, pstats, io, warnings, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy, statsmodels
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from runner import run, scenario
from dgps import generate_freedman_population, generate_nonlinear_population, randomize_treatment
from methods import estimate_effects as baseline
from optimized import estimate_effects as optimized
OUT=ROOT/'results'/'performance'; OUT.mkdir(parents=True,exist_ok=True)

def dump(name, obj):
    (OUT/name).write_text(json.dumps(obj,indent=2))

def environment():
    dump('environment.json',dict(python=sys.version,platform=platform.platform(),cpu_count=os.cpu_count(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,statsmodels=statsmodels.__version__,blas_threads={k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']},timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))

def full():
    spec=importlib.util.spec_from_file_location('original_simulation',ROOT/'baseline/simulation.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    mod.RAW_DIR=OUT/'original_baseline';mod.RAW_DIR.mkdir(exist_ok=True)
    rows=[];total=time.perf_counter()
    for d in ['freedman','nonlinear']:
        start=time.perf_counter();df=mod.run_simulation(dgp=d)
        rows.append(dict(dgp=d,seconds=time.perf_counter()-start,rows=len(df),nonfinite_mle=int((~np.isfinite(df.MLE_Beta2)).sum()),nonfinite_plugin=int((~np.isfinite(df.Plugin_Delta)).sum()),nonfinite_itt=int((~np.isfinite(df.ITT_Delta)).sum())))
    dump('baseline_full.json',dict(total_seconds=time.perf_counter()-total,conditions=rows,notes='Unmodified Unit 2 driver; output redirected. Includes computation and CSV writing, excludes interpreter imports and metrics plotting. Seed(None) remains as submitted.'))

def profile():
    for engine in ['baseline','optimized']:
        pr=cProfile.Profile();pr.enable();run(reps=100,ns=[1000,5000],engine=engine);pr.disable()
        pr.dump_stats(str(OUT/f'{engine}.prof'))
        stream=io.StringIO();pstats.Stats(pr,stream=stream).strip_dirs().sort_stats('cumulative').print_stats(30)
        (OUT/f'profile_{engine}.txt').write_text(stream.getvalue())

def benchmark():
    run(reps=2,ns=[100],engine='baseline');run(reps=2,ns=[100])
    rows=[]
    for repeat in range(3):
        configs=[('baseline',1),('optimized',1),('optimized',2)]
        if repeat%2:configs.reverse()
        for engine,workers in configs:
            start=time.perf_counter();df=run(reps=1000,engine=engine,workers=workers)
            rows.append(dict(repeat=repeat,engine=engine,workers=workers,seconds=time.perf_counter()-start,rows=len(df)))
            print(rows[-1],flush=True)
            if repeat==0:df.to_csv(OUT/f'paired_full_{engine}_{workers}.csv',index=False)
    df=pd.DataFrame(rows);df.to_csv(OUT/'benchmark.csv',index=False)
    a=pd.read_csv(OUT/'paired_full_baseline_1.csv');b=pd.read_csv(OUT/'paired_full_optimized_1.csv')
    evidence={}
    for col in ['MLE_Beta2','Plugin_Delta','ITT_Delta']:
        valid=np.isfinite(a[col])&np.isfinite(b[col]);diff=np.abs(a.loc[valid,col]-b.loc[valid,col])
        evidence[col]=dict(paired_finite=int(valid.sum()),finite_mismatch=int((np.isfinite(a[col])!=np.isfinite(b[col])).sum()),max_abs_difference=float(diff.max()),within_1e_7=int((diff<=1e-7).sum()))
    dump('full_regression.json',evidence)
    summary=df.groupby(['engine','workers']).seconds.agg(['median','min','max']).reset_index()
    fig,ax=plt.subplots(figsize=(7,4))
    ax.bar(['Statsmodels\nserial','Optimized\nserial','Optimized\n2 workers'],summary['median'],color=['#67768a','#15847b','#306cb0'])
    ax.set_ylabel('Wall time (seconds), median of 3 runs');ax.set_title('Full study: 8,000 simulation replicates')
    for i,v in enumerate(summary['median']):ax.text(i,v,f'{v:.2f}s',ha='center',va='bottom')
    ax.set_ylim(0,summary['median'].max()*1.18);fig.tight_layout();fig.savefig(OUT/'timing_comparison.png',dpi=180);plt.close(fig)

def complexity():
    rows=[]
    for n in [100,250,500,1000,2500,5000,10000,20000]:
        for d in ['freedman','nonlinear']:
            for repeat in range(5):
                engines=['baseline','optimized'] if repeat%2==0 else ['optimized','baseline']
                for engine in engines:
                    start=time.perf_counter();scenario((d,n,30,1234567891,engine));elapsed=time.perf_counter()-start
                    rows.append(dict(dgp=d,n=n,repeat=repeat,engine=engine,seconds_per_rep=elapsed/30))
    df=pd.DataFrame(rows);df.to_csv(OUT/'complexity.csv',index=False)
    slopes=[];fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,d in zip(axes,['freedman','nonlinear']):
        for e in ['baseline','optimized']:
            s=df[(df.dgp==d)&(df.engine==e)].groupby('n').seconds_per_rep.median()
            slope,intercept=np.polyfit(np.log(s.index),np.log(s.values),1)
            slopes.append(dict(dgp=d,engine=e,log_log_slope=float(slope)))
            line, = ax.loglog(s.index,s.values,'o-',label=f'{e} (slope {slope:.2f})')
            ax.loglog(s.index,np.exp(intercept)*s.index**slope,'--',color=line.get_color(),alpha=.5)
        ax.set_title(d.capitalize());ax.set_xlabel('Population size n');ax.set_ylabel('Seconds per replicate');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'complexity.png',dpi=180);plt.close(fig);dump('complexity_slopes.json',slopes)

def stability():
    # Expose warnings and swallowed exceptions in a copy; arithmetic unchanged.
    source=(ROOT/'baseline/methods.py').read_text()
    source=source.replace('warnings.simplefilter("ignore")','warnings.simplefilter("always")')
    source=source.replace('beta_hat_2 = model.params[1]', "diagnostic['converged'] = bool(model.mle_retvals.get('converged', False))\n            beta_hat_2 = model.params[1]")
    source=source.replace('except Exception:', "except Exception as exc:\n            diagnostic['exception'] = type(exc).__name__ + ': ' + str(exc)")
    namespace={'diagnostic':{}};exec(compile(source,'instrumented_baseline','exec'),namespace)
    rows=[];events=[];cases=[]
    for d,gen in [('freedman',generate_freedman_population),('nonlinear',generate_nonlinear_population)]:
        for n in [20,100,500,1000,5000]:
            z,c,t=gen(n,seed=908)
            for rep in range(100):
                x,y=randomize_treatment(c,t,seed=rep);cases.append((d,n,rep,x,z,y))
    x=np.tile([0,1],50);z=np.linspace(-2,2,100);y=np.tile([0,1,1,0],25)
    cases += [('empty_arm',100,0,np.ones(100),z,y),('constant_outcome',100,0,x,z,np.zeros(100)),('rank_deficient',100,0,x,np.zeros(100),y),('separated',100,0,x,z,(z>0).astype(int)),('large_covariate',100,0,x,1e8+1e5*z,y)]
    for d,n,rep,x,z,y in cases:
        for engine in ['baseline','optimized']:
            namespace['diagnostic']={}
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                if engine=='baseline':
                    value=namespace['estimate_effects'](x,z,y);info=namespace['diagnostic']
                    status='exception' if 'exception' in info else ('ok' if info.get('converged') else 'nonconverged')
                else:
                    value,info=optimized(x,z,y,diagnostics=True);status=info['status']
            rows.append(dict(condition=d,n=n,rep=rep,engine=engine,status=status,warnings=len(caught),nonfinite_mle=int(not np.isfinite(value[0])),nonfinite_plugin=int(not np.isfinite(value[1])),nonfinite_itt=int(not np.isfinite(value[2]))))
            if caught or status!='ok':events.append(dict(condition=d,n=n,rep=rep,engine=engine,status=status,details=info,warnings=[str(w.message) for w in caught]))
    df=pd.DataFrame(rows);df.to_csv(OUT/'stability.csv',index=False)
    df.groupby(['condition','n','engine','status']).agg(runs=('rep','size'),warnings=('warnings','sum'),nonfinite_mle=('nonfinite_mle','sum'),nonfinite_plugin=('nonfinite_plugin','sum'),nonfinite_itt=('nonfinite_itt','sum')).reset_index().to_csv(OUT/'stability_summary.csv',index=False)
    dump('stability_events.json',events)

if __name__=='__main__':
    environment()
    {'baseline-full':full,'profile':profile,'benchmark':benchmark,'complexity':complexity,'stability-check':stability}[sys.argv[1]]()

"""Educational QRC residual anomaly detector for Ausgrid GC smart-meter data.
No claim of quantum advantage: a small reproducible state-vector simulator.
"""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

I2 = np.eye(2, dtype=complex)
Z = np.diag([1, -1]).astype(complex)

def ry(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)

def kron3(a, b, c):
    return np.kron(np.kron(a, b), c)

def cnot(control, target):
    U = np.zeros((8, 8), complex)
    for i in range(8):
        bits = [(i >> (2-q)) & 1 for q in range(3)]
        if bits[control]: bits[target] ^= 1
        j = sum(bits[q] << (2-q) for q in range(3))
        U[j, i] = 1
    return U

C01, C12 = cnot(0,1), cnot(1,2)
OBS = [kron3(Z,I2,I2), kron3(I2,Z,I2), kron3(I2,I2,Z),
       kron3(Z,Z,I2), kron3(I2,Z,Z), kron3(Z,I2,Z)]

def reservoir(inputs, seed=42):
    """3-qubit untrained gate reservoir with carried quantum state and Z readouts."""
    rng = np.random.default_rng(seed)
    fixed = rng.uniform(-0.4, 0.4, size=(3, 3))
    U0 = kron3(ry(fixed[0,0]), ry(fixed[0,1]), ry(fixed[0,2]))
    U1 = kron3(ry(fixed[1,0]), ry(fixed[1,1]), ry(fixed[1,2]))
    U2 = kron3(ry(fixed[2,0]), ry(fixed[2,1]), ry(fixed[2,2]))
    state = np.zeros(8, complex); state[0]=1
    out = np.zeros((len(inputs), 6))
    for t, x in enumerate(inputs):
        enc = kron3(ry(1.5*x), ry(0.7*x), I2)
        state = U2 @ C12 @ U1 @ C01 @ U0 @ enc @ state
        # State is propagated through time; measurements are simulated expectations.
        out[t] = [np.real(np.vdot(state, op @ state)) for op in OBS]
    return out

def get_demo(n=2100, seed=5):
    """SYNTHETIC DEMONSTRATION ONLY: not Ausgrid observations."""
    rng=np.random.default_rng(seed)
    t=np.arange(n)
    daily = 0.34*np.sin(2*np.pi*t/48-0.8)+0.16*np.cos(4*np.pi*t/48)
    weekly=0.10*np.cos(2*np.pi*t/(48*7))
    val=np.maximum(0.04,0.8+daily+weekly+rng.normal(0,.085,n))
    return pd.Series(val,index=pd.date_range('2012-07-01',periods=n,freq='30min'),name='GC')

def load_ausgrid(path, customer=1, limit=2100):
    """Accept original Ausgrid wide CSV, or tidy datetime/value CSV.
    Original has Customer, Consumption Category, date and 48 half-hour time columns.
    """
    p=Path(path)
    if not p.exists(): raise FileNotFoundError(f'Missing data: {p}')
    df=pd.read_csv(p, low_memory=False, encoding='utf-8-sig', skiprows=1 if p.name in ('2011.csv','2012.csv','2013.csv') else 0)
    df.columns=[str(x).strip() for x in df.columns]
    lower={c.lower():c for c in df.columns}
    if 'datetime' in lower and ('gc' in lower or 'value' in lower or 'load' in lower):
        v=next(lower[x] for x in ['gc','value','load'] if x in lower)
        dates=pd.to_datetime(df[lower['datetime']],errors='coerce')
        s=pd.Series(pd.to_numeric(df[v],errors='coerce').to_numpy(),index=dates)
    else:
        cust=next((c for c in df if c.lower() in ['customer','customer id','customer_id']),None)
        cat=next((c for c in df if 'consumption category' in c.lower() or c.lower() in ['category','channel']),None)
        date=next((c for c in df if c.lower() in ['date','reading date']),None)
        if not all([cust,cat,date]):
            raise ValueError('Expected original Ausgrid fields Customer, Consumption Category, date, or tidy datetime/value')
        df=df[(pd.to_numeric(df[cust],errors='coerce')==customer)&(df[cat].astype(str).str.upper().str.strip()=='GC')].copy()
        if df.empty: raise ValueError('No GC readings for requested customer')
        timecols=[c for c in df.columns if ':' in c and c.split(':')[0].isdigit()]
        if len(timecols)<40: raise ValueError('Could not identify 48 half-hour columns')
        melted=df.melt(id_vars=[date],value_vars=timecols,var_name='time',value_name='kwh')
        raw_dates=pd.to_datetime(melted[date],dayfirst=True,errors='coerce')
        s=pd.Series(pd.to_numeric(melted['kwh'],errors='coerce').to_numpy(),
                    index=raw_dates+pd.to_timedelta(melted['time']+':00',errors='coerce')+pd.to_timedelta((melted['time']=='0:00').astype(int),unit='D'))
    s=s[~s.index.isna()].sort_index()
    s=s[~s.index.duplicated(keep='first')]
    s=s.resample('30min').mean().iloc[:limit]
    # Preserve zeros. Interpolate only short missing runs, then drop remaining missing.
    s=s.interpolate(method='time',limit=2).dropna()
    if len(s)<500: raise ValueError('At least 500 usable half-hour points required')
    return s.clip(lower=0).rename('GC')

def design(series, lags=48):
    a=series.to_numpy(dtype=float)
    # Lagged feature at time t uses only readings <=t-1; forecast y[t].
    X=np.stack([a[lags-k:len(a)-k] for k in [1,2,3,48]],axis=1)
    y=a[lags:]
    return X,y,series.index[lags:]

def ridge_fit(X,y,lam=1e-2):
    Xa=np.column_stack((np.ones(len(X)),X))
    reg=np.eye(Xa.shape[1])*lam; reg[0,0]=0
    return np.linalg.solve(Xa.T@Xa+reg,Xa.T@y)

def ridge_predict(X,w):
    return np.column_stack((np.ones(len(X)),X))@w

def scores(pred,y):
    err=y-pred
    return {'MAE':round(float(np.mean(np.abs(err))),5),
            'RMSE':round(float(np.sqrt(np.mean(err**2))),5)}

def run(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    series=get_demo(n=args.limit) if args.demo else load_ausgrid(args.csv,args.customer,args.limit)
    mode='SYNTHETIC DEMONSTRATION (NOT AUSGRID)' if args.demo else 'REAL AUSGRID GC DATA'
    Xraw,y,dates=design(series)
    n=len(y); ntrain=int(.6*n); nval=int(.2*n)
    a,b=ntrain,ntrain+nval
    if min(ntrain,nval,n-b)<40: raise ValueError('Too little data to split train/validation/test')
    # Scale features based on train only; no future information.
    mu=Xraw[:a].mean(axis=0); sd=Xraw[:a].std(axis=0)+1e-8
    X=(Xraw-mu)/sd
    ymu=y[:a].mean(); ysd=y[:a].std()+1e-8
    yn=(y-ymu)/ysd
    baseline_w=ridge_fit(X[:a],yn[:a])
    pred_base=ridge_predict(X,baseline_w)*ysd+ymu
    # QRC: input stream is last known load, standardized using training only.
    stream=np.clip(X[:,0],-4,4)
    qfeat=reservoir(stream,seed=args.seed)
    qX=np.column_stack([X,qfeat]) # Hybrid linear readout + quantum reservoir features.
    q_w=ridge_fit(qX[:a],yn[:a],lam=.1)
    pred_q=ridge_predict(qX,q_w)*ysd+ymu
    # Quantile threshold calibrated exclusively on validation data.
    qth=np.quantile(np.abs(y[a:b]-pred_q[a:b]),.99)
    bth=np.quantile(np.abs(y[a:b]-pred_base[a:b]),.99)
    # Evaluation of natural outliers gives alert rates only: no true labels.
    qalert=np.abs(y[b:]-pred_q[b:])>qth
    balert=np.abs(y[b:]-pred_base[b:])>bth
    result={'mode':mode,'customer':None if args.demo else args.customer,
        'samples':len(series),'train':a,'validation':nval,'test':n-b,
        'qrc':{**scores(pred_q[b:],y[b:]),'threshold_kwh':round(float(qth),5),'alert_rate':round(float(qalert.mean()),5)},
        'ridge_baseline':{**scores(pred_base[b:],y[b:]),'threshold_kwh':round(float(bth),5),'alert_rate':round(float(balert.mean()),5)}}
    # Controlled perturbations provide *labeled* test cases. Detection F1 here is
    # specifically synthetic-injection performance, not naturally labeled anomalies.
    rng=np.random.default_rng(args.seed+11)
    k=max(1,int(.05*(n-b)))
    indices=rng.choice(n-b,k,replace=False)
    injected=np.zeros(n-b,dtype=bool);injected[indices]=True
    ychanged=y[b:].copy()
    ychanged[indices]+=rng.choice([-1,1],k)*np.maximum(.55,2.5*ysd)
    for label,pred,th in [('qrc',pred_q[b:],qth),('ridge_baseline',pred_base[b:],bth)]:
        flag=np.abs(ychanged-pred)>th
        tp=int((flag&injected).sum()); fp=int((flag&~injected).sum()); fn=int((~flag&injected).sum())
        precision=tp/(tp+fp) if tp+fp else 0
        recall=tp/(tp+fn) if tp+fn else 0
        f1=2*precision*recall/(precision+recall) if precision+recall else 0
        result[label]['injected_precision']=round(float(precision),4)
        result[label]['injected_recall']=round(float(recall),4)
        result[label]['injected_F1']=round(float(f1),4)
    (out/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    pd.DataFrame({'datetime':dates[b:],'actual_kwh':y[b:],'qrc_prediction':pred_q[b:],
       'ridge_prediction':pred_base[b:],'qrc_residual':y[b:]-pred_q[b:],
       'qrc_alert':qalert,'ridge_alert':balert,'synthetic_injection_label':injected,
       'injected_test_kwh':ychanged}).to_csv(out/'test_predictions.csv',index=False)
    sl=slice(b,min(b+340,n))
    fig,ax=plt.subplots(figsize=(11,4));ax.plot(dates[sl],y[sl],label='Observed',lw=1)
    ax.plot(dates[sl],pred_q[sl],label='Hybrid QRC forecast',lw=1)
    ax.set(ylabel='Half-hour electricity (kWh)',title=mode+' — forecast on test set');ax.legend();fig.autofmt_xdate();fig.tight_layout()
    fig.savefig(out/'forecast.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4));error=np.abs(y[b:]-pred_q[b:]);ax.plot(dates[b:],error,lw=.75,label='Absolute residual');ax.axhline(qth,color='black',ls='--',label='Validation 99th percentile');ax.scatter(dates[b:][qalert],error[qalert],s=14,c='red',label='Alerts');ax.set(ylabel='Absolute residual (kWh)',title=mode+' — residual-based alerts');ax.legend();fig.autofmt_xdate();fig.tight_layout();fig.savefig(out/'anomalies.png',dpi=160);plt.close(fig)
    print(json.dumps(result,indent=2)); print('Outputs:',out.resolve())

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--csv',help='Original Ausgrid half-hour CSV or tidy datetime/value CSV')
    p.add_argument('--customer',type=int,default=1)
    p.add_argument('--limit',type=int,default=2100)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--out',default='outputs')
    p.add_argument('--demo',action='store_true',help='Run labeled synthetic demonstration; not Ausgrid')
    a=p.parse_args()
    if not a.demo and not a.csv: p.error('Use --csv path/to/Ausgrid.csv for real data, or --demo for synthetic demo')
    run(a)

"""Reproducible simulations for Covariance Kernels on Unordered Pair Spaces.

The script is platform independent and writes all results beneath --output-dir.
The full manuscript settings use seed 20260814 and Monte Carlo sizes
B_risk=5000, B_truncation=1000, B_perturbation=500. Use --quick for testing.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def args_parser():
    p=argparse.ArgumentParser()
    p.add_argument('--output-dir',type=Path,default=Path('results/simulations'))
    p.add_argument('--seed',type=int,default=20260814)
    p.add_argument('--quick',action='store_true')
    return p.parse_args()


def sym(A): return (A+A.T)/2

def eig_psd(A):
    w,U=np.linalg.eigh(sym(A)); idx=np.argsort(w)[::-1]
    w=w[idx]; U=U[:,idx]; w[np.abs(w)<1e-12]=0
    return np.maximum(w,0),U

def opnorm(A): return float(np.max(np.abs(np.linalg.eigvalsh(sym(A)))))

def pair_index(R,include_diag=False):
    return [(i,j) for i in range(R) for j in range(i if include_diag else i+1,R)]


def symmetric_tensor_operator(K):
    """Matrix of K tensor K in an orthonormal symmetric basis."""
    R=K.shape[0]; pairs=pair_index(R,True); N=len(pairs); B=np.zeros((R*R,N))
    def idx(r,s): return s*R+r
    for a,(r,s) in enumerate(pairs):
        if r==s: B[idx(r,s),a]=1.0
        else:
            B[idx(r,s),a]=1/np.sqrt(2.0); B[idx(s,r),a]=1/np.sqrt(2.0)
    return sym(B.T@np.kron(K,K)@B)

def pair_kernel(K,pairs):
    q=len(pairs); E=np.empty((q,q))
    for a,(i,j) in enumerate(pairs):
        for b,(u,v) in enumerate(pairs):
            E[a,b]=.5*(K[i,u]*K[j,v]+K[i,v]*K[j,u])
    return sym(E)

def make_base(R,d,decay,rng):
    Q,_=np.linalg.qr(rng.normal(size=(R,d)))
    j=np.arange(d)
    if decay=='exponential': lam=np.exp(-.35*j)
    elif decay=='power': lam=(j+1.)**-1.5
    else: lam=((d-j)/d)**2
    K=(Q*lam)@Q.T
    dd=np.sqrt(np.maximum(np.diag(K),1e-12)); K=K/np.outer(dd,dd)
    return sym(K)

def sample_psd(w,U,scale,rng):
    return U@(np.sqrt(scale*w)*rng.normal(size=len(w)))

def shrink(w,tau2,v): return tau2*w/(tau2*w+v)

def numerical_rank(w): return int(np.sum(w>1e-9*max(1.,float(w[0]))))


def experiment1(out,rng):
    rows=[]
    for R in [12,20,30]:
      for d in [4,8,12]:
       for decay in ['exponential','power','polynomial_taper']:
        K=make_base(R,d,decay,rng)
        Kfull=pair_kernel(K,pair_index(R,True)); Kloop=pair_kernel(K,pair_index(R,False))
        wf,_=eig_psd(Kfull); wl,_=eig_psd(Kloop)
        # Product-spectrum check uses the orthonormal symmetric-tensor operator,
        # not the unweighted finite Gram matrix Kfull.
        Tsym=symmetric_tensor_operator(K); ws,_=eig_psd(Tsym)
        wn,_=eig_psd(K); wn=wn[wn>1e-10*max(1.,wn[0])]
        prod=sorted([wn[i]*wn[j] for i in range(len(wn)) for j in range(i,len(wn))],reverse=True)
        mm=min(len(prod),len(ws)); prod_err=float(np.max(np.abs(ws[:mm]-np.asarray(prod[:mm]))))
        q=len(wl); violation=max(0.,float(np.max(wl-wf[:q])),float(np.max(wf[R:R+q]-wl)))
        rows.append(dict(R=R,d=d,decay=decay,nominal_pairs=R*(R-1)//2,
                         loopfree_rank=numerical_rank(wl),rank_bound=d*(d+1)//2,
                         product_spectrum_max_error=prod_err,interlacing_max_violation=violation,
                         trace_fraction_removed=(np.sum(wf)-np.sum(wl))/np.sum(wf)))
    df=pd.DataFrame(rows); df.to_csv(out/'experiment1_dimension_spectrum.csv',index=False)
    fig,ax=plt.subplots(figsize=(6.8,4.8))
    for d,g in df.groupby('d'):
        gg=g.groupby('R').first().reset_index()
        ax.plot(gg.R,gg.loopfree_rank/gg.nominal_pairs,marker='o',label=f'd={d}')
    ax.set(xlabel='Number of nodes R',ylabel='Edge rank / nominal pair dimension',ylim=(0,1.05)); ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(out/'experiment1_rank_summary.pdf'); plt.close(fig)
    return df


def experiment2(out,rng,B):
    R,d=25,8; K=make_base(R,d,'exponential',rng); E=pair_kernel(K,pair_index(R,False)); w,U=eig_psd(E); q=len(w); v=1.
    rows=[]
    for tau2 in [.25,1.,4.]:
        s=shrink(w,tau2,v)
        br=np.sum(v*tau2*w/(v+tau2*w)); rows.append(dict(scenario='Bayes_aligned',tau2=tau2,alpha=np.nan,exact_risk_ratio=br/(q*v)))
        norm=np.sqrt(q*v)
        for alpha in [0,.25,.5,.75,1.]:
            theta=np.zeros(q); theta[0]=np.sqrt(1-alpha)*norm; theta[max(0,numerical_rank(w)-1)]=np.sqrt(alpha)*norm
            risk=np.sum((1-s)**2*theta**2+v*s**2); rows.append(dict(scenario='Fixed_misalignment',tau2=tau2,alpha=alpha,exact_risk_ratio=risk/(q*v)))
    tau2=.05; s=shrink(w,tau2,v); theta=np.zeros(q); theta[max(0,numerical_rank(w)-1)]=3*np.sqrt(q*v)
    risk=np.sum((1-s)**2*theta**2+v*s**2); rows.append(dict(scenario='Severe_misspecification',tau2=tau2,alpha=1.,exact_risk_ratio=risk/(q*v)))
    df=pd.DataFrame(rows); df['replicates']=B; df.to_csv(out/'experiment2_risk_summary.csv',index=False)
    g=df[(df.scenario=='Fixed_misalignment')]
    fig,ax=plt.subplots(figsize=(6.8,4.8))
    for tau2,h in g.groupby('tau2'): ax.plot(h.alpha,h.exact_risk_ratio,marker='o',label=fr'$\tau^2={tau2:g}$')
    ax.axhline(1,ls='--',lw=1); ax.set(xlabel='Fraction of signal energy shifted toward weak direction',ylabel='Structured risk / unstructured risk'); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(out/'experiment2_misspecification_risk.pdf'); plt.close(fig)
    vals=[float(g[(g.tau2==1)&(g.alpha==a)].exact_risk_ratio.iloc[0]) for a in [0,.5,1]]+[float(df[df.scenario=='Severe_misspecification'].exact_risk_ratio.iloc[0])]
    fig,ax=plt.subplots(figsize=(6.8,4.8)); bars=ax.bar(range(4),vals,edgecolor='black'); ax.axhline(1,ls='--',lw=1); ax.set_ylim(0,10); ax.set_ylabel('Structured risk / unstructured risk'); ax.set_xticks(range(4),['Aligned','Moderate\nmisalignment','Weak-direction\nsignal','Severe\nmisspecification'])
    for b,z in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,z+.15,f'{z:.2f}',ha='center')
    fig.tight_layout(); fig.savefig(out/'experiment2_risk_scenarios.pdf'); plt.close(fig)
    return df


def experiment3(out,rng,B):
    R,d=30,10; tau2=v=1.; E=pair_kernel(make_base(R,d,'exponential',rng),pair_index(R,False)); w,U=eig_psd(E); pos=numerical_rank(w); ranks=np.unique(np.round(np.linspace(1,max(2,pos-1),15)).astype(int)); sf=shrink(w,tau2,v)
    rows=[]
    for r in ranks:
        exact=sf[r] if r<len(sf) else 0.; errs=[]; bounds=[]
        for _ in range(B):
            beta=sample_psd(w,U,tau2,rng); y=beta+rng.normal(size=len(w))*np.sqrt(v); z=U.T@y; full=U@(sf*z); st=sf.copy(); st[r:]=0; tr=U@(st*z); errs.append(np.linalg.norm(full-tr)); bounds.append(exact*np.linalg.norm(y))
        rows.append(dict(retained_rank=r,exact_shrinkage_operator_error=exact,mean_estimator_error=np.mean(errs),mean_theorem_bound=np.mean(bounds),trace_retained=np.sum(w[:r])/np.sum(w),replicates=B))
    df=pd.DataFrame(rows); df.to_csv(out/'experiment3_truncation.csv',index=False)
    fig,ax=plt.subplots(figsize=(6.8,4.8)); ax.plot(df.retained_rank,df.mean_estimator_error,marker='o',label='Mean actual error'); ax.plot(df.retained_rank,df.mean_theorem_bound,marker='o',ls='--',label='Mean theorem bound'); ax.set(xlabel='Retained rank',ylabel='Estimator discrepancy'); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(out/'experiment3_truncation_error.pdf'); plt.close(fig)
    return df


def perturb_from_features(K,delta,rng):
    w,U=eig_psd(K); keep=w>1e-12; X=U[:,keep]*np.sqrt(w[keep]); Xt=X+delta*rng.normal(size=X.shape); Kt=Xt@Xt.T; dd=np.sqrt(np.maximum(np.diag(Kt),1e-12)); return sym(Kt/np.outer(dd,dd))

def experiment4(out,rng,B):
    R,d=25,8; K=make_base(R,d,'exponential',rng); pairs=pair_index(R,False); E=pair_kernel(K,pairs); rows=[]
    for delta in [.01,.025,.05,.1,.2]:
      for _ in range(B):
        Kt=perturb_from_features(K,delta,rng); Et=pair_kernel(Kt,pairs); ne=opnorm(K-Kt); pe=opnorm(E-Et); bd=.5*(opnorm(K)+opnorm(Kt))*ne
        rows.append(dict(delta=delta,node_operator_error=ne,pair_operator_error=pe,pair_operator_bound=bd,pair_bound_ratio=pe/bd if bd else 0))
    raw=pd.DataFrame(rows); raw.to_csv(out/'experiment4_geometry_perturbation.csv',index=False)
    df=raw.groupby('delta',as_index=False).mean(numeric_only=True); df['replicates']=B; df.to_csv(out/'experiment4_geometry_perturbation_summary.csv',index=False)
    fig,ax=plt.subplots(figsize=(6.8,4.8)); ax.plot(df.delta,df.pair_operator_error,marker='o',label='Observed pair error'); ax.plot(df.delta,df.pair_operator_bound,marker='o',ls='--',label='Theorem bound'); ax.set(xlabel='Feature perturbation level',ylabel='Pair covariance operator error'); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(out/'experiment4_operator_perturbation.pdf'); plt.close(fig)
    return raw


def main():
    a=args_parser(); out=a.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True); rng=np.random.default_rng(a.seed)
    B2,B3,B4=(200,100,50) if a.quick else (5000,1000,500)
    e1=experiment1(out,rng); e2=experiment2(out,rng,B2); e3=experiment3(out,rng,B3); e4=experiment4(out,rng,B4)
    print('seed:',a.seed); print('max product-spectrum error:',e1.product_spectrum_max_error.max()); print('max interlacing violation:',e1.interlacing_max_violation.max()); print('max pair perturbation actual/bound ratio:',e4.pair_bound_ratio.max()); print('results:',out)

if __name__=='__main__': main()


"""
Research pipeline for one-participant eye-tracking CSV.
Run: python run_analysis.py --input Participant_1_GZ.csv --out results
Outputs: cleaned sample data, event tables, window features/labels, statistics, figures.
Important: all labels are exploratory algorithmic proxies, not validated psychological states.
"""
import argparse, os, warnings, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score

def contiguous_runs(mask):
    """Return half-open (start,end) index pairs for True runs."""
    x=np.asarray(mask, dtype=bool)
    if len(x)==0: return []
    edges=np.diff(np.r_[False,x,False].astype(int))
    return list(zip(np.where(edges==1)[0], np.where(edges==-1)[0]))

def robust_z(x):
    x=np.asarray(x,dtype=float)
    med=np.nanmedian(x)
    mad=np.nanmedian(np.abs(x-med))
    return (x-med)/(1.4826*mad) if mad>0 else (x-med)/(np.nanstd(x)+1e-9)

def add_run_lengths(labels, times, min_duration=0.0):
    """Convert sample labels into event records."""
    records=[]
    for a,b in contiguous_runs(pd.Series(labels).notna().to_numpy()):
        lab=labels[a]
        # split each contiguous segment whenever its label changes
        j=a
        while j<b:
            k=j+1
            while k<b and labels[k]==lab: k+=1
            dur=float(times[k-1]-times[j] + np.nanmedian(np.diff(times)))
            if dur>=min_duration:
                records.append((j,k,lab,float(times[j]),float(times[k-1]),dur))
            j=k
            if j<b: lab=labels[j]
    return records

def detect_events_ivt(d):
    # I-VT threshold is explicitly in normalized screen units / second.
    # The 90th percentile is a dataset-specific exploratory threshold;
    # use degrees/second only after screen geometry + calibration are known.
    v=d['gaze_speed_norm_s'].to_numpy()
    valid=d['gaze_valid'].to_numpy()
    threshold=float(np.nanquantile(v[valid],0.90)) if valid.any() else np.nan
    is_fix=valid & (v<=threshold)
    labels=np.where(~valid, None, np.where(is_fix,'fixation','saccade'))
    # remove very short fixations (< 60 ms) as a conservative event cleanup
    for a,b in contiguous_runs(is_fix):
        if d.timestamp.iloc[b-1]-d.timestamp.iloc[a] < 0.06:
            is_fix[a:b]=False
    labels=np.where(~valid,None,np.where(is_fix,'fixation','saccade'))
    records=[]
    for a,b,lab,t0,t1,dur in add_run_lengths(labels,d.timestamp.to_numpy(),0.0):
        coords=d[['gaze2d_x','gaze2d_y']].iloc[a:b].to_numpy()
        amp=float(np.linalg.norm(coords[-1]-coords[0])) if len(coords)>1 and np.isfinite(coords).all() else np.nan
        records.append({'algorithm':'I-VT','event':lab,'start_s':t0,'end_s':t1,'duration_s':dur,
                        'n_samples':b-a,'amplitude_norm':amp,'start_index':a,'end_index':b})
    return pd.DataFrame(records), threshold, is_fix

def detect_events_idt(d, dispersion_threshold=0.02, min_duration=0.10):
    # I-DT: sliding temporal window; dispersion is (x range + y range)
    x=d.gaze2d_x.to_numpy(); y=d.gaze2d_y.to_numpy(); t=d.timestamp.to_numpy()
    valid=d.gaze_valid.to_numpy()
    n=len(d); lab=np.full(n,None,dtype=object)
    i=0
    while i<n:
        if not valid[i]:
            i+=1; continue
        j=i
        # grow window until minimum duration; invalid sample ends a run
        while j<n and valid[j] and (t[j]-t[i] < min_duration):
            j+=1
        if j>n or j==i: j=min(n,i+1)
        if j-i>=2 and (np.nanmax(x[i:j])-np.nanmin(x[i:j]) + np.nanmax(y[i:j])-np.nanmin(y[i:j])) <= dispersion_threshold:
            # extend while dispersion remains below threshold
            k=j
            while k<n and valid[k]:
                xx=x[i:k+1]; yy=y[i:k+1]
                if (np.nanmax(xx)-np.nanmin(xx)+np.nanmax(yy)-np.nanmin(yy))>dispersion_threshold: break
                k+=1
            lab[i:k]='fixation'; i=k
        else:
            lab[i]='saccade'; i+=1
    lab[~valid]=None
    # fill unlabelled valid samples as saccade for mutually exclusive event output
    lab[valid & pd.isna(lab)]='saccade'
    records=[]
    for a,b,kind,t0,t1,dur in add_run_lengths(lab,t,0.0):
        pts=np.c_[x[a:b],y[a:b]]
        amp=float(np.linalg.norm(pts[-1]-pts[0])) if len(pts)>1 else 0.0
        records.append({'algorithm':'I-DT','event':kind,'start_s':t0,'end_s':t1,'duration_s':dur,
                        'n_samples':b-a,'amplitude_norm':amp,'start_index':a,'end_index':b})
    return pd.DataFrame(records)

def main(inp,out):
    os.makedirs(out,exist_ok=True)
    raw=pd.read_csv(inp)
    required=['timestamp','gaze2d_x','gaze2d_y','eyeleft.pupildiameter','eyeright.pupildiameter']
    missing=[c for c in required if c not in raw]
    if missing: raise ValueError('Missing required columns: '+str(missing))
    d=raw.copy().sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
    # Reject impossible normalized screen coordinates; retain original raw CSV untouched.
    d['gaze_valid']=d[['gaze2d_x','gaze2d_y']].notna().all(axis=1) & d.gaze2d_x.between(0,1) & d.gaze2d_y.between(0,1)


    # Pupil average: available eye if only one is present; avoid inventing missing values.
    d['pupil_mean_mm']=d[['eyeleft.pupildiameter','eyeright.pupildiameter']].mean(axis=1) # PUPILS
    d['pupil_mean_mm']=d['pupil_mean_mm'].where(d['pupil_mean_mm'].between(1.5,9.0)) # PUPILS
    # Interpolate only short gaps for derivative/rolling calculations; raw validity remains tracked.
    gx=d.gaze2d_x.where(d.gaze_valid).interpolate(limit=5,limit_area='inside')
    gy=d.gaze2d_y.where(d.gaze_valid).interpolate(limit=5,limit_area='inside')
    dt=d.timestamp.diff().replace(0,np.nan)
    dx=gx.diff(); dy=gy.diff()
    d['gaze_speed_norm_s']=np.sqrt(dx**2+dy**2)/dt
    d['gaze_accel_norm_s2']=d.gaze_speed_norm_s.diff()/dt
    d['gaze_step_norm']=np.sqrt(dx**2+dy**2)

    # SMOOTHS THE PUPIL
    d['pupil_smooth_mm']=d.pupil_mean_mm.rolling(25,center=True,min_periods=5).median()
    baseline=float(d.pupil_smooth_mm.median())
    d['pupil_change_pct']=100*(d.pupil_smooth_mm/baseline-1) if baseline else np.nan
    # I-VT and I-DT event detection
    ivt,ivt_threshold,ivt_fix=detect_events_ivt(d)
    idt=detect_events_idt(d)
    ivt.to_csv(os.path.join(out,'events_ivt.csv'),index=False)
    idt.to_csv(os.path.join(out,'events_idt.csv'),index=False)
    # Sample gaze clustering: DBSCAN on normalized 2D positions; sample every 5th valid point
    ix=np.where(d.gaze_valid.to_numpy())[0][::5]
    db_labels=np.full(len(d),-99,dtype=int)
    if len(ix)>10:
        db=DBSCAN(eps=0.015,min_samples=8).fit(d.loc[ix,['gaze2d_x','gaze2d_y']])
        db_labels[ix]=db.labels_
    d['dbscan_spatial_cluster']=db_labels
    # One-second windows for behaviour/attention clustering and PCA
    start=float(d.timestamp.min()); end=float(d.timestamp.max())
    d['window_id']=np.floor(d.timestamp-start).astype(int)
    rows=[]
    for wid,g in d.groupby('window_id'):
        validg=g[g.gaze_valid]
        speed=validg.gaze_speed_norm_s.replace([np.inf,-np.inf],np.nan).dropna()
        pup=g.pupil_smooth_mm.dropna()
        rows.append({'window_id':int(wid),'time_s':float(g.timestamp.mean()),
          'valid_gaze_fraction':float(g.gaze_valid.mean()),
          'mean_speed':float(speed.mean()) if len(speed) else np.nan,
          'median_speed':float(speed.median()) if len(speed) else np.nan,
          'speed_sd':float(speed.std()) if len(speed)>1 else np.nan,
          'speed_p90':float(speed.quantile(.9)) if len(speed) else np.nan,
          'gaze_dispersion':float(np.sqrt(validg.gaze2d_x.var()+validg.gaze2d_y.var())) if len(validg)>1 else np.nan,
          'path_length':float(validg.gaze_step_norm.sum()) if len(validg) else np.nan,
          'pupil_mean':float(pup.mean()) if len(pup) else np.nan,
          'pupil_sd':float(pup.std()) if len(pup)>1 else np.nan,
          'pupil_change_mean_pct':float(g.pupil_change_pct.mean()) if g.pupil_change_pct.notna().any() else np.nan,
          'ivt_fixation_fraction':float(ivt_fix[g.index].mean()) if len(g) else np.nan})
    w=pd.DataFrame(rows)
    features=['mean_speed','speed_sd','speed_p90','gaze_dispersion','path_length','pupil_mean','pupil_sd','pupil_change_mean_pct','ivt_fixation_fraction']
    imp=SimpleImputer(strategy='median')
    X=imp.fit_transform(w[features])
    scaler=StandardScaler()
    Xs=scaler.fit_transform(X)
    # KMeans: exploratory 3 movement regimes; labels are interpreted by cluster mean speed.
    km=KMeans(n_clusters=3,random_state=42,n_init=20)
    w['kmeans_cluster']=km.fit_predict(Xs)
    means=w.groupby('kmeans_cluster').mean_speed.mean().sort_values()
    names={means.index[0]:'low-movement / concentrated-gaze proxy',
           means.index[1]:'moderate-movement / intermediate proxy',
           means.index[2]:'high-movement / dispersed-gaze proxy'}
    w['gaze_behaviour_kmeans']=w.kmeans_cluster.map(names)
    # DBSCAN on standardized window features, separate from spatial DBSCAN.
    w['dbscan_window_cluster']=DBSCAN(eps=1.5,min_samples=5).fit_predict(Xs)
    # PCA for interpretable 2D projection
    pca=PCA(n_components=2,random_state=42)
    pcs=pca.fit_transform(Xs)
    w['PC1']=pcs[:,0]; w['PC2']=pcs[:,1]

    # Attention proxy: lower movement + lower spatial dispersion + more I-VT fixation time.
    att=( -robust_z(w.mean_speed.fillna(w.mean_speed.median()).to_numpy())
          -robust_z(w.gaze_dispersion.fillna(w.gaze_dispersion.median()).to_numpy())
          +robust_z(w.ivt_fixation_fraction.fillna(w.ivt_fixation_fraction.median()).to_numpy()) )
    w['attention_proxy_score']=att
    w['attention_kmeans']=np.where(w.kmeans_cluster==means.index[0],'more concentrated gaze proxy','more dispersed gaze proxy / intermediate')
    w.to_csv(os.path.join(out,'window_features_and_labels.csv'),index=False)
    d.to_csv(os.path.join(out,'sample_level_features.csv'),index=False)
    # Pupil anomaly detection on window features
    pupcols=['pupil_mean','pupil_sd','pupil_change_mean_pct']
    pX=SimpleImputer(strategy='median').fit_transform(w[pupcols])
    if len(w)>=20:
        iso=IsolationForest(contamination=0.03,random_state=42)
        w['pupil_anomaly']=iso.fit_predict(pX)==-1
    else: w['pupil_anomaly']=False
    w.to_csv(os.path.join(out,'window_features_and_labels.csv'),index=False)
    # Robust rolling change flags (3-MAD from centered 60-sample window)
    p=d.pupil_smooth_mm
    med=p.rolling(251,center=True,min_periods=30).median()
    mad=(p-med).abs().rolling(251,center=True,min_periods=30).median()
    d['pupil_change_flag']=(p-med).abs()>(3*1.4826*mad.clip(lower=0.01))
    # Statistics
    def event_stats(ev,alg):
        if ev.empty:return {'algorithm':alg,'n_fixations':0,'n_saccades':0,'median_fixation_s':np.nan,'mean_fixation_s':np.nan,'saccades_per_min':np.nan}
        fix=ev[ev.event=='fixation']; sac=ev[ev.event=='saccade']
        minutes=(d.timestamp.max()-d.timestamp.min())/60
        return {'algorithm':alg,'n_fixations':len(fix),'n_saccades':len(sac),
          'median_fixation_s':fix.duration_s.median() if len(fix) else np.nan,
          'mean_fixation_s':fix.duration_s.mean() if len(fix) else np.nan,
          'saccades_per_min':len(sac)/minutes if minutes>0 else np.nan,
          'median_saccade_amplitude_norm':sac.amplitude_norm.median() if len(sac) else np.nan}
    stats=[event_stats(ivt,'I-VT'),event_stats(idt,'I-DT')]
    report={'n_rows_raw':int(len(raw)),'n_rows_after_timestamp_dedup':int(len(d)),
      'duration_s':float(d.timestamp.max()-d.timestamp.min()),
      'median_sampling_hz':float(1/np.nanmedian(np.diff(d.timestamp))),
      'gaze_valid_n':int(d.gaze_valid.sum()),'gaze_valid_fraction':float(d.gaze_valid.mean()),
      'pupil_valid_fraction':float(d.pupil_mean_mm.notna().mean()),
      'ivt_velocity_threshold_norm_units_s':ivt_threshold,
      'pupil_baseline_mm':baseline,
      'pupil_anomaly_windows':int(w.pupil_anomaly.sum()),
      'pca_explained_variance_ratio':pca.explained_variance_ratio_.tolist(),
      'event_statistics':stats,
      'notes':['Gaze coordinates outside [0,1] are treated as invalid screen positions.',
      'Velocity is normalized screen units/s, not degrees/s.',
      'Attention labels are gaze-based proxies, not a validated measure of attention or cognition.',
      'Single participant: no population-level inference or between-person generalization.']}
    with open(os.path.join(out,'analysis_summary.json'),'w') as f: json.dump(report,f,indent=2)
    # Figures
    plt.figure(figsize=(13,4))
    plt.plot(d.timestamp,d.gaze2d_x,label='gaze X',alpha=.7)
    plt.plot(d.timestamp,d.gaze2d_y,label='gaze Y',alpha=.7)
    plt.xlabel('Time (s)');plt.ylabel('Normalized screen coordinate');plt.title('Gaze coordinates over time');plt.legend();plt.tight_layout()
    plt.savefig(os.path.join(out,'01_gaze_timeseries.png'),dpi=160);plt.close()
    plt.figure(figsize=(7,6))
    sc=plt.scatter(w.PC1,w.PC2,c=w.kmeans_cluster,s=22,alpha=.75)
    plt.xlabel('PC1');plt.ylabel('PC2');plt.title('PCA of 1-second gaze/pupil feature windows');plt.tight_layout()
    plt.savefig(os.path.join(out,'02_pca_kmeans.png'),dpi=160);plt.close()
    plt.figure(figsize=(13,4))
    plt.plot(d.timestamp,d.pupil_smooth_mm,label='smoothed mean pupil')
    plt.axhline(baseline,linestyle='--',label='recording median baseline')
    plt.xlabel('Time (s)');plt.ylabel('Pupil diameter (source units)');plt.title('Pupil diameter over time');plt.legend();plt.tight_layout()
    plt.savefig(os.path.join(out,'03_pupil_timeseries.png'),dpi=160);plt.close()
    plt.figure(figsize=(13,4))
    plt.plot(w.time_s,w.mean_speed,label='mean gaze speed')
    plt.plot(w.time_s,w.gaze_dispersion,label='gaze dispersion')
    plt.xlabel('Time (s)');plt.title('Windowed gaze movement and dispersion');plt.legend();plt.tight_layout()
    plt.savefig(os.path.join(out,'04_attention_features.png'),dpi=160);plt.close()
    plt.figure(figsize=(7,5))
    plt.scatter(d.loc[d.gaze_valid,'gaze2d_x'],d.loc[d.gaze_valid,'gaze2d_y'],s=2,alpha=.15)
    plt.gca().invert_yaxis();plt.xlabel('Normalized gaze X');plt.ylabel('Normalized gaze Y');plt.title('Gaze-position cloud');plt.tight_layout()
    plt.savefig(os.path.join(out,'05_gaze_position_cloud.png'),dpi=160);plt.close()
    pd.DataFrame(stats).to_csv(os.path.join(out,'event_summary_statistics.csv'),index=False)
    print(json.dumps(report,indent=2))
    print('Saved analysis to',out)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',required=True);ap.add_argument('--out',default='results')
    a=ap.parse_args();main(a.input,a.out)

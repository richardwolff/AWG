import numpy as np
import pandas as pd
import config
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.optimize import curve_fit
from scipy.stats import expon

SMALL_SIZE = 18
MEDIUM_SIZE = 25
BIGGER_SIZE = 40

#plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('xtick.major', size=.4*SMALL_SIZE)

plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

from matplotlib import rcParams
rcParams['mathtext.fontset'] = 'stixsans'

cmap_heatmap = ListedColormap(['silver', 'orangered', 'black', 'black','blue'], 'indexed')

## Default populations scanned by __main__, overridable with --populations
DEFAULT_POPULATIONS = ["AG","BF","GH","SW","KY","DM","all"]


def func(x,l_r,Nr,C):
    
    R = l_r*(1-np.exp(-x/l_r))

    num = C*(10 + 2*Nr*R)
    den = 22 + 26*Nr*R + 4*(Nr*R)**2

    return(num/den)


def func_jac(x, l_r, Nr, C):
    """
    Analytic Jacobian of `func` with respect to (l_r, Nr, C).

    Handing this to `curve_fit` removes the three extra evaluations of `func`
    per iteration that the default finite-difference approximation needs, which
    matters because the fit is repeated from several hundred starting points.

    With u = Nr * R and den = 22 + 26u + 4u^2:
        df/du   = C * (-216 - 80u - 8u^2) / den^2
        dR/dl_r = 1 - e - (x/l_r) e,     e = exp(-x/l_r)
    """
    x = np.asarray(x, dtype=float)

    t = x / l_r
    e = np.exp(-t)

    R = l_r * (1.0 - e)
    u = Nr * R

    den = 22.0 + 26.0 * u + 4.0 * u * u
    df_du = C * (-216.0 - 80.0 * u - 8.0 * u * u) / (den * den)

    jac = np.empty((x.shape[0], 3))
    jac[:, 0] = df_du * Nr * (1.0 - e - t * e)   # d/d l_r
    jac[:, 1] = df_du * R                        # d/d Nr
    jac[:, 2] = (10.0 + 2.0 * u) / den           # d/d C

    return jac


def take_triu(df):
    
    N = df.shape[0]
    p=np.triu_indices(N,k=1)
    
    return(df[p])


def get_gene_range(df,gene1,gene2,off_left=0,off_right=0):
    
    genes = df.index.get_level_values("gene_id")
    
    
    
    g1_loc = np.argwhere(genes == gene1).ravel()[0]
    g2_loc = np.argwhere(genes == gene2).ravel()[-1]
    
    return(df.iloc[g1_loc-off_left:g2_loc+off_right])


def return_fit_r2(xdata,ydata,popt4,func,ss_tot=None):
    
    residuals = ydata - func(xdata, *popt4)

    ss_res = np.sum(residuals**2)

    if ss_tot is None:
        ss_tot = np.sum((ydata-np.mean(ydata))**2)

    r_squared = 1 - (ss_res / ss_tot)

    return(r_squared)


def _bin_decay_curve(values, columns, d_bins):
    """
    Mean of every column, grouped by binned genomic distance.

    Equivalent to assigning `d_bins[np.digitize(D, d_bins) - 1]` as a column and
    calling `groupby("D_bins").mean()`, but the totals are accumulated per bin
    edge with `bincount` -- so nothing the size of the pairwise table is sorted
    or copied, and one pass replaces one pass per aggregation.

    The `- 1` wrap-around is kept as-is: distances below the first edge (and,
    as before, above the last) land in the final bin.
    """
    n_edges = d_bins.shape[0]

    codes = np.digitize(values["D"], d_bins) - 1
    codes[codes < 0] = n_edges - 1

    count = np.bincount(codes, minlength=n_edges)
    totals = {c: np.bincount(codes, weights=v, minlength=n_edges)
              for c, v in values.items()}

    # percentiles of an integer-valued distance repeat, and groupby keys off the
    # label *value*, so bins sharing an edge value collapse into one group
    labels, merge = np.unique(d_bins, return_inverse=True)
    n = labels.shape[0]

    count = np.bincount(merge, weights=count, minlength=n)
    occupied = count > 0

    means = {c: np.bincount(merge, weights=t, minlength=n)[occupied] / count[occupied]
             for c, t in totals.items()}

    return pd.DataFrame(means, index=pd.Index(labels[occupied], name="D_bins"))[columns]


def return_ld_fits(df4,min_dist = config.fit_min_dist,bin_scale = 100,max_dist=config.max_dist):
    
    sys.stderr.write("Estimating l_r and l_dd from decay of synonymous LD\n")
    
    # one combined mask over the raw arrays, rather than three chained `.loc`
    # calls each copying the whole pairwise table
    columns = list(df4.columns)
    values = {c: np.asarray(df4[c].values, dtype=float) for c in columns}

    keep = ((values["D"] < max_dist) & (values["D"] >= min_dist)
            & np.isfinite(values["r2"]))
    values = {c: v[keep] for c, v in values.items()}

    pp = np.array(list(np.linspace(1e-1,1e0,25))[:-1] + list(np.linspace(1e0,1e1,250))[:-1] + list(np.linspace(1e1,50,500)))
        
    d_bins = np.percentile(values["D"],pp)
    
    df4_mean = _bin_decay_curve(values, columns, d_bins)
    df4_mean = df4_mean.set_index("D")
    df4_mean = df4_mean.iloc[:-1]
    
    df4_mean_roll = df4_mean.rolling(25).mean().dropna()
    thresh = df4_mean_roll.iloc[-1] + (df4_mean_roll.iloc[0] - df4_mean_roll.iloc[-1])/10
    thresh = thresh.iloc[0]
    cumsum = np.cumsum(abs(df4_mean_roll) < thresh)
    cummean = cumsum.values.ravel()/np.arange(1,df4_mean_roll.shape[0] + 1)
    cut_value = 2*df4_mean_roll.index[np.argwhere(cummean==0)[-1][0]]
    
    ## apply smoothing w/ rolling mean
    #df4_mean = df4_mean.rolling(16).mean().shift(-8).dropna()
    
    eps = 1e30

    long_range4 = df4_mean["r2"].rolling(int(df4_mean.shape[0]/3)).mean().values[-1]


    ## loop over initializations to find best fit
    sys.stderr.write("\tLooping over initial conditions to find best fit")
    l_r_starts = [1e2,5*1e2] + list(np.logspace(3,4,40))
    CG_starts = [1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7]

    ydata = df4_mean["r2"].values
    xdata = np.asarray(df4_mean.index.values, dtype=float)

    # constant across every initialization
    ss_tot = np.sum((ydata - np.mean(ydata))**2)

    r2_dic = {}
    popt4_dic = {}
    for l_r_s in l_r_starts:
        for CG in CG_starts:

            try:
                popt4, pcov4 = curve_fit(func, xdata, ydata, p0=[l_r_s,CG/(2*long_range4),1/CG],
                                         bounds=((1,0,0),(1e5,eps,eps)),maxfev=10000,
                                         jac=func_jac)
            except RuntimeError:
                # this start did not converge; the remaining ones still can
                continue

            r2_dic[(l_r_s,CG)] = return_fit_r2(xdata,ydata,popt4,func,ss_tot)
            popt4_dic[(l_r_s,CG)] = popt4

    if not r2_dic:
        raise RuntimeError("No initialization of the LD decay fit converged.")

    r2_dic = pd.Series(r2_dic).sort_values() 
    popt4_dic = pd.DataFrame(popt4_dic,index=["l_r","Nr","C"]).T
    popt4 = popt4_dic.loc[r2_dic.index[-1]].values
    popt4_dic = popt4_dic.loc[r2_dic.index]
    popt4_dic["r2_fit"] = r2_dic
    
    
    l_r,Nrlr,C = popt4

    #decay_point (l_dd) set as 99th percentile of tract length distribution
    decay_point = expon(scale=l_r).ppf(.99)
    #decay_point = cut_value

    ## write parameters to file
    params = pd.Series([l_r,Nrlr,C,decay_point],index=["tract_length","rescaled_recombination","constant","decay_points"])
    sys.stderr.write(f"\nComplete\n\tl_DD = {decay_point}")
    
    return(df4_mean,popt4,params)


def plot_fit(df4_mean,popt4,params,species,pop,save=True,out_dir="tract_length_parameters"):
    
    
    l_r = params.loc["tract_length"]
    decay_point = params.loc["decay_points"]
    
    ### plot decay figure
    fig,ax = plt.subplots(figsize=(12,8))
    
    ax.spines[["top","right"]].set_visible(False)

    # sampled logarithmically: the axis below is a log axis, so this puts the
    # points where they are actually resolvable instead of spending a million
    # of them on the right-hand decade
    xx = np.logspace(np.log10(df4_mean.index.values[0]),
                     np.log10(df4_mean.index.values[-1]), 3000)

    ax.set_title(species,size=MEDIUM_SIZE,fontstyle="italic")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.plot(df4_mean.index,df4_mean["r2"],lw=10.5,color=cmap[3],zorder=1,label="Observed")
    ax.plot(df4_mean.index,df4_mean["r2"],lw=11,color="k",zorder=0)

    ax.plot(xx,func(xx,*popt4),ls="--",color=cmap[8],lw=4,label="Fit")

    ax.semilogx()

    alpha_val = .25

    ax.set_xlabel("Genomic distance",size=MEDIUM_SIZE)
    ax.set_ylabel(r"Linkage disequilibrium ($r^2$)",size=MEDIUM_SIZE)

    
    ax.axvline(decay_point,color="tomato",label=fr"$l_{{DD}}$: {int(decay_point)}bp")
    #ax.axvline(l_r,color="g",label=fr"$l_r$: {int(l_r)}bp")
    
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.865),prop={"size":SMALL_SIZE})
    fig.tight_layout()

    if save:
        fig.savefig(f"{out_dir}/{species}/{pop}/{species}_decay", bbox_inches="tight")    

    return fig,ax
    

    
    
cmap = ["#CC6677","#332288","#117733","#88CCEE","#882255","#DDCC77","#AA4499","#999933",
       "#EE7733","#225522","#EECC66","#004488"]*100

if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--species',
                        help="species for LD calculation",
                        default=None,
                        type=str)

    parser.add_argument('--ld_dir',
                        help="folder holding {species}/{pop}/{species}_df_r2.pq. Defaults to config.LD_dir",
                        default=getattr(config, "LD_dir", None),
                        type=str)

    parser.add_argument('--populations',
                        help="comma-separated populations to fit",
                        default=",".join(DEFAULT_POPULATIONS),
                        type=str)

    parser.add_argument('--out_dir',
                        help="folder to write fitted parameters and figures to",
                        default="tract_length_parameters",
                        type=str)
    
    args = parser.parse_args()
    
    species = args.species
    populations = [p for p in args.populations.split(",") if p]
    tract_directory = args.out_dir

    if args.ld_dir is None:
        parser.error("no LD directory: pass --ld_dir or set LD_dir in config.py")

    sites_color = getattr(config, "sites_color", {})
    
    df4_all = {}
    params_all = {}
    for pop in populations:

        print(pop)

        LD_dir = f'{args.ld_dir}/{species}/{pop}'

        if not os.path.exists(f'{LD_dir}/{species}_df_r2.pq'):
            continue

        df_r2 = pd.read_parquet(f'{LD_dir}/{species}_df_r2.pq')
        try:
            df4_mean,popt4,params = return_ld_fits(df_r2)
        except Exception as e:
            
            print("fit failed:" + pop + f" ({e})\n\n")
            
            continue
            
        df4_all[pop] = df4_mean
        params_all[pop] = params

        os.makedirs(f"{tract_directory}/{species}/{pop}",exist_ok=True)

        params.to_csv(f"{tract_directory}/{species}/{pop}/{species}_params.txt",sep="\t")

        df4_mean.to_csv(f"{tract_directory}/{species}/{pop}/{species}_df4_mean.txt")
        
        plot_fit(df4_mean,popt4,params,species,pop,out_dir=tract_directory)
    
    fig,ax = plt.subplots(figsize=(12,8))

    for i,(pop,df) in enumerate(df4_all.items()):
        
        dfr = df.rolling(3).mean().shift(-2).dropna()
        
        if pop != "all":

            ax.plot(dfr.index,dfr.r2,color=sites_color.get(pop,cmap[i]),lw=2.5,alpha=.7,label=pop)

        else:

            ax.plot(dfr.index,dfr.r2 ,color="k",lw=2.5,label=pop)

    ax.semilogx()

    ax.spines[["top","right"]].set_visible(False)

    ax.set_ylabel(r"Linkage disequilibrium $r^2$",size=20)
    ax.set_xlabel("Genomic distance (bp)",size=20)
    fig.legend()    
    
    os.makedirs(f"{tract_directory}/{species}",exist_ok=True)
    fig.savefig(f"{tract_directory}/{species}/decay_fig",bbox_inches="tight")

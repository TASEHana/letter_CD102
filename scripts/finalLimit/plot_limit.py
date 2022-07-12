import pandas as pd
import os
import matplotlib
from numpy             import *
from ou_Axion_limit    import analyse, Glimit
from matplotlib.pyplot import *
from glob              import glob
from scipy.signal      import savgol_filter
from scipy             import interpolate
from scipy.optimize    import curve_fit
import mplhep as hep
hep.style.use(hep.style.ROOT)

INPUT_EXPERIMENT_FLODER = "others_experiment"
INPUT_limit_PATH        = "gayy_unc_updateFF_06July.txt"  #"gayy_unc_updateNoise_C.txt"
OUTPUT_FILE_NAME        = "limit.png"

SHOW_OTHER_EXPERIMENT   = 1 #  1 : show others experiment data  0 : don't show
PLOT_SQUARE             = 1 #  1 : plot TASEH limit with suare  0 : don't do that
LOG_PLOT                = 1 #  1 : plot with log and add {admx sidecar}, {UF} and {RBF}. 0 :plot with linear
CAST                    = 1 #  1 : plot CAST

color_map ={
    "ADMX"   :"#7FF547",
    "HAYSTAC":"#FFDC4A",
    "CAPP"   :"#7157E8",
    "UF"     :"#9B6669",
    "RBF"    :"#534D6D",
    "CAST"   :"#9E7068"
}

########################################################################## functions
def ma_to_freq(ma):   # eat ev output GHz
    g = Glimit()
    return ma/( g.h_bar  * 2 * pi) * 1e-9

def freq_to_ma(freq): # eat GHz output ev
    g = Glimit()
    return freq *( g.h_bar  * 2 * pi) * 1e9

########################################################################## Read data
print(f"[*] Reading {INPUT_limit_PATH}\n")
df = pd.read_table(INPUT_limit_PATH,sep="\s+")
df.columns = ["Freq","limit_cen","noise_un","mis_un","QL_un","toal_un","Nan"]
print(df.head())


hein_limit_freq = df["Freq"     ].values * 1e9
center_limits   = df["limit_cen"].values
to_un           = df["toal_un"  ].values

upper__limits   = center_limits + sqrt(  to_un**2  )
lower__limits   = center_limits - sqrt(  to_un**2  )

######################################################################### remove two non-axion range

for i in range(len(hein_limit_freq)):
    if (4.747300e9<=hein_limit_freq[i] and  hein_limit_freq[i]<=4.747380e9):
        center_limits[i]   = None # np.nan
        upper__limits[i]   = None # np.nan
        lower__limits[i]   = None # np.nan
    if (4.710170e9<=hein_limit_freq[i] and  hein_limit_freq[i]<=4.710190e9):
        center_limits[i]   = None # np.nan
        upper__limits[i]   = None # np.nan
        lower__limits[i]   = None # np.nan


##########################################################################  plot
print(f"[*] Creating figures")

matplotlib.rcParams.update({'font.size': 14})
g = Glimit()
g.big_A= 77.6e6
upper_bound = 1e30
# create plot
fig        = figure(figsize=(8,6))
ax         = fig.add_subplot(111)
ax_x       = twinx()
inset_axes = ax.inset_axes([0.18, 0.56, 0.78, 0.39])

# Frequency and mu ev

fa_x           = linspace(1,5,1000)
ma_x           = freq_to_ma(fa_x)

# Two models

KSVZ_g_a_gamma = 0.97 * ma_x * g.alpha /(pi * g.big_A * g.big_A)  * 1e9
DFSZ_g_a_gamma = 0.36 * ma_x * g.alpha /(pi * g.big_A * g.big_A)  * 1e9

ax.fill_between(hein_limit_freq * 1e-9,upper__limits*1e14,lower__limits*1e14,color="blue")

# plot remove range
for i in range(len(center_limits)):
    if (center_limits[i] != center_limits[i]):
        if (flag):
            ax.vlines(x=hein_limit_freq[i]*1e-9,ymax=20,ymin=center_limits[i-1]*1e14,color="r")
            # print("[*]",center_limits[i-1])
            flag = not flag 
    elif(center_limits[i] == center_limits[i] and  not flag):
        ax.vlines(x=hein_limit_freq[i]*1e-9,ymax=20,ymin=center_limits[i+1]*1e14,color="r")
        # print(center_limits[i+1])
        flag = not flag


ax.plot(hein_limit_freq * 1e-9,center_limits*1e14,"r")
ax.vlines(x=hein_limit_freq[0]*1e-9,ymax=200,ymin=center_limits[0]*1e14,color="r")
ax.vlines(x=hein_limit_freq[-1]*1e-9,ymax=200,ymin=center_limits[-1]*1e14,color="r")
ax.plot(fa_x , KSVZ_g_a_gamma,"b--",alpha=0.8,label="KSVZ")
ax.plot(fa_x , DFSZ_g_a_gamma,"r--",alpha=0.8,label="DFSZ")
ax.fill_between(fa_x,KSVZ_g_a_gamma*4,DFSZ_g_a_gamma/4,color="yellow",label="model region")

# plot setting

ax.set_ylim(4.73e-14*1e14,1.34e-13*1e14)
ax.set_xlim(4.70,4.81)
ax.tick_params(which='both', width=2,top=None)
ax.set_yscale("linear")  # log
ax.set_ylabel(r"$|g_{a \gamma \gamma}|\ \ [10^{-14}\ GeV^{-1}]$",size=15)
# ax.grid(True, which='minor')
ax.set_xlabel("Frequency [GHz]",size=14)

ax_x.set_ylabel(ax.get_ylabel(),size=15)
ax_x.plot(hein_limit_freq * 1e-9,center_limits,"r",alpha=0)
ax_x.fill_between(fa_x,KSVZ_g_a_gamma*4,DFSZ_g_a_gamma/4,color="whitesmoke",label="model region",alpha=0)
ax_x.set_ylim(ax.get_ylim())
ax_x.set_xlim(ax.get_xlim())
# ax_x.grid(True, which='minor')
ax_x.tick_params(which='both', width=2)
ax_x.set_yscale(ax.get_yscale())
ax_y = twiny()
ax_y.plot(ma_x * 1e6, DFSZ_g_a_gamma,"r--",alpha=0)
ax_y.set_xlim(freq_to_ma(ax.get_xlim()[0])* 1e6,freq_to_ma(ax.get_xlim()[1])* 1e6)
ax_y.tick_params(which='both', width=2)
ax_y.set_xlabel(r"$m_{a}\ [{\mu}eV]$",size=15,labelpad=0)

# to G_gamma
center_limits_g_gamma = center_limits*  (pi * g.big_A * g.big_A) / (g.h_bar* 2*pi*hein_limit_freq * g.alpha  * 1e9) / 0.97
upper__limits_g_gamma = upper__limits*  (pi * g.big_A * g.big_A) / (g.h_bar* 2*pi*hein_limit_freq * g.alpha  * 1e9) / 0.97
lower__limits_g_gamma = lower__limits*  (pi * g.big_A * g.big_A) / (g.h_bar* 2*pi*hein_limit_freq * g.alpha  * 1e9) / 0.97

##########################################################################  inset plot

inset_axes.set_xlabel("Frequency [GHz]",size=10,loc="center",labelpad=1)
inset_axes.tick_params(which='both', width=1,labelsize=10)
inset_axes.set_ylabel(r"$|{g_{a\gamma\gamma}}\ /\ {g_{a\gamma\gamma}^{KSVZ}}|$",size=12,loc="bottom",labelpad=-3)
# inset_axes.grid(True, which='minor')
inset_axes.set_xlim(0.01,7)
inset_axes.set_ylim(-2,15)
inset_axes.hlines(abs(g.ksvz_g_gamma)/0.97,xmin=0,xmax=10,color="b",linestyles="--",label="KSVZ")
inset_axes.hlines(abs(g.dfsz_g_gamma)/0.97,xmin=0,xmax=10,color="r",linestyles="--",label="DFSZ")
inset_axes.xaxis.set_major_locator(MultipleLocator(1))
inset_axes.xaxis.set_minor_locator(MultipleLocator(0.2))
upper = 4
down  = abs(g.dfsz_g_gamma)/abs(g.ksvz_g_gamma) / 4
inset_axes.fill_between(linspace(0,10,1000),upper,down,color="#C0C0C0",label="model region",alpha=.8)
inset_axes.set_xlim(4.7,4.8)


mean_center = mean(center_limits_g_gamma[logical_not(isnan(center_limits_g_gamma))])

##########################################################################  plot others experiment

if (SHOW_OTHER_EXPERIMENT):
    print("[*] others experiement \t\t\t\t G_gamma / G_ksvz\n")
    ADMX_num = 40
    WANT = ["HAYSTAC","CAPP","ADMX"]

    if (LOG_PLOT):
        WANT.append("RBF")
        WANT.append("UF")
    if (CAST):
        WANT.append("CAST")

    for each_limit_file in glob(f"{INPUT_EXPERIMENT_FLODER}/*.csv"):
        flag = 0
        
        for want_ in WANT:
            if (want_ in each_limit_file):
                flag = 1
                this_color = color_map[want_]
                break
        
        if (not flag): continue
        if ("Projected" in each_limit_file):continue
        
        # read data
        df = pd.read_csv(each_limit_file,index_col=0)
        limit  = array(df["G_ap[GeV^-1]"].values,dtype=float)
        this_x = ma_to_freq(df["m_a [eV]"].values)
        this_g_gamma = limit*  (pi * g.big_A * g.big_A) / (g.h_bar* 2*pi*this_x * g.alpha*1e18)/ 0.97
        
        if ("RBF" in each_limit_file):
            sort_index = argsort(this_x)
            this_x = this_x[sort_index]

        if ("CAST" in each_limit_file):
            new_x       = linspace(this_x[0],this_x[3],1000)
            this_interp = interp(new_x , this_x[:5], limit[:5])
            this_g_gamma = this_interp*  (pi * g.big_A * g.big_A) / (g.h_bar*2*pi*new_x * g.alpha*1e18)/0.97
            this_x       = new_x

        print(f"\t{os.path.basename(each_limit_file):30} : {mean(this_g_gamma)}")

        inset_axes.fill_between(this_x, this_g_gamma,upper_bound,alpha=.5,color=this_color,label=os.path.basename(each_limit_file)[0:-4])

    if (PLOT_SQUARE):
        inset_axes.fill_between(hein_limit_freq * 1e-9,mean_center+center_limits_g_gamma*0,upper_bound,color="r")
    else:
        inset_axes.fill_between(hein_limit_freq * 1e-9,center_limits_g_gamma,upper_bound,color="r")


    if (LOG_PLOT):
        if (CAST):
            inset_axes.set_yscale("log")
            inset_axes.set_ylim(0.05,1e6)
            inset_axes.set_xlim(0.01,7)
            inset_axes.text(1.7 ,2.8,"CAPP",size=9)
            inset_axes.text(2.35,2.8,"CAPP",size=9)
            inset_axes.text(3.1 ,2.8,"CAPP",size=9)
            inset_axes.text(4.57,2.8,"CAPP",size=9)
            inset_axes.text(3.7 ,20,"HAYSTAC",size=9)
            inset_axes.text(5.4 ,20,"HAYSTAC",size=9)
            inset_axes.text(0.75,1000,"ADMX",size=9,ha="center")
            inset_axes.text(5.6 ,1000,"ADMX sidecar",size=9,ha="center")
            inset_axes.text(4.08 ,400,"TASEH\nCD102",size=9,color="r")
            
            inset_axes.text(1.4,800,"UF",size=9,ha="center")
            inset_axes.text(2.2,800,"RBF",size=9,ha="center")
            inset_axes.text(2,0.5e5,"CAST",size=12,ha="center")
            inset_axes.text(3.8,1.4,"KSVZ",size=7,c="b",ha="center")
            inset_axes.text(3.8,0.12,"DFSZ",size=7,c="r",ha="center")
        else:
            inset_axes.set_yscale("log")
            inset_axes.set_ylim(0.05,500)
            inset_axes.set_xlim(0.01,7)
            inset_axes.text(1.4,2,"CAPP",size=9)
            inset_axes.text(2.35,2.1,"CAPP",size=9)
            inset_axes.text(3.1,5,"CAPP",size=9)
            inset_axes.text(3.7,2.2,"HAYSTAC",size=9)
            inset_axes.text(5.4,2.2,"HAYSTAC",size=9)
            inset_axes.text(0.75,8,"ADMX",size=9,ha="center")
            inset_axes.text(5.4,100,"ADMX sidecar",size=9,ha="center")
            if (PLOT_SQUARE):
                    inset_axes.text(4.3,6,"TASEH CD102",size=9,color="r")
            else:
                inset_axes.text(4.3,4,"TASEH CD102",size=9,color="r")
            inset_axes.text(4.57,1.5,"CAPP",size=9)
            inset_axes.text(1.4,80,"UF",size=9,ha="center")
            inset_axes.text(2.2,80,"RBF",size=9,ha="center")
            inset_axes.text(4.6,0.47,"KSVZ",size=9,c="b",ha="center")
            inset_axes.text(4.6,0.17,"DFSZ",size=9,c="r",ha="center")

    elif (CAST):
        inset_axes.set_yscale("log")
        inset_axes.set_ylim(0.05,1e6)
        inset_axes.set_xlim(0.01,7)
        inset_axes.text(1.7 ,2.8,"CAPP",size=9)
        inset_axes.text(2.35,2.8,"CAPP",size=9)
        inset_axes.text(3.1 ,2.8,"CAPP",size=9)
        inset_axes.text(4.57,2.8,"CAPP",size=9)
        inset_axes.text(3.7 ,20,"HAYSTAC",size=9)
        inset_axes.text(5.4 ,20,"HAYSTAC",size=9)
        inset_axes.text(0.75,1000,"ADMX",size=9,ha="center")
        inset_axes.text(5.6 ,1000,"ADMX sidecar",size=9,ha="center")
        inset_axes.text(4.08 ,400,"TASEH\nCD102",size=9,color="r")
        inset_axes.text(2,0.5e5,"CAST",size=12,ha="center")
        inset_axes.text(3.8,1.4,"KSVZ",size=7,c="b",ha="center")
        inset_axes.text(3.8,0.12,"DFSZ",size=7,c="r",ha="center")
    else:
        inset_axes.text(1.4,2.4,"CAPP",size=9)
        inset_axes.text(2.35,2.5,"CAPP",size=9)
        inset_axes.text(3.1,9,"CAPP",size=9)
        inset_axes.text(0.75,8,"ADMX",size=9,ha="center")
        inset_axes.set_xlim(0.01,7)
        inset_axes.set_ylim(-2.5,20)
        w1 = inset_axes.text(4.775,1.4,"KSVZ",size=7,c="b",ha="center")
        w2 = inset_axes.text(4.775,-1.1,"DFSZ",size=7,c="r",ha="center")

        if (PLOT_SQUARE):
            inset_axes.text(3.7,5,"HAYSTAC",size=9)
            inset_axes.text(5.4,5,"HAYSTAC",size=9)
            inset_axes.text(4.3,8,"TASEH CD102",size=9,color="r")
        else:
            inset_axes.text(3.7,6.5,"HAYSTAC",size=9)
            inset_axes.text(5.4,6.5,"HAYSTAC",size=9)
            inset_axes.text(4.3,4.2,"TASEH CD102",size=9,color="r")
else:
    if (PLOT_SQUARE):
        inset_axes.fill_between(hein_limit_freq * 1e-9,mean_center+center_limits_g_gamma*0,upper_bound,color="r")
        inset_axes.text(4.75,7.5,"TASEH CD102",size=12,color="black",ha='center')
    else:

        # plot inset removed region

        for i in range(len(center_limits_g_gamma)):
            if (center_limits_g_gamma[i] != center_limits_g_gamma[i]):
                if (flag):
                    inset_axes.vlines(x=hein_limit_freq[i]*1e-9,ymax=200,ymin=center_limits_g_gamma[i-1],color="r")
                    flag = not flag 
            elif(center_limits[i] == center_limits[i] and  not flag):
                inset_axes.vlines(x=hein_limit_freq[i]*1e-9,ymax=200,ymin=center_limits_g_gamma[i+1],color="r")
                flag = not flag
        inset_axes.fill_between(hein_limit_freq * 1e-9,upper__limits_g_gamma*1e13,lower__limits_g_gamma*1e13,color="blue")
        ploy_line = inset_axes.plot(hein_limit_freq * 1e-9,center_limits_g_gamma,"r",linewidth=1)
        inset_axes.vlines(x=hein_limit_freq[0 ]*1e-9,ymax=upper_bound,ymin=center_limits_g_gamma[0],color="r")
        inset_axes.vlines(x=hein_limit_freq[-1]*1e-9,ymax=upper_bound,ymin=center_limits_g_gamma[-1],color="r")


savefig(OUTPUT_FILE_NAME,dpi=300)

print(f"[*] Saved in {OUTPUT_FILE_NAME}")

show()




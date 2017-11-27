import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
import itertools

pairs = []
for p in itertools.combinations(range(10),2):
    pairs.append(p)

data = pickle.load( open('learningProgress.p'))

errorRate = []
pairErrors = []

for i in range(len(data)):
    errorRate.append(data[i][0]*100./float(data[i][0] + data[i][1]))
    data[i][2][np.isnan(data[i][2])] = 0.
    data[i][3][np.isnan(data[i][3])] = 1.
    data[i][2][np.isinf(data[i][2])] = 0.
    data[i][3][np.isinf(data[i][3])] = 1.
    data[i][3][data[i][3]==0] = 1.
    #print data[i][3], data[i][2]
    pairErrors.append(data[i][2]*100./(data[i][2] + data[i][3]))

errorRate = np.asarray(errorRate)
pairErrors = np.asarray(pairErrors)


fig_width = 6 # width in inches
fig_height = 8  # height in inches
fig_size =  [fig_width,fig_height]
params = {'axes.labelsize': 14,
          'axes.titlesize': 13,
          'font.size': 11,
          'xtick.labelsize': 11,
          'ytick.labelsize': 11,
          'figure.figsize': fig_size,
          'savefig.dpi' : 600,
          'axes.linewidth' : 1.3,
          'ytick.major.size' : 4,      # major tick size in points
          'xtick.major.size' : 4      # major tick size in points
          #'edgecolor' : None
          #'xtick.major.size' : 2,
          #'ytick.major.size' : 2,
          }
rcParams.update(params)

# set sans-serif font to Arial
rcParams['font.sans-serif'] = 'Arial'

# create figure instance
fig = plt.figure()


# define sub-panel grid and possibly width and height ratios
gs = gridspec.GridSpec(2, 1,
                       #width_ratios=[1,1.2],
                       #height_ratios=[1,1]
                       )

# define vertical and horizontal spacing between panels
gs.update(wspace=0.3,hspace=0.4)

# possibly change outer margins of the figure
plt.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.1)

# sub-panel enumerations
#plt.figtext(0.06, 0.92, 'A',clip_on=False,color='black', weight='bold',size=22)
#plt.figtext(0.47, 0.92, 'B',clip_on=False,color='black', weight='bold',size=22)
#plt.figtext(0.06, 0.47, 'C',clip_on=False,color='black', weight='bold',size=22)
#plt.figtext(0.47, 0.47, 'D',clip_on=False,color='black', weight='bold',size=22)


# first sub-plot #######################################################
ax0 = plt.subplot(gs[0])

# title
ax0.set_title('Number recognition for range [0,...,9]')

# diplay of data
#ax0.axhline(y=0,ls='--',color='0.5',lw=2)
#ax0.axvline(x=0,ls='--',color='0.5',lw=2)
ax0.plot(errorRate)
#ax0.plot(x,cosy,label='cos')

# removes upper and right axes
# and moves left and bottom axes away
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.spines['bottom'].set_position(('outward', 10))
ax0.spines['left'].set_position(('outward', 10))
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')

# legends and labels
plt.legend(loc=1,frameon=False)

plt.xlabel('traning epochs')
plt.ylabel('error rate (%)')

# first sub-plot #######################################################
ax0 = plt.subplot(gs[1])

# title
ax0.set_title('Pair-wise classification [0,1], [0,2], ... ')

# diplay of data
#ax0.axhline(y=0,ls='--',color='0.5',lw=2)
#ax0.axvline(x=0,ls='--',color='0.5',lw=2)
for i in range(45):
    if pairErrors[-1,i]> 2.:
        ax0.plot(pairErrors[:,i],label='[%s,%s]' % (pairs[i][0],pairs[i][1]))
    else:
        ax0.plot(pairErrors[:,i])

#ax0.plot(x,cosy,label='cos')

# removes upper and right axes
# and moves left and bottom axes away
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.spines['bottom'].set_position(('outward', 10))
ax0.spines['left'].set_position(('outward', 10))
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')

# legends and labels
plt.legend(loc=(0.85,0.6),frameon=False)

leg = plt.gca().get_legend()
ltext  = leg.get_texts()
plt.setp(ltext, fontsize=8)

plt.xlabel('traning epochs')
plt.ylabel('error rate (%)')

## save figure ############################################################

fname = 'fig_perceptron_progress'

plt.savefig(fname+'.png')
plt.savefig(fname+'.pdf')
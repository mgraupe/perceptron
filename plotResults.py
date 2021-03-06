import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
import itertools
import matplotlib

pairs = []
for p in itertools.combinations(range(10),2):
    pairs.append(p)

data = pickle.load( open('learningProgress.p'))
test = pickle.load( open('testPerformance.p'))

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
fig_height = 12  # height in inches
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
gs = gridspec.GridSpec(3, 1,
                       #width_ratios=[1,1.2],
                       #height_ratios=[1,1]
                       )

# define vertical and horizontal spacing between panels
gs.update(wspace=0.3,hspace=0.4)

# possibly change outer margins of the figure
plt.subplots_adjust(left=0.13, right=0.95, top=0.95, bottom=0.06)

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
ax0.plot(501,test[0]*100./(test[0]+test[1]),'o',label='performance on test set')
ax0.plot(errorRate,label='learning on training set')
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

plt.xlabel('training epochs')
plt.ylabel('error rate (%)')

# first sub-plot #######################################################
ax0 = plt.subplot(gs[1])

# title
ax0.set_title('Pairwise classification [0,1], [0,2], ... ')

# diplay of data
#ax0.axhline(y=0,ls='--',color='0.5',lw=2)
#ax0.axvline(x=0,ls='--',color='0.5',lw=2)

#cmap = matplotlib.cm.hsv
#norm = matplotlib.colors.Normalize(vmin=0, vmax=45)

for i in range(45):
    #col = cmap(norm(i))
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

plt.xlabel('training epochs')
plt.ylabel('error rate (%)')

# first sub-plot #######################################################
ax2 = plt.subplot(gs[2])

# title
ax2.set_title('Pairwise classification [0,1], [0,2], ... ')

# diplay of data
#ax0.axhline(y=0,ls='--',color='0.5',lw=2)
#ax0.axvline(x=0,ls='--',color='0.5',lw=2)

#cmap = matplotlib.cm.hsv
#norm = matplotlib.colors.Normalize(vmin=0, vmax=45)

for i in range(45):
    #col = cmap(norm(i))
    if pairErrors[40,i] == 0.:
        ax2.plot(pairErrors[:,i],label='[%s,%s]' % (pairs[i][0],pairs[i][1]))
    #else:
    #    ax2.plot(pairErrors[:,i])

#ax0.plot(x,cosy,label='cos')

# removes upper and right axes
# and moves left and bottom axes away
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_position(('outward', 10))
ax2.spines['left'].set_position(('outward', 10))
ax2.yaxis.set_ticks_position('left')
ax2.xaxis.set_ticks_position('bottom')

ax2.set_xlim(left=-1,right=50)
# legends and labels
plt.legend(loc=1,frameon=False)

leg = plt.gca().get_legend()
ltext  = leg.get_texts()
plt.setp(ltext, fontsize=8)

plt.xlabel('training epochs')
plt.ylabel('error rate (%)')

## save figure ############################################################

fname = 'fig_perceptron_progress'

plt.savefig(fname+'.png')
plt.savefig(fname+'.pdf')

plt.clf()

## new figure ############################################################


d = pickle.load(open('numbersMissclassified.p'))

fig_width = 6 # width in inches
fig_height = 5  # height in inches
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
gs = gridspec.GridSpec(1, 1,
                       #width_ratios=[1,1.2],
                       #height_ratios=[1,1]
                       )

# define vertical and horizontal spacing between panels
gs.update(wspace=0.3,hspace=0.4)

# possibly change outer margins of the figure
plt.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.15)

# sub-panel enumerations
#plt.figtext(0.06, 0.92, 'A',clip_on=False,color='black', weight='bold',size=22)
#plt.figtext(0.47, 0.92, 'B',clip_on=False,color='black', weight='bold',size=22)
#plt.figtext(0.06, 0.47, 'C',clip_on=False,color='black', weight='bold',size=22)
#plt.figtext(0.47, 0.47, 'D',clip_on=False,color='black', weight='bold',size=22)


# first sub-plot #######################################################
ax0 = plt.subplot(gs[0])

# title
ax0.set_title('Misclassified numbers in test set')

# diplay of data
#ax0.axhline(y=0,ls='--',color='0.5',lw=2)
#ax0.axvline(x=0,ls='--',color='0.5',lw=2)
ax0.hist(d,bins=np.arange(-0.5,10.5,1),align='mid',rwidth=0.5)

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
#plt.legend(loc=1,frameon=False)

plt.xlabel('number')
plt.ylabel('mis-classifications')


fname = 'fig_test_classifications'

plt.savefig(fname+'.png')
plt.savefig(fname+'.pdf')
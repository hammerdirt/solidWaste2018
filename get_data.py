# This function will take the urls-make the API call and turn into dataframe
# and it checks that the dataframe that is returned has has data!
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.stats
from collections import Counter
def getData(url):
    a = url
    try:
        c = requests.get(url).json()
    except ValueError:
        print("The name you provided is not listd in the database. You can find the current list of all lakes, rivers and beaches at https://mwshovel.pythonanywhere.com/dirt/beach_litter.html")
    d = pd.DataFrame(c)
    if d.empty == True:
        raise ValueError('The dataframe is empty check the beach-name or lake-river name')
    else:
        return d

# This function will retireve the code definitions from the web-site
# and verifies that the code given by the user is valid
def codes():
    csvUrl = "https://mwshovel.pythonanywhere.com/static/newCriteria.csv"
    a = pd.read_csv(csvUrl)
    b = list(a.code.unique())
    if objectOfInterest in b:
        return a
    else:
        raise ValueError("The code is not included in the current list. Check to see that you entered it correctly ie.. 'G89'")

# This will return only the data from the desired beach:
# and verifies that the beach given is included with the given lake or river
def beachDf(beachData, beachName):
    a = list(beachData.location_id.unique())
    if beachName in a:
        b = beachData.loc[beachData.location_id == beachName].copy()
    else:
        raise ValueError('The beach is not included with the river or lake that was chosen. You can find the current list of all lakes, rivers and beaches at https://mwshovel.pythonanywhere.com/dirt/beach_litter.html')
    return b

# This will return the list of codes identified in a df and the number of codes identified
def codeFrequency(df):
    a = list(df.code_id)
    b = len(a)
    return a, b

# This will make a column of pcs_m for the dataFrame
def pcs_m(df):
    a = df.columns
    if "quantity" in a:
        df['pcs_m'] = df.quantity/df.length
    elif "total" in a:
        df['pcs_m'] = df.total/df.length
    return df

# This will remove any records with a pcs/m value of Zero
def greaterThanZero(df):
    df = df[df['pcs_m'] > 0]
    return df.copy()

# This creates a log column of the pcs/m value from a df
def logOfPcsMeter(df):
    df['ln_pcs'] = np.log(df['pcs_m'])
    return df.copy()

# This will create a list of objects from a beach that are greater than the percentile specified:
def getGreaterThan(aList, df, aBeach, p):
    c = []

    for b in aList:
        a = df.loc[df.code_id == b ][['location_id','code_id', 'pcs_m']]
        e = a.pcs_m.quantile(p)
        f = a[(a.location_id == aBeach) & (a.pcs_m > e)]
        # grab the value that is greater than the desired %
        if len(f['pcs_m'].values) > 0:
            c.append((b,e,f['pcs_m'].values[0]))
    return c
# This will print a string for the values from the getGreaterThan
def whatsMyPercentile(aList, p):
    a = "There were " + str(len(aList)) + " categories greater than the " + str(p) + "th %,"
    return a
# This returns a list of the codes from the getGreaterThan:
def codesListPercent(aList):
    a = [x[0] for x in aList]
    return a
# returns the number of entries in a df used for the number of samples
def numSamples(df):
    a = len(df)
    return a

def makeHist(series, bins, figSize, title, supTitle, xLabel, yLabel, saveName):
    fig, ax = plt.subplots(figsize=figSize)
    xOne = series
    nBins = bins
    ax.hist(xOne, bins=nBins)
    ax.set_xlabel(xLabel, size=14, labelpad=20)
    ax.set_ylabel(yLabel, size=14, labelpad=20)
    plt.suptitle(supTitle , fontsize=16, family='sans', horizontalalignment='center', y=1.01 )
    plt.title(title + str(len(xOne)), fontsize=14, family='sans', loc="center", y=1.03)
    plt.show()
    plt.savefig(saveName)
    plt.close()
def makeYearResults(date, dateTwo, dateThree, df, column):
    a = list(df[df.date < date][column])
    b = list(df[(df.date >= date) & (df.date < dateTwo)][column])
    c = list(df[(df.date >= dateTwo) & (df.date < dateThree)][column])
    return [sorted(a), sorted(b), sorted(c)]
def makeAve(a):
    return np.average(a)
def makeStd(a):
    return np.std(a)
def makePdf(a):
    return norm.pdf(a[0], a[1], a[2])
def makeXandY(a):
    f = []
    for l in a:
        b = makeAve(l)
        c = makeStd(l)
        g = np.random.normal(loc=b, scale=c, size=1000)
        d = [sorted(g), b, c]
        e = makePdf(d)
        h = len(l)
        f.append([sorted(g), e, b, c, h])
    return f
def makeYearOverYear(Data, figSize, colors, labels, saveFig):
    fig, ax = plt.subplots(figsize=figSize)
    ax.set_ymargin(0)
    samples = {}
    for i, data in enumerate(Data):
        samples.update({i:(data[4])})
        ax.plot(data[0], data[1], color=colors[i], label=labels[i])
        ax.fill(data[0], data[1], color=colors[i], alpha=0.2)
        if i == 0:
            vOne = np.round(data[2] + data[3], 3)
            vTwo = np.round(data[2] - data[3], 3)
            stdOne = np.round(data[3], 2)
            meanOne = np.round(data[2], 2)
            ax.axvline(vOne, ls="--", color="black", alpha=0.5)
            ax.axvline(vTwo, ls="--", color="black", alpha=0.5)
            ax.annotate("", xy=(vTwo, 0.45), xycoords='data',
                        xytext=(meanOne, 0.45), textcoords='data',
                        arrowprops=dict(arrowstyle="<|-|>", alpha=0.5,
                                        connectionstyle="arc3"))
            ax.annotate("", xy=(vOne, 0.45), xycoords='data',
                        xytext=(meanOne, 0.45), textcoords='data',
                        arrowprops=dict(arrowstyle="<|-|>", alpha=0.5,
                                        connectionstyle="arc3"))
            ax.scatter(meanOne, 0.45, s=20, c='black', alpha=0.5)
            ax.text(vTwo+(data[3]/2), 0.47, r'$\sigma$',
                    {'color': 'black', 'fontsize': 16, 'ha': 'center', 'va': 'center'},alpha=0.5)
            ax.text(vOne-(data[3]/2), 0.47, r'$\sigma$',
                    {'color': 'black', 'fontsize': 16, 'ha': 'center', 'va': 'center'},alpha=0.5)
    ax.legend()
    ax.set_ylim(top=0.5)
    ax.set_ylabel("Probability density", labelpad= 10, color='black', fontsize=14)
    ax.set_xlabel('Log of pieces of trash per meter', labelpad= 10, color='black', fontsize=14)
    left, right = plt.xlim()
    ax.text(left+0.1, 0.38, r'$\sigma$: ' + str(stdOne) + ' first year standard deviation',
                {'color': 'black', 'fontsize': 10, 'ha': 'left', 'va': 'center'})
    plt.suptitle("Year over year probability density of trash/meter 2015-2018" ,
                     fontsize=16, family='sans', horizontalalignment='center', y=0.98 )
    plt.title("Lake Geneva; year one n=" + str(samples[0]) + ", year two n=" + str(samples[1]) +" year three n=" + str(samples[2]), fontsize=14, family='sans', loc="center", pad=15)

    plt.savefig(saveFig)
    plt.show()
    plt.close()

def labelsCounts(a):
    f = []
    for b,c in a.items():
        f.append((b,c))
    return f

def countFrequency(aList):
    c = []
    for data in aList:
        a = Counter(data)
        b = labelsCounts(a)
        c.append(b)
    return c
def beachSamplesYear(aTuple,n, position):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(121)
    fig.subplots_adjust=0.3
    i = 0
    values = sorted(aTuple,key=lambda x: x[1], reverse=True, )
    for a in values:
        ax.bar(1, a[1], bottom=i, width=.4, label=a[0])
        i += a[1]
    ax.set_ylabel("Number of samples", labelpad= 10, color='black', fontsize=14)
    handles, labels = ax.get_legend_handles_labels()
    lgd1 = plt.legend(handles[::-1], labels[::-1], title='Beaches', title_fontsize=12, bbox_to_anchor=(1.02, 1.02))
    ax.set_title("Year " + str(position) + ": " + str(len(aTuple)) + " beaches surveyed; " + str(i) + " surveys.",
             fontsize=14, family='sans', loc="left", pad=20)
    plt.xticks([], [])
    plt.savefig("graphs/barCharts/"+"year" +str(position) +".svg", bbox_extra_artists=(lgd1))
    plt.show()
    plt.close()

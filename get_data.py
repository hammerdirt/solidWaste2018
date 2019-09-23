
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.stats
from collections import Counter
import matplotlib.dates as mdates
from matplotlib import colors as mcolors
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns



def getData(url):
    """
    This function will take the urls and make the API call, turn  that into dataframe
    and it checks that the dataframe that is returned has has data!
    """
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


def codes():
    """
    This function will retrieve the code definitions from the web-site
    and verifies that the code given by the user is valid
    """
    csvUrl = "https://mwshovel.pythonanywhere.com/static/newCriteria.csv"
    a = pd.read_csv(csvUrl)
    b = list(a.code.unique())
    return a, b


def beachDf(beachData, beachName):
    """
    This will return only the data from the desired beach:
    and verifies that the beach given is included with the given lake or river
    """
    c = beachData.columns
    if "location_id" in c:
        a = list(beachData.location_id.unique())
        if beachName in a:
            b = beachData.loc[beachData.location_id == beachName].copy()
    elif "location" in c:
        a = list(beachData.location.unique())
        if beachName in a:
            b = beachData.loc[beachData.location == beachName].copy()
    else:
        raise ValueError('The beach is not included with the river or lake that was chosen. You can find the current list of all lakes, rivers and beaches at https://mwshovel.pythonanywhere.com/dirt/beach_litter.html')
    return b


def codeFrequency(df):
    """
    This will return the list of codes identified in a df and the number of codes identified
    The list is not unique
    There must be a column called 'code_id'
    """
    a = list(df.code_id)
    b = len(a)
    return a, b


def pcs_m(df):
    """
    This will make a column of pcs_m for the dataFrame
    Note the df needs to have:
    a column called length and:
    a column called quantity or total
    """
    a = df.columns
    if "quantity" in a:
        df['pcs_m'] = df.quantity/df.length
    elif "total" in a:
        df['pcs_m'] = df.total/df.length
    return df


def greaterThanZero(df):
    """
    This will return a dataframe with a pcs/m value of > Zero
    """
    df = df[df['pcs_m'] > 0]
    return df.copy()


def logOfPcsMeter(df):
    """
    This creates a log column of the pcs/m value from a df
    Must have a 'pcs_m' column
    """
    df['ln_pcs'] = np.log(df['pcs_m'])
    return df.copy()


def getGreaterThan(aList, df, aBeach, p):
    """
    This will create a list of objects from a beach that are greater than the percentile specified:
    """
    c = []
    for b in aList:
        a = df.loc[df.code_id == b ][['location_id','code_id', 'pcs_m']]
        e = a.pcs_m.quantile(p)
        f = a[(a.location_id == aBeach) & (a.pcs_m > e)]
        # grab the value that is greater than the desired %
        if len(f['pcs_m'].values) > 0:
            c.append((b,e,f['pcs_m'].values[0]))
    return c

def whatsMyPercentile(aList, p):
    """
    This will print a string for the values from the getGreaterThan
    """
    a = "There were " + str(len(aList)) + " categories greater than the " + str(p) + "th %,"
    return a

def codesListPercent(aList):
    """
    This returns a list of the codes from the getGreaterThan:
    """
    a = [x[0] for x in aList]
    return a

def numSamples(df):
    """
    returns the number of entries in a df used for the number of samples
    """
    a = len(df)
    return a

def makeHist(series, bins, figSize, title, supTitle, xLabel, yLabel, saveName):
    """
    Makes a simple hsitogram, variables to specify:
    series: the data in array like format (pd.series works)
    bin: INT, the number of bins you whant
    figSize: Tuple, (float, flaot) the size of the figure (default matplotlib units)
    title & supTitle: String, value of the titles you want
    xLabel & yLabel: String, to label the x and y axis
    saveName - String: directory and file name to make the fiugre (inlcude type .svg, .jpg..)
    """
    fig, ax = plt.subplots(figsize=figSize)
    xOne = series
    nBins = bins
    ax.hist(xOne, bins=nBins, rwidth=0.9)
    ax.set_xlabel(xLabel, size=14, labelpad=10)
    ax.set_ylabel(yLabel, size=14, labelpad=10)
    plt.suptitle(supTitle , fontsize=16, family='sans', horizontalalignment='center', y=.98 )
    plt.title(title + str(len(xOne)), fontsize=14, family='sans', loc="center", y=1.0)
    plt.savefig(saveName)
    plt.show()

    plt.close()

def makeYearResults(listOfDates, df, column):
    """
    Seperates a data frame between the dates in a list and allows to select a column.
    The date provided is the "less than date" ie if "2017-10-05" is given, the function will
    well return all dates < "2017-10-05"
    column: Column name of interest
    Returns the column values in list format for each interval.
    """
    a = []
    b = len(listOfDates)
    if b == None:
        return "There are no dates in the list"
    if b == 1:
        c = list(df[df.date < listOfDates[0]][column])
        a.append(sorted(c))
    else:
        d = np.arange(b)
        for i,day in enumerate(d):
            if i == 0:
                c = list(df[df.date < listOfDates[0]][column])
                a.append(sorted(c))
            else:
                e = i-1
                c = list(df[(df.date >= listOfDates[e]) & (df.date < listOfDates[i])][column])
                a.append(sorted(c))
    return a
def getOneSurvey(df, date):
    """
    Returns the data from a specific date
    """
    return df[df.date == date]
def makeAve(a):
    """
    Returns the np.average of an array or list
    """
    return np.average(a)
def makeStd(a):
    """
    Returns the np.std of an array
    """
    return np.std(a)
def makePdf(a):
    """
    Returns the Probability density function of a list of values
    if list is 'a' then
    a[0] = list of values
    a[1] = np.average of a[0]
    a[2] = np.std of a[0]
    feeder for makeXandY
    """
    return norm.pdf(a[0], a[1], a[2])
def makeXandY(a):
    """
    Retuns x and y values to graph a pdf from list of a list of values.
    Feeder function for makeYearOverYear graphing function
    Uses the mean and standard deviation to create an array of 1000 points
    use the the mean and standard deviation with np.random.normal to create a theortical distribution
    calls the following functions:
    makeAve, makeStd, makePdf

    """
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
def makeYearOverYear(Data, figSize, colors, labels,title, subTitle, saveFig):
    """
    Returns filled plots of PDF for each array in Data.
    Accepts output from makeXandY, returns a filled plot for each element in array
    Data[0]: X values, float
    Data[1]: Y values, float
    Data[2]: Mean of X
    Data[3]: Standard deviation of X
    Data[4]: Number of graphs to draw
    figSize: Tuple of floats
    colors: List of colors for each data set
    labels: Labels for the legend
    title : String for the title
    subTitle: String for the subtitle
    saveFig: File path for the sved figure, with extension

    """
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
    plt.suptitle(title ,
                     fontsize=16, family='sans', horizontalalignment='center', y=0.98 )
    plt.title(subTitle, fontsize=14, family='sans', loc="center", pad=15)

    plt.savefig(saveFig)
    plt.show()
    plt.close()

def labelsCounts(a):
    """
    Turns a dict into a list of tuples
    """
    f = []
    for b,c in a.items():
        f.append((b,c))
    return f

def countFrequency(aList):
    """
    Calls collection.Counter on aList
    Turns the resulting dictionary into a list of tuples (key, value)
    Calls labelsCounts
    """
    c = []
    for data in aList:
        a = Counter(data)
        b = labelsCounts(a)
        c.append(b)
    return c
def beachSamplesYear(aTuple, year, colorDict):
    """
    Makes a stacked bar chart from a list of a list of tuples
    Sorts the list of tuples by tuple[1] to reverse order, applies this to the bar height
    tuple[0]: location name, used in the legend and to assign color
    Used to make barcharts of operations per location per year
    year: is the integer value of the year ie.. year one, year two etc..
    colorDict: is a dictionary that has key value pairs {"location":color}
    """
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(121)
    fig.subplots_adjust=0.3
    i = 0
    values = sorted(aTuple,key=lambda x: x[1], reverse=True)
    for a in values:
        ax.bar(1, a[1], bottom=i, width=.4, label=a[0], color=colorDict[a[0]])
        i += a[1]
    ax.set_ylabel("Number of surveys", labelpad= 10, color='black', fontsize=14)
    handles, labels = ax.get_legend_handles_labels()
    lgd1 = plt.legend(handles[::-1], labels[::-1], title='Beaches', title_fontsize=12, loc='upper left', bbox_to_anchor=(1.02, 1.02))
    ax.set_title("Year " + str(year) + ": " + str(len(aTuple)) + " beaches surveyed; " + str(i) + " surveys.",
             fontsize=14, family='sans', loc="left", pad=20)
    plt.ylim(top=85)
    plt.xticks([], [])
    plt.savefig("graphs/barCharts/"+"year" +str(year) +".svg", bbox_extra_artists=(lgd1))
    # plt.show()
    plt.close()
def surveysProjectYear(aList, keys, relation, df):
    """
    Returns an unstacked dataFrame of number of surveys per project between the supplied dates
    aList: list of dates
    keys: list of makeProjects
    relation: dictionary key=projectname, value = list of locations (call getProjectBeaches)
    """
    c = []
    for n,day in enumerate(aList):
        if n == 0:
            b = df[df.date < aList[n]]
        else:
            b = df[(df.date >= aList[n-1]) & (df.date < aList[n])]
        year = n+1
        for key in keys:
            e = list(b.location.unique())
            f = relation[key]
            g = [x for x in f if x in e]
            if len(g) > 0:
                aDf = b[b.location.isin(g)]
                h = len(aDf)
                i = aDf.pcs_m.mean()
                c.append({"Year":year, "Project":key, "Surveys":np.round(h, 0), "Average pcs/m":np.round(i, 2)})
            else:
                c.append({"Year":year, "Project":key, "Surveys":0, "Average pcs/m":0})
    d = pd.DataFrame(c, columns=["Project", "Year", "Surveys", "Average pcs/m"])
    d.set_index(["Year","Project"], inplace=True)
    return d.stack(0)

def getProjectBeaches(a):
    """
    Returns a dictionary project:list of locations
    Takes a df as input
    """
    b = {}
    c = list(a.project_id.unique())
    for project in c:
        d = list(a[a.project_id == project]["location"])
        b.update({project:d})
    return b, c
def changeProject(a, b, df):
    """
    Used to change project and location attribution
    If you don't know how or why to use this then you should move on
    """
    for name in a:
        for place in name[0]:
            df.loc[df.location == place, "project_id"] =name[1]
    for otherProject in b:
        df.loc[df.project_id == otherProject, "project_id"] = "MCBP"
def surveysPerProject(keys, relation, df):
    """
    Returns the total number of surveys per project
    keys is list of projects
    relation: dictionary key=projectname, value = list of locations (call getProjectBeaches)
    """
    b = {}
    for key in keys:
        a = df.pcs_m.groupby(df.location.isin(relation[key])).count()[True]
        b.update({key:a})
    c = pd.Series(b)
    return c
def makeTimeSeriesAll(figSize,data,projects, saveFig):
    """
    Returns a time series scatter plot of all data
    y = "pcs_m"
    data is a dataFrame
    project is a list of porjects from the dataframe
    """
    fig, ax = plt.subplots(figsize=figSize)
    ax.set_ymargin(0.01)
    ax.set_axisbelow(True)

    # define how the data is presented (year, month)
    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator()  # every month
    yearsFmt = mdates.DateFormatter('%Y')
    monthsFmt = mdates.DateFormatter('%b')

    # grab some colors here colors = dict(**mcolors.CSS4_COLORS) is your friend
    colors = ["darkblue", "darkcyan", "darkslategray", "darkolivegreen", "darkred"]

    # scatter for each project
    for i, project in enumerate(projects):
        newDf = data[data.Project == project]
        x = list(newDf.date)
        y = list(newDf.pcs_m)
        ax.scatter(x=x, y=y, color=colors[i], edgecolor="white",linewidth=1, s=130, label=project)
    numSamps = len(data)

    # lay down the x ticks
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(yearsFmt)
    ax.xaxis.set_minor_locator(months)
    ax.xaxis.set_minor_formatter(monthsFmt)
    for text in ax.get_xminorticklabels()[::2]:
        text.set_visible(False)
    ax.tick_params(axis='x', which="major", pad=22, labelrotation=90)
    ax.tick_params(axis='x', which="minor", labelsize=10, labelrotation=45,)

    ax.legend(fontsize=10)

    # create year labels and fills
    # xFive = pd.to_datetime("2016-05-01")
    # ax.text(xFive, 58, "Year one",
    #         {'color': 'black', 'fontsize': 16, 'ha': 'center', 'va': 'center'},alpha=0.6, zorder=1)
    # xSix = pd.to_datetime("2017-05-1")
    # ax.text(xSix, 58, "Year two",
    #         {'color': 'black', 'fontsize': 16, 'ha': 'center', 'va': 'center'},alpha=0.6, zorder=1)
    # xSeven = pd.to_datetime("2018-05-1")
    # ax.text(xSeven, 58, "Year three",
    #         {'color': 'black', 'fontsize': 16, 'ha': 'center', 'va': 'center'},alpha=0.6, zorder=1)



    # lay down the y ticks
    ml = MultipleLocator(2)
    ax.yaxis.set_minor_locator(ml)

    # lay down the gridlines
    ax.grid(b=True, which='major', axis='y', linewidth=1, color='slategray', alpha=0.5)
    ax.grid(b=True, which='minor', axis='y', linewidth=1, color='slategray', alpha=0.2)
    conv = np.vectorize(mdates.strpdate2num('%Y-%m-%d'))

    #fill between the dates
    # xOne, xTwo = pd.to_datetime("2015-11-15"), pd.to_datetime("2016-11-15"),
    # xThree, xFour = pd.to_datetime("2017-11-15"), pd.to_datetime("2018-11-15")
    # ax.axvspan(xOne,xTwo, color='r', alpha=0.2, zorder=0)
    # ax.axvspan(xTwo,xThree, color='g', alpha=0.2, zorder=0)
    # ax.axvspan(xThree, xFour, color='b', alpha=0.2, zorder=0)

    # axis labels and titles
    ax.set_ylabel("Total pieces of trash per meter of shoreline per sample", labelpad= 10, color='black', fontsize=12)
    plt.suptitle("Beach litter surveys 2015-2018" ,
                     fontsize=20, family='sans', horizontalalignment='center', y=0.98 )
    plt.title("Lake Geneva all sample sites, sorted by project/team n=" + str(numSamps),
              fontsize=12, family='sans', loc="center", pad=15)
    plt.savefig(saveFig)
    plt.show()
    plt.close()

def makeXandYProject(a):
    """
    Retuns x and y values to graph a pdf from list of a list of values.
    feeder function for makeYearOverYear graphing function
    Uses the mean and standard deviation to create a an array of 1000 points
    with np.random.normal
    calls the following functions:
    makeAve, makeStd, makePdf

    """
    f = []
    for n,l in a:
        b = makeAve(l)
        c = makeStd(l)
        g = np.random.normal(loc=b, scale=c, size=1000)
        d = [sorted(g), b, c]
        e = makePdf(d)
        h = len(l)
        f.append([n, sorted(g), e, b, c, h])
    return f

def makeProjectDist(Data, figSize, colors, saveFig):
    fig, ax = plt.subplots(figsize=figSize)
    ax.set_ymargin(0)
    samples = {}
    textLabel = []
    for i, data in enumerate(Data):
        samples.update({i:(data[5])})
        ax.plot(data[1], data[2], color=colors[i], label=data[0])
        ax.fill(data[1], data[2], color=colors[i], alpha=0.2)
        ymax = data[2].max()
        vOne = np.round(data[3] + data[4], 3)
        vTwo = np.round(data[3] - data[4], 3)
        stdOne = np.round(data[4], 2)
        meanOne = np.round(data[3], 2)
        textLabel.append([data[0], stdOne])
        plt.vlines(vOne, ymin=0, ymax=ymax, linestyles="--", color=colors[i], alpha=0.6)
        plt.vlines(vTwo, ymin=0, ymax=ymax, linestyles="--", color=colors[i], alpha=0.6)
        adjust = i/4
        ax.annotate("", xy=(vTwo, ymax-.04), xycoords='data',
                    xytext=(meanOne, ymax-.04), textcoords='data',
                    arrowprops=dict(arrowstyle="<|-|>", alpha=0.6, color=colors[i],
                                    connectionstyle="arc3"))
        ax.annotate("", xy=(vOne, ymax-.04), xycoords='data',
                    xytext=(meanOne, ymax-.04), textcoords='data',
                    arrowprops=dict(arrowstyle="<|-|>", alpha=0.6,color=colors[i],
                                    connectionstyle="arc3"))
        left, right = plt.xlim()
        ax.text(left+i, ymax-.04, r'$\sigma$: ' + data[0] + ' ' + str(stdOne),
                {'color': colors[i], 'fontsize': 10, 'ha': 'left', 'va': 'center'})
        ax.scatter(meanOne, ymax-.04, s=20, color=colors[i], alpha=0.6)
        ax.text(vTwo+(data[3]/2), ymax-.02, r'$\sigma$',
                {'color': colors[i], 'fontsize': 16, 'ha': 'center', 'va': 'center'},alpha=0.5)
        ax.text(vOne-(data[3]/2), ymax-.02, r'$\sigma$',
                {'color': colors[i], 'fontsize': 16, 'ha': 'center', 'va': 'center'},alpha=0.5)



    ax.legend(fontsize=14)
    ax.set_ylim(top=0.5)
    ax.set_ylabel("Probability density", labelpad= 10, color='black', fontsize=14)
    ax.set_xlabel('Log of pieces of trash per meter', labelpad= 10, color='black', fontsize=14)


    plt.suptitle("Probability density SLR vs MCBP 2015-2018" ,
                     fontsize=16, family='sans', horizontalalignment='center', y=0.98 )
    plt.title("Lake Geneva: log(pcs/m); MCBP n=" + str(samples[0]) + ", SLR n=" + str(samples[1])
              , fontsize=14, family='sans', loc="center", pad=15)

    plt.savefig(saveFig)
    plt.show()
    plt.close()
def makeCumlativeDist(x, mu, sigma, nbins, labelOne, labelTwo, chex, chex2, chex3):
    fig, ax = plt.subplots(figsize=(8, 6))
    # plot the cumulative hsitogram  from the data
    n, bins, patches = ax.hist(x, nbins, density=1, color='fuchsia', histtype='step',
                           cumulative=True, label=labelOne)

    # get the pairs (x=pcs/m, y=CDF)
    probDict = dict(zip(bins,n))

    # check the probability here
    def findTheProbability(a, b):
        c = list(a.keys())
        c = sorted(c)
        d = []
        f = []
        for number in b:
            for i,key in enumerate(c):
                if number < c[i] and number > c[i-1]:
                    d.append(number)
                    f.append(a[c[i]])
        return [d, f]
    checkMe = findTheProbability(probDict, chex)
    checkMe2 = findTheProbability(probDict, chex2)
    checkMe3 = findTheProbability(probDict, chex3)

    # mormalize that to see the proposed dist
    y = makePdf([bins, mu, sigma]).cumsum()
    y /= y[-1]

    # plot that
    ax.plot(bins, y, 'k--', linewidth=1.5, label=labelTwo)

    # plot the probability of the points
    ax.scatter(checkMe[0], checkMe[1], c="fuchsia", s=60, label="SWE")
    ax.scatter(checkMe2[0], checkMe2[1], c="peachpuff", s=60, label="GIS")
    ax.scatter(checkMe3[0], checkMe3[1], c="indigo", s=60, label="PC")

    # lay the grid down
    ax.grid(b=True, which='major', axis='y', linewidth=1, color='slategray', alpha=0.3)
    ax.grid(b=True, which='minor', axis='y', linewidth=1, color='slategray', alpha=0.1)
    ax.grid(b=True, which='major', axis='x', linewidth=1, color='slategray', alpha=0.3)
    ax.grid(b=True, which='minor', axis='x', linewidth=1, color='slategray', alpha=0.1)

    # format the ticks
    mlx = MultipleLocator(2)
    mly = MultipleLocator(.05)
    ax.yaxis.set_minor_locator(mly)
    ax.xaxis.set_minor_locator(mlx)
    ax.tick_params(axis='both', which="major", labelsize=12,pad=6)
    ax.tick_params(axis='both', which="minor", labelsize=8, )

    ax.set_ylabel("Probability of event", labelpad= 10, color='black', fontsize=14)
    ax.set_xlabel('Pieces of trash per meter', labelpad= 10, color='black', fontsize=14)
    ax.legend(loc='right', fontsize=14)

    plt.title("Lake Geneva: cumulative distribution MCBP & SLR 2015-2018",
              fontsize=14, family='sans', loc="center", pad=15)


    plt.savefig("graphs/distributions/cumSWE.svg")
    plt.show()
    return probDict, checkMe
def sideBysideBoxPlots(dataOne, dataTwo, colorsSamps, colorsBox, namesSamples, samplesOne,
                       samplesTwo, figTitle, axOneTitle, axTwoTitle, fileName):
    """
    Plots side by side box plots, empirical data and normalized
    dataOne and dataTwo: sorted array of float values for the boxplots
    colorsBox: the colors for the boxplot body and edges (two values)
    colorsSamples: the colors for the ransom samples len = the number of random sample groups
    namesSamples: legend names for the random sample groups
    samplesOne and samplesTwo: array of values for random samples
    figTitle, axOneTitle, axTwoTitle,: the respective titles in string format
    fileName: where you want to save the SVg

    """

    fig, axs = plt.subplots(1, 2, figsize=(8,8))

    def plotThese(aList, ax):
        for i,thisList in enumerate(aList):
            x = [1 for point in thisList]
            axs[ax].scatter(x, thisList, color=colorsSamps[i], edgecolor='white', s=140, zorder=2, label=namesSamples[i])

    flierprops = dict(marker='o', markerfacecolor=colorsBox[0], markersize=10,
                  markeredgecolor="white", alpha=0.4)
    whiskerprops = dict(color=colorsBox[1])

    boxprops=dict(facecolor=colorsBox[0], color=colorsBox[1], linewidth=2, alpha=0.4)


    # basic plot
    bPlot = axs[0].boxplot(dataOne, patch_artist=True, zorder=0,   flierprops=flierprops,
                           whiskerprops=whiskerprops, capprops=whiskerprops,
                           boxprops = boxprops, medianprops=whiskerprops, widths=.8)
    axs[0].set_title(axOneTitle, fontsize=14)
    axs[0].set_xticks([])
    plotThese(samplesOne, 0)

    # log plot
    bPlot2=axs[1].boxplot(dataTwo, patch_artist=True, zorder=0,flierprops=flierprops,
                          whiskerprops=whiskerprops, capprops=whiskerprops,
                          boxprops=dict(facecolor="darkblue", color="darkcyan", linewidth=2, alpha=0.4),
                          medianprops=whiskerprops, widths=.8)
    plotThese(samplesTwo, 1)

    axs[1].set_title(axTwoTitle, fontsize=14)
    axs[1].set_xticks([])
    plt.legend()
    fig.suptitle(figTitle,fontsize=14, family='sans', horizontalalignment="center",)

    plt.savefig(fileName)
    plt.show()
    plt.close()
def boxAndPdf(dataOne, dataOneDist, dataTwo, dataTwoDist, colorsSamps,colorsBox, namesSamples,
              samplesOne, samplesTwo, figTitle, axOneTitle, axTwoTitle, fileName):

    Fig = plt.figure( figsize=(10,9))
    axOne = plt.subplot2grid((3, 2), (0, 0))
    axTwo = plt.subplot2grid((3, 2), (1, 0), rowspan=2)
    axThree = plt.subplot2grid((3,2), (0,1))
    axFour = plt.subplot2grid((3,2), (1,1), rowspan=2)

    # plotter for random samples on boxplot
    def plotThese(aList, ax):
        for i,thisList in enumerate(aList):
            y = [1 for point in thisList]
            ax.scatter(thisList, y,color=colorsSamps[i],
                       s=140, zorder=3, edgecolor='white', label=namesSamples[i],)
        return
    # plotter for random samples on distributions
    def plotTheseDist(aList, distList, ax):
        for i,thisList in enumerate(aList):
            mu = makeAve(distList)
            sigma = makeStd(distList)
            ax.scatter(thisList,makePdf([thisList, mu, sigma]), color=colorsSamps[i],
                       s=140, zorder=3, edgecolor='white', label=namesSamples[i],)
        return
    # plotter for whiskers on distribution
    def graphThese(aList, ax):
        for point in aList:
            axTwo.axvline(point, ls="--", color=colors[0], alpha=0.6)
        return
    def makeAbox(data, ax):
        return ax.boxplot(data, patch_artist=True, zorder=0, vert=False, flierprops=flierprops,
                         whiskerprops=whiskerprops, capprops=whiskerprops,
                         boxprops = boxprops, medianprops=whiskerprops, widths=.8)
    def makeAFilledDist(data, ax):
        ax.plot(data[0], data[1], color=colorsBox[0],
               label="Combined PDF MCBP-SLR", alpha=0.8)
        ax.fill_between(data[0], data[1], y2=0, color=colorsBox[0], alpha=0.2)
        return


    flierprops = dict(marker='o', markerfacecolor=colorsBox[0], markersize=10,
                      markeredgecolor="white", alpha=0.4)
    whiskerprops = dict(color=colorsBox[1])

    boxprops=dict(facecolor=colorsBox[0], color=colorsBox[1], linewidth=2, alpha=0.4)

    # boxplot top left - basic plot

    axOne.set_xticks([])
    axOne.set_title(axOneTitle, fontsize=14)
    bplt = makeAbox(dataOne, axOne)
    # plot the random samples on the boxplot
    plotThese(samplesOne, axOne)

    # get the whisker coordinates from the top boxplot:
    fx = bplt["medians"][0].get_xdata()[0]
    fy = bplt["whiskers"][0].get_xdata()[0]
    fz = bplt["whiskers"][0].get_xdata()[1]
    fw = bplt["whiskers"][1].get_xdata()[1]

    anotherList = [fz, fw]


    # plot the distribution below the first boxplot
    axTwo.set_ymargin(0.005)
    axTwo.set_ylabel("Probability of event", labelpad= 10, color='black', fontsize=14)
    axTwo.set_xlabel('Pieces of trash per meter)', labelpad= 10, color='black', fontsize=14)
    makeAFilledDist(dataOneDist, axTwo)
    # plot the random samples on the dist
    plotTheseDist(samplesOne, dataOneDist[0], axTwo)
    axTwo.legend()





    # plot the second boxplot:
    axThree.set_xticks([])
    axThree.set_title(axTwoTitle, fontsize=14)
    bpltTwo = makeAbox(dataTwo, axThree)
    # plot the random samples on second boxplot
    plotThese(samplesTwo, axThree)


    # plot the second distribution
    axFour.set_xlabel('Log(pieces of trash per meter)', labelpad= 10, color='black', fontsize=14)
    makeAFilledDist(dataTwoDist, axFour)
    # plot the random samples on the second distribution
    plotTheseDist(samplesTwo,dataTwoDist[0], axFour)



    Fig.suptitle(figTitle,fontsize=14, family='sans', horizontalalignment="center",)
    plt.savefig(fileName)

    plt.show()
    plt.close()
def plotPcsMaxToFrequency(aDict):
    fig, ax = plt.subplots(figsize=(6,6))

    values = ((k,v["Frequency"],v["Max"], v["Mean"]) for k,v in aDict.items())
    freqMax = [[],[]]
    freqMean = [[],[]]

    for n,a in enumerate(values):
        freqMax[0].append(a[1])
        freqMax[1].append(a[2])
        freqMean[0].append(a[1])
        freqMean[1].append(a[3])
    ax.scatter(freqMax[0], freqMax[1], color="c", edgecolor="white",linewidth=2, s=130, label="Max pcs/m")
    ax.scatter(freqMean[0], freqMean[1], color="b", edgecolor="white",linewidth=2, s=130, label="Average pcs/m")
    ax.set_ylabel("Pieces per meter", fontsize=14, labelpad=10)
    ax.set_xlabel("Number of times identified (max = 148)", fontsize=14, labelpad=10)
    ax.set_title("Frequency of identification and max/mean pcs/m 2015-2018")
    ax.legend()
    plt.savefig("graphs/distributions/pcsMFrequencyMax.svg")
    plt.show()
    plt.close()
    return
def topTenLocation(aDict, dayPcsmDict, colorList, location):
    """
    Makes a stacked bar chart from the topTenDict:
    codeList, topTenDict = getCodesAndPcs(sortByPcs(someDfofcodevalues), 10, "code_id")
    Title info comes from getOneDayPcsM(lakeDailyPcsM, "2017-10-05", "Plage-de-St-Sulpice")
    Used to make barcharts of top ten items per day
    """
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(121)
    fig.subplots_adjust=0.3
    i = 0
    values = ((k,v["pcs_m"],v["source"], v["description"]) for k,v in aDict.items())
    values = sorted(values,key=lambda x: x[1], reverse=True, )
    for n,a in enumerate(values):
        ax.bar(1, a[1], bottom=i, width=.4, label=str(a[3]) +": " + str(a[2]), color=colorList[n])
        i += a[1]
    ax.set_ylabel("Pieces per meter", labelpad= 10, color='black', fontsize=14)
    ax.set_title("Top-ten " + dayPcsmDict["location"] + ", " + str(dayPcsmDict['date'].astype('datetime64[D]')),
                 fontsize=14, family='sans', loc="left", pad=20)
    handles, labels = ax.get_legend_handles_labels()

    lgd1 = plt.legend(handles[::-1], labels[::-1], title='Top ten',
                      title_fontsize=12, loc='upper left', bbox_to_anchor=(1.02, 1.02))

    plt.ylim(top= sum(v for name,v,v1, v2 in values) + 1)
    plt.xticks([], [])
    plt.savefig("graphs/barCharts/topTen"+ location + ".svg", bbox_extra_artists=(lgd1))
    plt.show()
    plt.close()
    return
def topTenBoxLocation(popData, sampData,x, y, ylimit, location):
    fig, ax = plt.subplots(figsize=(6,8))
    colorsBox = ["darkblue","darkcyan" ]
    boxprops=dict(facecolor=colorsBox[0], edgecolor=colorsBox[1], linewidth=2, alpha=0.4)
    whiskerprops = dict(color=colorsBox[1],linewidth=1 )
    flierprops = dict(marker='o', markerfacecolor=colorsBox[0], markersize=10,
                      markeredgecolor="white", alpha=0.4)
    ax.set_ylim(0, ylimit)
    sns.boxplot(x="code_id", y="pcs_m", data=popData,boxprops=boxprops, whiskerprops=whiskerprops,
            medianprops=whiskerprops, capprops=whiskerprops, flierprops=flierprops, ax=ax)
    sns.stripplot(x="code_id", y="pcs_m", color="fuchsia", jitter=False, size=12,
                  data=sampData, ax=ax)
    legend_elements = [ Line2D([0], [0], marker='o', color='w', label='Top ten ' + location,
                          markerfacecolor="fuchsia", markersize=12),
                   Line2D([0], [0], marker='o', color='w',label='Outliers all other samples',
                         markerfacecolor=colorsBox[0], markersize=12, alpha=0.4),
                   Patch(facecolor=colorsBox[0], edgecolor=colorsBox[1], alpha=0.5,
                         label='IQR all other samples')]
    ax.set_ylabel("Pieces per meter - outliers > " + str(ylimit) +"pcs/m not shown", labelpad=10, fontsize=14)
    ax.set_xlabel("Top-ten objects identified at " + location, labelpad=10, fontsize=14)
    ax.set_title("Top-ten items " + location +  " and all other samples", pad=15)
    lgd1 = plt.legend(handles=legend_elements, loc='upper right')
    plt.savefig("graphs/distributions/"+ location + "topTenBoxLocation.svg")
    plt.show()
    plt.close()
    return
def sortByPcs(a):
    """
    Sorts a data frame of code-data by pieces per meter in descending value
    """
    b = a.sort_values("pcs_m", ascending=False)
    return b
def getCodesAndPcs(a, n, column):
    """
    Gets the first n values from a dataframe of a single survey
    Will accept the string "all"
    Selects a column and returns a dictionary and a list:
    {columnName:value[i]}
    """
    b = list(a[column])
    if n == "all":
        c = {x:{"pcs_m":a[a[column] == x]["pcs_m"].values[0]} for x in b}
    else:
        c = {x:{"pcs_m":a[a[column] == x]["pcs_m"].values[0]} for x in b[:n]}
    return b, c
def getCodeDefs(codes, results):
    """
    Updates the output of the getCodesAndPcs fucntion with:
    material, source, description per key,value pair.
    """
    d = results
    a = list(results.keys())
    for b in a:
        c = codes[codes.code == b][["material","description", "source"]]
        d[b].update({"material":c.material.values[0], "source":c.source.values[0], "description":c.description.values[0]})
    return d
def getSummaryForLocation(dfPcsM, dfCode, codeList):
    # number of surveys
    a = len(dfPcsM)
    # the projects that surveyed this beach
    b = list(dfPcsM.project.unique())
    # The total number of objects
    c = dfPcsM.total.sum()
    # The average pieces per meter
    d = dfPcsM.pcs_m.mean()
    # first and last survey date
    e = [dfPcsM.date.values[0], dfPcsM.date.values[-1]]
    # the number of categories identified
    f = len(codeList)
    return {"Number of surveys":a, "Projects":b, "Total pieces":c, "Average pcs/m":d,
            "First sample":e[0], "Last sample":e[1], "# of MLW categories": f}

colorTopTen = ['blue', 'fuchsia', 'slategrey','maroon', 'darkgray', 'indigo',  'darksalmon',
              'palegoldenrod', 'seagreen', 'blueviolet', 'plum','lightseagreen', 'red', 'yellow', 'khaki',
               'lightcoral','maroon', 'dodgerblue', 'orange', 'thistle', 'darkslateblue', 'tomato', 'darkcyan',
               'midnightblue', 'peachpuff', 'darkred'
              ]

def getOneDayPcsM(df, date, location):
    """
    Returns one day value and all columns from the pcsM dataframe
    {columnName:value,columnNamevalue:value...}
    """
    a = df[(df.date == date) & (df.location == location)]

    b = {x:a[x].values[0] for x in a.columns}
    return b
def pointDistribution(popDist, fillXandY, samp, sampY,  pointName, distName,title, saveLocation):
    """
    Creates a PDF of a distribution and marks a point, fills the distribution to the point.
    Variables:
    popDist: An array [[x], [y]] of the data to make the distribution
    fillXandY: An array [[x], [y]] of the region to fill under the curve
    samp: The x-value of the right end of the fill area.
    sampY: The y-values of the right end of the fill area.
    aList: An array [x] with the x value of the point of interest
    distList: The data from the distribtution (the x-values)
    pointName: The name for the point (legend entry)
    distNAme: The name for the distribution (legend Entry)
    title: The title of the chart
    """

    fig, ax = plt.subplots(figsize=(6,6))

    colorsBox = ["darkblue","darkcyan","fuchsia" ]

    def makeAFilledDist(popDist,  fillXandY, samp,sampY):
        ax.plot(popDist[0], popDist[1], color=colorsBox[0],
               label="Combined PDF MCBP-SLR", alpha=0.8)
        ax.fill_between( fillXandY[0],  fillXandY[1], y2=0, color=colorsBox[0], alpha=0.2)
        ax.vlines(samp[0],0, sampY, color=colorsBox[2], linestyles='dashed',)
        return

    def plotTheseDist(aList, distList):
        ax.scatter(aList,distList, color=colorsBox[2],
                       s=140, zorder=3, edgecolor='white',)
        return

    makeAFilledDist(popDist,fillXandY, samp, sampY)
    plotTheseDist(samp, sampY)
    legend_elements = [ Line2D([0], [0], marker='o', color='w', label=pointName,
                          markerfacecolor=colorsBox[2], markersize=12),
                   Line2D([0], [0], color=colorsBox[0], lw=3, label=distName, alpha=0.4),
                 ]

    ax.set_ylabel("Probability density", labelpad=10, fontsize=14)
    ax.set_xlabel("Log(Pieces per meter of shoreline)", labelpad=10, fontsize=14)
    ax.set_title(title, pad=15)
    lgd1 = plt.legend(handles=legend_elements, loc='upper left')
    plt.savefig(saveLocation)
    plt.show()
    plt.close()
    return
def putInExcel(aDict,writer):
    """
    Component of function putListExcel
    Puts data to excel format
    """
    for k,v in aDict.items():
        v.to_excel(writer, k)

def putListExcel(aDict,bookName ):
    """
    Puts pandas data structures to excel format
    feeds a list of dicts {sheetName:data} to the putInExcel function
    """
    writer = pd.ExcelWriter(bookName)
    for data in aDict:
        putInExcel(data, writer)
    writer.save()
    return
def topTenLocations(aDict, dayPcsmDict, subject,  colorList, location):
    """
    Makes a stacked bar chart from the topTenDict:
    codeList, topTenDict = getCodesAndPcs(sortByPcs(someDfofcodevalues), 10, "code_id")
    Title info comes from getOneDayPcsM(lakeDailyPcsM, "2017-10-05", "Plage-de-St-Sulpice")
    Used to make barcharts of top ten items per day
    """
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(121)
    fig.subplots_adjust(left=0.05, bottom=0.1, right=0.7, top=0.9)
    i = 0
    values = ((k,v["pcs_m"],v["source"], v["description"]) for k,v in aDict.items())
    values = sorted(values,key=lambda x: x[1], reverse=True, )
    for n,a in enumerate(values):

        ax.bar(1, a[1], bottom=i, width=.3, label= a[2] + ": " + str(a[3]), color=colorList[n])
        i += a[1]
    ax.set_ylabel("Pieces per meter", labelpad= 10, color='black', fontsize=14)
    ax.set_title(subject + " " + dayPcsmDict["location"] + ", " + str(dayPcsmDict['date'].astype('datetime64[D]')),
                 fontsize=14, family='sans', loc="left", pad=20)
    handles, labels = ax.get_legend_handles_labels()
    lgd1 = plt.legend(handles[::-1], labels[::-1],
                      title_fontsize=12, bbox_to_anchor=(1.02, 1.02))

    plt.ylim(top= sum(v for name,v,v1, v2 in values) + 1)
    plt.xticks([], [])
    plt.savefig("graphs/barCharts/"+ location, bbox_extra_artists=(lgd1))
    plt.show()
    plt.close()
    return
def makeDescription(aDict, aList,keyOne):
    a = []
    b = {}
    for source in aList:
        c = source
        d = {source:{"pcs_m":0, "description":[], "source":source}}
        b.update(d)
    for k,v in aDict.items():
        b[v[keyOne]]["pcs_m"] += v["pcs_m"]
        b[v[keyOne]]["description"].append(v["description"])
    return b
def greaterThan(e, firstBlock, multiples, newList):
    a = e[:firstBlock]
    newList.append([", ".join(a)])
    b = e[firstBlock:]
    if len(b) < multiples:
        pass
    else:
        c = int(len(b)/multiples)
        for i in np.arange(c):
            f = i*multiples+multiples
            g = i
            d = b[i*multiples:f]
            newList.append([", ".join(d)])
            end = b[f:]
        if end:
            newList.append(end)
        else:
            pass
    return newList
def combineDescription(a):
    b = []
    for k, v in a.items():
        if v["pcs_m"] > 0:
            if len(v['description']) >1:
                it = []
                j = v["description"].copy()
                l = greaterThan(j, 1,2,it)
                v["description"] = l
        else:
            b.append(k)
    for name in b:
        a.pop(name)
    return a
def makeStrings(a):
    for k,v in a.items():
        stringList = []
        for b in v["description"]:
            if len(b) >2:
                stringList.append(b)
            else:
                stringList.append(b[0])

        c = "\n".join(stringList)
        v["description"]=c
    return a
def compareTwoLists(listOne, listTwo):
    setOne = set(l for l in listOne)
    setTwo = set(l for l in listTwo)
    notInListTwo = setOne - setTwo
    notInListOne = setTwo -setOne
    print("These are not in list two, but are in list one " + str(notInListTwo))
    print("These are not in list one, but are in list twon " + str(notInListOne))
    return

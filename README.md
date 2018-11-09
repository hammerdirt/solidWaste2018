## SolidWaste2018 -- Testing the probability of garbage
### Semester project EPFL Solid Waste Engineering

#### If you don't have an environment set up, or if you are unfamiliar with python you can see all the notebooks and use them interactiveley by clicking the badge below. The introduction note books will be coming up shortly! 

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/hammerdirt/solidWaste2018/master)


Masters students in Environmental Engineering from the [École Polytechnique Féderale de Lausanne](https://enac.epfl.ch/environmental-engineering) test the hypothesis that litter densities on Lac Léman are predicatable. The current method is based on the Probability Density Function derived from the logarithm of the pieces/meter of trash (pcs/m) from over 100 samples.

### Note on the workbooks:

1. Workbooks are incremental and intended to display results and methods.
2. As long as you have an internet connection the workbook will reflect the most recent data from the survey
3. The workboooks can be copied and the variables in the SECOND BLOCK can be changed to any location, body of water and object in the litter database

Contributions are limited to students in the group, EPFL professors and hammerdirt staff.

#### However if you want to clone this repo or provide alternative analysis methods. Go ahead and submit a pull request on a seperate notebook. We can always get better at what we do!

We use Anaconda to manage our environments:

1. The .yml file for this repo is included
2. Here are the instructions for creating an environment from a .yml file [go](https://conda.io/docs/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)

### Do not put copyrighted material in this repository:

1. This includes texts for which we (hammerdirt) have a valid license
2. Code samples from texts that are copyrighted
3. Links to copyrighted texts that have been uploaded to another server

There are plenty of resources out there. No need to steal anybodies work.



### Getting the data:
All the data for survey results is available through the hammeridrt API:

https://mwshovel.pythonanywhere.com/dirt/api_home.html

MLW code definitions are available in .csv format here:

https://mwshovel.pythonanywhere.com/static/newCriteria.csv

Check the first workbook for examples on how to acces the data(structure the URLs).

For more information contact me roger@hammerdirt.ch

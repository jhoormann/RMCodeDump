# RMCodeDump
In this repository I will put all the random bits of code that I wrote
while developing the OzDES RM code that didn't make it into the final
pipeline but will hopefully be useful to someone else.

# Files
Here are brief descriptions of all the codes contained in this
repository.  All the codes were developed using python 3.5.6.

## OzDES_Calculation.py
Here is a dump of all the functions I wrote for the OzDES RM Pipeline
(contained in OzDES_calibSpec/getPhoto/makeLC) plus some others I wrote
that are called in the other scripts presented here.

## OzDES_Plotting.py
This code specifically contains the functions which makes plots with
legible axis labels that are uniform.

## features.py
This code will look at a light curve and try to identify features which
will make it easier to recover a lag with reverberation mapping.  It
will look for significant jumps between observing seasons as well as
places where the slope of a best fit line between seasonal means changes
directions.
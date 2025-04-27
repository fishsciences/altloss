# Installation

Clone this repository and run `poetry build` in the top level directory. This
will create a directory called `dist` containing a wheel file called
`altloss-<version>-py3-none-any.whl`, which can be instaled using any python 
package manager. Once this is done, run `altloss setup` to download, compile,
and install CmdStan.

# Running the model

The model expects two input files: one file containing observations (usually
called `observations.csv`), and one containing tagging records (usually named
`tags.csv`). An example of such files, based on the 2017-2023 acoustic releases,
can be found in the `clean_data` directory.

To fit the model for fall Chinook salmon, for example, you would run:
```
 altloss fit -O clean_data/observations.csv -T clean_data/tags.csv -S FCS -o fcs_fit.csv
```
If this completes without errors, it will produce a file called `fcs_fit.csv`
containing complete histories for one million simulated fall Chinook salmon
individuals moving throughout the facility. Every row in this file contains 
a fish ID, the timestamp of a state transition for that fish, and the new state
the fish transitioned to. Sorting the records for one fish ID in chronological
order provides the complete movement history of that (simulated) fish. Almost
any statistic of interest can be computed based on this data.

To see all the other command arguments accepted, type `altloss fit --help`

# Preprocessing the acoustic data

Generally, the pre-processing of the data requires manual steps and/or changes
to the code, due to variations in the format of the data. The module `altloss.mrp`
contains functions that implement the bulk of the processing required to 
transform detections into residence observations in the format expected for the
`observations.csv` file. Assuming the "bookmarked" detections are in the same
format used in 2023, the following steps are sufficient:

1. start an ipython session
2. import the `mrp` module using `import altloss.mrp as mrp`
3. import pandas using `import pandas as pd`
4. read in the detections: `det = pd.read_csv('detections.csv')`
5. process the detections into observations: `obs = mrp.process_obs_data(det)`
6. write the ogenerated bservations to a file: `mrp.write_obs_data(obs, output='observations.csv')`

Processing the tagging records into the format expected for `tags.csv` requires 
manual inspection and processing, because the tagging records often come in multiple
files that do not have a unified format, and which can also have subtle differences,
such as fork lengths recorded in mm in one file and cm in another file.




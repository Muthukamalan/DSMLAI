# Effective Pandas
## Pattern for Data Manipulation

### Intro
- [X] Data 

### Data Structure
- [X] Summary

### Series
- [X] Index abstraction
- [X] pandas Series
- [X] NaN value
- [X] Optional Integer Support for NaN
- [X] Similar to Numpy
- [X] Categorical Data

### Series Deep Dive
- [X] Loading the Data
- [X] Series Attributes

### Operators(& Dunder Methods)
- [X] Intro
- [X] Dunder Methods
- [X] Index Alignment
- [X] Broadcasting
- [X] Iteration
- [X] Operator Methods
- [X] Chaining

### Aggregate Methods
- [X] Aggregation
- [X] Count and Mean of an Attribute
- [X] .agg and Aggregation Strings

### Conversion Methods
- [X] Automatic Conversion
- [X] Memory Usage
- [X] String and Category Types
- [X] Ordered Categories
- [X] Converting to Other Types

### Manipulation Methods
- [X] .apply and .where
- [X] If Else
- [X] Missing Data
- [X] Filling in Missing Data
- [X] Interpolating Data
- [X] Clipping 
- [X] Sorting Value
- [X] Sorting the Index
- [X] Dropping Duplicates
- [X] Ranking Data
- [X] Replacing Data
- [X] Binning

### Indexing Operations
- [X] Naming the index
- [X] Resetting the index
- [X] .loc Attributes
- [X] .iloc
- [X] heads and tails
- [X] Filtering Index Values
- [X] Reindexing

### String Manipulation
- [X] String and Objects
- [X] Categorical Strings
- [X] The .str Accessor
- [X] Searching
- [X] Splitting
- [ ] Optimizing .apply with Cython
- [X] Replacing Text

### Date and Time Manipulation
- [X] Date Theory
- [X] Loading UTC Time Data
- [X] Loading Local Time Data
- [X] Local Time to UTC
- [ ] Converting to Epochs
- [X] Manipulating Dates

### Dates in the Index
- [X] Finding the missing data
- [X] filling in missing data
- [X] intepolation
- [X] dropping missing values
- [X] shifting data
- [X] rolling average
- [X] resampling
- [X] gathering aggregate values(but keeping index)
- [ ] groupby operation
- [X] cumulative operations

### Plotting with a Series
- [X] Plotting in jupyter
- [X] .plot 
- [X] histogram
- [X] box
- [X] kde
- [X] line
- [X] line plot with multiple aggregations
- [X] bar
- [X] pie
- [ ] styling

### Categorical Manipulation
- [X] Categorical data
- [X] Frequency Counts
- [X] benefits of categories
- [X] conversion to Ordinal categories
- [X] .cat accessor
- [X] category gotchas (cons)

### Dataframes
- [ ] database and spreedsheet analogues
- [ ] simple python version
- [x] Dataframes
- [x] Construction
- [x] Dataframe Axis


### Similarities with Series and DataFrame
- [X] getting the data
- [X] view data

### Math Methods in DataFrames
- [X] Index alignment
- [X] duplicate index

### Looping and Aggregation
- [X] for loops
- [X] aggregaions
- [ ] .apply method

### Columns Types, .assign an Memory Usage
- [X] Conversion methods
- [X] memory usage

### Creating and Updating Columns
- [X] Loading the Data
- [x] more column cleanup

### Dealing with Missing and Duplicated Data
- [X] Missing data
- [X] duplicates

### Sorting Columns and Indexes
- [X] Soring columns
- [ ] Sorting columns Order
- [X] Setting and Sorting the index

### Filtering and Indexing Operations
- [X] Renaming an Index
- [X] Resetting the Index
- [X] Dataframe Indexing, Filtering & Querying
- [X] Indexing by Position

### Plotting with DataFrames
- [X] line plots
- [X] bar
- [X] scatter
- [X] area and stacked bar
- [X] column distributions with KDEs,Histograms and Boxplots


### Reshaping Dataframes with Dummies
- [X] dummy columns
- [X] undoing dummy columns

### Reshaping By Pivoting and Grouping
- [X] Using a custom Aggregation function
- [X] multiple aggregation
- [X] per column aggregations
- [X] grouping by hierarchy
- [X] grouping with functions


### More Aggregations
- [X] Aggregation while keeping rows
- [X] Filtering parts of groups


### Cross-tabulation Deep Dive
- [ ] cross-tabulation
- [ ] adding margins
- [ ] normalizing results
- [ ] hierarchical columns with cross tabulations
- [X] heatmaps

### Melting, Transpose and Stacking Data
- [X] melting data
- [X] Un-melting data
- [X] Transposing data
- [X] stacking and Unstacking
- [X] Flattening hierarchical index and columns


### Working with TimeSeries
- [X] loading the data
- [X] adding timezone info
- [X] exploring the data
- [X] slicing ts
- [X] missing ts data
- [X] exploring seasonality
- [ ] resampling data
- [ ] rules with offset aliases
- [ ] combining offset aliases
- [ ] anchored offset aliases
- [ ] resampling to finger-grain frequency
- [ ] grouping a date column with pd.Grouper



### Joining Dataframes
- [X] Adding rows to Dataframes
- [X] Adding cols to Dataframes
- [X] Joins
- [X] joins indicators
- [X] merge validations
- [ ] joining data
- [ ] validating joined data
- [ ] visualization of merged data

### Exporting data
- [X] create csv
- [ ] export to excel
- [ ] feather
- [x] sql
- [x] json

### Styling Dataframes
- [ ] Loading the Data
- [ ] Sparklines
- [ ] .style
- [ ] Formatting
- [ ] Embedding Bar plots
- [ ] Highlighting
- [ ] Heatmap and gradients
- [ ] Captions
- [ ] css properties
- [ ] stickiness and hidding
- [ ] hiding the index


### Debugging pandas
- [X] Checking if Dataframe are Equal
- [X] Debugging Chains
- [ ] Debugging Apply
- [ ] Memory Usage
- [X] Timing Info
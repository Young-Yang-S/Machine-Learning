## R Tips for R users
### Reading large table into R   From[Simplystats](https://simplystatistics.org/2011/10/07/r-workshop-reading-in-large-data-frames/)
- Get a roughly estimation of the size of your data 
Calculation formula : # rows * # columns * 8 bytes / 2^20  (MB unit)
- Read.table() <br />
The following options to ‘read.table()’ can affect R’s ability to read large tables:<br />
(1) colClasses<br />
This option takes a vector whose length is equal to the number of columns in the table. Specifying this option instead of using the default can make ‘read.table’ run MUCH faster.
In order to use this option, you have to know the of each column in your data frame. If all of the columns are “numeric”, for example, then you can just set ‘colClasses = “numeric”’.If the columns are all different classes, or perhaps you just don’t know, then you can have R do some of the work for you.<br />
You can read in just a few rows of the table and then create a vector of classes from just the few rows. For example, if I have a file called “datatable.txt”, I can read in the first 100 rows and determine the column classes from that:<br />
tab5rows <- read.table("datatable.txt", header = TRUE, nrows = 100)<br />
classes <- sapply(tab5rows, class)<br />
tabAll <- read.table("datatable.txt", header = TRUE, colClasses = classes)<br />
(2) nrows<br />
Specifying the ‘nrows’ argument doesn’t necessary make things go faster but it can help a lot with memory usage. R doesn’t know how many rows it’s going to read in so it first makes a guess, and then when it runs out of room it allocates more memory. The constant allocations can take a lot of time, and if R overestimates the amount of memory it needs, your computer might run out of memory. <br />
(3) comment.char<br />
If your file has no comments in it (e.g. lines starting with ‘#’), then setting ‘comment.char = “”’ will sometimes make ‘read.table()’ run faster. But not dramatic.<br />

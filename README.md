# ProBE:  Proportioning Privacy Budget for Complex Exploratory Decision Support

ProBE is a differentially private framework that answers complex decision support queries with strict accuracy guarantees while minimizing privacy loss.

## Pre-Requisites

1. Use the package manager [pip](https://pip.pypa.io/en/stable/) to install all ProBE dependencies.

```bash
$ pip install -r requirements.txt
```
2. Unzip file containing SQL dump for Taxi and Sales datasets ***TaxiandSalesDataDump.zip***.
3. Use MySQLWorkbench's Data Import to import SQL dump containing Taxi and Sales datasets.
4. Change database credentials/parameters in creds.py file.

***NOTE:*** *The UCI dataset is private data and thus was not included due to sharing policies in exchange for use. Special access to this dataset may be granted for review purposes only upon direct request.*


## Running Instructions
1. To run ProBE, run the following command in the directory:
```bash
$ python3 probeMain algorithm [dataset] [u=<value>] [b=<value>] [a=<value>] [t=<value>] [n=<value>]
```
where options in brackets [] are optional. Each option must not have any spaces within (e.g. only b=0.01 is acceptable). The options are explained below:

* **algorithm** options are *naive* (Naive Threshold Shift Laplace Mechanism), and *probe* (ProBE Mechanism).
* [dataset] options are *sales* and *taxi*. **The default option is taxi.**
* [u=\<value\>] provides a value for the uncertain region u for the Naive mechanism. u has to be a percentage of the data range (e.g. u=10 corresponds to 10%). **The default value is 12%.** The ProBe mechanism does not take a u value as it sets it internally.
* [b=\<value\>] provides a value for the false negative rate bound beta. **The default value is 0.05.**
* [a=\<value\>] provides a value for the false positive rate bound alpha (for ProBE only). **The default value is 0.15.**
* [t=\<value\>] provides a value for the type of complex query. The options are *0* for disjunction, *1* for conjunction and *2* for the combination of both. **The default option is 0 (disjunction).**
* [n=\<value\>] provides a value for the number of sub-queries in the complex decision support query. **The range supported is 1-6 and the default is 2.**
Below are examples of commands to run using specific parameters:
* Naive with the uncertain region u=14% and beta=0.008, with 2 sub-queries
```bash
$ python3 probeMain.py naive u=14 b=0.08
```
* ProBE with 4 sub-queries of type disjunction
```bash
$ python3 probeMain.py probe n=4
```
* ProBE with beta=0.02, alpha=0.1 and a conjunction query with 5 sub-queries
```bash
$ python3 probeMain.py probe a=0.1 b=0.02 t=1 n=5
```

***NOTE:*** *Queries are pre-defined and included in the *queries/* directory, and may be modified, but must comply with the definition of complex decision support queries as defined in the paper.*
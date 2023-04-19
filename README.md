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
$ probeMain [algorithm] [dataset] [u=<value>] [b=<value>] [m=<value>] [f=<value>] [e=<value>] [t=<value>]
```
where options in brackets [] are optional. The options are explained below:

* [algorithm] options are *tslm* (Threshold Shift Laplace Mechanism), *ppwlm* (Progressive Predicate-wise Laplace Mechanism) and *ddpwlm* (Data Dependent Predicate-wise Laplace Mechanism). **The default option is tslm.**
* [dataset] options are *sales* and *taxi*. **The default option is sales.**
* [u=\<value\>] (no spaces) provides a value for the uncertain region u. u has to be a percentage of the data range (e.g. u=10 corresponds to 10%). **The default value is 12%.**
* [b=\<value\>] (no spaces) provides a value for the false negative rate bound beta. **The default value is 0.005.**
* [m=\<value\>] (no spaces) provides a value for the number of steps for iterative algorithms (ppwlm,ddpwlm). **The default value is 4.**
* [f=\<value\>] (no spaces) provides a value for the number of fine-grained steps for the data-dependent algorithm (ddpwlm). **The default value is 3.**
* [e=\<value\>] (no spaces) provides a value for the starting privacy budget epsilon. **The default value is 0.01.**
* [t=\<value\>] (no spaces) provides a value for the type of complex query. The options are *0* for disjunction, *1* for conjunction and *2* for the combination of both. **The default option is 0 (disjunction).**

Below are examples of commands to run using specific parameters:
* ProBE TSLM with the uncertain region u=14% and beta=0.008
```bash
$ probeMain tslm u=14 b=0.008
```
* ProBE PPWLM with the number of steps m=3 and starting epsilon of e=0.0001
```bash
$ probeMain ppwlm m=3 e=0.0001
```
* ProBE DDPWLM with the number of fine-grained steps f=2, starting epsilon of e=0.1, and a combination of disjunction/conjunction query t=2
```bash
$ probeMain ddpwlm f=2 e=0.1 t=2
```

***NOTE:*** *Queries are pre-defined and included in the */queries* directory, and may be modified, but must comply with the definition of complex decision support queries as defined in the paper.*

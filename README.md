[![Build Status](https://travis-ci.org/torchnet/torchnet.svg)](https://travis-ci.org/torchnet/torchnet)

# torchnet

*torchnet* is a framework for [torch](http://torch.ch) which provides a set
of abstractions aiming at encouraging code re-use as well as encouraging
modular programming.

At the moment, *torchnet* provides four set of important classes:
  - [`Dataset`](#tntdataset): handling and pre-processing data in various ways.
  - [`Engine`](#tntengine): training/testing machine learning algorithm.
  - [`Meter`](#tntmeter): meter performance or any other quantity.
  - [`Log`](#tntlog): output performance or any other string to file / disk in a consistent manner.

For an overview of the *torchnet* framework, please also refer to [this paper](https://lvdmaaten.github.io/publications/papers/Torchnet_2016.pdf).


## Installation

Please install *torch* first, following instructions on
[torch.ch](http://torch.ch/docs/getting-started.html).  If *torch* is
already installed, make sure you have an up-to-date version of
[*argcheck*](https://github.com/torch/argcheck), otherwise you will get
weird errors at runtime.

Assuming *torch* is already installed, the *torchnet* core is only a set of
lua files, so it is straightforward to install it with *luarocks*
```
luarocks install torchnet
```

To run the MNIST example from the paper, install the `mnist` package:
```
luarocks install mnist
```

`cd` into the installed `torchnet` package directory and run:
```
th example/mnist.lua
```


## Documentation

Requiring *torchnet* returns a local variable containing all *torchnet*
class constructors.
```
local tnt = require 'torchnet'
```


### tnt.Dataset()

*torchnet* provides a variety of data containers, which can be easily
plugged between each others, allowing the user to easily concat, split,
batch, resample etc... datasets.

A instance `dataset` of a `tnt.Dataset()` implements two main methods:

  * `dataset:size()` which returns the size of the dataset.
  * `dataset:get(idx)` where `idx` is a number between 1 and the dataset size.

While it is easy to iterate over a dataset with a for loop, several
`DatasetIterator` iterators are nevertheless provided, allowing the user to
filter out some samples in an on-the-fly manner, or to parallelize easily
data fetching.

In *torchnet*, a sample returned by `dataset:get()` is supposed to be a Lua
`table`. Fields of the table can be arbitrary, even though many datasets
will only work with torch tensors.


### tnt.utils

*Torchnet* provides a set of util functions which are used all over torchnet.
<a name="utils.table.clone">
#### tnt.utils.table.clone(table)

This function do a deep copy of a table.

<a name="utils.table.merge">
#### tnt.utils.table.merge(dst, src)
```
({
   dst = table  -- 
   src = table  -- 
})
```

This function add to the destination table `dest`, the
element contained in the source table `source`.

The copy is shallow.

If a key exists in both tables, then the element in the source table
is preferred.
<a name="utils.table.foreach">
#### tnt.utils.table.foreach(tbl, closure[, recursive])
```
({
   tbl       = table     -- 
   closure   = function  -- 
  [recursive = boolean]  --  [default=false]
})
```

This function applies the function defined by `closure` to the
table `tbl`.

If `recursive` is given and set to `true`, the `closure` function
will be apply recursively to the table.
<a name="utils.table.canmergetensor">
#### tnt.utils.table.canmergetensor(tbl)

Check if a table can be merged into a tensor.
<a name="utils.table.mergetensor">
#### tnt.utils.table.mergetensor(tbl)
```
({
   tbl = table  -- 
})
```

Merge a table into a tensor in one extra dimension.

### tnt.transform

*Torchnet* provides a set of general data transformations.
These transformations are either directly on the data (e.g., normalization)
or on their structure. This is particularly handy
when manipulating [tnt.Dataset](#tnt.Dataset).

Most of the transformations are simple but can be [composed](#transform.compose) or
[merged](#transform.merge).
<a name="transform.identity">
#### transform.identity(...)

The identity transform takes any input and return it as it is.

For example, this function is useful when composing
transformations on data from multiple sources, and some of the sources
must not be transformed.
<a name = "transform.compose">
#### transform.compose(transforms)
```
({
   transforms = table  -- 
})
```

This function takes a `table` of functions and
composes them to return one transformation.

This function assumes that the table of transformations
is indexed by contiguous ordered keys starting at 1.
The transformations are composed in the ascending order.

For example, the following code:
```lua
> f = transform.compose{
        [1] = function(x) return 2*x end,
        [2] = function(x) return x + 10 end,
        foo = function(x) return x / 2 end,
        [4] = function(x) return x - x end
   }
   > f(3)
   16
```
is equivalent to compose the transformations stored in [1] and [2], i.e.,
defining the following transformation:
```lua
> f =  function(x) return 2*x + 10 end
 ```
Note that transformations stored with keys `foo` and `4` are ignored.
<a name = "transform.merge">
#### transform.merge(transforms)
```
({
   transforms = table  -- 
})
```

This function takes a `table` of transformations and
merge them into one transformation.
Once apply to an input, this transformation will produce a `table` of output,
containing the transformed input.

For example, the following code:
```lua
> f = transform.merge{
        [1] = function(x) return 2*x end,
        [2] = function(x) return x + 10 end,
        foo = function(x) return x / 2 end,
        [4] = function(x) return x - x end
   }
```
produces a function which applies a set of transformations to the same input:
```lua
> f(3)
   {
     1 : 6
     2 : 13
     foo : 1.5
     4 : 0
   }
```
<a name = "transform.tablenew">
#### transform.tablenew()

This function creates a new table of functions from an
existing table of functions.
<a name = "transform.tableapply">
#### transform.tableapply(transform)
```
({
   transform = function  -- 
})
```

This function applies a transformation to a table of input.
It return a table of output of the same size as the input.

For example, the following code:
```lua
> f = transform.tableapply(function(x) return 2*x end)
```
produces a function which multiplies any input by 2:
```lua
> f({[1] = 1, [2] = 2, foo = 3, [4] = 4})
   {
     1 : 2
     2 : 4
     foo : 6
     4 : 8
   }
```
<a name = "transform.tablemergekeys">
#### transform.tablemergekeys()

This function merges tables by key. More precisely, the input must be a
`table` of `table` and this function will reverse the table orderto
make the keys from the nested table accessible first.

For example, if the input is:
```lua
> x = { sample1 = {input = 1, target = "a"} , sample2 = {input = 2, target = "b", flag = "hard"}
```
Then apply this function will produce:
```lua
> transform.tablemergekeys(x)
{
   input :
         {
           sample1 : 1
           sample2 : 2
         }
   target :
          {
            sample1 : "a"
            sample2 : "b"
          }
   flag :
        {
           sample2: "hard"
        }
}
```
<a name = "transform.makebatch">
#### transform.makebatch([merge])
```
({
  [merge = function]  -- 
})
```

This function is used in many `tnt.Dataset` to format
samples in the format used by the `tnt.Engine`.

This function first [merges keys](#transform.tablemergekeys) to
produces a table of output. Then, transform this table into a tensor by
either using a `merge` transformation provided by the user or by
simply concatenating the table into a tensor directly.

This function uses the [compose](#transform.compose) transform to apply
successive transformations.
<a name = "transform.perm">
#### transform.perm(size)
```
({
   size = number  -- 
})
```

This function create a vector containing a permutation of the indices from 1 to `size`.
This vector is a `LongTensor` and  `size` must be a number.

Once the vector created, this function can be used to call a specific indices in it.

For example:
```lua
> p = transform.perm(3)
```
creates a function `p` which contains a permutation of indices:
```lua
> p(1)
2
> p(2)
1
> p(3)
3
```
<a name = "transform.normalize">
#### transform.normalize([threshold])
```
({
  [threshold = number]  --  [default=0]
})
```

This function normalizes data, i.e., it removes its mean and
divide it by its standard deviation.

The input must be a `Tensor`.

Once create, a `threshold` can be given (must be a number). Then,
the data will be divided by their standard deviation, only if this
deviation is greater than the `threshold`. This is handy, if the
deviation is small and deviding by it could lead to unstability.
<a name="ListDataset">
#### tnt.ListDataset(self, list, load[, path])
```
({
   self = tnt.ListDataset  -- 
   list = tds.Hash         -- 
   load = function         -- 
  [path = string]          -- 
})
```

Considering a `list` (can be a `tds.Hash`, `table` or a `torch.LongTensor`) the
i-th sample of a dataset will be returned by `load(list[i])`, where `load()` is
a closure provided by the user.

If `path` is provided, list is assumed to be a list of string, and will
each element `list[i]` will prefixed by `path/` when fed to `load()`.

Purpose: many low or medium-scale datasets can be seen as a list of files
(for example representing input samples). For this list of file, a target
can be often inferred in a simple manner.

#### tnt.ListDataset(self, filename, load[, maxload][, path])
```
({
   self     = tnt.ListDataset  -- 
   filename = string           -- 
   load     = function         -- 
  [maxload  = number]          -- 
  [path     = string]          -- 
})
```

The file specified by `filename` is interpreted as a list of strings (one
string per line). The i-th sample of a dataset will be returned by
`load(line[i])`, where `load()` is a closure provided by the user an
`line[i]` is the i-the line of `filename`.

If `path` is provided, list is assumed to be a list of string, and will
each element `list[i]` will prefixed by `path/` when fed to `load()`.

<a name = "TableDataset">
#### tnt.TableDataset(self, data)
```
({
   self = tnt.TableDataset  -- 
   data = table             -- 
})
```

`tnt.TableDataset` interfaces existing data
to torchnet. It is useful if you want to use torchnet on a small dataset.

The data must be contained in a `tds.Hash`.

`tnt.TableDataset` does a shallow copy of the data.

Data are loaded while constructing the `tnt.TableDataset`:
```lua
> a = tnt.TableDataset({1,2,3})
> print(a:size())
3
```
`tnt.TableDataset` assumes that table has contiguous keys starting at 1.
<a name="IndexedDataset">
#### tnt.IndexedDataset(self, fields[, path][, maxload][, mmap][, mmapidx])
```
{
   self    = tnt.IndexedDataset  -- 
   fields  = table               -- 
  [path    = string]             -- 
  [maxload = number]             -- 
  [mmap    = boolean]            --  [default=false]
  [mmapidx = boolean]            --  [default=false]
}
```

A `tnt.IndexedDataset()` is a data structure built upon (possibly several)
data archives containing a bunch of tensors of the same type.

See [tnt.IndexedDatasetWriter](#IndexedDatasetWriter) and
[tnt.IndexedDatasetReader](#IndexedDatasetReader) to see how to create and
read a single archive.

Purpose: large datasets (containing a lot of files) are often not very well
handled by filesystems (especially over network). `tnt.IndexedDataset`
provides a convenient and efficient way to bundle them into a single
archive file, associated with an indexed file.

If `path` is provided, then `fields` must be a Lua array (keys being
numbers), where values are string representing a filename prefix to a
(index,archive) pair.  In other word `path/field.{idx,bin}` must exist. The
i-th sample returned by this dataset will be a table containing each field
as key, and a tensor found at the corresponding archive at index i.

If `path` is not provided, then `fields` must be a Lua hash. Each key
represents sample fields and the corresponding value must be a table
containing the keys `idx` (for the index filename path) and `bin` (for the
archive filename path).

If provided (and positive), `maxload` limits the dataset size to the
specified size.

Archives and/or indexes can also be memory mapped with the `mmap` and
`mmapidx` flags.

<a name="IndexedDatasetWriter">
##### tnt.IndexedDatasetWriter(self, indexfilename, datafilename, type)
```
({
   self          = tnt.IndexedDatasetWriter  -- 
   indexfilename = string                    -- 
   datafilename  = string                    -- 
   type          = string                    -- 
})
```

Creates a (archive,index) file pair. The archive will contain tensors of the same specified `type`.

`type` must be a string chosen in {`byte`, `char`, `short`, `int`, `long`, `float`, `double` or `table`}.

`indexfilename` is the full path to the index file to be created.
`datafilename` is the full path to the data archive file to be created.

Tensors are added to the archive with [add()](#IndexedDataset.add).

Note that you *must* call [close()](#IndexedDataset.close) to ensure all
data is written on disk and to create the index file.

The type `table` is special: data will be stored into a CharTensor,
serialized from a Lua table object. IndexedDatasetReader will then
deserialize the CharTensor into a table at read time. This allows storing
heterogenous data easily into an IndexedDataset.

##### tnt.IndexedDatasetWriter(self, indexfilename, datafilename)
```
({
   self          = tnt.IndexedDatasetWriter  -- 
   indexfilename = string                    -- 
   datafilename  = string                    -- 
})
```

Opens an existing (archive,index) file pair for appending. The tensor type is inferred from the provided
index file.

`indexfilename` is the full path to the index file to be opened.
`datafilename` is the full path to the data archive file to be opened.

<a name="IndexedDatasetWriter.add">
###### tnt.IndexedDatasetWriter.add(self, tensor)
```
({
   self   = tnt.IndexedDatasetWriter  -- 
   tensor = torch.*Tensor             -- 
})
```

Add a tensor to the archive and record its index position. The tensor type must of the same type
than the one specified at the creation of the `tnt.IndexedDatasetWriter`.
###### tnt.IndexedDatasetWriter.add(self, filename)
```
({
   self     = tnt.IndexedDatasetWriter  -- 
   filename = string                    -- 
})
```

Convenience method which given a `filename` will open the corresponding
file in `binary` mode, and reads all data in there as if it was of the type
specified at the `tnt.IndexedDatasetWriter` construction.  A corresponding
tensor is then added to the archive/index pair.
###### tnt.IndexedDatasetWriter.add(self, table)
```
(
   self  = tnt.IndexedDatasetWriter  -- 
   table = table                     -- 
)
```

Convenience method only available for `table` type IndexedDataset.
The table will be serialized into a CharTensor.

###### tnt.IndexedDatasetWriter.add(self)
```
({
   self = tnt.IndexedDatasetWriter  -- 
})
```

Finalize the index, and Close the archive/index filename pair. This method
must be called to ensure the index is written and all the archive data is
flushed on disk.
<a name="IndexedDatasetReader">
##### tnt.IndexedDatasetReader(self, indexfilename, datafilename[, mmap][, mmapidx])
```
({
   self          = tnt.IndexedDatasetReader  -- 
   indexfilename = string                    -- 
   datafilename  = string                    -- 
  [mmap          = boolean]                  --  [default=false]
  [mmapidx       = boolean]                  --  [default=false]
})
```

Reads an archive/index pair previously created by
[tnt.IndexedDatasetWriter](#IndexedDatasetWriter).

`indexfilename` is the full path to the index file.
`datafilename` is the full path to the archive file.

Memory mapping can be specified for both the archive and index through the
optional `mmap` and `mmapidx` flags.

<a name="IndexedDatasetReader.size">
###### tnt.IndexedDatasetReader.size(self)

Returns the number of tensors present in the archive.
<a name="IndexedDatasetReader.get">
###### tnt.IndexedDatasetReader.get(self, index)

Returns the tensor at the specified `index` in the archive.
<a name="TransformDataset">
#### tnt.TransformDataset(self, dataset, transform[, key])
```
({
   self      = tnt.TransformDataset  -- 
   dataset   = tnt.Dataset           -- 
   transform = function              -- 
  [key       = string]               -- 
})
```

Given a closure `transform()`, and a `dataset`, `tnt.TransformDataset`
applies the closure in an on-the-fly manner when querying a sample with
`tnt.Dataset:get()`.

If key is provided, the closure is applied to the sample field specified
by `key` (only). The closure must return the new corresponding field value.

If key is not provided, the closure is applied on the full sample. The
closure must return the new sample table.

The size of the new dataset is equal to the size of the underlying `dataset`.

Purpose: when performing pre-processing operations, it is convenient to be
able to perform on-the-fly transformations to a
dataset.
<a name="TransformDataset">
#### tnt.TransformDataset(self, dataset, transforms)
```
({
   self       = tnt.TransformDataset  -- 
   dataset    = tnt.Dataset           -- 
   transforms = table                 -- 
})
```

Given a set of closures and a `dataset`, `tnt.TransformDataset` applies
these closures in an on-the-fly manner when querying a sample with
`tnt.Dataset:get()`.

Closures are provided in `transforms`, a Lua table, where a (key,value)
pair represents a (sample field name, corresponding closure to be applied
to the field name).

Each closure must return the new value of the corresponding field.
<a name="BatchDataset">
#### tnt.BatchDataset(self, dataset, batchsize[, perm][, merge][, policy])
```
({
   self      = tnt.BatchDataset  -- 
   dataset   = tnt.Dataset       -- 
   batchsize = number            -- 
  [perm      = function]         --  [has default value]
  [merge     = function]         -- 
  [policy    = string]           --  [default=include-last]
})
```

Given a `dataset`, `tnt.BatchDataset` merges samples from this dataset to
form a new sample which can be interpreted as a batch (of size
`batchsize`).

The `merge` function controls how the batch is performed. It is a closure
taking a Lua array as input containing all occurrences (for a given batch)
of a field of the sample, and returning the aggregated version of these
occurrences. By default the occurrences are supposed to be tensors, and
they aggregated along the first dimension.

More formally, if the i-th sample of the underlying dataset is written as:
```lua
{input=<input_i>, target=<target_i>}
```
assuming only two fields `input` and `target` in the sample, then `merge()`
will be passed tables of the form:
```lua
{<input_i_1>, <input_i_2>, ... <input_i_n>}
```
or
```lua
{<target_i_1>, <target_i_2>, ... <target_i_n>}
```
with `n` being the batch size.

It is often important to shuffle examples while performing the batch
operation. `perm(idx, size)` is a closure which returns the shuffled index
of the sample at position `idx` in the underlying dataset. For convenience,
the `size` of the underlying dataset is also passed to the closure. By
default, the closure is the identity.

The underlying dataset size might or might not be always divisible by
`batchsize`.  The optional `policy` string specify how to handle corner
cases:
  - `include-last` makes sure all samples of the underlying dataset will be seen, batches will be of size equal or inferior to `batchsize`.
  - `skip-last` will skip last examples of the underlying dataset if its size is not properly divisible. Batches will be always of size equal to `batchsize`.
  - `divisible-only` will raise an error if the underlying dataset has not a size divisible by `batchsize`.

Purpose: the concept of batch is problem dependent. In *torchnet*, it is up
to the user to interpret a sample as a batch or not. When one wants to
assemble samples from an existing dataset into a batch, then
`tnt.BatchDataset` is suited for the job. Sometimes it is however more
convenient to write a dataset from scratch providing "batched" samples.
<a name="CoroutineBatchDataset">
#### tnt.CoroutineBatchDataset(self, dataset, batchsize[, perm][, merge][, policy])
```
({
   self      = tnt.CoroutineBatchDataset  -- 
   dataset   = tnt.Dataset                -- 
   batchsize = number                     -- 
  [perm      = function]                  --  [has default value]
  [merge     = function]                  -- 
  [policy    = string]                    --  [default=include-last]
})
```

Given a `dataset`, `tnt.CoroutineBatchDataset` merges samples from this dataset
to form a new sample which can be interpreted as a batch (of size `batchsize`).

It behaves the same and has the same arguments as `tnt.BatchDataset` (see the
documentation there for additional details), with one important distinction:
it allows the underlying dataset to postpone returning the individual samples
once by doing a call to `coroutine.yield()` (from the underlying dataset).

This is useful when using datasets that are inefficient or slow when they need
to provide the required sample immediately after a call to `dataset:get()`. The
general pattern of code in the underlying `dataset:get()` would be:

```lua
FooDataset.get = function(self, idx)
   prepare(idx)  -- stores sample in self.__data[idx]
   coroutine.yield()
   return self.__data[idx]
end
```

Herein, the function `prepare(idx)` can implement, for instance, a buffering of
indices before actually fetching them.
<a name="ConcatDataset">
#### tnt.ConcatDataset(self, datasets)
```
{
   self     = tnt.ConcatDataset  -- 
   datasets = table              -- 
}
```

Given a Lua array (`datasets`) of [tnt.Dataset](#Dataset), concatenates
them into a single dataset.  The size of the new dataset is the sum of the
underlying dataset sizes.

Purpose: useful to assemble different existing datasets, possibly
large-scale datasets as the concatenation operation is done in an
on-the-fly manner.
<a name="ResampleDataset">
#### tnt.ResampleDataset(self, dataset[, sampler][, size])

Given a `dataset`, creates a new dataset which will (re-)sample from this
underlying dataset using the provided `sampler(dataset, idx)` closure.

If `size` is provided, then the newly created dataset will have the
specified `size`, which might be different than the underlying dataset
size.

If `size` is not provided, then the new dataset will have the same size
than the underlying one.

By default `sampler(dataset, idx)` is the identity, simply `return`ing `idx`.
`dataset` corresponds to the underlying dataset provided at construction, and
`idx` may take a value between 1 to `size`. It must return an index in the range
acceptable for the underlying dataset.

Purpose: shuffling data, re-weighting samples, getting a subset of the
data. Note that an important sub-class is ([tnt.ShuffleDataset](#ShuffleDataset)),
provided for convenience.
<a name="ShuffleDataset">
#### tnt.ShuffleDataset(self, dataset[, size][, replacement])
```
({
   self        = tnt.ShuffleDataset  -- 
   dataset     = tnt.Dataset         -- 
  [size        = number]             -- 
  [replacement = boolean]            --  [default=false]
})
```

`tnt.ShuffleDataset` is a sub-class of
[tnt.ResampleDataset](#ResampleDataset) provided for convenience.

It samples uniformly from the given `dataset` with, or without
`replacement`. The chosen partition can be redrawn by calling
[resample()](#ShuffleDataset.resample).

If `replacement` is `true`, then the specified `size` may be larger than
the underlying `dataset`.

If `size` is not provided, then the new dataset size will be equal to the
underlying `dataset` size.

Purpose: the easiest way to shuffle a dataset!
<a name="ShuffleDataset.resample">
##### tnt.ShuffleDataset.resample(self)

The permutation associated to `tnt.ShuffleDataset` is fixed, such that two
calls to the same index will return the same sample from the underlying
dataset.

Call `resample()` to draw randomly a new permutation.
<a name="SplitDataset">
#### tnt.SplitDataset(self, dataset, partitions)
```
({
   self       = tnt.SplitDataset  -- 
   dataset    = tnt.Dataset       -- 
   partitions = table             -- 
})
```

Partition a given `dataset`, according to the specified `partitions`.  Use
the method [select()](#SplitDataset.select) to select the current partition
in use.

The Lua hash table `partitions` is of the form (key, value) where key is a
user-chosen string naming the partition, and value is a number representing
the weight (in size) of the corresponding partition.

The sum of the partition weights may or may not sum to one
(`tnt.SplitDataset` will make them sum to one!).

Partionning is achieved linearly (no shuffling). See
[tnt.ShuffleDataset](#ShuffleDataset) if you want to shuffle the dataset
before partitioning.

Purpose: useful in machine learning to perform validation procedures.
<a name="SplitDataset.select">
##### tnt.SplitDataset.select(self, partition)
```
({
   self      = tnt.SplitDataset  -- 
   partition = string            -- 
})
```

Switch the current partition in use to the one specified by `partition`,
which must be a string corresponding to one of the names provided at
construction.

The current dataset size changes accordingly, as well as the samples returned
by the `get()` method.
### Dataset Iterators

It is easy to iterate over datasets using a for loop. However, sometimes
one wants to filter out samples in a on-the-fly manner or thread sample fetching.

Iterators are here for this particular cases. In general, refrain writing
iterators for handling custom cases, and write instead a `tnt.Dataset`

Iterators implement two methods:

  * `run()` which returns a Lua iterator usable in a for loop.
  * `exec(funcname, ...)` which execute a given funcname on the underlying dataset.

Typical usage is achieved with a for loop:
```lua
for sample in iterator:run() do
  <do something with sample>
end
```

Iterators implement the `__call` event, so one might also use the `()` operator:
```lua
for sample in iterator() do
  <do something with sample>
end
```

<a name="DatasetIterator">
#### tnt.DatasetIterator(self, dataset[, perm][, filter][, transform])
```
({
   self      = tnt.DatasetIterator  -- 
   dataset   = tnt.Dataset          -- 
  [perm      = function]            --  [has default value]
  [filter    = function]            --  [has default value]
  [transform = function]            --  [has default value]
})
```

The default dataset iterator.

`perm(idx)` is a permutation used to shuffle the examples. If shuffling
is needed, one can use this closure, or (better) use
[tnt.ShuffleDataset](#ShuffleDataset) on the underlying dataset.

`filter(sample)` is a closure which returns `true` if the given sample
should be considered or `false` if not.

`transform(sample)` is a closure which can perform online transformation of
samples. It returns a modified version of the given `sample`. It is the
identity by default. It is often more interesting to use
[tnt.TransformDataset](#TransformDataset) for that purpose.
<a name="DatasetIterator.exec">
#### tnt.DatasetIterator.exec(tnt.DatasetIterator, name, ...)

Execute the given method `name` on the underlying dataset, passing it the
subsequent arguments, and returns what the `name` method returns.
<a name="ParallelDatasetIterator">
#### tnt.ParallelDatasetIterator(self[, init], closure, nthread[, perm][, filter][, transform][, ordered])
```
({
   self      = tnt.ParallelDatasetIterator  -- 
  [init      = function]                    --  [has default value]
   closure   = function                     -- 
   nthread   = number                       -- 
  [perm      = function]                    --  [has default value]
  [filter    = function]                    --  [has default value]
  [transform = function]                    --  [has default value]
  [ordered   = boolean]                     --  [default=false]
})
```

Allows to iterate over a dataset in a thread
manner. `tnt.ParallelDatasetIterator:run()` guarantees that all samples
will be seen, but does not guarantee the order unless `ordered` is set to true.

The purpose of this class is to have a zero pre-processing cost.
When reading datasets on the fly from
disk (not loading them fully in memory), or performing complex
pre-processing this can be of interest.

The number of threads used to parallelize is specified by `nthread`.

`init(threadid)` (where threadid=1..nthread) is a closure which may
initialize the specified thread as needed, if needed. It is doing nothing
by default.

`closure(threadid)` will be called on each thread and must return a
`tnt.Dataset` instance.

`perm(idx)` is a permutation used to shuffle the examples. If shuffling is
needed, one can use this closure, or (better) use
[tnt.ShuffleDataset](#ShuffleDataset) on the underlying dataset (returned by
`closure()`).

`filter(sample)` is a closure which returns `true` if the given sample
should be considered or `false` if not. Note that filter is called _after_
fetching the data in a threaded manner.

`transform(sample)` is a function which maps the given sample to a new value.
This transformation occurs before filtering.

When `ordered` is set to true the ordering of samples returned by the iterator
is guaranteed. This option is particularly useful for repeatable experiments.
By default `ordered` is false, which means that order is not guaranteed by
`run()` (though often the ordering is similar in practice).

A common error raised by this dataset is when `closure()` is not
serializable. Make sure that all [upvalues](http://www.lua.org/pil/27.3.3.html) of `closure()` are
serializable. It is recommended to avoid [upvalues](http://www.lua.org/pil/27.3.3.html) at all cost,
and to make sure you require all the appropriate torch packages needed to (de-)serialize
`closure()` in the `init()` function.


For more information, check out the [threads package](https://github.com/torch/threads),
on which `tnt.ParallelDatasetIterator` relies.
<a name="ParallelDatasetIterator.execSingle">
#### tnt.ParallelDatasetIterator.execSingle(tnt.DatasetIterator, name, ...)

Execute the given method `name` on the dataset corresponding to the first
available thread, passing it the subsequent arguments, and returns what the
`name` method returns.

For example:
```lua
  local iterator = tnt.ParallelDatasetIterator{...}
  print(iterator:execSingle("size"))
```
will print the size of the dataset loaded in the first available thread.
<a name="ParallelDatasetIterator.exec">
#### tnt.ParallelDatasetIterator.exec(tnt.DatasetIterator, name, ...)

Execute the given method `name` on the underlying datasets in each thread,
passing to each of them the subsequent arguments, and returns a table
of what the `name` method returns for each thread.

For example:
```lua
  local iterator = tnt.ParallelDatasetIterator{...}
  for _, v in pairs(iterator:exec("size")) do
      print(v)
  end
```
will print the size of the datasets loaded in each thread.

### tnt.Engine

In experimenting with different models and datasets, the underlying training
procedure is often the same. The `Engine` module provides the boilerplate logic
necessary for the training and testing of models. This might include conducting
the interaction between model (`nn.Module`), `tnt.DatasetIterator`s,
`nn.Criterion`s, and `tnt.Meter`s.

An instance `engine` of a `tnt.Engine()` implements two main methods:

  * `engine:train()`, for training the model on data
        (i.e. sample data, forward prop, backward prop).
  * `engine:test()`,  for evaluating a model on data
        (optionally with respect to a `nn.Criterion`).

The `Engine` can be implemented for any common underlying training and testing
procedure involving a model and data. It can also be designed to allow user
control after certain events such as forward prop, criterion evaluation, or the
end of an epoch, by using coroutines (see `tnt.SGDEngine`).


### tnt.SGDEngine

The `SGDEngine` module implements the Stochastic Gradient Descent training
procedure in `train`, including data sampling, forward prop, back prop, and
parameter updates. It also operates as a coroutine allowing a user control
 (i.e. increment some sort of `tnt.Meter`) at events such as 'start',
'start-epoch', 'forward', 'forward-criterion', 'backward', etc.
The available hooks are the following:
```lua
hooks = {
   ['onStart']             = function() end, -- Right before training
   ['onStartEpoch']        = function() end, -- Before new epoch
   ['onSample']            = function() end, -- After getting a sample
   ['onForward']           = function() end, -- After model:forward
   ['onForwardCriterion']  = function() end, -- After criterion:forward
   ['onBackwardCriterion'] = function() end, -- After criterion:backward
   ['onBackward']          = function() end, -- After model:backward
   ['onUpdate']            = function() end, -- After UpdateParameters
   ['onEndEpoch']          = function() end, -- Right before completing epoch
   ['onEnd']               = function() end, -- After training
}
```
To specify a new closure for a given hook, we can access to it with
`engine.hooks.<onEvent>`. For example, we could reset a `Meter` before every
epoch by:
```lua
local engine = tnt.SGDEngine()
local meter  = tnt.AverageValueMeter()
engine.hooks.onStartEpoch = function(state)
   meter:reset()
end
```

Accordingly, `train` requires a network (`nn.Module`), a criterion expressing the
loss function (`nn.Criterion`), a dataset iterator (`tnt.DatasetIterator`), and a
learning rate, at the minimum. The `test` function allows for simple evaluation
of a model on a dataset.

A `state` is maintained for external access to outputs and parameters of modules
as well as sampled data. The content of the `state` table is the following, where
the passed values come from the arguments of `engine:train()`:
```lua
state = {
   ['network']     = network,
   ['criterion']   = criterion,
   ['iterator']    = iterator,
   ['lr']          = lr,
   ['lrcriterion'] = lrcriterion,
   ['maxepoch']    = maxepoch,
   ['sample']      = {},
   ['epoch']       = 0, -- epoch done so far
   ['t']           = 0, -- samples seen so far
   ['training']    = true
}
```

### tnt.OptimEngine

The `OptimEngine` module wraps the optimization functions from
https://github.com/torch/optim. At the start of training, the engine will call
`getParameters` on the provided network.

The `train` method requires the following parameters in addition to the
`SGDEngine.train` parameters:

  * `optimMethod` the optimization function (e.g `optim.sgd`)
  * `config` a table with configuration parameters for the optimizer

Example:
```lua
  local engine = tnt.OptimEngine()
  engine:train{
     network = model,
     criterion = criterion,
     iterator = iterator,
     optimMethod = optim.sgd,
     config = {
        learningRate = 0.1,
        momentum = 0.9,
     },
  }
```
### tnt.Meter

When training a model, you generally would like to measure how the model is
performing. Specifically, you may want to measure the average processing time
required per batch of data, the classification error or AUC of a classifier a
validation set, or the precision@k of a retrieval model.

Meters provide a standardized way to measure a range of different measures,
which makes it easy to measure a wide range of properties of your models.

Nearly all meters (except `tnt.TimeMeter`) implement three methods:

   * `add()` which adds an observation to the meter.
   * `value()` which returns the value of the meter, taking into account all observations.
   * `reset()` which removes all previously added observations, resetting the meter.

The exact input arguments to the `add()` method vary depending on the meter.
Most meters define the method as `add(output, target)`, where `output` is the
output produced by the model and `target` is the ground-truth label of the data.

The `value()` method is parameterless for most meters, but for measures that
have a parameter (such as the k parameter in precision@k), they may take an
input argument.

An example of a typical usage of a meter is as follows:
```lua
local meter = tnt.<Measure>Meter()  -- initialize meter
for state, event in tnt.<Optimization>Engine:train{
   network   = network,
   criterion = criterion,
   iterator  = iterator,
} do
  if state == 'start-epoch' then
     meter:reset()  -- reset meter
  elseif state == 'forward-criterion' then
     meter:add(state.network.output, sample.target)  -- add value to meter
  elseif state == 'end-epoch' then
     print('value of meter:' .. meter:value())  -- get value of meter
  end
end
```
<a name="APMeter">
#### tnt.APMeter(self)
```
({
   self = tnt.APMeter  -- 
})
```

The `tnt.APMeter` measures the average precision per class.

The `tnt.APMeter` is designed to operate on `NxK` Tensors `output` and `target`,
where (1) the `output` contains model output scores for `N` examples and `K`
classes that ought to be higher when the model is more convinced that the
example should be positively labeled, and smaller when the model believes the
example should be negatively labeled (for instance, the output of a sigmoid
function); and (2) the `target` contains only values 0 (for negative examples)
and 1 (for positive examples).

The `tnt.APMeter` has no parameters to be set.
<a name="AverageValueMeter">
#### tnt.AverageValueMeter(self)
```
({
   self = tnt.AverageValueMeter  -- 
})
```

The `tnt.AverageValueMeter` measures and returns the average value and the
standard deviation of any collection of numbers that are `add`ed to it. It is
useful, for instance, to measure the average loss over a collection of examples.

The `add()` function expects as input a Lua number `value`, which is the value
that needs to be added to the list of values to average. It also takes as input
an optional parameter `n` that assigns a weight to `value` in the average, in
order to facilitate computing weighted averages (default = 1).

The `tnt.AverageValueMeter` has no parameters to be set at initialization time.
<a name="AUCMeter">
#### tnt.AUCMeter(self)
```
({
   self = tnt.AUCMeter  -- 
})
```

The `tnt.AUCMeter` measures the area under the receiver-operating characteristic
(ROC) curve for binary classification problems. The area under the curve (AUC)
can be interpreted as the probability that, given a randomly selected positive
example and a randomly selected negative example, the positive example is
assigned a higher score by the classification model than the negative example.

The `tnt.AUCMeter` is designed to operate on one-dimensional Tensors `output`
and `target`, where (1) the `output` contains model output scores that ought to
be higher when the model is more convinced that the example should be positively
labeled, and smaller when the model believes the example should be negatively
labeled (for instance, the output of a signoid function); and (2) the `target`
contains only values 0 (for negative examples) and 1 (for positive examples).

The `tnt.AUCMeter` has no parameters to be set.
<a name="ConfusionMeter">
#### tnt.ConfusionMeter(self, k[, normalized])
```
{
   self       = tnt.ConfusionMeter  -- 
   k          = number              -- 
  [normalized = boolean]            --  [default=false]
}
```

The `tnt.ConfusionMeter` constructs a confusion matrix for a multi-class
classification problems. It does not support multi-label, multi-class problems:
for such problems, please use `tnt.MultiLabelConfusionMeter`.

At initialization time, the `k` parameter that indicates the number
of classes in the classification problem under consideration must be specified.
Additionally, an optional parameter `normalized` (default = `false`) may be
specified that determines whether or not the confusion matrix is normalized
(that is, it contains percentages) or not (that is, it contains counts).

The `add(output, target)` method takes as input an NxK tensor `output` that
contains the output scores obtained from the model for N examples and K classes,
and a corresponding N-tensor or NxK-tensor `target` that provides the targets
for the N examples. When `target` is an N-tensor, the targets are assumed to be
integer values between 1 and K. When target is an NxK-tensor, the targets are
assumed to be provided as one-hot vectors (that is, vectors that contain only
zeros and a single one at the location of the target value to be encoded).

The `value()` method has no parameters and returns the confusion matrix in a
KxK tensor. In the confusion matrix, rows correspond to ground-truth targets and
columns correspond to predicted targets.
<a name="mAPMeter">
#### tnt.mAPMeter(self)
```
({
   self = tnt.mAPMeter  -- 
})
```

The `tnt.mAPMeter` measures the mean average precision over all classes.

The `tnt.mAPMeter` is designed to operate on `NxK` Tensors `output` and `target`
where (1) the `output` contains model output scores for `N` examples and `K`
classes that ought to be higher when the model is more convinced that the
example should be positively labeled, and smaller when the model believes the
example should be negatively labeled (for instance, the output of a sigmoid
function); and (2) the `target` contains only values 0 (for negative examples)
and 1 (for positive examples).

The `tnt.mAPMeter` has no parameters to be set.
<a name="MultiLabelConfusionMeter">
#### tnt.MultiLabelConfusionMeter(self, k[, normalized])
```
{
   self       = tnt.MultiLabelConfusionMeter  -- 
   k          = number                        -- 
  [normalized = boolean]                      --  [default=true]
}
```

The `tnt.MultiLabelConfusionMeter` constructs a confusion matrix for multi-
label, multi-class classification problems. In constructing the confusion
matrix, the number of positive predictions is assumed to be equal to the number
of positive labels in the ground-truth. Correct predictions (that is, labels in
the prediction set that are also in the ground-truth set) are added to the
diagonal of the confusion matrix. Incorrect predictions (that is, labels in the
prediction set that are not in the ground-truth set) are equally divided over
all non-predicted labels in the ground-truth set.

At initialization time, the `k` parameter that indicates the number
of classes in the classification problem under consideration must be specified.
Additionally, an optional parameter `normalized` (default = `false`) may be
specified that determines whether or not the confusion matrix is normalized
(that is, it contains percentages) or not (that is, it contains counts).

The `add(output, target)` method takes as input an NxK tensor `output` that
contains the output scores obtained from the model for N examples and K classes,
and a corresponding NxK-tensor `target` that provides the targets for the N
examples using one-hot vectors (that is, vectors that contain only zeros and a
single one at the location of the target value to be encoded).

The `value()` method has no parameters and returns the confusion matrix in a
KxK tensor. In the confusion matrix, rows correspond to ground-truth targets and
columns correspond to predicted targets.
<a name="ClassErrorMeter">
#### tnt.ClassErrorMeter(self[, topk][, accuracy])
```
{
   self     = tnt.ClassErrorMeter  -- 
  [topk     = table]               --  [has default value]
  [accuracy = boolean]             --  [default=false]
}
```

The `tnt.ClassErrorMeter` measures the classification error (in %) of
classification models (zero-one loss). The meter can also measure the error of
predicting the correct label among the top-k scoring labels (for instance, in
the Imagenet competition, one generally measures classification@5 errors).

At initialization time, it takes to optional parameters: (1) a table
`topk` that contains the values at which the classification@k errors should be
measures (default = {1}); and (2) a boolean `accuracy` that makes the meter
output accuracies instead of errors (accuracy = 1 - error).

The `add(output, target)` method takes as input an NxK-tensor `output` that
contains the output scores for each of the N examples and each of the K classes,
and an N-tensor `target` that contains the targets corresponding to each of the
N examples (targets are integers between 1 and K). If only one example is
`add`ed, `output` may also be a K-tensor and target a 1-tensor.

Please note that `topk` (if specified) may not contain values larger than K.

The `value()` returns a table with the classification@k errors for all values
at k that were specified in `topk` at initialization time. Alternatively,
`value(k)` returns the classification@k error as a number; only values of `k`
that were element of `topk` are allowed. If `accuracy` was set to `true` at
initialization time, the `value()` method returns accuracies instead of errors.
<a name="TimeMeter">
#### tnt.TimeMeter(self[, unit])
```
({
   self = tnt.TimeMeter  -- 
  [unit = boolean]       --  [default=false]
})
```

The `tnt.TimeMeter` is designed to measure the time between events and can be
used to measure, for instance, the average processing time per batch of data.
It is different from most other meters in terms of the methods it provides:

At initialization time, an optional boolean parameter `unit` may be provided
(default = `false`). When set to `true`, the value returned by the meter
will be divided by the number of times that the `incUnit()` method is called.
This allows the user to compute, for instance, the average processing time per
batch by simply calling the `incUnit()` method after processing a batch.

The `tnt.TimeMeter` provides the following methods:

   * `reset()` resets the timer, setting the timer and unit counter to zero.
   * `stop()` stops the timer.
   * `resume()` resumes the timer.
   * `incUnit()` increments the unit counter by one.
   * `value()` returns the time passed since the last `reset()`; divided by the counter value when `unit=true`.

<a name="PrecisionAtKMeter">
#### tnt.PrecisionAtKMeter(self[, topk][, dim][, online])
```
{
   self   = tnt.PrecisionAtKMeter  -- 
  [topk   = table]                 --  [has default value]
  [dim    = number]                --  [default=2]
  [online = boolean]               --  [default=false]
}
```

The `tnt.PrecisionAtKMeter` measures the precision@k of ranking methods at pre-
specified levels k. The precision@k is the percentage of the k front-ranked
items according to the model that is in the list of correct (positive) targets.

At initialization time, a table `topk` may be given as input that specifies the
levels k at which the precision@k will be measures (default = `{10}`). In
addition, a number `dim` may be provided that specifies over which dimension the
precision@k should be computed (default = 2), and a boolean `online` may be
specified that indicates whether we see all inputs along dimension `dim` at once
(default = `false`).

The `add(output, target)` method takes two inputs. In the default mode (`dim=2`
and `online=false`), the inputs mean:
   * A NxC tensor that for each of the N examples (queries) contains a score
     indicating to what extent each of the C classes (documents) is relevant to
     the query, according to the model.
   * A binary NxC `target` tensor that encodes which of the C classes
     (documents) are actually relevant to the the N-th input (query). For
     instance, a row of {0, 1, 0, 1} indicates that the example is associated
     with classes 2 and 4.

The result of setting `dim` to `1` is identical to transposing the tensors
`output` and `target` in the above. The result of setting `online=true` is that
the function assumes that it is not the number of queries `N` that is growing
with repeated calls to `add()`, but the number of candidate documents `C`. (Use
this mode in scenarios where `C` is large but `N` is small.)

The `value()` method returns a table that contains the precision@k (that is, the
percentage of targets predicted correctly) at the cutoff levels in `topk` that
were specified at initialization time. Alternatively, the precision@k at
a specific level k can be obtained by calling `value(k)`. Note that the level
`k` should be an element of the table `topk` specified at initialization time.

Please note that the maximum value in `topk` cannot be higher than the total
number of classes (documents).
<a name="RecallMeter">
#### tnt.RecallMeter(self[, threshold][, perclass])
```
{
   self      = tnt.RecallMeter  -- 
  [threshold = table]           --  [has default value]
  [perclass  = boolean]         --  [default=false]
}
```

The `tnt.RecallMeter` measures the recall of ranking methods at pre-
specified thresholds. The recall is the percentage of the correct (positive)
targets that is in the list of positively labeled items according to the model.

At initialization time, the `tnt.RecallMeter` provides two optional
parameters. The first parameter is a table `threshold` that contains all
thresholds at which the recall is measured (default = {0.5}). Thresholds
should be numbers between 0 and 1. The second parameter is a boolean `perclass`
that makes the meter measure the recall per class when set to `true`
(default = `false`). When `perclass` is set to `false`, the recall is simply
averaged over all examples.

The `add(output, target)` method takes two inputs:
   * A NxK tensor that for each of the N examples indicates the probability
     of the example belonging to each of the K classes, according to the model.
     The probabilities should sum to one over all classes; that is, the row sums
     of `output` should all be one.
   * A binary NxK `target` tensor that encodes which of the K classes
     are associated with the N-th input. For instance, a row of {0, 1, 0, 1}
     indicates that the example is associated with classes 2 and 4.

The `value()` method returns a table containing the recall of the model
predictions measured at the `threshold`s specified at initialization time. The
`value(t)` method returns the recall at a particular threshold `t`. Note that
this threshold `t` should be an element of the `threshold` table specified at
initialization time of the meter.
<a name="PrecisionMeter">
#### tnt.PrecisionMeter(self[, threshold][, perclass])
```
{
   self      = tnt.PrecisionMeter  -- 
  [threshold = table]              --  [has default value]
  [perclass  = boolean]            --  [default=false]
}
```

The `tnt.PrecisionMeter` measures the precision of ranking methods at pre-
specified thresholds. The precision is the percentage of the positively labeled
items according to the model that is in the list of correct (positive) targets.

At initialization time, the `tnt.PrecisionMeter` provides two optional
parameters. The first parameter is a table `threshold` that contains all
thresholds at which the precision is measured (default = {0.5}). Thresholds
should be numbers between 0 and 1. The second parameter is a boolean `perclass`
that makes the meter measure the precision per class when set to `true`
(default = `false`). When `perclass` is set to `false`, the precision is simply
averaged over all examples.

The `add(output, target)` method takes two inputs:
   * A NxK tensor that for each of the N examples indicates the probability
     of the example belonging to each of the K classes, according to the model.
     The probabilities should sum to one over all classes; that is, the row sums
     of `output` should all be one.
   * A binary NxK `target` tensor that encodes which of the K classes
     are associated with the N-th input. For instance, a row of {0, 1, 0, 1}
     indicates that the example is associated with classes 2 and 4.

The `value()` method returns a table containing the precision of the model
predictions measured at the `threshold`s specified at initialization time. The
`value(t)` method returns the precision at a particular threshold `t`. Note that
this threshold `t` should be an element of the `threshold` table specified at
initialization time of the meter.
<a name="NDCGMeter">
#### tnt.NDCGMeter(self[, K])
```
({
   self = tnt.NDCGMeter  -- 
  [K    = table]         --  [has default value]
})
```

The `tnt.NDCGMeter` measures the normalized discounted cumulative gain (NDCG) of
a ranking produced by a model at prespecified levels k, and averages the NDCG
over all examples.

The discounted cumulative gain at level k is defined as:

DCG_k = rel_1 + \sum{i = 2}^k (rel_i / log_2(i))

Herein, rel_i is the relevance of item i as specified by an external rater.
Defining ideal DCG (IDCG) as the best possible DCG for a given example, the NDCG
at level k is defined as:

NDCG_k = DCG_k / IDCG_k

At initialization time, the meter takes as input a table `K` that contains all
the levels k at which the NDCG is computed.

The `add(output, relevance)` method takes as input (1) a NxC tensor of model
`outputs`, which scores for all C possible outputs for a batch of N examples;
and (2) a NxC tensor `relevance` that contains the corresponding relevances for
these scores, as provided by an external rater. Relevances are generally
obtained from human raters.

The `value()` method returns a table that contains the NDCG values for all
levels K that were provided at initialization time. Alternatively, the NDCG at
a specific level k can be obtained by calling `value(k)`. Note that the level
`k` should be an element of the table `K` specified at initialization time.

Please note that the number of outputs and relevances C should always be at
least as high as the highest NDCG level k that the meter is computing.
### tnt.Log

`Log` classes act as tables indexed by string keys. Allowed keys must be
provided at construction. A special key `__status__` can be also set the
convenience method `log:status()` to record basic messages.

Viewers closures can be attached to a `Log`, and called at different events:
   * `onSet(log, key, value)`: when setting a key to the `Log` with `log:set{}`.
   * `onGet(log, key)`: when querying a key with `log:get()`.
   * `onFlush(log)`: when flushing out the stored data of the `Log` with `log:flush()`.
   * `onClose(log)`: when closing a `Log` with `log:close()`.

Typical viewer closures are `text` or `json`, which allow to write to disk
or to the console a subset of the keys stored by the `Log`, in a particular
format. The special viewer closure `status` is made to be called on `set()`
events, and will print out only status records.

A typical use case would be the following:
```lua
tnt = require 'torchnet'

-- require the viewers we want
logtext = require 'torchnet.log.view.text'
logstatus = require 'torchnet.log.view.status'

log = tnt.Log{
   keys = {"loss", "accuracy"},
   onFlush = {
      -- write out all keys in "log" file
      logtext{filename='log.txt', keys={"loss", "accuracy"}, format={"%10.5f", "%3.2f"}},
      -- write out loss in a standalone file
      logtext{filename='loss.txt', keys={"loss"}},
      -- print on screen too
      logtext{keys={"loss", "accuracy"}},
   },
   onSet = {
      -- add status to log
      logstatus{filename='log.txt'},
      -- print status to screen
      logstatus{},
   }
}

-- set values
log:set{
  loss = 0.1,
  accuracy = 97
}

-- write some info
log:status("hello world")

-- flush out log
log:flush()
```
<a name="Log">
#### tnt.Log(self, keys[, onClose][, onFlush][, onGet][, onSet])
```
{
   self    = tnt.Log  -- 
   keys    = table    -- 
  [onClose = table]   -- 
  [onFlush = table]   -- 
  [onGet   = table]   -- 
  [onSet   = table]   -- 
}
```

Creates a new `Log` with allowed keys (strings) `keys`.  Specifiy event
closures with table of functions `onClose`, `onFlush`, `onGet` and `onSet`,
which will be called when `close()`, `flush()`, `get()`, and `set{}`
methods will be called, respectively.
<a name="Log.status">
#### tnt.Log:status(self[, message][, time])
```
({
   self    = tnt.Log   -- 
  [message = string]   -- 
  [time    = boolean]  --  [default=true]
})
```

Record a status message, with corresponding (optional) time of the event.
<a name="Log.set">
#### tnt.Log:set(self, keys)
```
(
   self = tnt.Log  -- 
   keys = table    -- 
)
```

Set a number of keys (a subset of the keys provided at construction) to
their corresponding values.

Closures attached to the `onSet(log, key, value)` event will be called.
<a name="Log.get">
#### tnt.Log:get(self, key)
```
({
   self = tnt.Log  -- 
   key  = string   -- 
})
```

Get the value of a given key.

Closures attached to the `onGet(log, key)` event will be called.
<a name="Log.flush">
#### tnt.Log:flush(self)
```
({
   self = tnt.Log  -- 
})
```

Flush (empty) the log data.

Closures attached to the `onFlush(log)` event will be called.
<a name="Log.close">
#### tnt.Log:close(self)
```
({
   self = tnt.Log  -- 
})
```

Close the log.

Closures attached to the `onClose(log)` event will be called.
<a name="Log.attach">
#### tnt.Log:attach(self, event, closures)
```
({
   self     = tnt.Log  -- 
   event    = string   -- 
   closures = table    -- 
})
```

Attach a set of functions (provided in a table) to a given event.
<a name="RemoteLog">
#### tnt.RemoteLog(self, keys[, server][, name][, onClose][, onFlush][, onGet][, onSet])
```
{
   self    = tnt.RemoteLog  -- 
   keys    = table          -- 
  [server  = string]        -- 
  [name    = string]        --  [default=default]
  [onClose = table]         -- 
  [onFlush = table]         -- 
  [onGet   = table]         -- 
  [onSet   = table]         -- 
}
```

Creates a new `RemoteLog` with allowed keys (strings) `keys`.  Specifiy event
closures with table of functions `onClose`, `onFlush`, `onGet` and `onSet`,
which will be called when `close()`, `flush()`, `get()`, and `set{}`
methods will be called, respectively.

If `server` is not provided, `RemoteLog` creates a server which can later be
reached at the address provided by `server()`.

If `server` is provided, `RemoteLog` will dialog with the given server to
store any values to be recorded by the `Log` (or query any of these values).

A given server can record different `Log`, with different names. The default name
is `default`, but can be specified with the `name` option.

At this time, it is important to call the `close()` method when `RemoteLog`
is not used anymore (before quitting the application).


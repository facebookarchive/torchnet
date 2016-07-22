--[[
   Copyright (c) 2016-present, Facebook, Inc.
   All rights reserved.

   This source code is licensed under the BSD-style license found in the
   LICENSE file in the root directory of this source tree. An additional grant
   of patent rights can be found in the PATENTS file in the same directory.
]]--

local tnt = require 'torchnet.env'
local argcheck = require 'argcheck'
local doc = require 'argcheck.doc'

doc[[

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

]]

local Dataset = torch.class('tnt.Dataset', tnt)

Dataset.__init =
   function()
   end

-- returns a number
Dataset.size =
   function(self)
      error(string.format(
               'size not implemented for class <%s>',
               torch.type(self)))
   end

-- execute a function
Dataset.exec =
   function(self, name, ...)
      if type(self[name]) == 'function' then
         return self[name](self, ...)
      elseif self.dataset then
         return self.dataset:exec(name, ...)
      elseif self.__dataset then
         return self.__dataset:exec(name, ...)
      else
         error(string.format('unknown function <%s>', name))
      end
   end

-- returns a table of tensors
Dataset.get =
   function(self)
      error(string.format(
               'get not implemented for class <%s>',
               torch.type(self)))
   end

Dataset.batch =
   function(...)
      return tnt.BatchDataset(...)
   end

Dataset.sample =
   function(...)
      return tnt.ResampleDataset(...)
   end

Dataset.shuffle =
   function(...)
      return tnt.ShuffleDataset(...)
   end

Dataset.split =
   function(...)
      return tnt.SplitDataset(...)
   end

Dataset.transform =
   function(...)
      return tnt.TransformDataset(...)
   end

Dataset.iterator =
   function(...)
      return tnt.DatasetIterator(...)
   end

Dataset.parallel = argcheck{
   {name='self', type='tnt.Dataset'},
   {name='init', type='function', default=function(idx) end},
   {name='nthread', type='number'},
   {name='perm', type='function', default=function(idx) return idx end},
   {name='filter', type='function', default=function(sample) return true end},
   {name='transform', type='function', default=function(sample) return sample end},
   {name='ordered', type='boolean', default=false},
   call =
   function(self, init, nthread, perm, filter, transform, ordered)
      local closure = function() return self end
      return tnt.ParallelDatasetIterator(init, closure, nthread, perm, filter, transform, ordered)
   end
}

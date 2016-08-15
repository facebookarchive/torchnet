--[[
   Copyright (c) 2016-present, Facebook, Inc.
   All rights reserved.

   This source code is licensed under the BSD-style license found in the
   LICENSE file in the root directory of this source tree. An additional grant
   of patent rights can be found in the PATENTS file in the same directory.
]]--

local tnt = require 'torchnet.env'
local argcheck = require 'argcheck'
local transform = require 'torchnet.transform'

local BatchDataset =
   torch.class('tnt.BatchDataset', 'tnt.Dataset', tnt)

BatchDataset.__init = argcheck{
   doc = [[
<a name="BatchDataset">
#### tnt.BatchDataset(@ARGP)
@ARGT

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
]],
   {name='self', type='tnt.BatchDataset'},
   {name='dataset', type='tnt.Dataset'},
   {name='batchsize', type='number'},
   {name='perm', type='function', default=function(idx, size) return idx end},
   {name='merge', type='function', opt=true},
   {name='policy', type='string', default='include-last'},
   {name='filter', type='function', default=function(sample) return true end},
   call =
      function(self, dataset, batchsize, perm, merge, policy, filter)
         assert(batchsize > 0 and math.floor(batchsize) == batchsize,
            'batchsize should be a positive integer number')
         self.dataset = dataset
         self.perm = perm
         self.batchsize = batchsize
         self.makebatch = transform.makebatch{merge=merge}
         self.policy = policy
         self.filter = filter
         self:size() -- check policy
      end
}

BatchDataset.size = argcheck{
   {name='self', type='tnt.BatchDataset'},
   call =
      function(self)
         local policy = self.policy
         if policy == 'include-last' then
            return math.ceil(self.dataset:size()/self.batchsize)
         elseif policy == 'skip-last' then
            return math.floor(self.dataset:size()/self.batchsize)
         elseif policy == 'divisible-only' then
            assert(self.dataset:size() % self.batchsize == 0, 'dataset size is not divisible by batch size')
            return self.dataset:size()/self.batchsize
         else
            error('invalid policy (include-last | skip-last | divisible-only expected)')
         end
      end
}

BatchDataset.get = argcheck{
   {name='self', type='tnt.BatchDataset'},
   {name='idx', type='number'},
   call =
      function(self, idx)
         assert(idx >= 1 and idx <= self:size(), 'index out of bound')
         local samples = {}
         local maxidx = self.dataset:size()
         for i=1,self.batchsize do
            local idx = (idx - 1)*self.batchsize + i
            if idx > maxidx then
               break
            end
            idx = self.perm(idx, maxidx)
            local sample = self.dataset:get(idx)
            if self.filter(sample) then table.insert(samples, sample) end
         end
         samples = self.makebatch(samples)
         collectgarbage()
         collectgarbage()
         return samples
      end
}

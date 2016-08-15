--[[
   Copyright (c) 2016-present, Facebook, Inc.
   All rights reserved.

   This source code is licensed under the BSD-style license found in the
   LICENSE file in the root directory of this source tree. An additional grant
   of patent rights can be found in the PATENTS file in the same directory.
]]--

local tnt = require 'torchnet.env'
local argcheck = require 'argcheck'

local CoroutineBatchDataset, BatchDataset =
   torch.class('tnt.CoroutineBatchDataset', 'tnt.BatchDataset', tnt)

CoroutineBatchDataset.__init = argcheck{
   doc = [[
<a name="CoroutineBatchDataset">
#### tnt.CoroutineBatchDataset(@ARGP)
@ARGT

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
]],
   {name = 'self',      type = 'tnt.CoroutineBatchDataset'},
   {name = 'dataset',   type = 'tnt.Dataset'},
   {name = 'batchsize', type = 'number'},
   {name = 'perm',      type = 'function', default = function(idx, size) return idx end},
   {name = 'merge',     type = 'function', opt = true},
   {name = 'policy',    type = 'string',   default = 'include-last'},
   {name='filter', type='function', default=function(sample) return true end},
   call = function(self, dataset, batchsize, perm, merge, policy, filter)
      BatchDataset.__init(self, dataset, batchsize, perm, merge, policy, filter)
   end
}

CoroutineBatchDataset.get = argcheck{
   {name = 'self', type = 'tnt.CoroutineBatchDataset'},
   {name = 'idx',  type = 'number'},
   call = function(self, idx)
      assert(idx >= 1 and idx <= self:size(), 'index out of bound')
      assert(idx == math.floor(idx), 'index should be integer value')

      -- create and start coroutines that perform get():
      local crs, samples, maxidx = {}, {}, self.dataset:size()
      for n = 1,self.batchsize do
         local idx = (idx - 1) * self.batchsize + n
         if idx > maxidx then break end

         -- start coroutine:
         crs[n] = coroutine.create(
            function() return self.dataset:get(self.perm(idx)) end
         )  -- create coroutine that gets example
         local status, sample = coroutine.resume(crs[n])  -- register sample
         if not status then
            error(string.format('dataset threw error: %s', sample))
         end

         -- if coroutine does not yield but dies, store sample:
         if coroutine.status(crs[n]) == 'dead' then samples[n] = sample end
      end

      -- get the samples from coroutines that are suspended:
      for n = 1,self.batchsize do
         if crs[n] and coroutine.status(crs[n]) == 'suspended' then
            local status, sample = coroutine.resume(crs[n])
            if not status then
               error(string.format('dataset threw error: %s', sample))
            end
            assert(coroutine.status(crs[n]) == 'dead', 'coroutine did not die')
            samples[n] = sample
         end
      end

      -- filter the samples:
      local filtered = {}
      for n = 1,self.batchsize do
         if self.filter(samples[n]) then table.insert(filtered, samples[n]) end
      end

      -- return batch:
      samples = self.makebatch(filtered)
      collectgarbage()
      return samples
   end
}

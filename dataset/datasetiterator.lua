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
### Dataset Iterators

It is easy to iterate over datasets using a for loop. However, sometimes
one wants to filter out samples in a on-the-fly manner or thread sample fetching.

Iterators are here for this particular cases. In general, refrain from writing
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

]]

local DatasetIterator = torch.class('tnt.DatasetIterator', tnt)

-- iterate over a dataset
DatasetIterator.__init = argcheck{
   doc = [[
<a name="DatasetIterator">
#### tnt.DatasetIterator(@ARGP)
@ARGT

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
]],
   {name='self', type='tnt.DatasetIterator'},
   {name='dataset', type='tnt.Dataset'},
   {name='perm', type='function', default=function(idx) return idx end},
   {name='filter', type='function', default=function(sample) return true end},
   {name='transform', type='function', default=function(sample) return sample end},
   call =
      function(self, dataset, perm, filter, transform)
         self.dataset = dataset
         function self.run()
            local size = dataset:size()
            local idx = 1
            return
               function()
                  while idx <= size do
                     local sample = transform(dataset:get(perm(idx)))
                     idx = idx + 1
                     if filter(sample) then
                        return sample
                     end
                  end
               end
         end
      end
}

-- iterates from another iterator
DatasetIterator.__init = argcheck{
   {name='self', type='tnt.DatasetIterator'},
   {name='iterator', type='tnt.DatasetIterator'},
   {name='filter', type='function', default=function(sample) return true end},
   {name='transform', type='function', default=function(sample) return sample end},
   overload = DatasetIterator.__init,
   call =
      function(self, iterator, filter, transform)
         self.iterator = iterator
         function self.run()
            local loop = iterator:run()
            return
               function()
                  repeat
                     local sample = loop()
                     if sample then
                        sample = transform(sample)
                        if filter(sample) then
                           return sample
                        end
                     end
                  until not sample
               end
         end
      end
}

DatasetIterator.__call__ =
   function(self, ...)
      return self:run(...)
   end

doc[[
<a name="DatasetIterator.exec">
#### tnt.DatasetIterator.exec(tnt.DatasetIterator, name, ...)

Execute the given method `name` on the underlying dataset, passing it the
subsequent arguments, and returns what the `name` method returns.
]]

DatasetIterator.exec =
   function(self, name, ...)
      if type(self[name]) == 'function' then
         return self[name](self, ...)
      elseif self.dataset then
         return self.dataset:exec(name, ...)
      elseif self.iterator then
         return self.iterator:exec(name, ...)
      else
         error(string.format('unknown function <%s>', name))
      end
   end

DatasetIterator.filter =
   function(self, filter)
      return tnt.DatasetIterator({ iterator = self, filter = filter })
   end

DatasetIterator.transform =
   function(self, transform)
      return tnt.DatasetIterator({ iterator = self, transform = transform })
   end

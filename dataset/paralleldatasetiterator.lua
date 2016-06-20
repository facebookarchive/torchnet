--[[
   Copyright (c) 2016-present, Facebook, Inc.
   All rights reserved.

   This source code is licensed under the BSD-style license found in the
   LICENSE file in the root directory of this source tree. An additional grant
   of patent rights can be found in the PATENTS file in the same directory.
]]--

local tnt = require 'torchnet.env'
local Threads = require 'threads'
local argcheck = require 'argcheck'
local doc = require 'argcheck.doc'

local ParallelDatasetIterator = torch.class('tnt.ParallelDatasetIterator', 'tnt.DatasetIterator', tnt)

ParallelDatasetIterator.__init = argcheck{
   doc = [[
<a name="ParallelDatasetIterator">
##### tnt.ParallelDatasetIterator(@ARGP)
@ARGT

Allows to iterate over a dataset in a thread
manner. `tnt.ParallelDatasetIterator:run()` guarantees that all samples
will be seen, but does not guarantee the order.

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

`perm(idx)` is a permutation used to shuffle the examples. While the order
is not guaranteed by `run()`, it is often ordered very similarly than the
original dataset. If shuffling is needed, one can use this closure, or
(better) use [tnt.ShuffleDataset](#ShuffleDataset) on the underlying
dataset (returned by `closure()`).

`filter(sample)` is a closure which returns `true` if the given sample
should be considered or `false` if not. Note that filter is called _after_
fetching the data in a threaded manner.

A common error raised by this dataset is when `closure()` is not
serializable. Make sure that all [upvalues](http://www.lua.org/pil/27.3.3.html) of `closure()` are
serializable. It is recommended to avoid [upvalues](http://www.lua.org/pil/27.3.3.html) at all cost,
and to make sure you require all the appropriate torch packages needed to (de-)serialize
`closure()` in the `init()` function.


For more information, check out the [threads package](https://github.com/torch/threads),
on which `tnt.ParallelDatasetIterator` relies.
]],
    {name='self', type='tnt.ParallelDatasetIterator'},
    {name='init', type='function', default=function(idx) end},
    {name='closure', type='function'},
    {name='nthread', type='number'},
    {name='perm', type='function', default=function(idx) return idx end},
    {name='filter', type='function', default=function(sample) return true end},
    {name='transform', type='function', default=function(sample) return sample end},
    call =
    function(self, init, closure, nthread, perm, filter, transform)
        local function main(idx)
            gdataset = closure(idx)
            assert(torch.isTypeOf(gdataset, 'tnt.Dataset'),
            'closure should return a Dataset class')
            return gdataset:size()
         end
         Threads.serialization('threads.sharedserialize')
         local threads, sizes = Threads(nthread, init, main)
         self.__threads = threads
         self.__nthread = nthread
         local size = sizes[1][1]
         local sample -- beware: do not put this line in loop()
         function self.run()
            -- loading size of the dataset each time run() is called
            threads:addjob(
                function()
                    local size = gdataset:size()
                    return size
                end,
                function(_size_)
                    size = _size_
                end
            )
            threads:dojob()
            local idx = 1
            local function enqueue()
                while idx <= size and threads:acceptsjob() do
                  threads:addjob(
                     function(idx)
                        local sample = gdataset:get(idx)
                        collectgarbage()
                        collectgarbage()
                        return sample
                    end,
                    function(_sample_)
                        sample = _sample_
                    end,
                    perm(idx)
                    )
                    idx = idx + 1
                end
            end

            enqueue()
            return
               function()
                  while threads:hasjob() do
                     enqueue()
                     threads:dojob()
                     if threads:haserror() then
                        threads:synchronize()
                     end
                     enqueue()
                     sample = transform(sample)
                     if filter(sample) then
                        return sample
                     end
                  end
               end
         end
    end
}

doc[[
<a name="ParallelDatasetIterator.execSingle">
##### tnt.ParallelDatasetIterator.execSingle(tnt.DatasetIterator, name, ...)

Execute the given method `name` on the dataset corresponding to the first
available thread, passing it the subsequent arguments, and returns what the
`name` method returns.

For example:
```lua
  local iterator = tnt.ParallelDatasetIterator{...}
  print(iterator:execSingle("size"))
```
will print the size of the dataset loaded in the first available thread.
]]

ParallelDatasetIterator.execSingle =
  function(self, name, ...)
      assert(not self.__threads:hasjob(), 'cannot exec during loop')
      local args = {...}
      local res
      self.__threads:addjob(
         function()
            return gdataset:exec(name, table.unpack(args))
         end,
         function(...)
            res = {...}
         end)
      self.__threads:synchronize()
      return table.unpack(res)
   end

doc[[
<a name="ParallelDatasetIterator.exec">
##### tnt.ParallelDatasetIterator.exec(tnt.DatasetIterator, name, ...)

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
]]

ParallelDatasetIterator.exec =
   function(self, name, ...)
      assert(not self.__threads:hasjob(), 'cannot execAll during loop')
      local args = {...}
      local res = {}
      self.__threads:specific(true)
      for i=1,self.__nthread do
         self.__threads:addjob(i,
            function()
               return gdataset:exec(name, table.unpack(args))
            end,
            function(...)
               local r = {...}
               res[i] = #r > 1 and r or r[1]
            end)
      end
      self.__threads:specific(false)
      return res
   end

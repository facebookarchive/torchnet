local tnt = require 'torchnet.env'
local tds = require 'tds'

local tester
local test = torch.TestSuite()

function test.DatasetIterator()
   local d = tnt.TableDataset{data = {1, 2, 3, 4, 5, 6}}

   local itr = tnt.DatasetIterator(d)
   local count = 0
   for sample in itr:run() do
      count = count + 1
      tester:eq(sample, count)
   end
   tester:eq(count, 6)
end

function test.DatasetIterator_filter()
   local d = tnt.TableDataset{data = {1, 2, 3, 4, 5, 6}}
   local itr = tnt.DatasetIterator{
      dataset = d,
      filter = function(x) return x % 2 == 0 end,
   }
   local count = 0
   for sample in itr:run() do
      count = count + 1
      tester:eq(sample, count * 2, 'error at ' .. count)
   end
   tester:eq(count, 3)
end

function test.DatasetIterator_transform()
   local d = tnt.TableDataset{data = {1, 2, 3, 4, 5, 6}}
   local itr = tnt.DatasetIterator{
      dataset = d,
      transform = function(x) return x - 1 end,
   }
   local count = 0
   for sample in itr:run() do
      count = count + 1
      tester:eq(sample, count - 1, 'error at ' .. count)
   end
   tester:eq(count, 6)
end

function test.DatasetIterator_exec()
   local itr = tnt.TableDataset{data = torch.range(1, 100):totable()}
      :shuffle()
      :transform(function(i) return i end)
      :iterator()

   local output1 = {}
   local output2 = {}

   for value in itr:run() do table.insert(output1, value) end
   itr:exec('resample')
   for value in itr:run() do table.insert(output2, value) end

   tester:ne(output1, output2, 'dataset not shuffled')

   table.sort(output1)
   table.sort(output2)
   tester:eq(output1, torch.range(1, 100):totable(), 'output1 incorrect')
   tester:eq(output2, torch.range(1, 100):totable(), 'output2 incorrect')
end

function test.DatasetIterator_perm()
   local d = tnt.TableDataset{data = {1, 2, 3, 4, 5, 6}}
   local itr = tnt.DatasetIterator{
      dataset = d,
      perm = function(x) return (x % 6) + 1 end,
   }
   local count = 0
   for sample in itr:run() do
      count = count + 1
      tester:eq(sample, (count % 6) + 1, 'error at ' .. count)
   end
   tester:eq(count, 6)
end

function test.ParallelDatasetIterator()
   local d = tnt.TableDataset{data = {1, 2, 3, 4, 5, 6}}
   local itr = tnt.ParallelDatasetIterator{
      closure = function() return d end,
      init = function() require 'torchnet' end,
      nthread = 3,
   }
   local count = 0
   local present = {}
   for sample in itr:run() do
      tester:eq(present[sample], nil, 'duplicate sample: ' .. tostring(sample))
      present[sample] = true
      count = count + 1
   end
   tester:eq(count, d:size())
   for i = 1, d:size() do
      tester:eq(present[i], true, 'missing sample: ' .. tostring(i))
   end
end

function test.ParallelDatasetIterator_ordered()
   -- Create a dataset in which the second item is likely to be returned out
   -- of order
   local tds = require 'tds'
   local c = tds.AtomicCounter(0)
   local d = tnt.TableDataset{data = {1, 2, 3, 4, 5, 6}}:transform(function(s)
      if s == 2 then
         repeat until c:get() ~= 0
      elseif s > 2 then
         c:inc()
      end
      return s
   end)

   local itr = tnt.ParallelDatasetIterator{
      closure = function() return d end,
      init = function() require 'torchnet'; require 'tds' end,
      nthread = 3,
      ordered = true,
   }

   local count = 0
   for sample in itr:run() do
      count = count + 1
      tester:eq(sample, count, 'sample out of order')
   end
   tester:eq(count, d:size())
end

return function(_tester_)
   tester = _tester_
   return test
end

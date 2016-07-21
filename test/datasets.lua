local tnt = require 'torchnet.env'
local tds = require 'tds'

local tester
local test = torch.TestSuite()

function test.TableDataset()
   local d = tnt.TableDataset{data = {1, 2, 3}}
   tester:eq(d:size(), 3)
   tester:eq(d:get(1), 1)
end

function test.ListDataset()
   local identity = function(...) return ... end

   local h = tds.hash({ 1, 2, 3})
   local d = tnt.ListDataset(h, identity)
   tester:eq(d:size(), 3)
   tester:eq(d:get(1), 1)

   local tbl = {1, 2, 3}
   local d = tnt.ListDataset(tbl, identity)
   tester:eq(d:size(), 3)
   tester:eq(d:get(1), 1)

   local tensor = torch.LongTensor{1, 2, 3}
   local d = tnt.ListDataset(tbl, identity)
   tester:eq(d:size(), 3)
   tester:eq(d:get(1), 1)
end

function test.ListDataset_path()
   -- With path option
   local prefix = function(x) return 'bar/' .. x end
   local tbl = {1, 2, 3}
   local d = tnt.ListDataset(tbl, prefix, 'foo')
   tester:eq(d:size(), 3)
   tester:eq(d:get(3), 'bar/foo/3')
end

function test.ListDataset_file()
   local filename = os.tmpname()
   local f = io.open(filename, 'w')
   for i = 1, 50 do
      f:write(tostring(i) .. '\n')
   end
   f:close()

   local identity = function(...) return ... end
   local d = tnt.ListDataset(filename, identity, 'foo')
   tester:eq(d:size(), 50)
   tester:eq(d:get(15), 'foo/15')

   os.remove(filename)
end

function test.TransformDataset()
   local d = tnt.TransformDataset{
      dataset = tnt.TableDataset{data = {1, 2, 3}},
      transform = function(x) return x * 2 end
   }
   tester:eq(d:size(), 3)
   tester:eq(d:get(2), 4)

   local data = {
      { input = 1, target = 1 },
      { input = 2, target = 2 },
      { input = 3, target = 3 },
   }

   local d = tnt.TransformDataset{
      dataset = tnt.TableDataset{data = data},
      transform = function(x) return x * 2 end,
      key = 'input',
   }
   tester:eq(d:size(), 3)
   tester:eq(d:get(2).input, 4)
   tester:eq(d:get(2).target, 2)


   local d = tnt.TransformDataset{
      dataset = tnt.TableDataset{data = data},
      transforms = {
         input = function(x) return x + 1 end,
         target = function(x) return x * 2 end,
      },
   }
   tester:eq(d:size(), 3)
   tester:eq(d:get(2).input, 3)
   tester:eq(d:get(2).target, 4)

   -- alternative way of expressing the same transform
   local d = tnt.TableDataset{data = data}
      :transform(function(x) return x + 1 end, 'input')
      :transform(function(x) return x * 2 end, 'target')
   tester:eq(d:size(), 3)
   tester:eq(d:get(2).input, 3)
   tester:eq(d:get(2).target, 4)
end

function test.ConcatDataset()
   local datasets = {
      tnt.TableDataset{data={1, 2, 3}},
      tnt.TableDataset{data={4, 5, 6}},
      tnt.TableDataset{data={7, 8, 9}},
   }
   local d = tnt.ConcatDataset{datasets=datasets}
   tester:eq(d:size(), 9)
   for i = 1, 9 do
      tester:eq(d:get(i), i)
   end
end

function test.ResampleDataset()
   local tbl = tnt.TableDataset{data={1, 2, 3}}
   local function sampler(dataset, i)
      return (i % 3) + 1
   end
   local d = tnt.ResampleDataset(tbl, sampler)
   tester:eq(d:size(), 3)
   tester:eq(d:get(1), 2)
   tester:eq(d:get(3), 1)

   local d = tnt.ResampleDataset(tbl, sampler, 2)
   tester:eq(d:size(), 2)
   tester:eq(d:get(1), 2)
   local ok, _ = pcall(function() d:get(3) end)
   tester:assert(not ok, 'should be out of range')
end

function test.ShuffleDataset()
   local tbl = tnt.TableDataset{data={1, 2, 3, 4, 5}}
   local d = tnt.ShuffleDataset(tbl, sampler)
   tester:eq(d:size(), 5)
   local present = {}
   for i = 1, d:size() do
      local val = d:get(i)
      tester:assert(not present[val], 'every item should appear exactly once')
      present[val] = true
   end
   for i = 1, d:size() do
      tester:assert(present[d:get(i)], 'every item should appear exactly once')
   end
end

-- function test.SplitDataset()
   -- local tbl = tnt.TableDataset{data={1, 2, 3, 4, 5, 6}}
   -- local d = tnt.SplitDataset(tbl, {train=2, val=4})
   -- -- partitions are sorted alphabetically, train comes before val
   -- tester:assert(d:size() == 2)
   --
   -- d:select('train')
   -- tester:eq(d:size(), 2)
   -- tester:eq(d:get(1), 1)
   --
   -- d:select('val')
   -- tester:eq(d:size(), 4)
   -- tester:eq(d:get(1), 3)
   --
   -- -- equal weight
   -- local d = tnt.SplitDataset(tbl, {train=0.2, val=0.3})
   -- d:select('train'); tester:eq(d:size(), 3)
   -- d:select('val'); tester:eq(d:size(), 3)
-- end

function test.BatchDataset()
   local data = {}
   for i = 1, 100 do
      data[i] = {
         input = i,
         target = torch.LongTensor{i, 2*i},
      }
   end
   local tbl = tnt.TableDataset{data=data}
   local d = tnt.BatchDataset(tbl, 30)
   tester:eq(d:size(), 4)

   local batch = d:get(2)
   tester:eq(torch.type(batch.input), 'table')
   tester:eq(#batch.input, 30)
   tester:eq(batch.input[1], 31)
   tester:eq(torch.type(batch.target), 'torch.LongTensor')
   tester:eq(batch.target:size(1), 30)
   tester:eq(batch.target:numel(), 60)
   tester:eq(batch.target[1][2], 62)

   -- last batch has the remainder
   tester:eq(#d:get(4).input, 10)

   local d = tnt.BatchDataset(tbl, 30, 'skip-last')
   tester:eq(d:size(), 3)
   tester:eq(#d:get(3).input, 30)

   -- divisible-only should trigger an error with batch size 30
   tester:assertErrorPattern(
      function() tnt.BatchDataset(tbl, 30, 'divisible-only') end,
      'not divisible')

   -- divisible-only should succeed with batch size 20
   local d = tnt.BatchDataset(tbl, 20, 'divisible-only')
   tester:eq(d:size(), 5)
   tester:eq(#d:get(3).input, 20)

   -- test with custom merge
   local d = tnt.BatchDataset{
      dataset = tbl,
      batchsize = 30,
      merge = tnt.transform.tableapply(function(field)
         if type(field[1]) == 'number' then
            return torch.IntTensor(field)
         else
            return tnt.utils.table.mergetensor(field)
         end
      end),
   }
   tester:eq(d:size(), 4)
   tester:eq(torch.type(d:get(1).input), 'torch.IntTensor')
   tester:eq(d:get(1).input:numel(), 30)
end

function test.IndexedDataset()
   local tmpdir = os.tmpname()
   os.remove(tmpdir) -- tmpname creates the file
   assert(paths.mkdir(tmpdir))

   -- write integers tensors
   local w = tnt.IndexedDatasetWriter{
      indexfilename = paths.concat(tmpdir, 'ints.idx'),
      datafilename = paths.concat(tmpdir, 'ints.bin'),
      type = 'int',
   }
   for i = 1, 10 do
      local t = torch.range(1, i):int()
      w:add(t)
   end
   w:close()

   -- write indexed tables
   local w = tnt.IndexedDatasetWriter{
      indexfilename = paths.concat(tmpdir, 'tables.idx'),
      datafilename = paths.concat(tmpdir, 'tables.bin'),
      type = 'table',
   }
   for i = 1, 10 do
      w:add({'a', 'b', i})
   end
   w:close()

   local d = tnt.IndexedDataset{
      path = tmpdir,
      fields = {'ints','tables'},
   }
   tester:eq(d:size(), 10)
   tester:eq(d:get(4), {
      ints = torch.range(1, 4):int(),
      tables = {'a', 'b', 4 },
   })

   assert(os.remove(paths.concat(tmpdir, 'ints.idx')))
   assert(os.remove(paths.concat(tmpdir, 'ints.bin')))
   assert(os.remove(paths.concat(tmpdir, 'tables.idx')))
   assert(os.remove(paths.concat(tmpdir, 'tables.bin')))
   assert(os.remove(tmpdir))
end

function test.CoroutineBatchDataset_basic()
   -- CoroutineBatchDataset without coroutines should work like BatchDataset
   local data = {}
   for i = 1, 100 do
      data[i] = {
         input = i,
         target = torch.LongTensor{i, 2*i},
      }
   end
   local tbl = tnt.TableDataset{data=data}
   local d = tnt.CoroutineBatchDataset(tbl, 20)
   tester:eq(d:size(), 5)
   tester:eq(d:get(2).input, torch.range(21, 40):totable())
   tester:eq(d:get(2).target:size(), torch.LongStorage{20, 2})
end

function test.CoroutineBatchDataset()
   local base = tnt.Dataset()
   base.__keys = {}
   base.__buffer = {}
   function base:size()
      return 100
   end
   function base:get(i)
      table.insert(self.__keys, i)
      coroutine.yield()
      if self.__buffer[i] == nil then
         tester:assert(i % 20 == 1, 'expected to be the first sample')
         tester:eq(#self.__keys, 20, 'expected batch of 20')
         for _, k in ipairs(self.__keys) do
            self.__buffer[k] = { input = k * 2 }
         end
         self.__keys = {}
      end
      tester:eq(#self.__keys, 0, 'keys should be empty')
      local val = self.__buffer[i]
      tester:assert(val ~= nil)
      self.__buffer[i] = nil
      return val
   end
   local d = tnt.CoroutineBatchDataset(base, 20)
   tester:eq(d:size(), 5)
   tester:eq(d:get(2), { input = torch.range(42, 80, 2):totable() })
end

return function(_tester_)
   tester = _tester_
   return test
end

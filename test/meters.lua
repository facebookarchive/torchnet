local tnt = require 'torchnet.env'

local tester
local test = torch.TestSuite()

function test.AverageValueMeter()
   local mtr = tnt.AverageValueMeter()

   mtr:add(1)
   local avg, var = mtr:value()

   tester:eq(avg, 1)
   tester:assert(var ~= var, "Variance for a single value is undefined")

   mtr:add(3)
   avg, var = mtr:value()

   tester:eq(avg, 2)
   tester:eq(var, math.sqrt(2))
end

function test.ClassErrorMeter()
   local mtr = tnt.ClassErrorMeter{topk = {1}}

   local output = torch.Tensor({{1,0,0},{0,1,0},{0,0,1}})
   local target = torch.Tensor({1,2,3})
   mtr:add(output, target)
   local error = mtr:value()

   tester:eq(error, {0}, "All should be correct")

   target[1] = 2
   target[2] = 1
   target[3] = 1
   mtr:add(output, target)

   error = mtr:value()
   tester:eq(error, {50}, "Half, i.e. 50%, should be correct")
end

function test.ConfusionMeter()
   local mtr = tnt.ConfusionMeter{k = 3}

   -- The max value is the one that is correct
   local output = torch.Tensor({{.8,0.1,0.1},{10,11,10},{0.2,0.2,.3}})
   local target = torch.Tensor({1,2,3})
   mtr:add(output, target)
   local conf_mtrx = mtr:value()

   tester:eq(conf_mtrx:sum(), 3, "All should be correct")
   tester:eq(torch.diag(conf_mtrx):sum(), 3, "All should be correct")

   target[1] = 2
   target[2] = 1
   target[3] = 1
   mtr:add(output, target)

   tester:eq(conf_mtrx:sum(), 6, "Six tests should give six values")
   tester:eq(torch.diag(conf_mtrx):sum(), 3, "Shouldn't have changed since all new values were false")
   tester:eq(conf_mtrx[1]:sum(), 3, "All top have gotten one guess")
   tester:eq(conf_mtrx[2]:sum(), 2, "Two first at the 2nd row have a guess")
   tester:eq(conf_mtrx[2][3], 0, "The last one should be empty")
   tester:eq(conf_mtrx[3]:sum(), 1, "Bottom row has only the first test correct")
   tester:eq(conf_mtrx[3][3], 1, "Bottom row has only the first test correct")

   -- Test normalized version
   mtr = tnt.ConfusionMeter{k = 4, normalized=true}
   output = torch.Tensor({
      {.8,0.1,0.1,0},
      {10,11,10,0},
      {0.2,0.2,.3,0},
      {0,0,0,1}
   })

   target = torch.Tensor({1,2,3,4})
   mtr:add(output, target)
   conf_mtrx = mtr:value()

   tester:eq(conf_mtrx:sum(), output:size(2), "All should be correct")
   tester:eq(torch.diag(conf_mtrx):sum(), output:size(2), "All should be correct")

   target[1] = 2
   target[2] = 1
   target[3] = 1
   mtr:add(output, target)
   conf_mtrx = mtr:value()

   tester:eq(conf_mtrx:sum(), output:size(2), "The noramlization should sum all values to 1")
   for i=1,output:size(2) do
      tester:eq(conf_mtrx[i]:sum(), 1, "Row no " .. i .. " fails to sum to one in normalized mode")
   end
end

function test.MultilabelConfusionMeter()
   local mtr = tnt.MultiLabelConfusionMeter{k = 3, normalized=false}

   -- The max value is the one that is correct
   local output = torch.Tensor({{.8,0.1,0.1},{10,11,10},{0.2,0.2,.3}})
   local target = torch.LongTensor({1,2,3})
   local one_hot = torch.zeros(output:size())
   one_hot:scatter(2, target:view(-1,1), 1)
   mtr:add(output, one_hot)
   local conf_mtrx = mtr:value()

   tester:eq(conf_mtrx, torch.eye(3), "All should be correct")

   target[1] = 2
   target[2] = 1
   target[3] = 1
   one_hot = torch.zeros(output:size())
   one_hot:scatter(2, target:view(-1,1), 1)
   mtr:add(output, one_hot)
   conf_mtrx = mtr:value()

   tester:eq(conf_mtrx:sum(), 6, "Six tests should give six values")
   tester:eq(torch.diag(conf_mtrx):sum(), 3, "Shouldn't have changed since all new values were false")
   tester:eq(conf_mtrx[1]:sum(), 3, "All top have gotten one guess")
   tester:eq(conf_mtrx[2]:sum(), 2, "Two first at the 2nd row have a guess")
   tester:eq(conf_mtrx[2][3], 0, "The last one should be empty")
   tester:eq(conf_mtrx[3]:sum(), 1, "Bottom row has only the first test correct")
   tester:eq(conf_mtrx[3][3], 1, "Bottom row has only the first test correct")

   -- Test normalized version
   mtr = tnt.MultiLabelConfusionMeter{k = 4, normalized=true}
   output = torch.Tensor({
      {.8,0.1,0.1,0},
      {10,11,10,0},
      {0.2,0.2,.3,0},
      {0,0,0,1}
   })

   target = torch.LongTensor({1,2,3,4})
   one_hot = torch.zeros(output:size())
   one_hot:scatter(2, target:view(-1,1), 1)
   mtr:add(output, one_hot)
   conf_mtrx = mtr:value()

   tester:eq(conf_mtrx:sum(), output:size(2), "All should be correct", 10^-3)
   tester:eq(torch.diag(conf_mtrx):sum(), output:size(2), "All should be correct", 10^-3)

   target[1] = 2
   target[2] = 1
   target[3] = 1
   one_hot = torch.zeros(output:size())
   one_hot:scatter(2, target:view(-1,1), 1)
   mtr:add(output, one_hot)
   conf_mtrx = mtr:value()

   tester:eq(conf_mtrx:sum(), output:size(2), "The noramlization should sum all values to 1", 10^-3)
   for i=1,output:size(2) do
      tester:eq(conf_mtrx[i]:sum(), 1, "Row no " .. i .. " fails to sum to one in normalized mode", 10^-3)
   end
end

function test.AUCMeter()
   local mtr = tnt.AUCMeter()

   local test_size = 10^3
   mtr:add(torch.rand(test_size), torch.zeros(test_size))
   mtr:add(torch.rand(test_size), torch.Tensor(test_size):fill(1))
   local err = mtr:value()
   tester:eq(err, 0.5, "Random guesses should provide a AUC close to 0.5", 10^-1)

   mtr:add(torch.Tensor(test_size):fill(0), torch.zeros(test_size))
   mtr:add(torch.Tensor(test_size):fill(.4), torch.zeros(test_size))
   mtr:add(torch.Tensor(test_size):fill(1), torch.Tensor(test_size):fill(1))
   err = mtr:value()
   tester:eq(err, 1, "Only correct guesses should provide a AUC close to 1", 10^-1)

   -- Simulate a random situation where all the guesses are correct
   mtr:reset()
   local output = torch.abs(torch.rand(test_size)-.5)*2/3
   mtr:add(output, torch.zeros(test_size))
   output = torch.min(
      torch.cat(torch.rand(test_size) + .75,
                torch.Tensor(test_size):fill(1),
                2),
      2)
   mtr:add(output:fill(1), torch.Tensor(test_size):fill(1))
   err = mtr:value()
   tester:eq(err, 1, "Simulated random correct guesses should provide a AUC close to 1", 10^-1)
end


function test.APMeter()
   local mtr = tnt.APMeter()

   local target = torch.Tensor{0,1,0,1}
   local output = torch.Tensor{.1,.2,.3,4}
   mtr:add(output, target)

   local ap = mtr:value()
   tester:eq(ap[1], (1*1 + 0*1/2 + 2*1/3 + 0*1/4)/2)

   mtr:reset()

   target = torch.Tensor{0,1,0,1}
   output = torch.Tensor{4,3,2,1}
   mtr:add(output, target)

   ap = mtr:value()
   tester:eq(ap[1], (0*1 + 1*1/2 + 0*1/3 + 2*1/4)/2)

   mtr:reset()
   target = torch.Tensor{0,1,0,1}
   output = torch.Tensor{1,4,2,3}
   mtr:add(output, target)

   ap = mtr:value()
   tester:eq(ap[1], (1*1 + 2*1/2 + 0*1/3 + 0*1/4)/2)

   mtr:reset()
   target = torch.Tensor{0,0,0,0}
   output = torch.Tensor{1,4,2,3}
   mtr:add(output, target)

   ap = mtr:value()
   tester:eq(ap[1], 0)

   mtr:reset()
   target = torch.Tensor{1,1,0}
   output = torch.Tensor{3,1,2}
   mtr:add(output, target)

   ap = mtr:value()
   tester:eq(ap[1], (1*1 + 0*1/2 + 2*1/3)/2)

   -- Test multiple K:s
   mtr:reset()
   target = torch.Tensor{
      {0,1,0,1},
      {0,1,0,1}
   }:transpose(1,2)
   output = torch.Tensor{
      {.1,.2,.3,4},
      {4,3,2,1}
   }:transpose(1,2)
   mtr:add(output, target)

   ap = mtr:value()
   tester:eq(
      ap,
      torch.DoubleTensor{
         (1*1 + 0*1/2 + 2*1/3 + 0*1/4)/2,
         (0*1 + 1*1/2 + 0*1/3 + 2*1/4)/2
      }
   )
end


function test.mAPMeter()
   local mtr = tnt.mAPMeter()

   local target = torch.Tensor{0,1,0,1}
   local output = torch.Tensor{.1,.2,.3,4}
   mtr:add(output, target)

   local ap = mtr:value()
   tester:eq(ap, (1*1 + 0*1/2 + 2*1/3 + 0*1/4)/2)

   -- Test multiple K:s
   mtr:reset()
   target = torch.Tensor{
      {0,1,0,1},
      {0,1,0,1}
   }:transpose(1,2)
   output = torch.Tensor{
      {.1,.2,.3,4},
      {4,3,2,1}
   }:transpose(1,2)
   mtr:add(output, target)

   ap = mtr:value()
   tester:eq(
      ap,
      torch.DoubleTensor{
         (1*1 + 0*1/2 + 2*1/3 + 0*1/4)/2,
         (0*1 + 1*1/2 + 0*1/3 + 2*1/4)/2
      }:mean()
   )
end

return function(_tester_)
   tester = _tester_
   return test
end

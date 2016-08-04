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

function test.AUCMeter()
   local mtr = tnt.AUCMeter()

   -- From http://stats.stackexchange.com/questions/145566/how-to-calculate-area-under-the-curve-auc-or-the-c-statistic-by-hand
   local samples = torch.Tensor{
      {33,6,6,11,2}, --normal
      {3,2,2,11,33} -- abnormal
   }
   for i=1,samples:size(2) do
      local target = torch.Tensor():resize(samples:narrow(2,i,1):sum()):zero()
      target:narrow(1,1,samples[2][i]):fill(1)
      local output = torch.Tensor(target:size(1)):fill(i)
      mtr:add(output, target)
   end

   local error, tpr, fpr = mtr:value()

   tester:assert(math.abs(error - 0.8931711) < 10^-3,
      ("The AUC error does not match: %.3f is not equal to 0.893"):format(error))
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

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

   torch.manualSeed(41412)
   local test_size = 10^3
   mtr:add(torch.rand(test_size), torch.zeros(test_size))
   mtr:add(torch.rand(test_size), torch.Tensor(test_size):fill(1))
   local err = mtr:value()
   tester:eq(err, 0.5, "Random guesses should provide a AUC close to 0.5", 10^-1)

   mtr:reset()
   mtr:add(torch.Tensor(test_size):fill(0), torch.zeros(test_size))
   mtr:add(torch.Tensor(test_size):fill(.1), torch.zeros(test_size))
   mtr:add(torch.Tensor(test_size):fill(.2), torch.zeros(test_size))
   mtr:add(torch.Tensor(test_size):fill(.3), torch.zeros(test_size))
   mtr:add(torch.Tensor(test_size):fill(.4), torch.zeros(test_size))
   mtr:add(torch.Tensor(test_size):fill(1), torch.Tensor(test_size):fill(1))
   err = mtr:value()
   tester:eq(err, 1, "Only correct guesses should provide a AUC close to 1", 10^-1)
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

function test.NDCGMeter()
   local mtr = tnt.NDCGMeter{K = {6}}

   -- From: https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG
   local relevance = torch.DoubleTensor{3,2,3,0,1,2}
   local output = torch.linspace(relevance:size(1), 1, relevance:size(1)):double()
   mtr:add(output, relevance)

   local est = mtr:value()
   tester:eq(est[6], 0.932, "Problematic nDGC with K=6", 10^-3)
end

function test.PrecisionMeter()
   local mtr = tnt.PrecisionMeter{}

   local target = torch.zeros(10)
   target:narrow(1, 3, 7):fill(1)
   local output = torch.zeros(10)
   output:narrow(1, 2, 6):fill(1)

   mtr:add(output, target)

   local tp = 5
   local fp = 1
   tester:eq(mtr:value()[0.5], 100 * tp / (tp + fp), "Basic test", 10^-2)

   mtr = tnt.PrecisionMeter{
      threshold = {.5, .7}
   }

   target = torch.zeros(10)
   target:narrow(1, 3, 7):fill(1)
   output = torch.zeros(10)
   output:narrow(1, 2, 3):fill(.5)
   output:narrow(1, 5, 3):fill(.7)

   mtr:add(output, target)
   tp = 5
   fp = 1
   tester:eq(mtr:value()[0.5], 100 * tp / (tp + fp), "Cutoff at 0.5", 10^-2)
   tester:eq(mtr:value()[0.7], 100, "Cutoff at 0.7", 10^-2)
end

function test.PrecisionAtKMeter()
   local mtr = tnt.PrecisionAtKMeter{topk = {1, 2, 3}}

   local target = torch.eye(3)
   local output = torch.Tensor{
      {.5,.1,.4},
      {.1,.5,.4},
      {.1,.4,.5}
   }
   mtr:add(output, target)
   tester:eq(mtr:value()[1], 100*3/3, "Top 1 matches", 10^-3)
   tester:eq(mtr:value()[2], 100*3/(3*2), "Top 2 matches", 10^-3)
   tester:eq(mtr:value()[3], 100*3/(3*3), "Top 3 matches", 10^-3)

   mtr:add(output, target) -- Adding the same twice shouldn't change anything
   tester:eq(mtr:value()[1], 100*3/3, "Top 1 matches", 10^-3)
   tester:eq(mtr:value()[2], 100*3/(3*2), "Top 2 matches", 10^-3)
   tester:eq(mtr:value()[3], 100*3/(3*3), "Top 3 matches", 10^-3)

   mtr:reset()
   target[1][3] = 1

   mtr:add(output, target)
   tester:eq(mtr:value()[1], 100*3/3, "Top 1 matches", 10^-3)
   tester:eq(mtr:value()[2], 100*(3 + 1)/(3 * 2), "Top 2 matches", 10^-3)
   tester:eq(mtr:value()[3], 100*(3 + 1)/(3 * 3), "Top 3 matches", 10^-3)

   mtr:reset()
   output = torch.Tensor{
      {.1,.5,.4},
      {.1,.5,.4},
      {.5,.4,.1}
   }
   mtr:add(output, target)
   tester:eq(mtr:value()[1], 100*1/3, "Top 1 matches", 10^-3)
   tester:eq(mtr:value()[2], 100*(1 + 1)/(3 * 2), "Top 2 matches", 10^-3)
   tester:eq(mtr:value()[3], 100*(1 + 1 + 2)/(3 * 3), "Top 3 matches", 10^-3)

   local mtr = tnt.PrecisionAtKMeter{topk = {1, 2, 3}, online = true}

   local target = torch.eye(3)
   local output = torch.Tensor{
      {.5,.1,.4},
      {.1,.5,.4},
      {.1,.4,.5}
   }
   mtr:add(output, target)
   tester:eq(mtr:value()[1], 100*3/3, "Top 1 matches", 10^-3)
   tester:eq(mtr:value()[2], 100*3/(3*2), "Top 2 matches", 10^-3)
   tester:eq(mtr:value()[3], 100*3/(3*3), "Top 3 matches", 10^-3)

   mtr:add(output, target)
   tester:eq(mtr:value()[1], 100*3/3, "Top 1 matches", 10^-3)
   tester:eq(mtr:value()[2], 100*3*2/(3*2), "Top 2 matches", 10^-3)
   tester:eq(mtr:value()[3], 100*3*2/(3*3), "Top 3 matches", 10^-3)
end

function test.RecallMeter()
   local mtr = tnt.RecallMeter{}

   local target = torch.zeros(10)
   target:narrow(1, 3, 7):fill(1)
   local output = torch.zeros(10)
   output:narrow(1, 2, 6):fill(1)

   mtr:add(output, target)

   local tp = 5
   local fn = 2
   tester:eq(mtr:value()[0.5], 100 * tp / (tp + fn), "Basic test", 10^-2)

   mtr = tnt.RecallMeter{
      threshold = {.5, .7}
   }

   target = torch.zeros(10)
   target:narrow(1, 3, 7):fill(1)
   output = torch.zeros(10)
   output:narrow(1, 2, 3):fill(.5)
   output:narrow(1, 5, 3):fill(.7)

   mtr:add(output, target)
   tester:eq(mtr:value()[0.5], 100 * tp / (tp + fn), "Cutoff at 0.5", 10^-2)
   tp = tp - 2
   fn = fn + 2
   tester:eq(mtr:value()[0.7], 100 * tp / (tp + fn), "Cutoff at 0.7", 10^-2)
end

function test.TimeMeter()
   local mtr = tnt.TimeMeter()

   local function wait(seconds)
     local start = os.time()
     repeat until os.time() > start + seconds
   end

   mtr:reset()
   wait(1)
   local passed_time = mtr:value()
   tester:assert(passed_time < 2,
                ("Too long time passed: %.1f sec >= 2 sec"):format(passed_time))
   tester:assert(passed_time > .5,
                ("Too short time passed:  %.1f sec <= 0.5 sec"):format(passed_time))
end

return function(_tester_)
   tester = _tester_
   return test
end

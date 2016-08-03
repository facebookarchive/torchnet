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

return function(_tester_)
   tester = _tester_
   return test
end

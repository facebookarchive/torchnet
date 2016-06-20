--[[
   Copyright (c) 2016-present, Facebook, Inc.
   All rights reserved.

   This source code is licensed under the BSD-style license found in the
   LICENSE file in the root directory of this source tree. An additional grant
   of patent rights can be found in the PATENTS file in the same directory.
]]--

local tnt = require 'torchnet.env'
local argcheck = require 'argcheck'

local PrecisionMeter = torch.class('tnt.PrecisionMeter', 'tnt.Meter', tnt)

PrecisionMeter.__init = argcheck{
   doc = [[
<a name="PrecisionMeter">
#### tnt.PrecisionMeter(@ARGP)
@ARGT

The `tnt.PrecisionMeter` measures the precision of ranking methods at pre-
specified thresholds. The precision is the percentage of the positively labeled
items according to the model that is in the list of correct (positive) targets.

At initialization time, the `tnt.PrecisionMeter` provides two optional
parameters. The first parameter is a table `threshold` that contains all
thresholds at which the precision is measured (default = {0.5}). Thresholds
should be numbers between 0 and 1. The second parameter is a boolean `perclass`
that makes the meter measure the precision per class when set to `true`
(default = `false`). When `perclass` is set to `false`, the precision is simply
averaged over all examples.

The `add(output, target)` method takes two inputs:
   * A NxK tensor that for each of the N examples indicates the probability
     of the example belonging to each of the K classes, according to the model.
     The probabilities should sum to one over all classes; that is, the row sums
     of `output` should all be one.
   * A binary NxK `target` tensor that encodes which of the K classes
     are associated with the N-th input. For instance, a row of {0, 1, 0, 1}
     indicates that the example is associated with classes 2 and 4.

The `value()` method returns a table containing the precision of the model
predictions measured at the `threshold`s specified at initialization time. The
`value(t)` method returns the precision at a particular threshold `t`. Note that
this threshold `t` should be an element of the `threshold` table specified at
initialization time of the meter.
]],
    noordered = true,
    {name="self", type="tnt.PrecisionMeter"},
    {name="threshold", type="table", default={0.5}},
    {name="perclass", type="boolean", default=false},
    call = function(self, threshold, perclass)
        self.threshold = {}
        for _,n in pairs(threshold) do
            assert(n >= 0 and n <= 1, 'threshold should be between 0 and 1')
            table.insert(self.threshold, n)
        end
        table.sort(self.threshold)
        self.perclass = perclass
        self:reset()
    end
}

PrecisionMeter.reset = argcheck{
    {name="self", type="tnt.PrecisionMeter"},
    call = function(self)
        self.tp = {}
        self.tpfp = {}
        for _,t in ipairs(self.threshold) do
            self.tp[t]   = torch.Tensor()
            self.tpfp[t] = torch.Tensor()
        end
    end
}

PrecisionMeter.add = argcheck{
    {name="self", type="tnt.PrecisionMeter"},
    {name="output", type="torch.*Tensor"},
    {name="target", type="torch.*Tensor"}, -- target is k-hot vector
    call = function(self, output, target)
        output = output:squeeze()
        if output:nDimension() == 1 then
            output = output:view(1, output:size(1))
        else
            assert(
                output:nDimension() == 2,
                'wrong output size (1D or 2D expected)'
            )
        end
        if target:nDimension() == 1 then
            target = target:view(1, target:size(1))
        else
            assert(
                target:nDimension() == 2,
                'wrong target size (1D or 2D expected)'
            )
        end
        for s = 1,#output:size() do
            assert(
                output:size(s) == target:size(s),
                string.format('target and output do not match on dim %d', s)
            )
        end

        -- initialize upon receiving first inputs:
        for _,t in ipairs(self.threshold) do
            if self.tp[t]:nElement() == 0 then
                self.tp[t]:resize(  target:size(2)):typeAs(output):fill(0)
                self.tpfp[t]:resize(target:size(2)):typeAs(output):fill(0)
            end
        end

        -- scores of true positives:
        local true_pos = output:clone()
        true_pos[torch.eq(target, 0)] = -1

        -- sum all the things:
        for _,t in ipairs(self.threshold) do
            self.tp[t]:add( torch.ge(true_pos, t):typeAs(output):sum(1):squeeze())
            self.tpfp[t]:add(torch.ge(output,  t):typeAs(output):sum(1):squeeze())
        end
    end
}

PrecisionMeter.value = argcheck{
   {name="self", type="tnt.PrecisionMeter"},
   {name="t", type="number", opt=true},
   call = function(self, t)
         if t then
            assert(
                self.tp[t] and self.tpfp[t],
                string.format('%f is an incorrect threshold [not set]', t)
            )
            if self.perclass then
                local precisionPerClass =
                    torch.cdiv(self.tp[t], self.tpfp[t]):mul(100)
                precisionPerClass[torch.eq(self.tpfp[t], 0)] = 100
                return precisionPerClass
            else
                if self.tpfp[t]:sum() == 0 then return 100 end
                return self.tp[t]:sum() / self.tpfp[t]:sum() * 100
            end
         else
            local value = {}
            for _,t in ipairs(self.threshold) do
               value[t] = self:value(t)
            end
            return value
         end
      end
}

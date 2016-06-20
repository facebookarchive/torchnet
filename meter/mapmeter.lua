--[[
   Copyright (c) 2016-present, Facebook, Inc.
   All rights reserved.

   This source code is licensed under the BSD-style license found in the
   LICENSE file in the root directory of this source tree. An additional grant
   of patent rights can be found in the PATENTS file in the same directory.
]]--

local tnt = require 'torchnet.env'
local argcheck = require 'argcheck'

local mAPMeter = torch.class('tnt.mAPMeter', 'tnt.Meter', tnt)


mAPMeter.__init = argcheck{
   doc = [[
<a name="mAPMeter">
#### tnt.mAPMeter(@ARGP)
@ARGT

The `tnt.mAPMeter` measures the mean average precision over all classes.

The `tnt.mAPMeter` is designed to operate on `NxK` Tensors `output` and `target`
where (1) the `output` contains model output scores for `N` examples and `K`
classes that ought to be higher when the model is more convinced that the
example should be positively labeled, and smaller when the model believes the
example should be negatively labeled (for instance, the output of a sigmoid
function); and (2) the `target` contains only values 0 (for negative examples)
and 1 (for positive examples).

The `tnt.mAPMeter` has no parameters to be set.
]],
   {name="self", type="tnt.mAPMeter"},
   call = function(self)
      self.apmeter = tnt.APMeter()
   end
}

mAPMeter.reset = argcheck{
   {name="self", type="tnt.mAPMeter"},
   call = function(self)
      self.apmeter:reset()
   end
}

mAPMeter.add = argcheck{
   {name="self", type="tnt.mAPMeter"},
   {name="output", type="torch.*Tensor"},
   {name="target", type="torch.*Tensor"},
   call = function(self, output, target)
      self.apmeter:add{output = output, target = target}
   end
}

mAPMeter.value = argcheck{
   {name="self", type="tnt.mAPMeter"},
   call = function(self)
      return self.apmeter:value():mean()
   end
}

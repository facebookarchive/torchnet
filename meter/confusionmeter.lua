--[[
   Copyright (c) 2016-present, Facebook, Inc.
   All rights reserved.

   This source code is licensed under the BSD-style license found in the
   LICENSE file in the root directory of this source tree. An additional grant
   of patent rights can be found in the PATENTS file in the same directory.
]]--

local tnt = require 'torchnet.env'
local argcheck = require 'argcheck'

local ConfusionMeter = torch.class('tnt.ConfusionMeter', 'tnt.Meter', tnt)

ConfusionMeter.__init = argcheck{
   doc = [[
<a name="ConfusionMeter">
#### tnt.ConfusionMeter(@ARGP)
@ARGT

The `tnt.ConfusionMeter` constructs a confusion matrix for a multi-class
classification problems. It does not support multi-label, multi-class problems:
for such problems, please use `tnt.MultiLabelConfusionMeter`.

At initialization time, the `k` parameter that indicates the number
of classes in the classification problem under consideration must be specified.
Additionally, an optional parameter `normalized` (default = `false`) may be
specified that determines whether or not the confusion matrix is normalized
(that is, it contains percentages) or not (that is, it contains counts).

The `add(output, target)` method takes as input an NxK tensor `output` that
contains the output scores obtained from the model for N examples and K classes,
and a corresponding N-tensor or NxK-tensor `target` that provides the targets
for the N examples. When `target` is an N-tensor, the targets are assumed to be
integer values between 1 and K. When target is an NxK-tensor, the targets are
assumed to be provided as one-hot vectors (that is, vectors that contain only
zeros and a single one at the location of the target value to be encoded).

The `value()` method has no parameters and returns the confusion matrix in a
KxK tensor. In the confusion matrix, rows correspond to ground-truth targets and
columns correspond to predicted targets.
]],
   noordered = true,
   {name="self", type="tnt.ConfusionMeter"},
   {name="k", type="number"},
   {name="normalized", type="boolean", default=false},
   call =
      function(self, k, normalized)
         self.conf = torch.LongTensor(k, k)
         self.normalized = normalized
         self:reset()
      end
}

ConfusionMeter.reset = argcheck{
   {name="self", type="tnt.ConfusionMeter"},
   call =
      function(self)
         self.conf:zero()
      end
}

ConfusionMeter.add = argcheck{
   {name="self", type="tnt.ConfusionMeter"},
   {name="output", type="torch.*Tensor"},
   {name="target", type="torch.*Tensor"},
   call =
      function(self, output, target)
         target = target:squeeze()
         output = output:squeeze()
         if output:nDimension() == 1 then
            output = output:view(1, output:size(1))
            if type(target) == 'number' then
               target = torch.Tensor(1):fill(target)
            end
         end
         local onehot = not (target:nDimension() == 1)
         assert(
            target:size(1) == output:size(1),
            'number of targets and outputs do not match'
         )
         assert(
            output:size(2) == self.conf:size(1),
            'number of outputs does not match size of confusion matrix'
         )
         assert(
            not onehot or target:size(2) == output:size(2),
            'target should be 1D Tensor or have size of output (one-hot)'
         )
         if onehot then
            assert(
               torch.eq(torch.eq(target, 0):add(torch.eq(target, 1)), 1):all(),
               'in one-hot encoding, target values should be 0 or 1'
            )
            assert(
               torch.eq(target:sum(2), 1):all(),
               'multi-label setting is not supported'
            )
         end

         -- update confusion matrix:
         local pos
         local _,pred = output:double():max(2)
         for n = 1,pred:size(1) do
            if onehot then _,pos = target[n]:max(1); pos = pos[1]
            else             pos = target[n] end
            self.conf[pos][pred[n][1]] = self.conf[pos][pred[n][1]] + 1
         end
      end
}

ConfusionMeter.value = argcheck{
   {name="self", type="tnt.ConfusionMeter"},
   call =
      function(self)
         local confmat
         if self.normalized then
            confmat = torch.DoubleTensor(self.conf:size()):copy(self.conf)
            confmat:cdiv(confmat:sum(2):cmax(1e-12):expandAs(confmat))
         else
            confmat = self.conf
         end
         return confmat
      end
}

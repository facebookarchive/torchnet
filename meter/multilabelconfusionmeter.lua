--[[
   Copyright (c) 2016-present, Facebook, Inc.
   All rights reserved.

   This source code is licensed under the BSD-style license found in the
   LICENSE file in the root directory of this source tree. An additional grant
   of patent rights can be found in the PATENTS file in the same directory.
]]--

local tnt = require 'torchnet.env'
local tds = require 'tds'
local argcheck = require 'argcheck'

local MultiLabelConfusionMeter =
   torch.class('tnt.MultiLabelConfusionMeter', 'tnt.Meter', tnt)

MultiLabelConfusionMeter.__init = argcheck{
   doc = [[
<a name="MultiLabelConfusionMeter">
#### tnt.MultiLabelConfusionMeter(@ARGP)
@ARGT

The `tnt.MultiLabelConfusionMeter` constructs a confusion matrix for multi-
label, multi-class classification problems. In constructing the confusion
matrix, the number of positive predictions is assumed to be equal to the number
of positive labels in the ground-truth. Correct predictions (that is, labels in
the prediction set that are also in the ground-truth set) are added to the
diagonal of the confusion matrix. Incorrect predictions (that is, labels in the
prediction set that are not in the ground-truth set) are equally divided over
all non-predicted labels in the ground-truth set.

At initialization time, the `k` parameter that indicates the number
of classes in the classification problem under consideration must be specified.
Additionally, an optional parameter `normalized` (default = `false`) may be
specified that determines whether or not the confusion matrix is normalized
(that is, it contains percentages) or not (that is, it contains counts).

The `add(output, target)` method takes as input an NxK tensor `output` that
contains the output scores obtained from the model for N examples and K classes,
and a corresponding NxK-tensor `target` that provides the targets for the N
examples using one-hot vectors (that is, vectors that contain only zeros and a
single one at the location of the target value to be encoded).

The `value()` method has no parameters and returns the confusion matrix in a
KxK tensor. In the confusion matrix, rows correspond to ground-truth targets and
columns correspond to predicted targets.
]],
   noordered = true,
   {name="self", type="tnt.MultiLabelConfusionMeter"},
   {name="k", type="number"},
   {name="normalized", type="boolean", default=true},
   call =
      function(self, k, normalized)
         self.conf = torch.DoubleTensor(k, k)
         self.normalized = normalized
         self:reset()
      end
}

MultiLabelConfusionMeter.reset = argcheck{
   {name="self", type="tnt.MultiLabelConfusionMeter"},
   call =
      function(self)
         self.conf:zero()
      end
}

MultiLabelConfusionMeter.add = argcheck{
   {name="self", type="tnt.MultiLabelConfusionMeter"},
   {name="output", type="torch.*Tensor"},
   {name="target", type="torch.*Tensor"},
   call =
      function(self, output, target)
         target = target:squeeze()
         output = output:squeeze()
         if output:nDimension() == 1 then
            output = output:view(1, output:size(1))
         end
         if target:nDimension() == 1 then
            target = target:view(1, output:size(1))
         end
         assert(
            target:nDimension() == output:nDimension() and
            torch.eq(
               torch.LongTensor(target:size()),
               torch.LongTensor(output:size())
            ):all(),
            'number of targets and outputs do not match'
         )
         assert(
            torch.eq(torch.eq(target, 0):add(torch.eq(target, 1)), 1):all(),
            'target values should be 0 or 1'
         )
         assert(
            target:size(2) == self.conf:size(1),
            'target size does not match size of confusion matrix'
         )

         -- update confusion matrix:
         local nc = output:size(2)
         local _,pred = output:double():sort(2, true)
         for n = 1,pred:size(1) do

            -- convert targets and predictions to sets:
            local targetTable, predTable = tds.hash(), tds.hash()
            local pos = torch.range(1, nc)[torch.eq(target[n], 1)]
            for k = 1,pos:nElement() do
               targetTable[pos[k]]   = 1
               predTable[pred[n][k]] = 1
            end

            -- loop over correct predictions:
            for key,_ in pairs(targetTable) do
               if predTable[key] then
                  self.conf[key][key] = self.conf[key][key] + 1
                  targetTable[key] = nil
                  predTable[key]   = nil
               end
            end

            -- equally distribute mass of incorrect predictions:
            local weight = 1 / #predTable
            for key1,_ in pairs(targetTable) do
               for key2,_ in pairs(predTable) do
                  self.conf[key1][key2] = self.conf[key1][key2] + weight
               end
            end
         end
      end
}

MultiLabelConfusionMeter.value = argcheck{
   {name="self", type="tnt.MultiLabelConfusionMeter"},
   call =
      function(self)
         local conf = torch.DoubleTensor(self.conf:size()):copy(self.conf)
         if self.normalized then
            conf:cdiv(conf:sum(2):expandAs(conf):add(1e-8))
         end
         return conf
      end
}

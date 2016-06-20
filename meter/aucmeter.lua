--[[
   Copyright (c) 2016-present, Facebook, Inc.
   All rights reserved.

   This source code is licensed under the BSD-style license found in the
   LICENSE file in the root directory of this source tree. An additional grant
   of patent rights can be found in the PATENTS file in the same directory.
]]--

local tnt = require 'torchnet.env'
local argcheck = require 'argcheck'

local AUCMeter = torch.class('tnt.AUCMeter', 'tnt.Meter', tnt)

AUCMeter.__init = argcheck{
   doc = [[
<a name="AUCMeter">
#### tnt.AUCMeter(@ARGP)
@ARGT

The `tnt.AUCMeter` measures the area under the receiver-operating characteristic
(ROC) curve for binary classification problems. The area under the curve (AUC)
can be interpreted as the probability that, given a randomly selected positive
example and a randomly selected negative example, the positive example is
assigned a higher score by the classification model than the negative example.

The `tnt.AUCMeter` is designed to operate on one-dimensional Tensors `output`
and `target`, where (1) the `output` contains model output scores that ought to
be higher when the model is more convinced that the example should be positively
labeled, and smaller when the model believes the example should be negatively
labeled (for instance, the output of a signoid function); and (2) the `target`
contains only values 0 (for negative examples) and 1 (for positive examples).

The `tnt.AUCMeter` has no parameters to be set.
]],
   {name="self", type="tnt.AUCMeter"},
   call =
      function(self)
         self:reset()
      end
}

AUCMeter.reset = argcheck{
   {name="self", type="tnt.AUCMeter"},
   call =
      function(self)
         self.scores  = torch.DoubleTensor()
         self.targets = torch.LongTensor()
      end
}

AUCMeter.add = argcheck{
   {name="self", type="tnt.AUCMeter"},
   {name="output", type="torch.*Tensor"},
   {name="target", type="torch.*Tensor"},
   call =
      function(self, output, target)
         target = target:squeeze()
         output = output:squeeze()
         assert(
            output:nDimension() == 1,
            'dimensionality of output should be 1 (e.g., nn.Sigmoid output)'
         )
         assert(
            target:nDimension() == 1,
            'dimensionality of targets should be 1'
         )
         assert(
            output:size(1) == target:size(1),
            'number of outputs and targets does not match'
         )
         assert(
            torch.eq(torch.eq(target, 0):add(torch.eq(target, 1)), 1):all(),
            'targets should be binary (0 or 1)'
         )

         -- store scores and targets in storage:
         local offset1 = self.scores:nElement()
         local offset2 = self.targets:nElement()
         self.scores:resize( offset1 + output:nElement())
         self.targets:resize(offset2 + target:nElement())
         self.scores:narrow( 1, offset1 + 1, output:nElement()):copy(output)
         self.targets:narrow(1, offset2 + 1, target:nElement()):copy(target)
      end
}

AUCMeter.value = argcheck{
   {name="self", type="tnt.AUCMeter"},
   call =
      function(self)

         -- sort the scores:
         local scores, sortind = torch.sort(self.scores, 1, true)

         -- construct the ROC curve:
         local tpr = torch.DoubleTensor(scores:nElement() + 1):zero()
         local fpr = torch.DoubleTensor(scores:nElement() + 1):zero()
         for n = 2,scores:nElement() + 1 do
            if self.targets[sortind[n - 1]] == 1 then
               tpr[n], fpr[n] = tpr[n - 1] + 1, fpr[n - 1]
            else
               tpr[n], fpr[n] = tpr[n - 1], fpr[n - 1] + 1
            end
         end
         tpr:div(self.targets:sum())
         fpr:div(torch.mul(self.targets, -1):add(1):sum())

         -- compute AUC:
         local auc = torch.div(tpr, fpr:nElement()):sum()

         -- return AUC and ROC curve:
         return auc, tpr, fpr
      end
}

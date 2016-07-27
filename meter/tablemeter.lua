--[[
   Copyright (c) 2016-present, Facebook, Inc.
   All rights reserved.

   This source code is licensed under the BSD-style license found in the
   LICENSE file in the root directory of this source tree. An additional grant
   of patent rights can be found in the PATENTS file in the same directory.
]]--

local tnt = require 'torchnet.env'
local argcheck = require 'argcheck'

local TableMeter = torch.class('tnt.TableMeter', 'tnt.Meter', tnt)

TableMeter.__init = argcheck{
   doc = [[
<a name="TableMeter">
#### tnt.TableMeter(@ARGP)
@ARGT

The `tnt.TableMeter` allows you to take in outputs from a `nn.ConcatTable` construct
that instead of a tensor returns a table of tensors. This is useful when working with
multilabel classification tasks where there may be a varying number of outputs.

If `k` is omitted then the meters will be created at the first `add` call

]],
   noordered = true,
   {name="self", type="tnt.TableMeter"},
   {name="k", type="number", opt=true,
    doc="The number of subelements to the `nn.ConcatTable`, i.e. table length."},
   {name="class", type="table",
    doc="A class for the meter that should be applied to each table element, e.g. tnt.AverageValueMeter"},
   {name="classargs", type="table", default={},
    doc="Arguments for the meter class"},
   call = function(self, k, class, classargs)
      self.meters = {}
      self.class = class
      self.classargs = classargs

      if (k) then
         self:_createMeters(k)
      end
   end
}

TableMeter._createMeters = argcheck{
   {name="self", type="tnt.TableMeter"},
   {name="k", type="number"},
   call=function(self, k)
      assert(k > 0, "The number of meters must be positive")

      for i=1,k do
         -- Named arguments for consructor then classargs[1] is nil
         if (self.classargs[1] == nil) then
            self.meters[i] = self.class(self.classargs)
         elseif(unpack) then
            -- Hack for Lua version compatibility
            self.meters[i] = self.class(unpack(self.classargs))
         else
            self.meters[i] = self.class(table.unpack(self.classargs))
         end
      end
   end
}

TableMeter.reset = argcheck{
   {name="self", type="tnt.TableMeter"},
   call = function(self)
      for i=1,#self.meters do
         self.meters[i]:reset()
      end
   end
}

TableMeter.add = argcheck{
   {name="self", type="tnt.TableMeter"},
   {name="output", type="table"},
   {name="target", type="torch.*Tensor"},
   call = function(self, output, target)
      assert(#output == target:size(2),
            ([[The output  length (%d) doesn't match the length of the tensor's
            second dimension (%d). The first dimension in the target should be
            the batch size for tensors.]]):format(#output, target:size(2)))

      local table_target = {}
      for i=1,#output do
         table_target[i] = target[{{},{i}}]:squeeze():clone()
      end

      return self:add(output, table_target)
   end
}


TableMeter.add = argcheck{
   {name="self", type="tnt.TableMeter"},
   {name="output", type="table"},
   {name="target", type="table"},
   overload=TableMeter.add,
   call = function(self, output, target)
      assert(#output == #target,
             ("The output size (%d) and the target (%d) don't match"):format(#output, #target))

      if (not self.meters[1]) then
         self:_createMeters(#output)
      end
      assert(#output == #self.meters,
            ("The output size (%d) and the number of meters that you've specified (%d) don't match"):format(#output, #target))

      for i=1,#self.meters do
         self.meters[i]:add(output[i], target[i])
      end

   end
}

TableMeter.value = argcheck{
   {name="self", type="tnt.TableMeter"},
   {name="k", type="number", opt=true},
   {name="parameters", type="table", opt=true,
      doc="Parameters that should be passed to the underlying meter"},
   call = function(self, k, parameters)

      -- Odd hack as argcheck seems to encapsulate parameters inside its own table
      if (parameters and parameters.parameters) then
         parameters = parameters.parameters
      end

      if k then
         assert(self.meters[k],
               ('invalid k (%d), i.e. there is no output corresponding to this meter'):format(k))

         if (not parameters) then
            return self.meters[k]:value()
         elseif (parameters[1] == nil) then
            return self.meters[k]:value(parameters)
         elseif(unpack) then
            -- Hack for Lua version compatibility
            return self.meters[k]:value(unpack(parameters))
         else
            return self.meters[k]:value(table.unpack(parameters))
         end

      else
         local value = {}
         for meter_no=1,#self.meters do
            value[meter_no] = self:value(meter_no, parameters)
         end
         return value
      end
   end
}

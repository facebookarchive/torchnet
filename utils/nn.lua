--[[
   Copyright (c) 2016-present, Facebook, Inc.
   All rights reserved.

   This source code is licensed under the BSD-style license found in the
   LICENSE file in the root directory of this source tree. An additional grant
   of patent rights can be found in the PATENTS file in the same directory.
]]--

local argcheck = require 'argcheck'

local unn = {}

local function isz(okw, odw, kw, dw, pw)
   dw = dw or 1
   pw = pw or 0
   okw = okw*dw + kw-dw-pw*2
   odw = odw*dw
   return okw, odw
end

-- in specs, list of {kw=,dw=,pw=} (dw and pw optionals)
-- kw (kernel width)
-- dw (kernel stride)
-- pw (padding)
unn.inferinputsize = argcheck{
   {name="specs", type="table"},
   {name="size", type="number", default=1},
   {name="verbose", type="boolean", default=false},
   call =
      function(specs, size, verbose)
         local okw, odw = size, 1
         for i=#specs,1,-1 do
            if specs[i].kw then
               okw, odw = isz(okw, odw, specs[i].kw, specs[i].dw, specs[i].pw)
            end
            if verbose then
               print(string.format(
                        "|| layer %d: size=%d stride=%d",
                        i,
                        okw,
                        odw))
            end
         end
         return okw, odw
      end
}

local function iszr(okw, odw, kw, dw, pw)
   dw = dw or 1
   pw = pw or 0
   okw = math.floor((okw+2*pw-kw)/dw)+1
   odw = odw * dw
   return okw, odw
end

unn.inferoutputsize = argcheck{
   {name="specs", type="table"},
   {name="size", type="number"},
   {name="verbose", type="boolean", default=false},
   call =
      function(specs, size, verbose)
         local okw, odw = size, 1
         for i=1,#specs do
            okw, odw = iszr(okw, odw, specs[i].kw, specs[i].dw, specs[i].pw)
            if verbose then
               print(string.format("|| layer %d: size=%dx stride=%d", okw, odw))
            end
         end
         return okw, odw
      end
}

return unn

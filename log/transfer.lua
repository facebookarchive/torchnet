--[[
   Copyright (c) 2016-present, Facebook, Inc.
   All rights reserved.

   This source code is licensed under the BSD-style license found in the
   LICENSE file in the root directory of this source tree. An additional grant
   of patent rights can be found in the PATENTS file in the same directory.
]]--

local transfer = {}

function transfer.send(c, data)
   data = torch.serialize(data)
   c:send(string.format("0x%0.16x", #data))
   c:send(data)
end

function transfer.receive(c)
   local sz, err = c:receive(18)
   if err then
      return
   end
   sz = tonumber(sz)
   local data, err = c:receive(sz)
   if err then
      return
   end
   local status, data = pcall(torch.deserialize, data)
   if status then
      return data
   end
end

return transfer

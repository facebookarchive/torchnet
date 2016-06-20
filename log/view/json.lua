--[[
   Copyright (c) 2016-present, Facebook, Inc.
   All rights reserved.

   This source code is licensed under the BSD-style license found in the
   LICENSE file in the root directory of this source tree. An additional grant
   of patent rights can be found in the PATENTS file in the same directory.
]]--

local argcheck = require 'argcheck'

local json = argcheck{
   noordered=true,
   {name="filename", type="string", opt=true},
   {name="keys", type="table"},
   {name="format", type="table", opt=true},
   {name="append", type="boolean", default=false},
   call =
      function(filename, keys__, format__, append)
         local keys = {}
         for idx, key in ipairs(keys__) do
            local format = format__ and format__[idx]
            if not format then
               table.insert(keys, {name=key, format=function(value) return string.format("%s %s", key, value) end})
            elseif type(format) == 'function' then
               table.insert(keys, {name=key, format=format})
            elseif type(format) == 'string' then
               table.insert(keys, {name=key, format=function(value) return string.format(format, value) end})
            else
               error('format must be a string or a function')
            end
         end
         if filename and not append then
            local f = io.open(filename, 'w') -- reset the file
            assert(f, string.format("could not open file <%s> for writing", filename))
            f:close()
         end
         return function(log)
            local txt = {}
            for _, key in ipairs(keys) do
               local format = key.format(log:get(key.name))
               assert(type(format) == 'string', string.format("value for key %s cannot be converted to string", key))
               table.insert(txt, string.format('"%s": "%s"', key.name, format))
            end
            txt = string.format("{%s}", table.concat(txt, ", "))
            if filename then
               local f = io.open(filename, 'a+') -- append
               assert(f, string.format("could not open file <%s> for writing", filename))
               f:write(txt)
               f:write("\n")
               f:close()
            else
               print(txt)
            end
         end
      end
}

return json

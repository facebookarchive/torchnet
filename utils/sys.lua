--[[
   Copyright (c) 2016-present, Facebook, Inc.
   All rights reserved.

   This source code is licensed under the BSD-style license found in the
   LICENSE file in the root directory of this source tree. An additional grant
   of patent rights can be found in the PATENTS file in the same directory.
]]--

local md5 = require 'md5'
local lfs = require 'lfs'
local tds = require 'tds'

local sys = {}

function sys.md5(obj)
   local str = torch.serialize(obj)
   return md5.sumhexa(str)
end

function sys.mkdir(path)
   assert(
      os.execute(string.format('mkdir -p %s', path)),
      'could not create directory'
   )
end

local function cmdlinecode(code, env)
   assert(type(code) == 'string', 'string expected')
   local msg
   if loadstring then -- lua 5.1
      code, msg = loadstring(code)
      if code then
         setfenv(code, env)
      end
   else
      code, msg = load(code, nil, nil, env) -- lua 5.2
   end
   if not code then
      error(string.format('compilation error: %s', msg))
   end
   assert(not getmetatable(env), 'env should have no metatable')
   setmetatable(env, {__index=_G})
   local status, msg = pcall(code)
   setmetatable(env, nil)
   if not status then
      error(msg)
   end
end

function sys.cmdline(arg, env)
   for _, code in ipairs(arg) do
      cmdlinecode(code, env)
   end
end

function sys.loadlist(path, revert, maxload)
   local lst = tds.hash()
   local idx = 0
   for elem in io.lines(path) do
      idx = idx + 1
      lst[idx] = elem
      if revert then
         lst[elem] = idx
      end
      if maxload and maxload == idx then
         break
      end
   end
   return lst
end

function sys.listimgfiles(path, lst)
   lst = lst or tds.hash()
   for filename in lfs.dir(path) do
      if filename ~= '.' and filename ~= '..' then
         local fullpath = string.format('%s/%s', path, filename)
         if lfs.attributes(fullpath, 'mode') == 'directory' then
            sys.listimgfiles(fullpath, lst)
         else
            local ext = filename:match('[^%.]+$')
            if ext then
               ext = ext:lower()
               if ext == 'jpeg' or ext == 'jpg' or ext == 'png' then
                  lst[#lst+1] = fullpath
               else
                  print(string.format('ignoring <%s>', fullpath))
               end
            end
         end
      end
   end
   return lst
end

return sys
